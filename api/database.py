"""Interaction log and user feedback, stored with SQLAlchemy.

Every /ask call is recorded (question, intent, answer, sources, latency,
Langfuse trace id). A later /feedback call attaches a thumbs up/down and an
optional comment to that row. The Streamlit dashboard reads the same table.

SQLite by default, in data/ so it lives on the Docker volume and survives
container restarts. The URL comes from INTERACTIONS_DB_URL; switching to
PostgreSQL in production is a change of URL, not of code.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DEFAULT_DB_URL = "sqlite:///data/interactions.db"


class Base(DeclarativeBase):
    pass


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    question: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(20))
    answer: Mapped[str] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 = 👍, -1 = 👎
    feedback_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def as_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class InteractionStore:
    """Small repository around the interactions table. A class rather than
    module-level functions so the API and the tests can each hold their own
    store (tests use a temporary SQLite file)."""

    def __init__(self, url: str | None = None):
        url = url or os.environ.get("INTERACTIONS_DB_URL", DEFAULT_DB_URL)
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI serves requests from a thread pool.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)  # idempotent: creates the table once

    def log_interaction(
        self,
        question: str,
        intent: str,
        answer: str,
        sources: list[dict],
        latency_ms: float,
        trace_id: str | None = None,
    ) -> int:
        with self.Session() as db:
            row = Interaction(
                question=question, intent=intent, answer=answer,
                sources=sources, latency_ms=latency_ms, trace_id=trace_id,
            )
            db.add(row)
            db.commit()
            return row.id

    def log_feedback(self, interaction_id: int, score: int, comment: str | None = None) -> Interaction | None:
        """Attach feedback to an interaction. Returns the row, or None when the
        id is unknown. A second vote on the same answer replaces the first."""
        with self.Session() as db:
            row = db.get(Interaction, interaction_id)
            if row is None:
                return None
            row.feedback_score = score
            row.feedback_comment = (comment or "").strip() or None
            row.feedback_at = datetime.now(timezone.utc)
            db.commit()
            return row

    def list_interactions(self, limit: int = 200, feedback: str | None = None) -> list[dict]:
        """Newest first. feedback: None (all), "positive", "negative", "none"."""
        query = select(Interaction).order_by(Interaction.created_at.desc()).limit(limit)
        if feedback == "positive":
            query = query.where(Interaction.feedback_score == 1)
        elif feedback == "negative":
            query = query.where(Interaction.feedback_score == -1)
        elif feedback == "none":
            query = query.where(Interaction.feedback_score.is_(None))
        with self.Session() as db:
            return [row.as_dict() for row in db.scalars(query)]

    def summary(self) -> dict:
        """Headline numbers for the dashboard."""
        with self.Session() as db:
            total = db.scalar(select(func.count(Interaction.id))) or 0
            positive = db.scalar(select(func.count()).where(Interaction.feedback_score == 1)) or 0
            negative = db.scalar(select(func.count()).where(Interaction.feedback_score == -1)) or 0
            by_intent = dict(db.execute(select(Interaction.intent, func.count()).group_by(Interaction.intent)).all())
        rated = positive + negative
        return {
            "total": total,
            "rated": rated,
            "positive": positive,
            "negative": negative,
            "satisfaction": positive / rated if rated else None,  # share of 👍 among rated answers
            "feedback_rate": rated / total if total else None,  # share of answers that got a vote
            "by_intent": by_intent,
        }
