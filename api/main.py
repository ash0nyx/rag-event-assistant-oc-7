"""REST API exposing the RAG assistant.

Endpoints:
    GET  /health   liveness check (used by Docker and the demo)
    POST /ask      question in, augmented answer + sources out (logged, with an id)
    POST /feedback thumbs up/down and optional comment on a logged answer
    POST /rebuild  re-fetch Open Agenda, rebuild the FAISS index, reload it
                   (protected by an X-API-Key header: it is slow and costs API calls)

The assistant (FAISS index + LLM) is loaded once at startup and kept in
`app.state`, not rebuilt per request. Business logic stays in rag/ and
scripts/; this file only does HTTP.

Run locally:
    uv run uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from api.database import InteractionStore
from rag.chain import INDEX_DIR, RagAssistant, build_assistant, flush_traces, push_feedback_score

warnings.filterwarnings("ignore")  # HF tokenizer notice, langchain-community sunset


# --- Request / response models (Pydantic validates input and documents Swagger) ---


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["Quels concerts de jazz ce week-end ?"],
        description="Question in natural language about cultural events in Paris.",
    )


class Source(BaseModel):
    uid: int
    title: str
    date_range: str | None = None
    venue: str | None = None
    url: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(description="Events retrieved from the index and given to the LLM.")
    intent: Literal["events", "chat", "off_topic"] = Field(
        description="events: answered by the RAG; chat / off_topic: fixed answer, no retrieval."
    )
    interaction_id: int | None = Field(description="Id to send back with POST /feedback. None if logging failed.")
    latency_ms: float


class FeedbackRequest(BaseModel):
    interaction_id: int
    score: Literal[1, -1] = Field(description="1 = thumbs up, -1 = thumbs down.")
    comment: str | None = Field(None, max_length=1000, examples=["La date de l'événement est fausse."])


class FeedbackResponse(BaseModel):
    interaction_id: int
    score: int
    comment: str | None


class RebuildRequest(BaseModel):
    max_events: int = Field(500, ge=1, le=3000, description="Events to fetch from Open Agenda.")
    from_date: date | None = Field(
        None,
        description="Keep events with a session on or after this date (YYYY-MM-DD). Default: today.",
        examples=["2026-09-25"],
    )
    to_date: date | None = Field(
        None,
        description="Keep events with a session on or before this date (YYYY-MM-DD). Default: from_date + 60 days.",
        examples=["2026-10-31"],
    )

    @model_validator(mode="after")
    def fill_defaults_and_check_order(self) -> "RebuildRequest":
        self.from_date = self.from_date or date.today()
        self.to_date = self.to_date or self.from_date + timedelta(days=60)
        if self.to_date < self.from_date:
            raise ValueError("to_date must be on or after from_date")
        return self


class RebuildResponse(BaseModel):
    fetched: int
    cleaned: int
    chunks: int
    vectors: int


# --- App lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once before the first request: load index + LLM into memory."""
    load_dotenv()
    app.state.assistant = build_assistant()
    app.state.store = InteractionStore()  # data/interactions.db by default
    yield
    flush_traces()  # on shutdown: push any Langfuse trace still in the buffer


app = FastAPI(
    title="Puls-Events RAG API",
    description="Assistant de recommandation d'événements culturels à Paris (RAG : FAISS + Mistral via LangChain).",
    version="0.3.0",
    lifespan=lifespan,
)


def get_assistant() -> RagAssistant:
    """Dependency: tests override this to inject a fake assistant."""
    return app.state.assistant


def get_store() -> InteractionStore:
    """Dependency: tests override this with a store on a temporary file."""
    return app.state.store


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dependency guarding /rebuild. The expected key comes from the environment;
    if none is configured, the endpoint is closed rather than open."""
    expected = os.environ.get("RAG_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header.")


# --- Routes ---


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(
    payload: AskRequest,
    assistant: RagAssistant = Depends(get_assistant),
    store: InteractionStore = Depends(get_store),
) -> AskResponse:
    """Answer a question using only the indexed events (RAG). Greetings and
    off-topic questions get a fixed answer without retrieval. Every call is
    logged; the returned interaction_id is what POST /feedback expects."""
    if not payload.question.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Question is empty.")
    start = time.perf_counter()
    result = assistant.ask(payload.question.strip())
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    try:
        interaction_id = store.log_interaction(
            question=result.question, intent=result.intent, answer=result.answer,
            sources=result.sources, latency_ms=latency_ms, trace_id=result.trace_id,
        )
    except Exception:
        # Logging must never cost the user their answer.
        logging.exception("could not log interaction")
        interaction_id = None
    return AskResponse(
        question=result.question, answer=result.answer, sources=result.sources,
        intent=result.intent, interaction_id=interaction_id, latency_ms=latency_ms,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["rag"])
def feedback(payload: FeedbackRequest, store: InteractionStore = Depends(get_store)) -> FeedbackResponse:
    """Record a thumbs up (1) or down (-1), with an optional comment, on an
    answer returned by /ask. Voting again on the same answer replaces the vote.
    Also sent to Langfuse as a score on the question's trace when tracing is on."""
    row = store.log_feedback(payload.interaction_id, payload.score, payload.comment)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Unknown interaction_id {payload.interaction_id}.")
    push_feedback_score(row.trace_id, payload.score, row.feedback_comment)
    return FeedbackResponse(interaction_id=row.id, score=row.feedback_score, comment=row.feedback_comment)


@app.post("/rebuild", response_model=RebuildResponse, tags=["ops"], dependencies=[Depends(require_api_key)])
def rebuild(payload: RebuildRequest | None = None) -> RebuildResponse:
    """Re-fetch Open Agenda, rebuild the FAISS index and reload it in memory.
    Requires the X-API-Key header. Takes a minute or two."""
    payload = payload or RebuildRequest()
    counts = rebuild_index(max_events=payload.max_events, since=payload.from_date, until=payload.to_date)
    app.state.assistant = build_assistant()  # swap in the fresh index
    return RebuildResponse(**counts)


def rebuild_index(max_events: int, since: date, until: date) -> dict:
    """The three pipeline scripts, called as functions. Kept separate from the
    route so it can be tested and reused without HTTP."""
    import json

    from scripts.build_index import INDEX_DIR as index_dir
    from scripts.build_index import PROCESSED_PATH, build_index, get_embeddings, load_events, make_documents
    from scripts.clean_events import clean_events
    from scripts.fetch_events import AGENDA_UID, RAW_PATH, fetch_events

    api_key = os.environ.get("OPENAGENDA_API_KEY")
    if not api_key:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OPENAGENDA_API_KEY is not configured.")

    events = fetch_events(api_key, AGENDA_UID, since, max_events, until=until)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps({"events": events}, ensure_ascii=False), encoding="utf-8")

    df = clean_events(events)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")

    documents = make_documents(load_events(PROCESSED_PATH))
    index = build_index(documents, get_embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_dir))

    return {"fetched": len(events), "cleaned": len(df), "chunks": len(documents), "vectors": index.index.ntotal}
