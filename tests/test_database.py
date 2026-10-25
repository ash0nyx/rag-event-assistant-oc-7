"""Unit tests for api/database.py, on a real SQLite file in a temp folder.

The course mocks SQLAlchemy's session; a throwaway SQLite database is just
as fast and also checks the SQL itself (filters, counts, JSON column).
"""

import pytest

from api.database import InteractionStore


@pytest.fixture
def store(tmp_path):
    return InteractionStore(f"sqlite:///{tmp_path}/test.db")


def log(store, question="Une expo ?", intent="events", latency_ms=1200.0):
    return store.log_interaction(
        question=question, intent=intent, answer="Réponse",
        sources=[{"uid": 1, "title": "Expo A"}], latency_ms=latency_ms, trace_id="abc",
    )


def test_log_interaction_returns_increasing_ids_and_keeps_fields(store):
    first, second = log(store), log(store, question="Un concert ?")
    assert second == first + 1
    rows = store.list_interactions()
    assert [r["question"] for r in rows] == ["Un concert ?", "Une expo ?"]  # newest first
    assert rows[0]["sources"] == [{"uid": 1, "title": "Expo A"}]  # JSON round-trip
    assert rows[0]["trace_id"] == "abc"


def test_log_feedback_updates_and_can_be_changed(store):
    interaction_id = log(store)
    store.log_feedback(interaction_id, 1)
    row = store.log_feedback(interaction_id, -1, "Mauvais lieu")  # user changed their mind
    assert row.feedback_score == -1
    assert row.feedback_comment == "Mauvais lieu"


def test_log_feedback_unknown_id_returns_none(store):
    assert store.log_feedback(42, 1) is None


def test_blank_comment_is_stored_as_none(store):
    row = store.log_feedback(log(store), 1, "   ")
    assert row.feedback_comment is None


def test_list_interactions_filters_by_feedback(store):
    up, down, unrated = log(store), log(store), log(store)
    store.log_feedback(up, 1)
    store.log_feedback(down, -1)
    assert [r["id"] for r in store.list_interactions(feedback="positive")] == [up]
    assert [r["id"] for r in store.list_interactions(feedback="negative")] == [down]
    assert [r["id"] for r in store.list_interactions(feedback="none")] == [unrated]
    assert len(store.list_interactions(limit=2)) == 2


def test_summary_counts_and_rates(store):
    assert store.summary()["satisfaction"] is None  # nothing rated yet: no division by zero
    ids = [log(store) for _ in range(3)] + [log(store, intent="off_topic")]
    store.log_feedback(ids[0], 1)
    store.log_feedback(ids[1], 1)
    store.log_feedback(ids[2], -1)

    summary = store.summary()
    assert summary["total"] == 4
    assert summary["rated"] == 3
    assert summary["satisfaction"] == pytest.approx(2 / 3)
    assert summary["feedback_rate"] == pytest.approx(3 / 4)
    assert summary["by_intent"] == {"events": 3, "off_topic": 1}
