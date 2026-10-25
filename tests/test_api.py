"""Unit tests for api/main.py.

FastAPI's TestClient calls the app in-process (no server, no network).
The real assistant is never built: `get_assistant` is overridden with a fake,
and the lifespan (which would load FAISS + Mistral) is not triggered because
we do not enter the client as a context manager.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.database import InteractionStore
from rag.chain import RagAnswer


class FakeAssistant:
    """Records the question and returns a canned answer with one source."""

    def __init__(self):
        self.questions: list[str] = []

    def ask(self, question: str) -> RagAnswer:
        self.questions.append(question)
        return RagAnswer(
            question=question,
            answer="Je vous recommande Expo A.",
            sources=[{
                "uid": 1,
                "title": "Expo A",
                "date_range": "1 - 2 octobre",
                "venue": "Grand Palais",
                "url": "https://openagenda.com/fr/que-faire-a-paris/events/expo-a",
            }],
        )


@pytest.fixture
def store(tmp_path):
    """A real SQLite database in a temporary folder: tests the actual SQL,
    not a mock of it, and leaves nothing behind."""
    return InteractionStore(f"sqlite:///{tmp_path}/interactions.db")


@pytest.fixture
def client(store):
    fake = FakeAssistant()
    api_main.app.dependency_overrides[api_main.get_assistant] = lambda: fake
    api_main.app.dependency_overrides[api_main.get_store] = lambda: store
    yield TestClient(api_main.app), fake
    api_main.app.dependency_overrides.clear()


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    test_client, fake = client
    response = test_client.post("/ask", json={"question": "  Une expo ?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Je vous recommande Expo A."
    assert body["sources"][0]["title"] == "Expo A"
    assert body["sources"][0]["url"].endswith("/expo-a")
    assert fake.questions == ["Une expo ?"]  # stripped before reaching the chain


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "ab"}, {"question": "   "}])
def test_ask_rejects_empty_or_too_short_question(client, payload):
    test_client, fake = client
    response = test_client.post("/ask", json=payload)
    assert response.status_code == 422
    assert fake.questions == []  # the chain was never called


def test_ask_rejects_malformed_body(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"query": "Une expo ?"})
    assert response.status_code == 422


def test_rebuild_requires_api_key(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setenv("RAG_API_KEY", "secret")

    assert test_client.post("/rebuild").status_code == 401
    assert test_client.post("/rebuild", headers={"X-API-Key": "wrong"}).status_code == 401


def test_rebuild_is_closed_when_no_key_configured(client, monkeypatch):
    test_client, _ = client
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    assert test_client.post("/rebuild", headers={"X-API-Key": "anything"}).status_code == 401


def test_rebuild_runs_pipeline_and_reloads_assistant(client, monkeypatch):
    """With the right key, /rebuild calls the pipeline (faked here) and
    replaces the in-memory assistant with a fresh one."""
    test_client, _ = client
    monkeypatch.setenv("RAG_API_KEY", "secret")

    calls: list[dict] = []
    def fake_rebuild(max_events, since, until):
        calls.append({"max_events": max_events, "since": since, "until": until})
        return {"fetched": 50, "cleaned": 48, "chunks": 120, "vectors": 120}

    monkeypatch.setattr(api_main, "rebuild_index", fake_rebuild)
    monkeypatch.setattr(api_main, "build_assistant", lambda: "fresh-assistant")

    body = {"max_events": 50, "from_date": "2026-09-25", "to_date": "2026-10-31"}
    response = test_client.post("/rebuild", json=body, headers={"X-API-Key": "secret"})

    assert response.status_code == 200
    assert response.json() == {"fetched": 50, "cleaned": 48, "chunks": 120, "vectors": 120}
    assert calls == [{"max_events": 50, "since": date(2026, 9, 25), "until": date(2026, 10, 31)}]
    assert api_main.app.state.assistant == "fresh-assistant"


def test_rebuild_defaults_to_today_plus_60_days(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setenv("RAG_API_KEY", "secret")
    calls: list[dict] = []
    monkeypatch.setattr(
        api_main, "rebuild_index",
        lambda max_events, since, until: calls.append({"since": since, "until": until})
        or {"fetched": 0, "cleaned": 0, "chunks": 0, "vectors": 0},
    )
    monkeypatch.setattr(api_main, "build_assistant", lambda: "fresh-assistant")

    assert test_client.post("/rebuild", headers={"X-API-Key": "secret"}).status_code == 200
    assert calls[0]["since"] == date.today()
    assert calls[0]["until"] == date.today() + timedelta(days=60)


def test_rebuild_rejects_reversed_dates(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setenv("RAG_API_KEY", "secret")
    body = {"from_date": "2026-10-31", "to_date": "2026-09-25"}
    response = test_client.post("/rebuild", json=body, headers={"X-API-Key": "secret"})
    assert response.status_code == 422
    assert api_main.app.state.assistant == "fresh-assistant"


# --- Logging and feedback -----------------------------------------------------


def test_ask_logs_the_interaction_and_returns_its_id(client, store):
    test_client, _ = client
    body = test_client.post("/ask", json={"question": "Une expo ?"}).json()

    assert body["intent"] == "events"
    assert isinstance(body["interaction_id"], int)
    assert body["latency_ms"] >= 0
    rows = store.list_interactions()
    assert len(rows) == 1
    assert rows[0]["id"] == body["interaction_id"]
    assert rows[0]["question"] == "Une expo ?"
    assert rows[0]["sources"][0]["title"] == "Expo A"
    assert rows[0]["feedback_score"] is None


def test_feedback_is_stored_on_the_interaction(client, store):
    test_client, _ = client
    interaction_id = test_client.post("/ask", json={"question": "Une expo ?"}).json()["interaction_id"]

    response = test_client.post(
        "/feedback", json={"interaction_id": interaction_id, "score": -1, "comment": "  Mauvaise date  "}
    )

    assert response.status_code == 200
    assert response.json() == {"interaction_id": interaction_id, "score": -1, "comment": "Mauvaise date"}
    row = store.list_interactions(feedback="negative")[0]
    assert row["feedback_comment"] == "Mauvaise date"
    assert row["feedback_at"] is not None


def test_feedback_on_unknown_interaction_is_404(client):
    test_client, _ = client
    response = test_client.post("/feedback", json={"interaction_id": 999, "score": 1})
    assert response.status_code == 404


@pytest.mark.parametrize("score", [0, 2, "yes"])
def test_feedback_rejects_scores_other_than_plus_or_minus_one(client, score):
    test_client, _ = client
    response = test_client.post("/feedback", json={"interaction_id": 1, "score": score})
    assert response.status_code == 422


def test_answer_is_returned_even_if_logging_fails(client, store, monkeypatch):
    test_client, _ = client

    def broken(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "log_interaction", broken)
    response = test_client.post("/ask", json={"question": "Une expo ?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Je vous recommande Expo A."
    assert response.json()["interaction_id"] is None
