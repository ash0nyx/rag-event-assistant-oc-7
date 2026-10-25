"""Headless tests of the Streamlit pages with streamlit.testing.v1.AppTest.

AppTest runs a page script in-process, without a browser: we can check what
it renders and click its widgets. The API is faked by monkeypatching
`requests`; the dashboard reads a temporary SQLite database.
"""

from pathlib import Path

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from api.database import InteractionStore

ROOT = Path(__file__).resolve().parent.parent  # paths are resolved against this file
CHAT = str(ROOT / "ui" / "Chat.py")
DASHBOARD = str(ROOT / "ui" / "pages" / "1_Tableau_de_bord.py")


class FakeResponse:
    def __init__(self, payload, ok=True):
        self._payload, self.ok = payload, ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


@pytest.fixture
def fake_api(monkeypatch):
    """Records POSTs and answers like the real API."""
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], json))
        if url.endswith("/ask"):
            return FakeResponse({
                "question": json["question"], "answer": "Je vous recommande Expo A.", "intent": "events",
                "interaction_id": 7, "latency_ms": 1500.0,
                "sources": [{"uid": 1, "title": "Expo A", "date_range": "1 - 2 octobre", "venue": "Grand Palais",
                             "url": "https://openagenda.com/fr/que-faire-a-paris/events/expo-a"}],
            })
        return FakeResponse({"interaction_id": 7, "score": json["score"], "comment": json.get("comment")})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: FakeResponse({"status": "ok"}))
    return calls


def test_chat_shows_answer_sources_and_sends_a_vote(fake_api):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    assert not at.exception

    at.chat_input[0].set_value("Une expo ?").run()

    assert ("ask", {"question": "Une expo ?"}) in fake_api
    assert any("Je vous recommande Expo A." in m.value for m in at.markdown)
    assert any("Expo A" in m.value and "openagenda.com" in m.value for m in at.markdown)  # source link

    at.feedback[0].set_value(1).run()  # thumbs up
    assert ("feedback", {"interaction_id": 7, "score": 1, "comment": None}) in fake_api


def test_chat_down_vote_offers_a_comment_box(fake_api):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.chat_input[0].set_value("Une expo ?").run()
    at.feedback[0].set_value(0).run()  # thumbs down

    assert ("feedback", {"interaction_id": 7, "score": -1, "comment": None}) in fake_api
    # A form sends its fields only on submit: type and click in the same run.
    at.text_input[0].set_value("Mauvaise date")
    at.button(key="FormSubmitter:comment_7-Envoyer").click().run()
    assert ("feedback", {"interaction_id": 7, "score": -1, "comment": "Mauvaise date"}) in fake_api


def test_chat_reports_an_unreachable_api(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError

    monkeypatch.setattr(requests, "post", down)
    monkeypatch.setattr(requests, "get", down)
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.chat_input[0].set_value("Une expo ?").run()
    assert not at.exception
    assert any("L'API ne répond pas" in m.value for m in at.markdown)


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/interactions.db"
    monkeypatch.setenv("INTERACTIONS_DB_URL", url)
    st.cache_data.clear()  # the dashboard caches its query for 30 s
    return url


def test_dashboard_with_no_data_says_so(db_url):
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception
    assert "Aucune interaction" in at.info[0].value


def test_dashboard_shows_kpis_and_negative_feedback(db_url):
    store = InteractionStore(db_url)
    for q, intent, score in [("Une expo ?", "events", 1), ("Un concert ?", "events", -1), ("Météo ?", "off_topic", None)]:
        i = store.log_interaction(question=q, intent=intent, answer="Réponse", sources=[], latency_ms=2000.0)
        if score:
            store.log_feedback(i, score, "Mauvaise date" if score == -1 else None)

    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Questions"] == "3"
    assert metrics["Satisfaction"] == "50%"
    assert metrics["Latence médiane (RAG)"] == "2.0 s"
    assert any("Un concert ?" in e.label for e in at.expander)  # the negative one is listed


def test_example_buttons_ask_the_question_and_history_shows_no_stray_values(fake_api):
    """Click a suggestion, then rerun with a second question: earlier messages
    are re-rendered from history and must not print stray values like None."""
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.button(key="main_example_3").click().run()  # "Quel temps fera-t-il demain ?"
    assert fake_api[0] == ("ask", {"question": "Quel temps fera-t-il demain ?"})

    at.chat_input[0].set_value("Bonjour").run()

    # Streamlit magic renders a stray None as the markdown `None`.
    assert not [m.value for m in at.markdown if "None" in m.value]
    assert len([m for m in at.chat_message]) == 4  # 2 questions + 2 answers, each once


def test_examples_stay_clickable_after_a_first_question(fake_api):
    """Your scenario: ask a question, then click an example in the sidebar.
    It starts a fresh conversation with that question and its answer."""
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.chat_input[0].set_value("Bonjour").run()
    assert not [b for b in at.button if b.key and b.key.startswith("main_example_")]  # hidden once a question is asked

    at.button(key="side_example_2").click().run()

    assert fake_api[-1] == ("ask", {"question": "Une exposition de photographie en novembre ?"})
    # A fresh conversation: only the example and its answer remain.
    assert len(at.chat_message) == 2
    assert at.chat_message[0].markdown[0].value == "Une exposition de photographie en novembre ?"
