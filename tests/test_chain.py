"""Unit tests for rag/chain.py.

No Mistral calls. Two fakes:
  - FakeVectorStore: returns a scripted list of chunks for any question.
  - FakeLLM: a LangChain chat model that echoes the prompt it received,
    so tests can check what the LLM was actually asked.
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from datetime import date

from rag.chain import (
    RagAssistant,
    dedupe_by_event,
    describe_period,
    describe_today,
    drop_past_events,
    format_context,
)


def make_doc(uid: int, title: str, chunk_index: int = 0) -> Document:
    """A chunk shaped like the ones build_index.py produces."""
    return Document(
        page_content=f"Événement : {title}\nDates : 1 - 2 octobre\n\nDescription du chunk {chunk_index}.",
        metadata={
            "uid": uid,
            "title": title,
            "date_range": "1 - 2 octobre",
            "venue": "Lieu test",
            "url": f"https://openagenda.com/fr/que-faire-a-paris/events/event-{uid}",
            "chunk_index": chunk_index,
        },
    )


class FakeVectorStore:
    """Stands in for FAISS: ignores the question, returns the scripted hits
    and remembers the k it was asked for."""

    def __init__(self, hits: list[Document]):
        self.hits = hits
        self.last_k: int | None = None

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        self.last_k = k
        return self.hits[:k]


def test_dedupe_keeps_first_chunk_per_event_and_caps_count():
    hits = [
        make_doc(1, "Expo A", chunk_index=2),  # best-ranked chunk of event 1
        make_doc(1, "Expo A", chunk_index=0),  # same event: dropped
        make_doc(2, "Concert B"),
        make_doc(3, "Atelier C"),
        make_doc(4, "Spectacle D"),  # beyond max_events: dropped
    ]
    kept = dedupe_by_event(hits, max_events=3)
    assert [d.metadata["uid"] for d in kept] == [1, 2, 3]
    assert kept[0].metadata["chunk_index"] == 2  # the first (best) one survived


def test_describe_today_computes_the_coming_weekend_in_french():
    """Python, not the LLM, works out which days the weekend is."""
    thursday = describe_today(date(2026, 9, 24))
    assert thursday == (
        "Nous sommes le jeudi 24 septembre 2026. "
        "Ce week-end désigne uniquement le samedi 26 septembre 2026 et le dimanche 27 septembre 2026. "
        "La saison en cours est l'automne (du mardi 22 septembre 2026 au dimanche 20 décembre 2026)."
    )
    # On a Saturday, the weekend is today and tomorrow.
    assert "samedi 26 septembre 2026 et le dimanche 27 septembre 2026" in describe_today(date(2026, 9, 26))
    # On a Sunday, the weekend is yesterday and today, not next week's.
    assert "samedi 26 septembre 2026 et le dimanche 27 septembre 2026" in describe_today(date(2026, 9, 27))
    # On a Monday, it is the coming one.
    assert "samedi 3 octobre 2026 et le dimanche 4 octobre 2026" in describe_today(date(2026, 9, 28))


def test_needs_calendar_detects_relative_time_expressions():
    from rag.chain import needs_calendar

    assert needs_calendar("Quels concerts de jazz ce week-end ?")
    assert needs_calendar("Un spectacle de danse en novembre ?")
    assert needs_calendar("Un festival de cinéma cet automne ?")
    assert needs_calendar("Une expo demain ?")
    assert not needs_calendar("Un atelier gratuit pour les enfants le mercredi ?")
    assert not needs_calendar("Une pièce de théâtre pour adolescents ?")


def test_describe_today_omits_weekend_for_undated_questions():
    text = describe_today(date(2026, 9, 24), "Une pièce de théâtre pour adolescents ?")
    assert text == (
        "Nous sommes le jeudi 24 septembre 2026. "
        "La question ne précise pas de période : tous les événements du contexte conviennent."
    )
    dated = describe_today(date(2026, 9, 24), "Quels concerts ce week-end ?")
    assert "samedi 26 septembre 2026" in dated and "automne" in dated


def test_current_season_covers_every_date():
    from rag.chain import current_season

    assert current_season(date(2026, 9, 24))[0] == "automne"
    assert current_season(date(2026, 12, 25)) == ("hiver", date(2026, 12, 21), date(2027, 3, 19))
    assert current_season(date(2026, 1, 5)) == ("hiver", date(2025, 12, 21), date(2026, 3, 19))
    assert current_season(date(2026, 4, 1))[0] == "printemps"
    assert current_season(date(2026, 7, 14))[0] == "été"


def test_drop_past_events_removes_finished_ones_only():
    """Python filters by date; the LLM is never asked to ignore past events."""
    over = make_doc(1, "Concert passé")
    over.metadata["last_end"] = "2026-09-26T23:00:00+02:00"
    today_event = make_doc(2, "Concert ce soir")
    today_event.metadata["last_end"] = "2026-09-27T22:00:00+02:00"  # same day: kept
    future = make_doc(3, "Concert futur")
    future.metadata["last_end"] = "2026-10-04T22:00:00+02:00"
    unknown = make_doc(4, "Sans date")  # no last_end: kept, better safe than silent
    broken = make_doc(5, "Date cassée")
    broken.metadata["last_end"] = "n/a"

    kept = drop_past_events([over, today_event, future, unknown, broken], today=date(2026, 9, 27))
    assert [d.metadata["uid"] for d in kept] == [2, 3, 4, 5]


def test_retrieve_filters_past_events_before_dedup():
    past = make_doc(1, "Ancien")
    past.metadata["last_end"] = "2026-01-01T00:00:00+01:00"
    store = FakeVectorStore([past, make_doc(2, "Actuel")])
    assistant = RagAssistant(store, FakeListChatModel(responses=["ok"]))
    docs = assistant.retrieve("q", today=date(2026, 9, 27))
    assert [d.metadata["uid"] for d in docs] == [2]


def test_describe_period_lists_covered_months():
    """Python lists the months so the LLM never has to infer that a range
    like September to January includes October."""
    text = describe_period("2026-09-29T10:00:00+02:00", "2027-01-31T18:00:00+01:00")
    assert text.startswith("Période : du mardi 29 septembre 2026 au dimanche 31 janvier 2027")
    assert "octobre 2026, novembre 2026, décembre 2026, janvier 2027" in text
    assert describe_period(None, "2026-10-01") is None
    assert describe_period("not-a-date", "2026-10-01") is None


def test_format_context_numbers_the_events_and_adds_period():
    docs = [make_doc(1, "Expo A"), make_doc(2, "Concert B")]
    docs[0].metadata.update(first_begin="2026-10-01T10:00:00+02:00", last_end="2026-10-02T18:00:00+02:00")
    text = format_context(docs)
    assert text.startswith("[Événement 1]\nPériode : du jeudi 1 octobre 2026 au vendredi 2 octobre 2026")
    assert "Événement : Expo A" in text
    assert "[Événement 2]\nÉvénement : Concert B" in text  # no timestamps: no period line


def test_ask_returns_answer_and_sources():
    store = FakeVectorStore([make_doc(1, "Expo A"), make_doc(1, "Expo A", 1), make_doc(2, "Concert B")])
    llm = FakeListChatModel(responses=["Je vous recommande Expo A."])
    assistant = RagAssistant(store, llm, top_k_chunks=20, max_events=5)

    result = assistant.ask("Une expo ?")

    assert result.question == "Une expo ?"
    assert result.answer == "Je vous recommande Expo A."
    assert store.last_k == 20  # retrieval asked FAISS for top_k_chunks
    assert [s["uid"] for s in result.sources] == [1, 2]  # deduplicated, in rank order
    assert result.sources[0]["url"].endswith("/event-1")


def test_prompt_contains_rules_context_and_question():
    """Check what the LLM is really asked: the rules, the retrieved chunks
    and the user's question all end up in the prompt."""
    store = FakeVectorStore([make_doc(1, "Expo A")])
    llm = FakeListChatModel(responses=["ok"])
    assistant = RagAssistant(store, llm)

    # Render the prompt exactly as the chain does, without calling a model.
    docs = assistant.retrieve("Une expo de peinture ?")
    messages = assistant.chain.first.format_messages(
        context=format_context(docs),
        question="Une expo de peinture ?",
        today=describe_today(date(2026, 9, 24), "Une expo de peinture ?"),
    )

    system, human = messages
    assert "uniquement à partir des événements fournis" in system.content
    assert "Nous sommes le jeudi 24 septembre 2026" in system.content  # the LLM has no clock
    assert "Événement : Expo A" in system.content
    assert human.content == "Une expo de peinture ?"


def test_generate_retries_on_rate_limit(monkeypatch):
    """Two 429s then success: the answer comes back after two sleeps."""
    import httpx

    from rag import chain as chain_module

    store = FakeVectorStore([make_doc(1, "Expo A")])
    assistant = RagAssistant(store, FakeListChatModel(responses=["ok"]))

    calls: list[int] = []
    sleeps: list[float] = []

    class RateLimitedChain:
        """Replaces the LCEL chain: fails twice with 429, then answers."""

        def invoke(self, inputs, config=None):
            calls.append(1)
            if len(calls) < 3:
                resp = httpx.Response(429, request=httpx.Request("POST", "https://api.mistral.ai"))
                raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
            return "réponse"

    assistant.chain = RateLimitedChain()
    monkeypatch.setattr(chain_module.time, "sleep", sleeps.append)  # no real waiting

    assert assistant.generate("q", assistant.retrieve("q")) == "réponse"
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]  # exponential backoff


def test_tracing_is_off_without_langfuse_keys(monkeypatch):
    """Without LANGFUSE_* keys the assistant must not even import langfuse."""
    from rag.chain import tracing_enabled

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing_enabled() is False
    assistant = RagAssistant(FakeVectorStore([make_doc(1, "Expo A")]), FakeListChatModel(responses=["ok"]))
    assert assistant.tracing is False and assistant._callbacks == []
    assert assistant.ask("q").answer == "ok"

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert tracing_enabled() is True
    # Explicit override wins, so tests and offline runs never hit the network.
    assert RagAssistant(FakeVectorStore([]), FakeListChatModel(responses=["ok"]), tracing=False).tracing is False


def test_ask_with_no_hits_still_answers():
    store = FakeVectorStore([])
    llm = FakeListChatModel(responses=["Aucun événement ne correspond."])
    result = RagAssistant(store, llm).ask("Un opéra ?")
    assert result.sources == []
    assert result.answer == "Aucun événement ne correspond."
