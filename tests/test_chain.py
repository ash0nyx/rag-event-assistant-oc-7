"""Unit tests for rag/chain.py.

No Mistral calls. Two fakes:
  - FakeVectorStore: returns a scripted list of chunks for any question.
  - FakeLLM: a LangChain chat model that echoes the prompt it received,
    so tests can check what the LLM was actually asked.
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from datetime import date

from rag.chain import RagAssistant, dedupe_by_event, describe_today, format_context


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
        "Le prochain week-end est le samedi 26 septembre 2026 et le dimanche 27 septembre 2026."
    )
    # On a Sunday, "this weekend" is today; the coming Saturday is 6 days away.
    assert "samedi 3 octobre 2026" in describe_today(date(2026, 9, 27))
    # On a Saturday, the weekend is today and tomorrow.
    assert "samedi 26 septembre 2026 et le dimanche 27 septembre 2026" in describe_today(date(2026, 9, 26))


def test_format_context_numbers_the_events():
    text = format_context([make_doc(1, "Expo A"), make_doc(2, "Concert B")])
    assert text.startswith("[Événement 1]\nÉvénement : Expo A")
    assert "[Événement 2]\nÉvénement : Concert B" in text


def test_ask_returns_answer_and_sources():
    store = FakeVectorStore([make_doc(1, "Expo A"), make_doc(1, "Expo A", 1), make_doc(2, "Concert B")])
    llm = FakeListChatModel(responses=["Je vous recommande Expo A."])
    assistant = RagAssistant(store, llm, top_k_chunks=10, max_events=5)

    result = assistant.ask("Une expo ?")

    assert result.question == "Une expo ?"
    assert result.answer == "Je vous recommande Expo A."
    assert store.last_k == 10  # retrieval asked FAISS for top_k_chunks
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
        today=describe_today(date(2026, 9, 24)),
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

        def invoke(self, inputs):
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


def test_ask_with_no_hits_still_answers():
    store = FakeVectorStore([])
    llm = FakeListChatModel(responses=["Aucun événement ne correspond."])
    result = RagAssistant(store, llm).ask("Un opéra ?")
    assert result.sources == []
    assert result.answer == "Aucun événement ne correspond."
