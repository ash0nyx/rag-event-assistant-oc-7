"""The RAG chain: question -> retrieve chunks from FAISS -> prompt Mistral -> answer.

Two phases, kept separate on purpose (a classic source of confusion):
  1. Retrieval: embed the question, find the k closest chunks in FAISS.
     No LLM involved. Pure vector similarity.
  2. Generation: put those chunks in a prompt and ask the LLM to answer
     using only that context. The LLM never sees the whole database.

`RagAssistant` is a plain class so it can be used from a script
(scripts/ask_question.py), from the API, and from tests with fake models.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = "mistral-embed"
# Mistral's free tier does not enable mistral-small/medium (0 req/min); the
# ministral family is open. 14B is the strongest of those (30 req/min).
CHAT_MODEL = "ministral-14b-latest"
TOP_K_CHUNKS = 10  # chunks fetched from FAISS
MAX_EVENTS = 5  # distinct events kept after deduplication
RATE_LIMIT_RETRIES = 4  # Mistral free tier answers 429 when called too fast
RATE_LIMIT_WAIT = 2.0  # seconds; doubled after each failed attempt

# The prompt is the contract with the LLM. Key rules: answer only from the
# context, say so when the context has nothing, always give dates and venue.
SYSTEM_PROMPT = """Tu es l'assistant de Puls-Events. Tu recommandes des événements culturels à Paris.
{today}

Règles :
- Tiens compte de la date du jour pour interpréter "ce week-end", "ce mois-ci", etc. et ne recommande pas d'événement déjà terminé.
- Réponds uniquement à partir des événements fournis dans le contexte ci-dessous.
- Si aucun événement du contexte ne correspond à la question, dis-le clairement et ne propose rien d'autre.
- Pour chaque événement recommandé, donne son titre, ses dates, son lieu et son tarif s'ils sont connus.
- Réponds en français, de façon concise et structurée (liste si plusieurs événements).
- N'invente jamais d'événement, de date ou de lieu.

Contexte :
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# French names, hard-coded so the prompt does not depend on the machine locale.
_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def format_date_fr(d: date) -> str:
    return f"{_DAYS[d.weekday()]} {d.day} {_MONTHS[d.month - 1]} {d.year}"


def describe_today(today: date) -> str:
    """The date sentence injected in the prompt. The coming weekend is
    computed here: LLMs are unreliable at calendar arithmetic, so we never
    ask the model to work out which days "ce week-end" are."""
    days_to_saturday = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_to_saturday)
    sunday = saturday + timedelta(days=1)
    return (
        f"Nous sommes le {format_date_fr(today)}. "
        f"Le prochain week-end est le {format_date_fr(saturday)} et le {format_date_fr(sunday)}."
    )


@dataclass
class RagAnswer:
    """What the assistant returns: the text plus the sources behind it, so
    the API can show links and the evaluation can check faithfulness."""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)


def format_context(documents: list[Document]) -> str:
    """Turn retrieved chunks into the text block pasted into the prompt.
    Chunks already start with their event header (title, dates, venue)."""
    blocks = [f"[Événement {i}]\n{doc.page_content}" for i, doc in enumerate(documents, start=1)]
    return "\n\n".join(blocks)


def dedupe_by_event(documents: list[Document], max_events: int) -> list[Document]:
    """Keep the best-ranked chunk per event. FAISS often returns several
    chunks of the same long event; the LLM only needs one to recommend it,
    and we want variety in the answer."""
    seen: set = set()
    kept: list[Document] = []
    for doc in documents:
        uid = doc.metadata.get("uid")
        if uid in seen:
            continue
        seen.add(uid)
        kept.append(doc)
        if len(kept) == max_events:
            break
    return kept


class RagAssistant:
    """Wraps a vector store and a chat model into one `ask()` method."""

    def __init__(
        self,
        vector_store: FAISS,
        llm: BaseChatModel,
        top_k_chunks: int = TOP_K_CHUNKS,
        max_events: int = MAX_EVENTS,
    ):
        self.vector_store = vector_store
        self.top_k_chunks = top_k_chunks
        self.max_events = max_events
        # LCEL chain: prompt -> model -> plain string. Inputs: context, question.
        self.chain = PROMPT | llm | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        """Phase 1: vector search, then one chunk per event."""
        hits = self.vector_store.similarity_search(question, k=self.top_k_chunks)
        return dedupe_by_event(hits, self.max_events)

    def generate(self, question: str, documents: list[Document], today: date | None = None) -> str:
        """Phase 2: prompt the LLM with the retrieved context.
        `today` is injected so the model can resolve "ce week-end"; the LLM
        has no clock of its own. Tests pass a fixed date.
        The LangChain wrapper only retries network errors, not HTTP 429
        (rate limit), so we add a small exponential backoff ourselves."""
        today = today or date.today()
        inputs = {
            "context": format_context(documents),
            "question": question,
            "today": describe_today(today),
        }
        wait = RATE_LIMIT_WAIT
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return self.chain.invoke(inputs)
            except httpx.HTTPStatusError as err:
                if err.response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(wait)
                wait *= 2
        raise RuntimeError("unreachable")  # loop always returns or raises

    def ask(self, question: str) -> RagAnswer:
        """Phase 1 + phase 2."""
        documents = self.retrieve(question)
        answer = self.generate(question, documents)
        sources = [
            {k: doc.metadata.get(k) for k in ("uid", "title", "date_range", "venue", "url")}
            for doc in documents
        ]
        return RagAnswer(question=question, answer=answer, sources=sources)


def load_vector_store(index_dir: Path = INDEX_DIR, embeddings: Embeddings | None = None) -> FAISS:
    """Load the FAISS index saved by scripts/build_index.py.
    allow_dangerous_deserialization: the .pkl holds our own documents; it is
    only unsafe when loading a pickle from an untrusted source."""
    if embeddings is None:
        from langchain_mistralai import MistralAIEmbeddings

        load_dotenv()
        embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL, max_retries=5, wait_time=2)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def build_assistant(index_dir: Path = INDEX_DIR, chat_model: str = CHAT_MODEL) -> RagAssistant:
    """Production wiring: real index, real Mistral models."""
    from langchain_mistralai import ChatMistralAI

    load_dotenv()
    llm = ChatMistralAI(model=chat_model, temperature=0.2, max_retries=5)
    return RagAssistant(load_vector_store(index_dir), llm)
