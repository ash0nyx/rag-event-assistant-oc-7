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

import re
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
TOP_K_CHUNKS = 20  # chunks fetched from FAISS; some are dropped as past events
MAX_EVENTS = 5  # distinct events kept after deduplication
RATE_LIMIT_RETRIES = 4  # Mistral free tier answers 429 when called too fast
RATE_LIMIT_WAIT = 2.0  # seconds; doubled after each failed attempt

# The prompt is the contract with the LLM. Key rules: answer only from the
# context, say so when the context has nothing, always give dates and venue.
SYSTEM_PROMPT = """Tu es l'assistant de Puls-Events. Tu recommandes des événements culturels à Paris.
{today}

Règles :
- Tous les événements du contexte sont à venir. Ne les écarte pour une raison de date que si la question demande explicitement une période ("ce week-end", "en novembre", "cet automne").
- Dans les mots-clés, "Jeunes" désigne les adolescents et jeunes adultes, "Enfants" les moins de 12 ans, "Tout public" tout le monde.
- Un événement qui se déroule sur une période (par exemple "29 septembre 2026 - 31 janvier 2027") correspond à toute date ou tout mois inclus dans cette période.
- Réponds uniquement à partir des événements fournis dans le contexte ci-dessous.
- Si aucun événement du contexte ne correspond exactement à la question, dis-le clairement, puis propose les événements du contexte qui s'en rapprochent le plus (même thème, même période) en précisant en quoi ils diffèrent. Ne propose jamais un événement absent du contexte.
- Pour chaque événement recommandé, donne son titre, ses dates, son lieu et son tarif s'ils sont connus.
- Réponds en français, de façon concise et structurée (une ligne par information, un tiret par événement).
- Texte brut uniquement : pas de markdown, pas d'astérisques, pas de titres, pas de séparateurs.
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


# Words that make a question relative to the calendar. Only then does the
# prompt get the weekend and season sentences: given to every question, they
# made the model restrict undated questions ("un atelier le mercredi ?") to
# the coming weekend.
_RELATIVE_TIME = re.compile(
    r"\b(aujourd'hui|ce soir|demain|week-?end|cette semaine|ce mois|ce trimestre|"
    r"cet été|cet automne|cet hiver|ce printemps|prochain|prochaine|bientôt|"
    r"en (janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre))\b",
    re.IGNORECASE,
)


def needs_calendar(question: str) -> bool:
    """True when the question refers to a period relative to today."""
    return bool(_RELATIVE_TIME.search(question))


def describe_today(today: date, question: str = "") -> str:
    """The date sentence injected in the prompt. Weekend and season bounds are
    computed here (LLMs are unreliable at calendar arithmetic) and only added
    when the question needs them; otherwise the model is told that no date
    constraint applies."""
    sentence = f"Nous sommes le {format_date_fr(today)}."
    if question and not needs_calendar(question):
        return sentence + " La question ne précise pas de période : tous les événements du contexte conviennent."
    if today.weekday() == 6:  # Sunday: "ce week-end" is the one we are in
        saturday, sunday = today - timedelta(days=1), today
    else:
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        sunday = saturday + timedelta(days=1)
    season_name, season_start, season_end = current_season(today)
    return (
        f"{sentence} "
        f"Ce week-end désigne uniquement le {format_date_fr(saturday)} et le {format_date_fr(sunday)}. "
        f"La saison en cours est l'{season_name} (du {format_date_fr(season_start)} au {format_date_fr(season_end)})."
    )


def current_season(today: date) -> tuple[str, date, date]:
    """Name and bounds of the season containing `today` (approximate
    astronomical dates). "Cet automne" is then a date range, not a guess."""
    y = today.year
    seasons = [
        ("hiver", date(y - 1, 12, 21), date(y, 3, 19)),
        ("printemps", date(y, 3, 20), date(y, 6, 20)),
        ("été", date(y, 6, 21), date(y, 9, 21)),
        ("automne", date(y, 9, 22), date(y, 12, 20)),
        ("hiver", date(y, 12, 21), date(y + 1, 3, 19)),
    ]
    for name, start, end in seasons:
        if start <= today <= end:
            return name, start, end
    raise ValueError("unreachable")  # every date falls in one of the ranges above


@dataclass
class RagAnswer:
    """What the assistant returns: the text plus the sources behind it, so
    the API can show links and the evaluation can check faithfulness."""

    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)


def describe_period(first_begin: str | None, last_end: str | None) -> str | None:
    """Explicit period line computed from the ISO timestamps in the metadata,
    e.g. 'Période : du 29 septembre 2026 au 31 janvier 2027 (mois couverts :
    septembre 2026, octobre 2026, ...)'. Small LLMs fail to infer that such a
    range includes October; listing the months removes the inference."""
    if not first_begin or not last_end:
        return None
    try:
        start = date.fromisoformat(first_begin[:10])
        end = date.fromisoformat(last_end[:10])
    except ValueError:
        return None
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month) and len(months) < 12:
        months.append(f"{_MONTHS[m - 1]} {y}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    covered = ", ".join(months) + (", ..." if (y, m) <= (end.year, end.month) else "")
    return f"Période : du {format_date_fr(start)} au {format_date_fr(end)} (mois couverts : {covered})"


def format_context(documents: list[Document]) -> str:
    """Turn retrieved chunks into the text block pasted into the prompt.
    Chunks already start with their event header (title, dates, venue); an
    explicit period line is added from the metadata."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        period = describe_period(doc.metadata.get("first_begin"), doc.metadata.get("last_end"))
        lines = [f"[Événement {i}]"] + ([period] if period else []) + [doc.page_content]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def drop_past_events(documents: list[Document], today: date) -> list[Document]:
    """Remove chunks whose event is already over (last session before today).
    Vector search knows nothing about dates: "concert de jazz ce week-end"
    ranks last week's concert as high as next week's. Filtering here, in
    Python, is exact; asking the LLM to ignore past events is not.
    Chunks without a usable end date are kept."""
    kept = []
    for doc in documents:
        last_end = doc.metadata.get("last_end")
        try:
            if last_end and date.fromisoformat(last_end[:10]) < today:
                continue
        except ValueError:
            pass
        kept.append(doc)
    return kept


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

    def retrieve(self, question: str, today: date | None = None) -> list[Document]:
        """Phase 1: vector search, drop finished events, one chunk per event."""
        hits = self.vector_store.similarity_search(question, k=self.top_k_chunks)
        hits = drop_past_events(hits, today or date.today())
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
            "today": describe_today(today, question),
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
        today = date.today()
        documents = self.retrieve(question, today)
        answer = self.generate(question, documents, today)
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
    # temperature 0: the same question and context give the same answer, which
    # makes the evaluation reproducible and the demo predictable.
    llm = ChatMistralAI(model=chat_model, temperature=0, max_retries=5)
    return RagAssistant(load_vector_store(index_dir), llm)
