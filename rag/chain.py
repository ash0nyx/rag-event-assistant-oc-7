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

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

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
PERIOD_FETCH_FACTOR = 4  # with a period, price or audience constraint, fetch 4x more candidates before filtering
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
- Si des événements du contexte répondent à la question, recommande-les directement, sans phrase d'introduction sur ce qui manquerait. Une question large (« que faire », « quoi faire », « quels événements ») est satisfaite par tout événement du contexte qui correspond à son thème et à sa période.
- Seulement si aucun événement du contexte ne répond à la question, dis-le clairement, puis propose les événements qui s'en rapprochent le plus (même thème, même période) en précisant en quoi ils diffèrent. Ne propose jamais un événement absent du contexte.
- Réponds en français. Garde l'ordre du contexte, déjà trié par date. Pour chaque événement recommandé, écris un bloc dans ce format exact, sans tiret ni puce, et sépare les blocs par une ligne vide :
Titre de l'événement
Dates : ...
Lieu : ...
Tarif : ...
- N'ajoute aucune autre ligne au bloc. Omets une ligne si l'information est inconnue. Une phrase d'introduction ou de conclusion est permise seulement si elle apporte une information utile (par exemple, ce qui ne correspond pas exactement).
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
    saturday, sunday = weekend_dates(today)
    season_name, season_start, season_end = current_season(today)
    return (
        f"{sentence} "
        f"Ce week-end désigne uniquement le {format_date_fr(saturday)} et le {format_date_fr(sunday)}. "
        f"La saison en cours est l'{season_name} (du {format_date_fr(season_start)} au {format_date_fr(season_end)})."
    )


def weekend_dates(today: date) -> tuple[date, date]:
    """Saturday and Sunday of "ce week-end": the current one on a Sunday,
    the coming one otherwise (on Saturday, today and tomorrow)."""
    if today.weekday() == 6:
        return today - timedelta(days=1), today
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    return saturday, saturday + timedelta(days=1)


def _seasons(y: int) -> list[tuple[str, date, date]]:
    """Approximate astronomical seasons touching year y."""
    return [
        ("hiver", date(y - 1, 12, 21), date(y, 3, 19)),
        ("printemps", date(y, 3, 20), date(y, 6, 20)),
        ("été", date(y, 6, 21), date(y, 9, 21)),
        ("automne", date(y, 9, 22), date(y, 12, 20)),
        ("hiver", date(y, 12, 21), date(y + 1, 3, 19)),
    ]


def current_season(today: date) -> tuple[str, date, date]:
    """Name and bounds of the season containing `today` (approximate
    astronomical dates). "Cet automne" is then a date range, not a guess."""
    for name, start, end in _seasons(today.year):
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
    intent: str = "events"  # events | chat | off_topic, see rag/intent.py
    trace_id: str | None = None  # Langfuse trace, to attach user feedback to it


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


_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_SEASON_RE = re.compile(r"\bcet(?:te)? (hiver|printemps|été|automne)\b|\bce (printemps)\b", re.IGNORECASE)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    next_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return first, next_first - timedelta(days=1)


def requested_period(question: str, today: date) -> tuple[date, date] | None:
    """The date window a question asks for, or None if it asks for none.
    Computed in Python so retrieval can keep only events in that window:
    vector similarity reads "novembre" as a word, not as a constraint.
    Handles: aujourd'hui / ce soir, demain, ce week-end, cette semaine,
    ce mois-ci, le mois prochain, a month name, cet été / automne / hiver,
    ce printemps. Past months and seasons mean the next occurrence."""
    q = question.lower()
    if re.search(r"\b(aujourd'hui|ce soir|cet après-midi|ce midi)\b", q):
        return today, today
    if re.search(r"\bdemain\b", q):
        return today + timedelta(days=1), today + timedelta(days=1)
    if re.search(r"\bweek-?end\b", q):
        return weekend_dates(today)
    if re.search(r"\bcette semaine\b", q):
        return today, today + timedelta(days=6 - today.weekday())
    if re.search(r"\bce mois(-ci)?\b", q):
        return today, _month_bounds(today.year, today.month)[1]
    if re.search(r"\bmois prochain\b", q):
        y, m = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return _month_bounds(y, m)
    if match := _MONTH_RE.search(q):
        month = _MONTHS.index(match.group(1).lower()) + 1
        year = today.year if month >= today.month else today.year + 1
        start, end = _month_bounds(year, month)
        return max(start, today), end
    if match := _SEASON_RE.search(q):
        name = (match.group(1) or match.group(2)).lower()
        for year in (today.year, today.year + 1):
            for season, start, end in _seasons(year):
                if season == name and end >= today:
                    return max(start, today), end
    return None


def in_period(doc: Document, start: date, end: date) -> bool:
    """True if the event has a session window overlapping [start, end].
    Uses the first and last session dates only: an event running every
    Thursday from October to December overlaps a weekend window although it
    never happens on a weekend (the list of sessions is not indexed).
    Chunks without usable dates are kept."""
    try:
        first = date.fromisoformat(doc.metadata["first_begin"][:10])
        last = date.fromisoformat(doc.metadata["last_end"][:10])
    except (KeyError, TypeError, ValueError):
        return True
    return first <= end and last >= start


_FREE_RE = re.compile(r"\bgratuit(e|s|es|ement)?\b", re.IGNORECASE)
_PRICE_RE = re.compile(r"\d\s*(€|eur)", re.IGNORECASE)
# Audience words in the question -> agenda keywords that satisfy them.
# "Tout public" fits every audience.
_AUDIENCES = [
    (re.compile(r"\b(enfants?|famille|petits|bébés?|kids)\b", re.IGNORECASE), ("Enfants", "Tout public")),
    (re.compile(r"\b(ados?|adolescents?|jeunes)\b", re.IGNORECASE), ("Jeunes", "Tout public")),
]


def is_free(doc: Document) -> bool:
    """Free events: price text starting with "Gratuit". "Gratuit sous
    condition" usually means free with registration, unless it states a price
    ("8 à 15€"), which then does not count as free."""
    conditions = (doc.metadata.get("conditions") or "").strip()
    if not conditions.lower().startswith("gratuit"):
        return False
    return not (conditions.lower().startswith("gratuit sous condition") and _PRICE_RE.search(conditions))


def requested_filters(question: str, today: date) -> list[tuple[str, Callable[[Document], bool]]]:
    """Constraints the question states explicitly, as exact metadata tests:
    the period ("en novembre"), the price ("gratuit") and the audience
    ("pour les enfants", "pour ados"). Vector similarity treats these words as
    vague topics; Python applies them as hard filters."""
    filters: list[tuple[str, Callable[[Document], bool]]] = []
    if period := requested_period(question, today):
        filters.append(("période", lambda d, p=period: in_period(d, *p)))
    if _FREE_RE.search(question):
        filters.append(("gratuit", is_free))
    for pattern, keywords in _AUDIENCES:
        if pattern.search(question):
            filters.append((
                "public",
                lambda d, k=keywords: any(kw in (d.metadata.get("keywords") or "") for kw in k),
            ))
    return filters


def apply_filters(documents: list[Document], filters) -> list[Document]:
    """Apply the filters one by one. A filter that would leave nothing is
    skipped, so the model still gets the closest events and can say what is
    missing ("nothing free this weekend") instead of answering from nothing."""
    for _, keep in filters:
        kept = [d for d in documents if keep(d)]
        if kept:
            documents = kept
    return documents


_MD_EMPHASIS = re.compile(r"\*\*|__")  # **bold** / __bold__
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)  # "## Titre"
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", re.MULTILINE)  # "- ", "• ", "1. "


def strip_markdown(text: str) -> str:
    """Bring the answer back to the agreed plain format: no bold, no
    headings, no list markers (the model sometimes puts a dash on every
    line). Done in Python because a prompt rule is a request, not a
    guarantee. Presentation (bold titles) is the front-end's job."""
    text = _MD_HEADING.sub("", _MD_EMPHASIS.sub("", text))
    return _LIST_MARKER.sub("", text).strip()


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


def sort_by_date(documents: list[Document], today: date) -> list[Document]:
    """Show the selected events in chronological order. Retrieval ranks by
    similarity, which is the right way to choose the events but a confusing
    order to read them in. Key: the next session (index-time value; if it
    is already past, the event is ongoing, so it counts as today). Same
    day: by start time. Unknown dates go last, in similarity order."""

    def key(doc: Document) -> str:
        for field in ("next_begin", "first_begin"):
            value = doc.metadata.get(field)
            try:
                date.fromisoformat(value[:10])  # validate
            except (TypeError, ValueError):
                continue
            # "2026-10-31T10:30" (Paris local time): ISO strings sort like dates.
            return max(value[:16], today.isoformat())
        return "9999"  # no date: last

    return sorted(documents, key=key)


# --- Observability (optional) -------------------------------------------------
# Langfuse records each question as a trace: retrieval (which events, how
# long), the exact prompt, the model's answer, tokens and latency. It is the
# "flight recorder" complementing the offline Ragas evaluation. Enabled only
# when the LANGFUSE_* keys are in the environment; without them nothing is
# imported and the chain behaves exactly as before.


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def flush_traces() -> None:
    """Send pending traces now. Needed before a short-lived process exits
    (the CLI script); the API flushes at shutdown."""
    if tracing_enabled():
        from langfuse import get_client

        get_client().flush()


def push_feedback_score(trace_id: str | None, score: int, comment: str | None = None) -> None:
    """Attach a user's thumbs up/down to the question's Langfuse trace, so the
    dashboard can filter traces by satisfaction. No-op without tracing or
    trace id; never raises (feedback is already stored in the database)."""
    if not (trace_id and tracing_enabled()):
        return
    try:
        from langfuse import get_client

        get_client().create_score(
            name="user_feedback", value=1 if score > 0 else 0, data_type="BOOLEAN",
            trace_id=trace_id, comment=comment,
        )
    except Exception:
        pass


class RagAssistant:
    """Wraps a vector store and a chat model into one `ask()` method."""

    def __init__(
        self,
        vector_store: FAISS,
        llm: BaseChatModel,
        top_k_chunks: int = TOP_K_CHUNKS,
        max_events: int = MAX_EVENTS,
        tracing: bool | None = None,
        classifier=None,
    ):
        self.vector_store = vector_store
        # Optional IntentClassifier (rag/intent.py). None = every question goes
        # to the RAG, which is what the evaluation and the unit tests want.
        self.classifier = classifier
        self.top_k_chunks = top_k_chunks
        self.max_events = max_events
        # LCEL chain: prompt -> model -> plain string. Inputs: context, question.
        self.chain = PROMPT | llm | StrOutputParser()
        self.tracing = tracing_enabled() if tracing is None else tracing
        # LangChain callback: records the prompt/LLM step under the current trace.
        self._callbacks: list = []
        if self.tracing:
            from langfuse.langchain import CallbackHandler

            self._callbacks = [CallbackHandler()]

    def retrieve(self, question: str, today: date | None = None) -> list[Document]:
        """Phase 1: vector search, drop finished events, apply the
        constraints the question states (period, free, audience), one chunk
        per event.

        With constraints, 4x more candidates are fetched (still instant with
        an exact index) because many will not pass. A constraint that would
        leave nothing is skipped, so the model can say "nothing free this
        weekend" and offer the nearest events instead of answering from an
        empty context."""
        today = today or date.today()
        filters = requested_filters(question, today)
        k = self.top_k_chunks * PERIOD_FETCH_FACTOR if filters else self.top_k_chunks
        hits = drop_past_events(self.vector_store.similarity_search(question, k=k), today)
        if filters:
            filtered = apply_filters(hits, filters)
            # Nothing passed any filter: keep the best unfiltered candidates.
            hits = filtered if filtered is not hits else hits[: self.top_k_chunks]
        return sort_by_date(dedupe_by_event(hits, self.max_events), today)

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
        config = {"callbacks": self._callbacks, "run_name": "generate"} if self._callbacks else None
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return strip_markdown(self.chain.invoke(inputs, config=config))
            except httpx.HTTPStatusError as err:
                if err.response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(wait)
                wait *= 2
        raise RuntimeError("unreachable")  # loop always returns or raises

    def ask(self, question: str) -> RagAnswer:
        """Phase 0 (intent) + phase 1 (retrieval) + phase 2 (generation)."""
        if not self.tracing:
            return self._ask(question)

        # One trace per question: a root span "ask" (question in, answer out),
        # a child span "retrieve" (events found), and the LangChain "generate"
        # step attached automatically by the callback handler.
        from langfuse import get_client

        langfuse = get_client()
        with langfuse.start_as_current_observation(name="ask", as_type="span", input={"question": question}) as root:
            result = self._ask(question, langfuse)
            result.trace_id = langfuse.get_current_trace_id()
            root.update(
                output={"answer": result.answer},
                metadata={"n_sources": len(result.sources), "intent": result.intent},
            )
        return result

    def _ask(self, question: str, langfuse=None) -> RagAnswer:
        # Phase 0: small talk and off-topic questions get a fixed answer,
        # without retrieval or the 14B model.
        intent = self.classifier.classify(question) if self.classifier is not None else "events"
        if intent != "events":
            from rag.intent import FIXED_ANSWERS

            return RagAnswer(question=question, answer=FIXED_ANSWERS[intent], intent=intent)

        today = date.today()
        if langfuse is None:
            documents = self.retrieve(question, today)
        else:
            with langfuse.start_as_current_observation(name="retrieve", as_type="span", input={"question": question}) as span:
                documents = self.retrieve(question, today)
                span.update(output=[{"uid": d.metadata.get("uid"), "title": d.metadata.get("title")} for d in documents])
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


def build_assistant(
    index_dir: Path = INDEX_DIR,
    chat_model: str = CHAT_MODEL,
    tracing: bool | None = None,
    with_intent: bool = True,
) -> RagAssistant:
    """Production wiring: real index, real Mistral models, intent classifier
    in front. Tracing follows the LANGFUSE_* environment unless forced (the
    evaluation script forces it off so 14 synthetic questions do not drown
    the real ones in the dashboard)."""
    from langchain_mistralai import ChatMistralAI

    load_dotenv()
    # temperature 0: the same question and context give the same answer, which
    # makes the evaluation reproducible and the demo predictable.
    llm = ChatMistralAI(model=chat_model, temperature=0, max_retries=5)
    assistant = RagAssistant(load_vector_store(index_dir), llm, tracing=tracing)
    if with_intent:
        from rag.intent import build_classifier

        # Same Langfuse callback as the generation, so the classification
        # shows up as a step of the question's trace.
        assistant.classifier = build_classifier(callbacks=assistant._callbacks)
    return assistant
