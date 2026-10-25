"""Intent classifier: decide whether a question needs the RAG at all.

Three intents:
  events     a question about outings, culture, leisure in Paris -> full RAG
  chat       greetings, thanks, "who are you?" -> fixed friendly answer
  off_topic  anything unrelated (weather, maths, code, news) -> fixed refusal

Why: retrieval + a 14B generation cost ~3 s and tokens for every message.
"Bonjour" or "quel temps fait-il ?" deserve neither. A fixed answer is also
predictable, whereas a refusal written by the LLM varies from run to run.

Two stages, cheapest first:
  1. rules: obvious greetings and thanks, no model call;
  2. a small model (ministral-3b, ~0.3 s, 750 requests/min on the free tier)
     answering with a single label.
When the model's output cannot be parsed, the default is "events": better
to run the RAG for nothing than to refuse a real question.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

INTENT_MODEL = "ministral-3b-latest"
EVENTS, CHAT, OFF_TOPIC = "events", "chat", "off_topic"
INTENTS = (EVENTS, CHAT, OFF_TOPIC)
DEFAULT_INTENT = EVENTS

# Fixed answers: no model involved, same wording every time.
FIXED_ANSWERS = {
    CHAT: (
        "Bonjour ! Je suis l'assistant de Puls-Events. Je recommande des événements culturels à Paris : "
        "concerts, expositions, ateliers, spectacles, balades. Posez-moi par exemple "
        "« Quels concerts de jazz ce week-end ? » ou « Une activité gratuite pour les enfants ? »."
    ),
    OFF_TOPIC: (
        "Je ne peux répondre qu'aux questions sur les événements culturels et les sorties à Paris. "
        "Essayez par exemple « Une exposition de photographie en novembre ? »."
    ),
}

# Short messages that are clearly small talk. Anchored on the whole message
# so "merci de me trouver un concert" still goes to the model.
_SMALL_TALK = re.compile(
    r"^\s*(bonjour|bonsoir|salut|coucou|hello|hey|hi|merci( beaucoup| bien)?|"
    r"au revoir|bonne (journée|soirée)|ok|d'accord|super|parfait)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
# Whole-message questions about the assistant itself, optionally after a
# greeting. The small model sent "Que sais-tu faire ?" to the RAG: a rule is
# exact and free. Anchored, so "que peux-tu me conseiller ce soir ?" still
# goes to the model (and then to the RAG).
_ABOUT_ASSISTANT = re.compile(
    r"^\s*((bonjour|bonsoir|salut|coucou|hello)\s*[,!.]?\s*)?"
    r"(qui es[- ]tu|qui êtes[- ]vous|tu es qui|"
    r"que sais[- ]tu faire|que peux[- ]tu faire|qu'est[- ]ce que tu sais faire|"
    r"à quoi (sers[- ]tu|tu sers)|tu sers à quoi|"
    r"comment (fonctionnes[- ]tu|tu fonctionnes|ça marche|ca marche)|"
    r"(comment )?(ça|ca) va|tu vas bien)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Tu classes les messages envoyés à un assistant qui recommande des événements culturels à Paris.
Réponds par un seul mot parmi : events, chat, off_topic.

- events : toute demande de sortie, d'événement, d'activité culturelle ou de loisir (concert, exposition, spectacle, atelier, conférence, balade, sport, cinéma, fête), même si l'événement demandé n'existe peut-être pas, même sans date ni lieu.
- chat : salutations, remerciements, questions sur l'assistant lui-même.
- off_topic : tout le reste (météo, actualité, mathématiques, programmation, culture générale, conseils personnels).

En cas de doute, réponds events.

Exemples :
"Quels concerts de jazz ce week-end ?" -> events
"Y a-t-il un opéra de Wagner ce mois-ci ?" -> events
"Un cours de cuisine japonaise ?" -> events
"Que puis-je faire avec mes enfants mercredi ?" -> events
"Bonjour, qui es-tu ?" -> chat
"Que sais-tu faire ?" -> chat
"Quel temps fera-t-il demain ?" -> off_topic
"Qui a gagné la coupe du monde 1998 ?" -> off_topic
"Écris-moi une fonction Python" -> off_topic"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def parse_intent(raw: str) -> str:
    """Map the model's text to a label. Tolerates case, punctuation and extra
    words ("Réponse : off_topic."); anything else falls back to events."""
    text = raw.strip().lower().replace("-", "_")
    for label in (OFF_TOPIC, CHAT, EVENTS):  # off_topic first: "events" never appears inside it
        if label in text:
            return label
    return DEFAULT_INTENT


class IntentClassifier:
    """Rules first, then a small LLM. `llm=None` disables the model stage
    (every non-small-talk message is then treated as events)."""

    def __init__(self, llm: BaseChatModel | None = None, callbacks: list | None = None):
        self.chain = PROMPT | llm | StrOutputParser() if llm is not None else None
        self.callbacks = callbacks or []

    def classify(self, question: str) -> str:
        if _SMALL_TALK.match(question) or _ABOUT_ASSISTANT.match(question):
            return CHAT
        if self.chain is None:
            return DEFAULT_INTENT
        try:
            config = {"callbacks": self.callbacks, "run_name": "classify_intent"} if self.callbacks else None
            return parse_intent(self.chain.invoke({"question": question}, config=config))
        except Exception:
            # The classifier must never block an answer: on any API error,
            # fall back to the RAG.
            return DEFAULT_INTENT


def build_classifier(callbacks: list | None = None) -> IntentClassifier:
    """Production wiring: ministral-3b, deterministic, one short label."""
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(model=INTENT_MODEL, temperature=0, max_tokens=8, max_retries=3)
    return IntentClassifier(llm, callbacks)
