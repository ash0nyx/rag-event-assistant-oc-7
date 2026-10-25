"""Unit tests for rag/intent.py and the routing in RagAssistant.

The small model is replaced by LangChain's FakeListChatModel returning
scripted labels, so we test the rules, the parsing of the model's answer,
the fallback, and that off-topic questions never reach retrieval.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import RagAssistant
from rag.intent import CHAT, EVENTS, FIXED_ANSWERS, OFF_TOPIC, IntentClassifier, parse_intent
from tests.test_chain import FakeVectorStore, make_doc


@pytest.mark.parametrize("message", ["Bonjour", "bonjour !", "Merci beaucoup.", "salut", "OK", "Au revoir"])
def test_small_talk_is_caught_by_rules_without_calling_the_model(message):
    llm = FakeListChatModel(responses=["events"])  # would say events if called
    assert IntentClassifier(llm).classify(message) == CHAT


def test_greeting_followed_by_a_request_goes_to_the_model():
    llm = FakeListChatModel(responses=["events"])
    assert IntentClassifier(llm).classify("Bonjour, un concert ce soir ?") == EVENTS


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("events", EVENTS),
        ("off_topic", OFF_TOPIC),
        ("OFF-TOPIC.", OFF_TOPIC),  # case, dash and punctuation tolerated
        ("Réponse : chat", CHAT),
        ("je ne sais pas", EVENTS),  # unparseable: safe default
        ("", EVENTS),
    ],
)
def test_parse_intent(raw, expected):
    assert parse_intent(raw) == expected


def test_model_error_falls_back_to_events():
    class BrokenChain:
        def invoke(self, inputs, config=None):
            raise RuntimeError("API down")

    classifier = IntentClassifier(FakeListChatModel(responses=["x"]))
    classifier.chain = BrokenChain()
    assert classifier.classify("Une expo ?") == EVENTS


def test_without_a_model_everything_but_small_talk_is_events():
    classifier = IntentClassifier(llm=None)
    assert classifier.classify("Quel temps fait-il ?") == EVENTS
    assert classifier.classify("Merci") == CHAT


def test_off_topic_question_gets_fixed_answer_and_skips_retrieval():
    store = FakeVectorStore([make_doc(1, "Expo A")])
    rag_llm = FakeListChatModel(responses=["should not be used"])
    classifier = IntentClassifier(FakeListChatModel(responses=["off_topic"]))
    assistant = RagAssistant(store, rag_llm, tracing=False, classifier=classifier)

    result = assistant.ask("Quel temps fera-t-il demain ?")

    assert result.intent == OFF_TOPIC
    assert result.answer == FIXED_ANSWERS[OFF_TOPIC]
    assert result.sources == []
    assert store.last_k is None  # FAISS was never queried


def test_events_question_goes_through_the_rag():
    store = FakeVectorStore([make_doc(1, "Expo A")])
    classifier = IntentClassifier(FakeListChatModel(responses=["events"]))
    assistant = RagAssistant(store, FakeListChatModel(responses=["Expo A."]), tracing=False, classifier=classifier)

    result = assistant.ask("Une expo ?")

    assert result.intent == EVENTS
    assert result.answer == "Expo A."
    assert [s["uid"] for s in result.sources] == [1]
