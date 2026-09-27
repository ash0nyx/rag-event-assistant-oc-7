"""Tests for the evaluation step.

Two kinds:
  - unit tests of the pure helpers in scripts/evaluate_rag.py (offline);
  - a regression gate that reads evaluation/results/latest.json and fails if
    a score fell below its threshold. It never calls an LLM: the scores come
    from the last run of scripts/evaluate_rag.py. Skipped when no results
    file exists (fresh clone), so the offline suite stays green.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.evaluate_rag import retrieval_precision, retrieval_recall, summarize

RESULTS_PATH = Path("evaluation/results/latest.json")
TEST_SET_PATH = Path("evaluation/test_set.json")

# Minimum acceptable averages. Deliberately modest for a POC; raise them as
# the system improves. The point is to catch regressions, not to look good.
THRESHOLDS = {
    "retrieval_recall": 0.6,
    "retrieval_precision": 0.4,
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
}


def test_retrieval_recall():
    assert retrieval_recall([1, 2, 3], [1, 2]) == 1.0
    assert retrieval_recall([1, 9], [1, 2]) == 0.5
    assert retrieval_recall([9], [1, 2]) == 0.0
    assert retrieval_recall([1, 2], []) is None  # no expectation: not scored


def test_retrieval_precision():
    assert retrieval_precision([1, 2], [1, 2, 3]) == 1.0  # everything retrieved was expected
    assert retrieval_precision([1, 9, 8, 7], [1, 2]) == 0.25
    assert retrieval_precision([], [1, 2]) is None
    assert retrieval_precision([1], []) is None


def test_summarize_ignores_missing_values():
    rows = [
        {"retrieval_recall": 1.0, "faithfulness": 0.8},
        {"retrieval_recall": None, "faithfulness": 0.6},  # a 'no match' question
    ]
    summary = summarize(rows)
    assert summary["retrieval_recall"] == 1.0  # None not counted as 0
    assert summary["faithfulness"] == 0.7
    assert summary["answer_relevancy"] is None  # metric absent from every row


def test_test_set_is_well_formed():
    """The annotated set is a deliverable: check its shape so a typo is
    caught before an expensive evaluation run."""
    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    date.fromisoformat(test_set["today"])  # the evaluation date must be valid
    items = test_set["items"]
    assert len(items) >= 10
    assert len({i["id"] for i in items}) == len(items)  # unique ids
    for item in items:
        assert item["question"].strip()
        assert item["ground_truth"].strip()
        assert item["type"] in ("recommendation", "no_match")
        assert isinstance(item["expected_uids"], list)
        if item["type"] == "recommendation":
            assert item["expected_uids"], f"{item['id']} should list expected events"


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="no evaluation results yet; run scripts/evaluate_rag.py")
def test_latest_scores_meet_thresholds():
    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    summary = report["summary"]
    failures = [
        f"{metric} = {summary.get(metric)} < {threshold}"
        for metric, threshold in THRESHOLDS.items()
        if summary.get(metric) is not None and summary[metric] < threshold
    ]
    assert not failures, "; ".join(failures)
