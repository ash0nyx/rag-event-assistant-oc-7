"""Measure the intent classifier on evaluation/intent_set.json.

Cheap: one ministral-3b call per message (well under a minute for ~30
messages). Writes evaluation/results/intent_latest.json, which the
regression gate in tests/test_evaluation.py checks.

Usage:
    uv run python scripts/evaluate_intent.py
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

from rag.intent import INTENT_MODEL, INTENTS, build_classifier  # noqa: E402

SET_PATH = Path("evaluation/intent_set.json")
OUT_PATH = Path("evaluation/results/intent_latest.json")


def main() -> None:
    load_dotenv()
    classifier = build_classifier()
    items = json.loads(SET_PATH.read_text(encoding="utf-8"))["items"]

    rows = []
    for item in items:
        start = time.perf_counter()
        predicted = classifier.classify(item["question"])
        rows.append({**item, "predicted": predicted, "latency_ms": round((time.perf_counter() - start) * 1000)})

    correct = sum(r["predicted"] == r["intent"] for r in rows)
    confusion = Counter((r["intent"], r["predicted"]) for r in rows)
    # The costly mistake is refusing a real events question: track it apart.
    events_refused = sum(r["intent"] == "events" and r["predicted"] != "events" for r in rows)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": INTENT_MODEL,
        "n": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "events_refused": events_refused,
        "median_latency_ms": sorted(r["latency_ms"] for r in rows)[len(rows) // 2],
        "confusion": {f"{t}->{p}": n for (t, p), n in sorted(confusion.items())},
        "rows": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"accuracy {report['accuracy']} on {len(rows)} messages, events refused: {events_refused}, "
          f"median latency {report['median_latency_ms']} ms")
    for (t, p), n in sorted(confusion.items()):
        print(f"  {t:10s} -> {p:10s} {n}")
    for r in rows:
        if r["predicted"] != r["intent"]:
            print(f"  MISS [{r['intent']} -> {r['predicted']}] {r['question']}")
    assert set(INTENTS) >= {r["predicted"] for r in rows}


if __name__ == "__main__":
    main()
