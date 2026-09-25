"""Functional test of the running API: real HTTP calls, real index, real LLM.

This is the integration counterpart of tests/test_api.py (which is offline).
Start the API first, then run this script. It is also the demo script.

Usage:
    uv run uvicorn api.main:app            # terminal 1
    uv run python scripts/api_test.py      # terminal 2
    uv run python scripts/api_test.py --url http://127.0.0.1:8000 --question "Une expo gratuite ?"
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

DEFAULT_URL = "http://127.0.0.1:8000"
DEMO_QUESTIONS = [
    "Quels concerts de jazz ce week-end ?",
    "Une activité gratuite pour les enfants ?",
    "Y a-t-il un opéra de Wagner ce mois-ci ?",  # expected: a clear "nothing matches"
]


def check_health(url: str) -> None:
    response = requests.get(f"{url}/health", timeout=10)
    response.raise_for_status()
    assert response.json() == {"status": "ok"}, response.text
    print(f"[ok] {url}/health")


def ask(url: str, question: str) -> None:
    start = time.perf_counter()
    response = requests.post(f"{url}/ask", json={"question": question}, timeout=120)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    assert body["answer"], "empty answer"
    assert isinstance(body["sources"], list)

    print(f"\n[ok] POST /ask ({elapsed:.1f}s)  Q: {question}")
    print(body["answer"])
    print("Sources :")
    for s in body["sources"]:
        print(f"  - {s['title']} | {s['date_range']} | {s['url']}")


def check_validation(url: str) -> None:
    response = requests.post(f"{url}/ask", json={"question": ""}, timeout=10)
    assert response.status_code == 422, f"expected 422, got {response.status_code}"
    print("\n[ok] POST /ask with empty question -> 422")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--question", help="ask a single question instead of the demo set")
    args = parser.parse_args()

    try:
        check_health(args.url)
        for q in [args.question] if args.question else DEMO_QUESTIONS:
            ask(args.url, q)
        check_validation(args.url)
    except (requests.RequestException, AssertionError) as err:
        print(f"\n[FAIL] {err}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
