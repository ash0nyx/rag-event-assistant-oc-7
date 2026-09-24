"""Ask the RAG assistant a question from the command line.

Lets you test the whole chain (retrieval + generation) without the API.

Usage:
    uv run python scripts/ask_question.py "Quels concerts de jazz ce week-end ?"
    uv run python scripts/ask_question.py "Une expo gratuite pour enfants ?" --show-context
"""

from __future__ import annotations

import argparse
import warnings

from rag.chain import build_assistant, format_context

warnings.filterwarnings("ignore")  # HF tokenizer download notice, etc.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    args = parser.parse_args()

    assistant = build_assistant()

    if args.show_context:
        print("=== Contexte récupéré ===")
        print(format_context(assistant.retrieve(args.question)))
        print()

    result = assistant.ask(args.question)
    print("=== Réponse ===")
    print(result.answer)
    print()
    print("=== Sources ===")
    for s in result.sources:
        print(f"- {s['title']} | {s['date_range']} | {s['venue']} | {s['url']}")


if __name__ == "__main__":
    main()
