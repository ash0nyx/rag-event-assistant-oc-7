"""Evaluate the RAG assistant on the annotated test set with Ragas.

For each question in evaluation/test_set.json:
  1. run the chain: retrieve chunks, generate an answer;
  2. score retrieval deterministically against the annotated uids (no LLM):
       - retrieval_recall:    share of expected events that were retrieved;
       - retrieval_precision: share of retrieved events that were expected;
  3. score generation with Ragas and an LLM judge (Mistral, same key):
       - faithfulness:       is every claim in the answer supported by the retrieved context?
       - answer_relevancy:   does the answer address the question?

Why not Ragas' context_precision / context_recall: they need the judge to
grade one chunk at a time against the reference answer, and the 14B model
available on the free tier grades the whole reference instead, returning 0
for chunks that are plainly relevant. The uid-based metrics are exact and
cost nothing, so they are the better tool here.

Outputs evaluation/results/<timestamp>.json plus evaluation/results/latest.json
(per-question scores and averages), used by tests/test_evaluation.py as a
regression gate.

Usage:
    uv run python scripts/evaluate_rag.py                 # full set
    uv run python scripts/evaluate_rag.py --limit 3       # quick check
    uv run python scripts/evaluate_rag.py --skip-ragas    # retrieval metrics only, no judge calls
    uv run python scripts/evaluate_rag.py --from-results evaluation/results/latest.json --metrics faithfulness
                                                          # re-score one metric on saved answers, no chain calls
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

from rag.chain import RagAssistant, build_assistant  # noqa: E402

TEST_SET_PATH = Path("evaluation/test_set.json")
RESULTS_DIR = Path("evaluation/results")
JUDGE_MODEL = "ministral-14b-latest"  # same free-tier constraint as the chain
PAUSE_BETWEEN_QUESTIONS = 2.0  # seconds; keeps the chain under 30 req/min


def retrieval_recall(retrieved_uids: list[int], expected_uids: list[int]) -> float | None:
    """Share of expected events that were retrieved. None when nothing is
    expected (the 'no match' questions), so they do not skew the average."""
    if not expected_uids:
        return None
    return len(set(retrieved_uids) & set(expected_uids)) / len(expected_uids)


def retrieval_precision(retrieved_uids: list[int], expected_uids: list[int]) -> float | None:
    """Share of retrieved events that were expected. None when nothing is
    expected or nothing was retrieved."""
    if not expected_uids or not retrieved_uids:
        return None
    return len(set(retrieved_uids) & set(expected_uids)) / len(set(retrieved_uids))


def run_chain(assistant: RagAssistant, items: list[dict], pause: float, today: date) -> list[dict]:
    """Ask every question once; keep answer, contexts and sources.
    `today` is the date the reference answers were written for (from the
    test set), not the real clock: "ce week-end" and the past-event filter
    must resolve the same way whenever the evaluation is run."""
    rows = []
    for item in items:
        documents = assistant.retrieve(item["question"], today)
        answer = assistant.generate(item["question"], documents, today)
        retrieved_uids = [d.metadata["uid"] for d in documents]
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": answer,
            "contexts": [d.page_content for d in documents],
            "retrieved_uids": retrieved_uids,
            "expected_uids": item["expected_uids"],
            "ground_truth": item["ground_truth"],
            "retrieval_recall": retrieval_recall(retrieved_uids, item["expected_uids"]),
            "retrieval_precision": retrieval_precision(retrieved_uids, item["expected_uids"]),
        })
        print(f"  [{item['id']}] recall = {rows[-1]['retrieval_recall']}, precision = {rows[-1]['retrieval_precision']}")
        time.sleep(pause)
    return rows


def make_judge_llm(model: str):
    """ChatMistralAI with a fix for a bug in langchain-mistralai 1.1.6:
    when several generations are merged (Ragas asks for 3 at once in
    answer_relevancy), the library sums token_usage values with `+=`, but
    Mistral now returns nested dicts in there, and dict += dict crashes.
    We override the merge to sum numbers only."""
    from langchain_mistralai import ChatMistralAI

    class PatchedChatMistralAI(ChatMistralAI):
        def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
            usage: dict = {}
            for output in llm_outputs:
                for k, v in ((output or {}).get("token_usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage[k] = usage.get(k, 0) + v
            return {"token_usage": usage, "model_name": self.model}

    return PatchedChatMistralAI(model=model, temperature=0, max_retries=5)


RAGAS_METRICS = ("faithfulness", "answer_relevancy")
ALL_METRICS = ("retrieval_recall", "retrieval_precision", *RAGAS_METRICS)


def score_with_ragas(rows: list[dict], metric_names: tuple[str, ...] = RAGAS_METRICS, workers: int = 2) -> None:
    """Add Ragas metrics to each row, in place. The judge is a Mistral chat
    model wrapped for Ragas; embeddings are Mistral too (answer_relevancy
    compares the question with questions generated from the answer).

    Concurrency: the free tier appears to queue concurrent requests, so more
    workers make every call slower until they time out (8 workers: 13 of 14
    faithfulness jobs lost). Two workers and a long timeout are the safe
    setting; expect 15 to 25 minutes for the full set."""
    from langchain_mistralai import MistralAIEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness

    judge = LangchainLLMWrapper(make_judge_llm(JUDGE_MODEL))
    embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(model="mistral-embed", max_retries=5, wait_time=2))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rows
    ])
    available = {"faithfulness": Faithfulness, "answer_relevancy": AnswerRelevancy}
    metrics = [available[m]() for m in metric_names]
    run_config = RunConfig(max_workers=workers, max_retries=10, max_wait=90, timeout=900)

    result = evaluate(dataset, metrics=metrics, llm=judge, embeddings=embeddings, run_config=run_config)
    scores = result.to_pandas()
    for row, (_, s) in zip(rows, scores.iterrows()):
        for m in metric_names:
            value = s.get(m)
            row[m] = None if value is None or value != value else round(float(value), 3)  # NaN -> None


def summarize(rows: list[dict]) -> dict:
    """Averages over questions where the metric is defined."""
    summary = {}
    for m in ALL_METRICS:
        values = [r[m] for r in rows if r.get(m) is not None]
        summary[m] = round(sum(values) / len(values), 3) if values else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=TEST_SET_PATH)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--skip-ragas", action="store_true", help="retrieval metrics only, no judge calls")
    parser.add_argument(
        "--metrics", nargs="+", choices=RAGAS_METRICS, default=list(RAGAS_METRICS),
        help="which Ragas metrics to score (default: all)",
    )
    parser.add_argument("--workers", type=int, default=2, help="parallel judge calls (keep low on the free tier)")
    parser.add_argument(
        "--from-results", type=Path,
        help="reuse answers and contexts from a previous results file instead of re-running the chain; "
        "scores already present are kept unless re-scored",
    )
    parser.add_argument("--pause", type=float, default=PAUSE_BETWEEN_QUESTIONS)
    parser.add_argument(
        "--index-dir", type=Path, default=Path("data/index"),
        help="FAISS index to evaluate against; use evaluation/index_snapshot for the snapshot the test set was annotated on",
    )
    args = parser.parse_args()

    load_dotenv()
    test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    items = test_set["items"][: args.limit] if args.limit else test_set["items"]

    if args.from_results:
        previous = json.loads(args.from_results.read_text(encoding="utf-8"))
        wanted = {item["id"] for item in items}
        rows = [r for r in previous["rows"] if r["id"] in wanted]
        print(f"Reusing {len(rows)} answers from {args.from_results}")
    else:
        today = date.fromisoformat(test_set["today"])
        print(f"Running the chain on {len(items)} questions as of {today} ...")
        rows = run_chain(build_assistant(index_dir=args.index_dir), items, args.pause, today)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def save() -> dict:
        """Write the report now. Called before and after the judge phase so a
        network drop during the (long) Ragas run cannot lose the answers."""
        summary = summarize(rows)
        report = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "test_set": str(args.test_set),
            "index_built_on": test_set.get("index_built_on"),
        "index_dir": str(args.index_dir),
            "n_questions": len(rows),
            "judge_model": None if args.skip_ragas else JUDGE_MODEL,
            "summary": summary,
            "rows": rows,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for path in (RESULTS_DIR / f"{stamp}.json", RESULTS_DIR / "latest.json"):
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    save()
    if not args.skip_ragas:
        print(f"Scoring {', '.join(args.metrics)} with Ragas ({args.workers} workers) ...")
        score_with_ragas(rows, tuple(args.metrics), args.workers)
    summary = save()

    print("\nSummary:")
    for metric, value in summary.items():
        print(f"  {metric:20s} {value}")
    print(f"\nSaved to {RESULTS_DIR / 'latest.json'}")


if __name__ == "__main__":
    main()
