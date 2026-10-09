# Index snapshot for the evaluation

FAISS index built on 2026-09-25 from the "Que faire à Paris" agenda (events from 2026-09-25 to 2026-12-31, 2650 events, 5499 chunks). The reference answers in `../test_set.json` were written on this snapshot, so the evaluation must run against it, not against the live index in `data/index/`.

Used by `scripts/evaluate_rag.py --index-dir evaluation/index_snapshot` and by the "Evaluate RAG" GitHub Actions workflow.

Rebuild it only when the test set is re-annotated: copy `data/index/` here and update the dates above and in `test_set.json`.
