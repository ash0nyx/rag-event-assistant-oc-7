# Puls-Events RAG Assistant

Proof of concept for a chatbot that answers questions about upcoming cultural events. It uses Retrieval-Augmented Generation (RAG): event data from the Open Agenda API is embedded with Mistral, indexed in FAISS, and queried through a LangChain pipeline exposed as a REST API.

OpenClassrooms AI Engineer path, project 7.

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- LangChain, FAISS (vector store), Mistral (`mistral-embed` for embeddings, `ministral-14b-latest` for generation)
- FastAPI and uvicorn (REST API), pytest, Ragas (evaluation), Docker (local deployment)

## Setup

1. Clone the repository and install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file at the project root with your API keys (see `.env.example`):

   ```
   MISTRAL_API_KEY=your_key_here
   OPENAGENDA_API_KEY=your_key_here
   RAG_API_KEY=any_secret_string
   ```

   Get a Mistral key at [console.mistral.ai](https://console.mistral.ai) and an Open Agenda key from your account settings at [openagenda.com](https://openagenda.com). Both free tiers are enough. Note that the Mistral free tier does not enable `mistral-small` or `mistral-medium` (0 requests per minute); the project uses `ministral-14b-latest`, which is open on that tier.

3. Verify the setup:

   ```bash
   uv run python -c "from langchain_community.vectorstores import FAISS; print('ok')"
   ```

Without uv, `requirements.txt` (exported from `uv.lock`) works with a plain virtualenv: `pip install -r requirements.txt`.

## Data pipeline

The data comes from the Open Agenda API, agenda "Que faire à Paris" (uid 648405), the official City of Paris calendar. Scope for the POC: events in Paris with at least one session in a date window, by default today to 60 days ahead, capped at 500 events. About 2000 events are available for two months, so the cap is a sample size, not a limit of the source.

1. Fetch raw events:

   ```bash
   uv run python scripts/fetch_events.py                        # 500 events, today to +60 days
   uv run python scripts/fetch_events.py --since-days 30 --until-days 90 --max-events 1000
   ```

   The script paginates with the API cursor and saves the raw JSON to `data/raw/events_raw.json`. The date window is sent to the API as `timings[gte]` and `timings[lte]`.

2. Clean them into a flat table:

   ```bash
   uv run python scripts/clean_events.py
   ```

   This strips markdown and HTML from descriptions, keeps French text, flattens location and dates, filters to Paris, deduplicates, and saves `data/processed/events.csv`. Each row holds the text to embed (title, description, long description) and the metadata to store next to each vector (dates, venue, address, price, keywords, public URL).

3. Build the vector index:

   ```bash
   uv run python scripts/build_index.py
   uv run python scripts/build_index.py --chunk-size 600 --chunk-overlap 80
   ```

   Each event is split into chunks of 800 characters with a 100-character overlap, cut on paragraph, line, then sentence boundaries. Every chunk starts with a short header (title, dates, venue, price, keywords) so it stays self-describing when retrieved alone. Chunks are embedded with `mistral-embed` (1024 dimensions) and stored in a FAISS `IndexFlatL2` index, saved to `data/index/`. Each vector carries the event metadata (uid, title, dates, venue, address, price, keywords, URL) so a search hit returns both the text and where and when the event happens.

   Current numbers: 196 events, 469 chunks, 469 vectors. At this size an exact index is instant; approximate indexes (IVF, HNSW) only pay off at millions of vectors.

The `data/` folder is git-ignored. Run the three scripts above to rebuild it.

Note on `langchain-community`: the package is being sunset, but it still ships the only official LangChain wrapper for FAISS. The import is kept with its deprecation warning silenced.

## Asking a question

```bash
uv run python scripts/ask_question.py "Quels concerts de jazz ce week-end ?"
uv run python scripts/ask_question.py "Une expo gratuite pour enfants ?" --show-context
```

The chain lives in `rag/chain.py` (`RagAssistant` class) and runs in two phases:

1. **Retrieval.** The question is embedded with `mistral-embed` and FAISS returns the 20 closest chunks. Chunks whose event is already over (last session before today) are dropped, then one chunk per event is kept (best-ranked first), at most 5 events. No LLM is involved in this phase. The date filter is done in Python because vector similarity knows nothing about dates: "concert de jazz ce week-end" ranks last week's concert as high as next week's.
2. **Generation.** The kept chunks are pasted into a French system prompt together with today's date, and `ministral-14b-latest` writes the answer. The prompt tells the model to use only the provided events, to give title, dates, venue and price, and, when nothing matches exactly, to say so and then offer the closest events from the context with the difference stated. The prompt is in French on purpose: Mistral models are trained heavily on French, the data and the expected answer are French, and mixing languages costs a small model more than it gains.

The result holds the answer and the sources (uid, title, dates, venue, URL), so the API can show links and the evaluation step can check that answers stay faithful to the context.

Two details worth knowing:

- The LLM has no clock, and small models are unreliable at calendar arithmetic. Today's date, the dates of the weekend (the current one on a Sunday, the coming one otherwise) and the bounds of the current season are computed in Python and injected in the prompt, in French, so that "ce week-end", "ce mois-ci" or "cet automne" are resolved correctly. Each retrieved event also gets an explicit list of the months it covers. These calendar sentences are only added when the question contains a relative time expression (detected by a regular expression); for undated questions the prompt says instead that every event in the context fits, because the calendar text made the model restrict "un atelier le mercredi" to the coming weekend.
- The chat model runs at temperature 0: same question and context, same answer. Creativity brings nothing to a recommendation grounded in the context, and determinism makes the evaluation reproducible and the demo predictable.
- The LangChain Mistral wrapper only retries network errors. A small exponential backoff on HTTP 429 (rate limit) is added around the LLM call.

Conversation history is out of scope for the POC.

## REST API

```bash
uv run uvicorn api.main:app --reload
```

Swagger documentation is generated at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The index and the LLM are loaded once at startup, not on each request.

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/health` | Liveness check, used by Docker and the demo. |
| `POST` | `/ask` | Body `{"question": "..."}`. Returns the answer and the retrieved sources. Questions shorter than 3 characters or blank are rejected with a 422. |
| `POST` | `/rebuild` | Re-fetches Open Agenda, rebuilds the FAISS index and reloads it in memory. Optional body `{"max_events": 500, "from_date": "2026-09-25", "to_date": "2026-10-31"}`; defaults are 500 events from today to today + 60 days. Requires the `X-API-Key` header matching `RAG_API_KEY`; without a configured key the route is closed. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts de jazz ce week-end ?"}'

curl -X POST http://127.0.0.1:8000/rebuild \
  -H "X-API-Key: $RAG_API_KEY" -H "Content-Type: application/json" \
  -d '{"from_date": "2026-09-25", "to_date": "2026-10-31"}'
```

Functional test against a running API (also the demo script):

```bash
uv run python scripts/api_test.py
uv run python scripts/api_test.py --question "Une expo gratuite ?"
```

It checks `/health`, asks three demo questions, and verifies that an empty question returns a 422. Unlike the unit tests, it uses the real index and the real LLM.

## Docker

Run the API in a container, without installing Python or uv on the machine. The index must exist first (`data/index/`, built by the pipeline above or by a `/rebuild` call).

```bash
docker compose up --build        # build the image and start the API on port 8000
docker compose up -d             # same, in the background
docker compose logs -f           # follow the logs
docker compose down              # stop and remove the container
```

Without Compose:

```bash
docker build -t puls-events-rag .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data puls-events-rag
```

Then `http://localhost:8000/docs` and `uv run python scripts/api_test.py` work exactly as with a local server.

How the image is built and run, and why:

- **In the image**: Python 3.12, the exact versions from `uv.lock` (installed with `uv sync --frozen`), and the code. Dependencies are installed before the code is copied, so a code change reuses the cached install.
- **Outside the image**: secrets, passed at run time with `--env-file .env`, and `data/`, mounted as a volume at `/app/data`. A `/rebuild` call from the container writes the new index to the host folder, so refreshing the data never needs a new image.
- **Ports**: uvicorn listens on `0.0.0.0:8000` inside the container (not `127.0.0.1`, which would be unreachable from outside), and `-p 8000:8000` publishes it on the host.
- **Non-root**: the process runs as user `app` (uid 1000), so files written to the volume belong to the host user and a compromised API cannot act as root.
- **Health check**: Docker calls `/health` every 30 s; `docker ps` shows the container as healthy or unhealthy.
- `.dockerignore` keeps `.venv`, `data/`, `.env`, tests and private docs out of the build.

Image size is about 1.2 GB, mostly the tokenizer and numeric stack; acceptable for a local POC.

Troubleshooting: a VPN with a kill switch (NordVPN and similar) blocks traffic between the host and Docker's network, so `localhost:8000` hangs even though the container reports healthy. Disconnect the VPN or allowlist Docker's subnet (`nordvpn allowlist add subnet 172.17.0.0/16`).

## Evaluation

```bash
uv run python scripts/evaluate_rag.py                 # full annotated set, about 25 minutes
uv run python scripts/evaluate_rag.py --limit 3       # quick check
uv run python scripts/evaluate_rag.py --skip-ragas    # retrieval metrics only, no judge calls
uv run python scripts/evaluate_rag.py --from-results evaluation/results/latest.json --metrics faithfulness
                                                      # re-score one metric on saved answers
```

### Annotated test set

`evaluation/test_set.json` holds 14 questions with a hand-written reference answer and the uids of the events a good retrieval should surface. The references were written from the indexed data (index of 2026-09-25, events from 2026-09-25 to 2026-12-31) and reviewed by hand. Two question types:

- `recommendation` (12 questions): matching events exist; the answer should list them with dates, venue and price.
- `no_match` (2 questions): nothing really fits (a Wagner opera, a Japanese cooking class); the answer should say so instead of inventing an event.

### Metrics

| Metric | Computed by | Question |
|--------|-------------|----------|
| `retrieval_recall` | exact, from the annotated uids | Which share of the expected events did the retriever return? |
| `retrieval_precision` | exact, from the annotated uids | Which share of the returned events were expected? |
| `faithfulness` | Ragas, LLM judge | Is every claim in the answer supported by the retrieved chunks? Detects hallucinations. |
| `answer_relevancy` | Ragas, LLM judge | Does the answer address the question? Non-answers score low. |

The judge is `ministral-14b-latest`, the strongest model open on the Mistral free tier. Ragas' `context_precision` and `context_recall` were tried and dropped: they ask the judge to grade one chunk at a time against the reference answer, and this judge grades the whole reference instead, returning 0 for chunks that are plainly relevant. The uid-based metrics are exact, free, and answer the same question.

Results are written to `evaluation/results/latest.json` (per question and averages). `tests/test_evaluation.py` reads that file and fails if an average drops below its threshold, so a regression is caught by `pytest` without re-running the judge.

### Results (index of 2026-09-25, 14 questions)

| Metric | Average | Threshold |
|--------|---------|-----------|
| retrieval_recall | 0.96 | 0.6 |
| retrieval_precision | 0.71 | 0.4 |
| faithfulness | 0.83 | 0.5 |
| answer_relevancy | 0.73 | 0.5 |

Reading: the retriever finds almost every expected event (the one miss is a broad Halloween question with 7 expected events and a 5-event cap). Precision is lower on questions with a date constraint ("ce week-end"), because vector similarity ignores dates and returns jazz concerts of every date; only finished events are filtered out, matching a requested period before the search is the next improvement. Answers stay faithful to the retrieved events. The "no match" questions get a clear negative answer followed by the closest alternatives; Ragas scores such answers low on relevancy by design (it treats "nothing matches exactly" as non-committal), which is why one relevancy score is 0 for an answer that is actually right.

The successive runs, all on the same 14 questions and judge, show how the evaluation drove the changes:

| Run | Change | precision | faithfulness | relevancy |
|-----|--------|-----------|--------------|-----------|
| 1 | baseline (10 chunks, temperature 0.2) | 0.75 | 0.83 | 0.77 |
| 2 | 20 chunks, past events filtered | 0.71 | 0.68 | 0.75 |
| 3 | season dates in every prompt, temperature 0 | 0.71 | 0.55 | 0.73 |
| 4 | calendar text only for dated questions, near-match rule | 0.71 | 0.83 | 0.73 |

Run 3 is the useful lesson: a prompt addition that looked harmless (season bounds for every question) made the model restrict undated questions to the coming weekend and cost 0.28 of faithfulness. The evaluation caught it, and run 4 recovered it. Precision between runs 1 and 2 reflects the annotation as much as the system: with 20 candidates, questions with only two expected events now return three extra relevant ones that the reference does not list.

Variance: a single run of a 14B judge moves individual scores by 0.1 to 0.3, so conclusions should rest on averages and on reading the answers, not on one number.

The evaluation runs "as of" the date stored in the test set (`today`), not the real clock, so "ce week-end" and the past-event filter resolve the same way whenever it is run. The full run takes about 25 minutes on the free tier with 2 parallel judge calls. More workers looked faster but made every call slower until they timed out (8 workers: 13 of 14 faithfulness jobs lost). `--from-results` re-scores saved answers without re-running the chain, and results are saved before the judge phase so a network drop cannot lose them.

Two limits of the evaluation, worth knowing:

- A 14B judge on English metric prompts with French content is noisy. A larger judge (`mistral-large`) would be both faster and more reliable; it is not available on the free tier.
- Reference answers are tied to one index snapshot. After a rebuild, the test set must be re-annotated.

### What the evaluation caught

Running the set surfaced four generation failures of the 14B model, each fixed by a prompt rule or by computing in Python what the model got wrong:

- it could not infer that an event running from 29 September to 31 January happens "in October": an explicit list of covered months is now added to each retrieved event;
- it miscounted the days of the coming weekend: the dates are now computed and stated in the prompt;
- it restricted undated questions ("un atelier le mercredi ?") to the coming weekend: a rule now says that undated questions accept any upcoming event;
- it did not map the agenda keyword "Jeunes" to adolescents: the audience keywords are now explained in the prompt;
- given weekend and season dates on every question, it restricted undated questions to the coming weekend: the calendar sentences are now only added when the question contains a relative time expression;
- told to "propose nothing else" when nothing matched, it refused defensible near matches (a public rehearsal for "un spectacle de danse contemporaine"): it now states the mismatch and offers the closest events from the context;
- on a Sunday evening it still listed Saturday's concert for "ce week-end": finished events are now filtered out in Python before the prompt.

Before these fixes, faithfulness was below 0.5 on the affected questions; after, the average is 0.83.

## Tests

```bash
uv run pytest -v
```

Tests are written in Python with [pytest](https://docs.pytest.org/). No other test library is used: fakes are plain Python classes, and `monkeypatch`, `tmp_path` and `pytest.raises` are built into pytest.

Unit tests never call the real APIs. Network calls and embeddings are replaced by fakes so the tests run offline and stay deterministic. Each test checks one rule of the pipeline.

### `tests/test_fetch_events.py`

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_concatenates_pages_and_passes_cursor_back` | Events from several pages are joined in order, and the `after` cursor returned by one page is sent on the next request. | Pagination is the part most likely to silently drop or duplicate events. |
| `test_stops_at_max_events_and_shrinks_last_page` | Fetching stops exactly at `max_events`, the last request asks only for the missing count, and no extra request is made. | Keeps the dataset size predictable and avoids useless API calls. |
| `test_sends_since_date_and_detailed_flag` | The API key, the `timings[gte]` and `timings[lte]` date window and `detailed=1` are really present in the request. | The date scope and the `longDescription` field depend on these parameters. |
| `test_no_upper_bound_when_until_is_omitted` | Without an end date, no `timings[lte]` parameter is sent. | The window end must be optional. |
| `test_raises_when_api_reports_failure` | A response with `success: false` raises an error carrying the API message. | A bad key or quota error must fail loudly, not produce an empty dataset. |

### `tests/test_clean_events.py`

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_clean_text_strips_markdown_and_normalizes_spaces` | Bold markers, headings, links, HTML tags and non-breaking spaces are removed; text and line breaks are kept. | Markup adds noise to the embeddings and to the answer shown to the user. |
| `test_clean_text_handles_missing_text` | `None` and empty strings become `""` without crashing. | Some fields are missing on some events. |
| `test_pick_lang_prefers_french_then_falls_back` | French text is picked when present, otherwise any available language. | Open Agenda text fields are multilingual dicts. |
| `test_flatten_event_produces_flat_record` | One nested event becomes one flat row: keywords joined, sessions counted, venue and public URL filled, no nested values left. | The row must be CSV-friendly and carry the metadata stored next to each vector. |
| `test_clean_events_filters_city_and_duplicates` | Only Paris events are kept ("Paris, 8e" normalized to "Paris"), duplicate uids are dropped. | Enforces the geographic scope of the POC and a clean index. |
| `test_clean_events_drops_events_without_description` | Events with an empty long description are removed. | Nothing to embed means nothing to retrieve. |

### `tests/test_build_index.py`

Embeddings are replaced by a fake model that maps a text to a 4-dimensional vector from letter counts. It is deterministic and needs no API key, yet enough to check that FAISS is wired correctly.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_event_header_contains_who_when_where` | The header prepended to each chunk holds the title, dates, venue and price. | A chunk from the middle of a long text must still say which event it belongs to. |
| `test_short_event_gives_one_chunk_with_header_and_metadata` | A short description gives exactly one chunk, with header, text and full metadata. | The common case must be simple and complete. |
| `test_long_event_is_split_into_overlapping_chunks_sharing_metadata` | A long description gives several chunks, numbered in order, each with the header and the same event uid. | Chunking is where text can get lost or detached from its event. |
| `test_falls_back_to_short_description_when_long_is_empty` | When the long description is empty, the short one is embedded instead. | Never index an event with an empty body. |
| `test_index_roundtrip_and_search` | The index is built, saved, reloaded, and a search returns the right chunk with its metadata. | This is exactly what the API will do at startup and on every question. |
| `test_load_events_reads_csv_without_nan_in_text` | Missing text cells are read as empty strings, not NaN. | NaN in a document would break the header and the embedding call. |

### `tests/test_chain.py`

FAISS is replaced by a fake store that returns scripted chunks, and the LLM by LangChain's `FakeListChatModel`, which returns scripted answers. This isolates the chain logic from both external services.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_dedupe_keeps_first_chunk_per_event_and_caps_count` | Only the best-ranked chunk of each event survives, and the list stops at `max_events`. | A long event must not fill every slot of the answer. |
| `test_describe_today_computes_the_coming_weekend_in_french` | The date sentence gives today and the coming Saturday and Sunday in French, for a weekday, a Saturday and a Sunday. | Python does the calendar arithmetic, not the LLM, which gets it wrong. |
| `test_format_context_numbers_the_events` | Retrieved chunks are rendered as numbered `[Événement i]` blocks. | The prompt must be readable and stable for the LLM. |
| `test_ask_returns_answer_and_sources` | `ask()` returns the LLM answer, asks FAISS for `top_k_chunks`, and lists deduplicated sources in rank order with their URL. | This is the contract the API relies on. |
| `test_prompt_contains_rules_context_and_question` | The rendered prompt holds the rules, today's date, the retrieved chunks and the user question. | The prompt is the whole contract with the model; a missing piece silently degrades answers. |
| `test_generate_retries_on_rate_limit` | Two HTTP 429 responses then a success: the call is retried with waits of 2 s then 4 s. | The free tier rate-limits; the chain must survive it without real waiting in tests. |
| `test_ask_with_no_hits_still_answers` | With no retrieved chunk, the chain still returns an answer and an empty source list. | Empty context is a normal case, not an error. |

### `tests/test_api.py`

FastAPI's `TestClient` calls the app in-process. The assistant dependency is overridden with a fake, so no index is loaded and no LLM is called.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_health` | `/health` returns 200 and `{"status": "ok"}`. | Docker and the demo rely on it. |
| `test_ask_returns_answer_and_sources` | `/ask` returns the answer and sources, and the question is stripped before reaching the chain. | This is the contract with the product and marketing teams. |
| `test_ask_rejects_empty_or_too_short_question` | Missing, blank or 2-character questions return 422 and the chain is never called. | Bad input must fail fast, before spending an LLM call. |
| `test_ask_rejects_malformed_body` | A body without the `question` field returns 422. | Pydantic validation is on. |
| `test_rebuild_requires_api_key` | Missing or wrong `X-API-Key` returns 401. | Rebuilding is slow and costs API calls; it must not be open. |
| `test_rebuild_is_closed_when_no_key_configured` | With no `RAG_API_KEY` in the environment, every call returns 401. | A missing config must close the route, not open it. |
| `test_rebuild_runs_pipeline_and_reloads_assistant` | With the right key, the pipeline is called with the given dates and cap, and the in-memory assistant is replaced. | The fresh index must be served without restarting the API. |
| `test_rebuild_defaults_to_today_plus_60_days` | With no body, the window is today to today + 60 days. | A scheduled rebuild with no parameters must always cover the near future. |
| `test_rebuild_rejects_reversed_dates` | `to_date` before `from_date` returns 422. | Catch a typo before spending a fetch and a re-index. |

### `tests/test_evaluation.py`

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_retrieval_recall` | Share of expected uids found; `None` when nothing is expected. | Exact retrieval metric, no judge needed. |
| `test_retrieval_precision` | Share of retrieved uids that were expected; `None` when nothing is expected or retrieved. | Measures noise in what reaches the LLM. |
| `test_summarize_ignores_missing_values` | Averages skip `None` values instead of counting them as 0. | `no_match` questions must not drag retrieval scores down. |
| `test_test_set_is_well_formed` | Unique ids, non-empty question and reference, valid type, expected uids on recommendation questions. | The annotated set is a deliverable; a typo must fail before a 10-minute run. |
| `test_latest_scores_meet_thresholds` | Averages in `evaluation/results/latest.json` are above the thresholds. Skipped when no results exist. | Regression gate: a prompt or model change that degrades answers fails `pytest`. |

## Project structure

```
.
├── README.md
├── .env.example             # keys to set in .env
├── pyproject.toml           # dependencies (managed by uv); rag/, scripts/ and api/ are installed as packages
├── requirements.txt         # exported from uv.lock, for pip users
├── Dockerfile               # image: Python 3.12 + locked dependencies + code, non-root
├── docker-compose.yml       # one-command run: ports, .env, data volume
├── .dockerignore
├── api/
│   └── main.py              # FastAPI app: /health, /ask, /rebuild
├── rag/
│   └── chain.py             # RagAssistant: retrieval (FAISS) + generation (Mistral), prompt
├── scripts/
│   ├── fetch_events.py      # Open Agenda API -> data/raw/events_raw.json
│   ├── clean_events.py      # raw JSON -> data/processed/events.csv
│   ├── build_index.py       # events.csv -> chunks -> Mistral embeddings -> data/index/
│   ├── ask_question.py      # command-line access to the chain, without the API
│   ├── api_test.py          # functional test of a running API (real index, real LLM)
│   └── evaluate_rag.py      # Ragas + retrieval metrics on the annotated set -> evaluation/results/
├── evaluation/
│   ├── test_set.json        # 14 annotated questions (reference answer, expected event uids)
│   └── results/             # latest.json and timestamped runs
└── tests/
    ├── test_fetch_events.py
    ├── test_clean_events.py
    ├── test_build_index.py
    ├── test_chain.py
    ├── test_api.py
    └── test_evaluation.py
```

## Status

All six steps of the mission are implemented: environment, data pipeline, FAISS index, RAG chain, REST API, evaluation, Docker. Remaining: technical report and presentation.

Known limits: the date filter only removes finished events, it does not yet match a requested period ("en novembre") before the vector search; the index is a snapshot and must be rebuilt to stay current; the data source is the City of Paris agenda, rich on public and cultural events, thin on commercial nightlife; the 14B model available on the free tier still misreads some borderline questions, and a single judge run carries noticeable variance.