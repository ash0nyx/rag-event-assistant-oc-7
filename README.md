# Puls-Events RAG Assistant

Proof of concept for a chatbot that answers questions about upcoming cultural events in Paris. It uses Retrieval-Augmented Generation (RAG): event data from the Open Agenda API is embedded with Mistral, indexed in FAISS, and queried through a LangChain pipeline exposed as a REST API. A Streamlit app provides the chat, with thumbs up/down feedback, and a dashboard of usage and satisfaction. An intent classifier keeps greetings and off-topic questions away from the RAG.

OpenClassrooms AI Engineer path, project 7. The technical report (in French, following the OpenClassrooms template) is in [docs/rapport_technique.md](docs/rapport_technique.md); this README is the developer documentation.

## Quick start

With Docker, once `.env` is filled in (see [Setup](#setup)) and the index is built (see [Data pipeline](#data-pipeline)):

```bash
docker compose up --build
```

| URL | What |
|-----|------|
| http://localhost:8501 | Chat and dashboard (Streamlit) |
| http://localhost:8000/docs | REST API with Swagger |

A VPN with a kill switch can block these local URLs; see [Docker](#docker).

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- LangChain, FAISS (vector store), Mistral (`mistral-embed` for embeddings, `ministral-14b-latest` for generation)
- FastAPI and uvicorn (REST API), Streamlit (chat and dashboard), SQLAlchemy + SQLite (interaction log and feedback)
- pytest, Ragas (evaluation), Docker (local deployment), Langfuse (optional tracing)

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
| `POST` | `/ask` | Body `{"question": "..."}`. Returns the answer, the retrieved sources, the detected `intent`, the `latency_ms` and an `interaction_id`. Every call is logged. Questions shorter than 3 characters or blank are rejected with a 422. |
| `POST` | `/feedback` | Body `{"interaction_id": 12, "score": 1, "comment": "..."}` with `score` 1 (thumbs up) or -1 (thumbs down). Stores the vote on the logged answer (a second vote replaces the first) and sends it to Langfuse as a score. Unknown id: 404. |
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

## Chat app, intent classifier and user feedback

```bash
uv run uvicorn api.main:app            # terminal 1: the API
uv run streamlit run ui/Chat.py        # terminal 2: http://localhost:8501
```

### How to use the app

1. Open http://localhost:8501. The chat page shows four example questions; click one, or type your own in the box at the bottom.
2. Read the answer. Under it, **Sources** lists the events used, with links to their Open Agenda pages.
3. Rate the answer with the thumbs. After a thumbs down, a box asks what was wrong; type a comment and click **Envoyer**. You can change your vote.
4. The sidebar shows whether the API is reachable, the same example questions (a click starts a new conversation with that question), and **Nouvelle conversation** to clear the chat.
5. Open **Tableau de bord** in the sidebar's page list to see the statistics. Click **Rafraîchir** to load the latest interactions (the page caches its data for 30 seconds).

Try "Bonjour" or "Quel temps fera-t-il demain ?" to see the intent classifier answer instantly without the RAG.

### What each page does

The Streamlit app has two pages:

- **Chat** (`ui/Chat.py`): ask a question, read the answer with its sources as links, rate it with a thumbs up or down, and after a thumbs down explain what was wrong. The page only talks to the API over HTTP (`/ask`, `/feedback`): it holds no model and no index, so any other front-end could replace it.
- **Tableau de bord** (`ui/pages/1_Tableau_de_bord.py`): headline numbers (questions, satisfaction rate, feedback rate, median RAG latency, share answered without RAG), questions per day, split by intent, latency over time, the answers users rated badly with their comments, the full interaction table, and the latest offline evaluation scores. It reads the interactions table directly: it needs data, not the model.

**Intent classifier** (`rag/intent.py`). Before any retrieval, each message is classified as `events` (goes to the RAG), `chat` (greeting, thanks, "who are you?") or `off_topic` (weather, maths, code, news). The last two get a fixed answer, with no retrieval and no 14B call. Two stages, cheapest first: regular expressions for obvious small talk, then `ministral-3b-latest` answering with one label. When the model's answer cannot be parsed or the call fails, the default is `events`: running the RAG for nothing is cheaper than refusing a real question.

Measured on `evaluation/intent_set.json` (32 labelled messages: the 14 RAG test questions plus extra event, small-talk and off-topic ones) with `uv run python scripts/evaluate_intent.py`:

| Accuracy | Real event questions refused | Median latency |
|----------|------------------------------|----------------|
| 0.97 | 0 | 0.24 s |

The one miss sends "Que sais-tu faire ?" to the RAG instead of the fixed presentation, the safe direction. End to end in Docker, a greeting is answered in under a millisecond, an off-topic question in about 0.5 s, and an events question in about 3.5 s.

**Interaction log and feedback** (`api/database.py`). Every `/ask` call is stored with SQLAlchemy in `data/interactions.db` (SQLite, on the Docker volume): question, intent, answer, sources, latency, Langfuse trace id, then the user's vote and comment. Switching to PostgreSQL is a change of `INTERACTIONS_DB_URL`, not of code. Logging never blocks an answer: if the write fails, the answer is still returned without an id. With Langfuse enabled, each vote is also attached to the question's trace as a `user_feedback` score, so traces can be filtered by satisfaction.

Not implemented on purpose: conversation history. Each question is answered on its own, as in the brief; follow-up questions like "et le dimanche ?" would need the last exchanges in the prompt.

## Docker

Run the API and the Streamlit app in containers, without installing Python or uv on the machine. The index must exist first (`data/index/`, built by the pipeline above or by a `/rebuild` call).

```bash
docker compose up --build        # build the image, start the API (port 8000) and the app (port 8501)
docker compose up -d             # same, in the background
docker compose logs -f           # follow the logs
docker compose down              # stop and remove the containers
```

Both services use the same image. The `ui` service waits for the API to be healthy, reaches it at `http://api:8000` on the compose network, and mounts `data/` and `evaluation/results/` read-only for the dashboard.

Without Compose:

```bash
docker build -t puls-events-rag .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data puls-events-rag
```

Then `http://localhost:8000/docs`, `http://localhost:8501` and `uv run python scripts/api_test.py` work exactly as with a local server.

How the image is built and run, and why:

- **In the image**: Python 3.12, the exact versions from `uv.lock` (installed with `uv sync --frozen`), and the code. Dependencies are installed before the code is copied, so a code change reuses the cached install.
- **Outside the image**: secrets, passed at run time with `--env-file .env`, and `data/`, mounted as a volume at `/app/data`. A `/rebuild` call from the container writes the new index to the host folder, so refreshing the data never needs a new image.
- **Ports**: uvicorn listens on `0.0.0.0:8000` inside the container (not `127.0.0.1`, which would be unreachable from outside), and `-p 8000:8000` publishes it on the host.
- **Non-root**: the process runs as user `app` (uid 1000), so files written to the volume belong to the host user and a compromised API cannot act as root.
- **Health check**: Docker calls `/health` every 30 s; `docker ps` shows the container as healthy or unhealthy.
- `.dockerignore` keeps `.venv`, `data/`, `.env`, tests and private docs out of the build.

Image size is about 1.2 GB, mostly the tokenizer and numeric stack; acceptable for a local POC.

Troubleshooting: a VPN with a kill switch (NordVPN and similar) blocks traffic between the host and Docker's network, so `localhost:8000` hangs even though the container reports healthy. Disconnect the VPN or allowlist Docker's subnet (`nordvpn allowlist add subnet 172.17.0.0/16`).

## Observability with Langfuse (optional)

The evaluation below scores the system once, on a fixed test set. Langfuse records what happens on every real question: a trace per `/ask` call with the question, the events retrieved and how long that took, the exact prompt, the model's answer, latency and token usage. The dashboard shows traces, latency percentiles and the questions that got a "nothing matches" answer.

Setup: create a free project at [cloud.langfuse.com](https://cloud.langfuse.com) (EU region), create API keys, and add them to `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

That is all: `rag/chain.py` detects the keys and attaches Langfuse's LangChain callback to the chain, with a root span `ask` and a child span `retrieve`. Without the keys, nothing is imported and nothing is sent. The evaluation script forces tracing off so its 14 synthetic questions do not mix with real ones. The CLI script flushes traces before exiting; the API flushes at shutdown.

Why Langfuse and not only Ragas: Ragas answers "how good is the system on these questions" before shipping; Langfuse answers "what is it doing right now" after shipping, and its evaluations can run on real traffic. Langfuse is open source and can be self-hosted (Docker Compose with Postgres, ClickHouse, Redis and MinIO), the option a company would take for private data; the cloud free tier is enough here because the traces only contain public event listings.

## Evaluation

```bash
uv run python scripts/evaluate_rag.py --index-dir evaluation/index_snapshot   # full annotated set, about 25 minutes
uv run python scripts/evaluate_rag.py --index-dir evaluation/index_snapshot --limit 3       # quick check
uv run python scripts/evaluate_rag.py --index-dir evaluation/index_snapshot --skip-ragas    # retrieval metrics only, no judge calls
uv run python scripts/evaluate_rag.py --from-results evaluation/results/latest.json --metrics faithfulness
                                                      # re-score one metric on saved answers
```

The evaluation runs against `evaluation/index_snapshot/`, a committed copy (28 MB) of the index the test set was annotated on. The live index in `data/index/` changes with every rebuild; the snapshot does not, so the reference answers stay valid and the run is reproducible on any machine, including the GitHub Actions runner.

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

### Results (14 questions, index snapshot of 2026-09-25, last run 2026-09-27)

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

## Continuous integration

Two GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | every push to `main` and every pull request | Installs the locked dependencies with uv and runs the 97 unit tests with coverage. Offline, no secret, about a minute. A failing test, or coverage under 85 %, blocks the pull request. The HTML coverage report is uploaded as an artifact. |
| `evaluate.yml` | manual (Actions tab, "Run workflow") | Runs `evaluate_rag.py` against the committed index snapshot, then the regression gate, and uploads `evaluation/results/` as a downloadable artifact. Needs the `MISTRAL_API_KEY` repository secret. About 25 minutes; a `skip_ragas` checkbox gives the exact retrieval metrics in 2 minutes. |

The evaluation is not run on every push on purpose: it costs API calls and half an hour. It is run after a prompt, model or retrieval change.

## Tests

```bash
uv run pytest -v
uv run pytest --cov --cov-report=term --cov-report=html   # with coverage; open htmlcov/index.html
```

### Coverage

Coverage is the share of our code lines executed by the tests. 97 tests, 95 % overall:

| File | Coverage | Not covered |
|------|----------|-------------|
| `api/database.py` | 100 % | |
| `api/main.py` | 100 % | |
| `scripts/clean_events.py` | 100 % | |
| `scripts/fetch_events.py` | 100 % | |
| `scripts/build_index.py` | 95 % | creation of the real Mistral embeddings client |
| `ui/Chat.py` | 94 % | network-error toasts |
| `ui/pages/1_Tableau_de_bord.py` | 93 % | a few empty-data branches |
| `rag/intent.py` | 92 % | creation of the real `ministral-3b` client |
| `rag/chain.py` | 89 % | loading the real index and Mistral clients |
| **Total** | **95 %** | |

What is not covered is the wiring to the real Mistral API, exercised by `scripts/api_test.py` and the evaluation instead. Four command-line tools that only run against the real APIs (`api_test.py`, `ask_question.py`, `evaluate_rag.py`, `evaluate_intent.py`) are excluded from the measure in `pyproject.toml`; their logic lives in measured modules.

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
| `test_traced_ask_records_spans_and_returns_the_trace_id` | With a fake Langfuse client, a question opens the `ask` and `retrieve` spans with the right content and returns the trace id. | The trace id links a user's vote to the trace. |
| `test_push_feedback_score_sends_a_boolean_score` | A thumbs down becomes a `user_feedback` score of 0 with its comment on the trace; no trace id, nothing sent. | Feedback visible in Langfuse. |
| `test_push_feedback_score_is_silent_without_keys` / `test_flush_traces_only_with_keys` | Without keys nothing is sent and nothing breaks. | Observability stays optional. |
| `test_tracing_is_off_without_langfuse_keys` | Without `LANGFUSE_*` keys tracing is off and `langfuse` is not imported; an explicit `tracing=False` wins over the keys. | Observability must stay optional: tests, CI and Docker work without an account. |

### `tests/test_pipeline.py`

The whole pipeline and the command-line entry points, offline: Open Agenda faked, `FakeEmbeddings`, files in a temporary folder.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_rebuild_index_runs_fetch_clean_and_index` | The `/rebuild` pipeline fetches with the requested window, cleans (suburb dropped), indexes, writes the three outputs and returns the counts. | The route that refreshes production data. |
| `test_rebuild_index_without_openagenda_key_is_a_server_error` | No Open Agenda key: a clear 500, nothing written. | Misconfiguration fails loudly. |
| `test_fetch_events_cli_writes_the_raw_dump_with_its_window` | `--from-date`/`--to-date` reach the API and the dump records its window. | The CI evaluation and the manual refresh use this command. |
| `test_fetch_events_cli_stops_without_api_key` | Missing key: the script stops with an explicit message. | |
| `test_clean_then_build_index_clis` | The clean and build commands chain from raw JSON to a saved FAISS index. | The documented way to rebuild by hand. |
| `test_api_startup_loads_the_assistant_and_the_database` | The API startup builds the assistant once and opens the interaction database. | The lifespan wiring. |

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
| `test_ask_logs_the_interaction_and_returns_its_id` | `/ask` returns `intent`, `latency_ms` and an `interaction_id`, and the row is in the database with its sources. | The id is what links a vote to an answer. |
| `test_feedback_is_stored_on_the_interaction` | `/feedback` stores the score and the trimmed comment on the right row. | The feedback loop of the course. |
| `test_feedback_on_unknown_interaction_is_404` | A vote on a missing id returns 404. | No silent loss of feedback. |
| `test_feedback_rejects_scores_other_than_plus_or_minus_one` | 0, 2 or a string are rejected with 422. | Only thumbs up or down reach the database. |
| `test_answer_is_returned_even_if_logging_fails` | With a broken database the user still gets the answer, without an id. | Logging must never cost an answer. |

### `tests/test_intent.py`

The small model is replaced by `FakeListChatModel` returning scripted labels.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_small_talk_is_caught_by_rules_without_calling_the_model` | "Bonjour", "Merci beaucoup.", "OK"... are `chat` without a model call. | Free and instant for the most frequent messages. |
| `test_greeting_followed_by_a_request_goes_to_the_model` | "Bonjour, un concert ce soir ?" is not swallowed by the rules. | Rules only take whole small-talk messages. |
| `test_parse_intent` | Case, dashes, punctuation and extra words are tolerated; unparseable output falls back to `events`. | Small models do not always answer with exactly one word. |
| `test_model_error_falls_back_to_events` | An API error during classification sends the question to the RAG. | The classifier must never block an answer. |
| `test_without_a_model_everything_but_small_talk_is_events` | With no model configured, only the rules apply. | Safe degraded mode. |
| `test_off_topic_question_gets_fixed_answer_and_skips_retrieval` | An off-topic question gets the fixed answer, no sources, and FAISS is never queried. | The point of the classifier: no retrieval, no 14B call. |
| `test_events_question_goes_through_the_rag` | An events question still gets the RAG answer and its sources. | The classifier must not change the main path. |

### `tests/test_database.py`

A real SQLite file in a temporary folder rather than mocks: as fast, and it also checks the SQL.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_log_interaction_returns_increasing_ids_and_keeps_fields` | Ids increase, newest first, sources survive the JSON round trip, trace id kept. | The dashboard and the feedback depend on these fields. |
| `test_log_feedback_updates_and_can_be_changed` | A second vote replaces the first. | Users change their mind. |
| `test_log_feedback_unknown_id_returns_none` | Unknown id: nothing written, `None` returned. | Lets the API answer 404. |
| `test_blank_comment_is_stored_as_none` | A comment of spaces is stored as empty. | Clean data for the dashboard. |
| `test_list_interactions_filters_by_feedback` | Positive, negative, unrated filters and the row limit. | The dashboard's filter. |
| `test_summary_counts_and_rates` | Totals, satisfaction and feedback rates, split by intent; no division by zero when nothing is rated. | The headline numbers of the dashboard. |

### `tests/test_ui.py`

Streamlit's `AppTest` runs each page headlessly; the API is faked by patching `requests`, the dashboard reads a temporary database.

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_chat_shows_answer_sources_and_sends_a_vote` | A question shows the answer and the source links; a thumbs up posts score 1. | The main user path. |
| `test_chat_down_vote_offers_a_comment_box` | A thumbs down posts -1 at once, then the typed comment is sent. | Negative feedback is the useful kind; the comment says why. |
| `test_chat_reports_an_unreachable_api` | With the API down the page shows a clear message instead of crashing. | Demo-day resilience. |
| `test_example_buttons_ask_the_question_and_history_shows_no_stray_values` | A suggestion button sends its question; after a second question, the history is re-rendered once, with no stray `None`. | Regression test for a bug found by hand: Streamlit "magic" displayed the value of a one-line `if/else`. |
| `test_examples_stay_clickable_after_a_first_question` | After a first question, the main-page suggestions are hidden; a sidebar example starts a fresh conversation with that question and its answer. | Regression test for a lost click found by hand. |
| `test_dashboard_with_no_data_says_so` | Empty database: an explanatory message, no error. | First run. |
| `test_dashboard_shows_kpis_and_negative_feedback` | Question count, satisfaction, median latency and the badly rated answer are shown. | The dashboard's numbers come from the data. |

### `tests/test_evaluation.py`

| Test | What it checks | Why it matters |
|------|----------------|----------------|
| `test_retrieval_recall` | Share of expected uids found; `None` when nothing is expected. | Exact retrieval metric, no judge needed. |
| `test_retrieval_precision` | Share of retrieved uids that were expected; `None` when nothing is expected or retrieved. | Measures noise in what reaches the LLM. |
| `test_summarize_ignores_missing_values` | Averages skip `None` values instead of counting them as 0. | `no_match` questions must not drag retrieval scores down. |
| `test_test_set_is_well_formed` | Unique ids, non-empty question and reference, valid type, expected uids on recommendation questions. | The annotated set is a deliverable; a typo must fail before a 10-minute run. |
| `test_latest_scores_meet_thresholds` | Averages in `evaluation/results/latest.json` are above the thresholds. Skipped when no results exist. | Regression gate: a prompt or model change that degrades answers fails `pytest`. |
| `test_intent_set_is_well_formed` | The intent set covers the three classes, no empty message. | A typo must fail before a run. |
| `test_intent_classifier_meets_thresholds` | Latest intent results: accuracy at least 0.85 and zero real event questions refused. | Refusing a real question is the costly error. |

## Project structure

```
.
├── README.md
├── docs/
│   ├── rapport_technique.md # technical report (French, OC template)
│   ├── Sazonova_Kseniia_2_PRES_082026.pdf # defense slides (French, 15 slides)
│   ├── architecture.dot     # architecture diagram source (Graphviz)
│   └── architecture.png
├── .env.example             # keys to set in .env
├── pyproject.toml           # dependencies (managed by uv); rag/, scripts/ and api/ are installed as packages
├── requirements.txt         # exported from uv.lock, for pip users
├── Dockerfile               # image: Python 3.12 + locked dependencies + code, non-root
├── docker-compose.yml       # one-command run: api (8000) and ui (8501), .env, data volume
├── .dockerignore
├── api/
│   ├── main.py              # FastAPI app: /health, /ask, /feedback, /rebuild
│   └── database.py          # SQLAlchemy interaction log and feedback (SQLite by default)
├── rag/
│   ├── chain.py             # RagAssistant: intent, retrieval (FAISS), generation (Mistral), prompt
│   └── intent.py            # intent classifier: rules + ministral-3b, fixed answers
├── ui/
│   ├── Chat.py              # Streamlit chat with thumbs up/down and comments (talks to the API)
│   └── pages/1_Tableau_de_bord.py  # Streamlit dashboard: usage, satisfaction, latency, evaluation
├── scripts/
│   ├── fetch_events.py      # Open Agenda API -> data/raw/events_raw.json
│   ├── clean_events.py      # raw JSON -> data/processed/events.csv
│   ├── build_index.py       # events.csv -> chunks -> Mistral embeddings -> data/index/
│   ├── ask_question.py      # command-line access to the chain, without the API
│   ├── api_test.py          # functional test of a running API (real index, real LLM)
│   ├── evaluate_rag.py      # Ragas + retrieval metrics on the annotated set -> evaluation/results/
│   └── evaluate_intent.py   # intent classifier accuracy on evaluation/intent_set.json
├── .github/workflows/
│   ├── ci.yml               # unit tests on every push and pull request
│   └── evaluate.yml         # on-demand RAG evaluation, results uploaded as artifact
├── evaluation/
│   ├── test_set.json        # 14 annotated questions (reference answer, expected event uids)
│   ├── intent_set.json      # 32 messages labelled events / chat / off_topic
│   ├── index_snapshot/      # the FAISS index the test set was annotated on (committed, 28 MB)
│   └── results/             # latest.json and timestamped runs
└── tests/
    ├── test_fetch_events.py
    ├── test_clean_events.py
    ├── test_build_index.py
    ├── test_chain.py
    ├── test_api.py
    ├── test_intent.py
    ├── test_database.py
    ├── test_ui.py
    ├── test_pipeline.py
    └── test_evaluation.py
```

## Status

All six steps of the mission are implemented: environment, data pipeline, FAISS index, RAG chain, REST API, evaluation, Docker. Defended in October 2026; technical report and slides in `docs/`. Added after the defense, following the end of the OpenClassrooms RAG course: Streamlit chat and dashboard, intent classifier, interaction log with user feedback.

Known limits: the date filter only removes finished events, it does not yet match a requested period ("en novembre") before the vector search; the index is a snapshot and must be rebuilt to stay current; the data source is the City of Paris agenda, rich on public and cultural events, thin on commercial nightlife; the 14B model available on the free tier still misreads some borderline questions, and a single judge run carries noticeable variance.