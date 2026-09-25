# Puls-Events RAG Assistant

Proof of concept for a chatbot that answers questions about upcoming cultural events. It uses Retrieval-Augmented Generation (RAG): event data from the Open Agenda API is embedded with Mistral, indexed in FAISS, and queried through a LangChain pipeline exposed as a REST API.

OpenClassrooms AI Engineer path, project 7.

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- LangChain, FAISS (vector store), Mistral (`mistral-embed` for embeddings, `ministral-14b-latest` for generation)
- FastAPI and uvicorn (REST API), pytest, Ragas (planned)

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

1. **Retrieval.** The question is embedded with `mistral-embed` and FAISS returns the 10 closest chunks. One chunk per event is kept (best-ranked first), at most 5 events. No LLM is involved in this phase.
2. **Generation.** The kept chunks are pasted into a French system prompt together with today's date, and `ministral-14b-latest` writes the answer. The prompt tells the model to use only the provided events, to give title, dates, venue and price, to account for today's date, and to say clearly when nothing matches.

The result holds the answer and the sources (uid, title, dates, venue, URL), so the API can show links and the evaluation step can check that answers stay faithful to the context.

Two details worth knowing:

- The LLM has no clock, and small models are unreliable at calendar arithmetic. Today's date and the dates of the coming weekend are computed in Python and injected in the prompt, in French, so that "ce week-end" or "ce mois-ci" are resolved correctly and past events are not recommended.
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

## Project structure

```
.
├── README.md
├── .env.example             # keys to set in .env
├── pyproject.toml           # dependencies (managed by uv); rag/, scripts/ and api/ are installed as packages
├── api/
│   └── main.py              # FastAPI app: /health, /ask, /rebuild
├── rag/
│   └── chain.py             # RagAssistant: retrieval (FAISS) + generation (Mistral), prompt
├── scripts/
│   ├── fetch_events.py      # Open Agenda API -> data/raw/events_raw.json
│   ├── clean_events.py      # raw JSON -> data/processed/events.csv
│   ├── build_index.py       # events.csv -> chunks -> Mistral embeddings -> data/index/
│   ├── ask_question.py      # command-line access to the chain, without the API
│   └── api_test.py          # functional test of a running API (real index, real LLM)
└── tests/
    ├── test_fetch_events.py
    ├── test_clean_events.py
    ├── test_build_index.py
    ├── test_chain.py
    └── test_api.py
```

The structure will grow as the project advances: evaluation, Docker, and documentation.

## Status

Work in progress. Done: environment setup, data collection and cleaning, FAISS vector index, RAG chain, REST API. Next: annotated test set and Ragas evaluation.

Known limits, to address after a first evaluation baseline: retrieval is purely semantic, so events are not filtered by date before reaching the LLM; the index is a snapshot and must be rebuilt to stay current.