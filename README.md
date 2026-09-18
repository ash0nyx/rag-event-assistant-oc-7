# Puls-Events RAG Assistant

Proof of concept for a chatbot that answers questions about upcoming cultural events. It uses Retrieval-Augmented Generation (RAG): event data from the Open Agenda API is embedded with Mistral, indexed in FAISS, and queried through a LangChain pipeline exposed as a REST API.

OpenClassrooms AI Engineer path, project 7.

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- LangChain, FAISS (vector store), Mistral (embeddings and LLM)
- FastAPI (planned), pytest, Ragas (planned)

## Setup

1. Clone the repository and install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file at the project root with your API keys:

   ```
   MISTRAL_API_KEY=your_key_here
   OPENAGENDA_API_KEY=your_key_here
   ```

   Get a Mistral key at [console.mistral.ai](https://console.mistral.ai) and an Open Agenda key from your account settings at [openagenda.com](https://openagenda.com). Both free tiers are enough.

3. Verify the setup:

   ```bash
   uv run python -c "from langchain_community.vectorstores import FAISS; print('ok')"
   ```

## Data pipeline

The data comes from the Open Agenda API, agenda "Que faire à Paris" (uid 648405), the official City of Paris calendar. Scope for the POC: events in Paris with at least one date in the last 365 days or upcoming, capped at 200 events.

1. Fetch raw events:

   ```bash
   uv run python scripts/fetch_events.py            # 200 events by default
   uv run python scripts/fetch_events.py --max-events 50
   ```

   The script paginates with the API cursor and saves the raw JSON to `data/raw/events_raw.json`.

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
| `test_sends_since_date_and_detailed_flag` | The API key, the `timings[gte]` date filter and `detailed=1` are really present in the request. | The "less than one year old" rule and the `longDescription` field depend on these parameters. |
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

## Project structure

```
.
├── README.md
├── pyproject.toml           # dependencies (managed by uv)
├── scripts/
│   ├── fetch_events.py      # Open Agenda API -> data/raw/events_raw.json
│   ├── clean_events.py      # raw JSON -> data/processed/events.csv
│   └── build_index.py       # events.csv -> chunks -> Mistral embeddings -> data/index/
└── tests/
    ├── test_fetch_events.py
    ├── test_clean_events.py
    └── test_build_index.py
```

The structure will grow as the project advances: RAG chain, API, and documentation.

## Status

Work in progress. Done: environment setup, data collection and cleaning, FAISS vector index. Next: RAG chain with LangChain and Mistral.