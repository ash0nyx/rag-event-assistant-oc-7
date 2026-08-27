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

2. Create a `.env` file at the project root with your Mistral API key:

   ```
   MISTRAL_API_KEY=your_key_here
   ```

   Get a key at [console.mistral.ai](https://console.mistral.ai). The free tier is enough.

3. Verify the setup:

   ```bash
   uv run python -c "from langchain_community.vectorstores import FAISS; print('ok')"
   ```

## Project structure

```
.
├── README.md
├── pyproject.toml     # dependencies (managed by uv)
└── main.py            # placeholder, will be replaced by scripts/ and api/
```

The structure will grow as the project advances: data scripts, indexing scripts, RAG chain, API, tests, and documentation.

## Status

Work in progress. Current step: environment setup done, next is Open Agenda data collection.