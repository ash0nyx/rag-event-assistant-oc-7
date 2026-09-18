"""Build the FAISS vector index from the cleaned events table.

Pipeline: events.csv -> one Document per chunk (text + metadata)
          -> Mistral embeddings (1024 floats per chunk) -> FAISS index on disk.

Why chunks: an embedding is a fixed-size summary of its input. A 8000-char
description squeezed into one vector loses detail, so long texts are split
into overlapping pieces and each piece gets its own vector. Each chunk starts
with a short header (title, dates, venue, price) so it stays self-describing
when retrieved on its own.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --chunk-size 600 --chunk-overlap 80
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# langchain-community is being sunset but still ships the only official FAISS
# wrapper. We silence its deprecation warning here; see README for the note.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

PROCESSED_PATH = Path("data/processed/events.csv")
INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = "mistral-embed"
CHUNK_SIZE = 800  # characters; roughly 200 tokens of French text
CHUNK_OVERLAP = 100  # repeated tail so a sentence cut in two is still whole in one chunk

# Columns copied into each chunk's metadata: what the API returns next to an answer.
METADATA_COLUMNS = [
    "uid", "title", "date_range", "first_begin", "last_end", "next_begin",
    "venue", "address", "postal_code", "city", "conditions", "keywords", "url",
]


def load_events(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Read the cleaned table. Text columns are forced to str so NaN never
    leaks into the documents."""
    df = pd.read_csv(path)
    text_cols = ["title", "description", "long_description", "keywords", "conditions", "date_range", "venue"]
    df[text_cols] = df[text_cols].fillna("").astype(str)
    return df


def event_header(row: pd.Series) -> str:
    """Compact, human-readable summary prepended to every chunk of an event.
    Gives the retriever (and the LLM) the who/when/where even if the chunk
    itself is the middle of a long description."""
    parts = [f"Événement : {row['title']}"]
    if row["date_range"]:
        parts.append(f"Dates : {row['date_range']}")
    if row["venue"]:
        parts.append(f"Lieu : {row['venue']}, {row['postal_code']} {row['city']}")
    if row["conditions"]:
        parts.append(f"Tarif : {row['conditions']}")
    if row["keywords"]:
        parts.append(f"Mots-clés : {row['keywords']}")
    return "\n".join(parts)


def make_documents(
    df: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """One event -> one or more Documents (chunk text + metadata)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to cut on paragraph, then line, then sentence, then word boundaries.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for _, row in df.iterrows():
        header = event_header(row)
        body = row["long_description"] or row["description"]
        chunks = splitter.split_text(body)
        metadata = {col: row[col] for col in METADATA_COLUMNS}
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=f"{header}\n\n{chunk}",
                    metadata={**metadata, "chunk_index": i, "n_chunks": len(chunks)},
                )
            )
    return documents


def build_index(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed every document and store vectors + documents in a FAISS index.

    FAISS.from_documents uses IndexFlatL2: exact nearest-neighbour search.
    At a few hundred vectors that is instant; approximate indexes (IVF, HNSW)
    only pay off at millions of vectors."""
    return FAISS.from_documents(documents, embeddings)


def get_embeddings() -> Embeddings:
    """Real Mistral embeddings. Imported lazily so tests never need the SDK
    or an API key."""
    from langchain_mistralai import MistralAIEmbeddings

    load_dotenv()
    # Free tier is rate-limited: retry with a small pause instead of failing.
    return MistralAIEmbeddings(model=EMBEDDING_MODEL, max_retries=5, wait_time=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=INDEX_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    df = load_events(args.input)
    documents = make_documents(df, args.chunk_size, args.chunk_overlap)
    print(f"{len(df)} events -> {len(documents)} chunks (size={args.chunk_size}, overlap={args.chunk_overlap})")

    print(f"Embedding with {EMBEDDING_MODEL} ...")
    index = build_index(documents, get_embeddings())

    args.output.mkdir(parents=True, exist_ok=True)
    index.save_local(str(args.output))
    print(f"Saved index with {index.index.ntotal} vectors to {args.output}")


if __name__ == "__main__":
    main()
