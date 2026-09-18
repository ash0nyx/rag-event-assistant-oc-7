"""Unit tests for scripts/build_index.py.

No Mistral calls: a FakeEmbeddings class turns text into vectors with a
cheap, deterministic rule. That is enough to check chunking, metadata and
that FAISS is wired correctly (save, load, search)."""

import math

import pandas as pd
from langchain_core.embeddings import Embeddings

from scripts.build_index import build_index, event_header, load_events, make_documents


class FakeEmbeddings(Embeddings):
    """Maps a text to a 4-dim vector: counts of the letters a, e, i, o
    (normalized). Texts that share letters land close together, which is all
    a search test needs. Real embeddings are 1024-dim and semantic."""

    def _one(self, text: str) -> list[float]:
        counts = [text.lower().count(c) for c in "aeio"]
        norm = math.sqrt(sum(c * c for c in counts)) or 1.0
        return [c / norm for c in counts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


def make_df(long_description: str = "Une exposition de peinture.") -> pd.DataFrame:
    """Smallest table with the columns build_index.py reads."""
    return pd.DataFrame([{
        "uid": 1,
        "title": "Expo Hilma af Klint",
        "description": "Courte description",
        "long_description": long_description,
        "keywords": "Expo, Peinture",
        "conditions": "Payant, 15 euros",
        "date_range": "19 - 30 août",
        "first_begin": "2026-08-19T08:00:00+02:00",
        "last_end": "2026-08-30T18:00:00+02:00",
        "next_begin": "2026-08-28T08:00:00+02:00",
        "n_timings": 12,
        "venue": "Grand Palais",
        "address": "3 avenue du Général Eisenhower, 75008, Paris",
        "postal_code": 75008,
        "city": "Paris",
        "latitude": 48.86,
        "longitude": 2.31,
        "url": "https://openagenda.com/fr/que-faire-a-paris/events/expo-1",
    }])


def test_event_header_contains_who_when_where():
    header = event_header(make_df().iloc[0])
    assert "Expo Hilma af Klint" in header
    assert "19 - 30 août" in header
    assert "Grand Palais" in header
    assert "15 euros" in header


def test_short_event_gives_one_chunk_with_header_and_metadata():
    docs = make_documents(make_df())
    assert len(docs) == 1
    assert docs[0].page_content.startswith("Événement : Expo Hilma af Klint")
    assert "Une exposition de peinture." in docs[0].page_content
    assert docs[0].metadata["uid"] == 1
    assert docs[0].metadata["url"].endswith("/expo-1")
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[0].metadata["n_chunks"] == 1


def test_long_event_is_split_into_overlapping_chunks_sharing_metadata():
    # 20 paragraphs of ~100 chars: far above a 300-char chunk size.
    long_text = "\n\n".join(f"Paragraphe {i} : " + "blabla " * 12 for i in range(20))
    docs = make_documents(make_df(long_text), chunk_size=300, chunk_overlap=50)

    assert len(docs) > 1
    assert all(d.metadata["uid"] == 1 for d in docs)  # every chunk knows its event
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    assert all(d.page_content.startswith("Événement :") for d in docs)  # header on each


def test_falls_back_to_short_description_when_long_is_empty():
    docs = make_documents(make_df(long_description=""))
    assert len(docs) == 1
    assert "Courte description" in docs[0].page_content


def test_index_roundtrip_and_search(tmp_path):
    # Two events with very different letters so the fake embedding separates them.
    df = pd.concat([make_df("aaaa aaaa"), make_df("oooo oooo")], ignore_index=True)
    df.loc[1, ["uid", "title"]] = [2, "Concert de jazz"]
    docs = make_documents(df)
    index = build_index(docs, FakeEmbeddings())

    assert index.index.ntotal == 2  # one vector per chunk

    # Save then reload, like the API will do at startup.
    index.save_local(str(tmp_path))
    reloaded = type(index).load_local(str(tmp_path), FakeEmbeddings(), allow_dangerous_deserialization=True)
    assert reloaded.index.ntotal == 2

    # Search returns the chunk *and* its metadata.
    hit = reloaded.similarity_search("oooo", k=1)[0]
    assert hit.metadata["uid"] == 2
    assert hit.metadata["title"] == "Concert de jazz"


def test_load_events_reads_csv_without_nan_in_text(tmp_path):
    csv = tmp_path / "events.csv"
    make_df().assign(conditions=None).to_csv(csv, index=False)
    df = load_events(csv)
    assert df.loc[0, "conditions"] == ""  # NaN became empty string
