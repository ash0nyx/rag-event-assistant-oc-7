"""Turn the raw Open Agenda dump into a flat, clean table of events.

Why a separate step: the raw JSON is nested (multilingual dicts, location
sub-object, hundreds of timings per event) and full of markdown. The RAG
pipeline needs one clean text per event plus flat metadata (dates, venue,
price) that FAISS can store alongside each vector.

Usage:
    uv run python scripts/clean_events.py
    uv run python scripts/clean_events.py --input data/raw/events_raw.json --output data/processed/events.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/events_raw.json")
PROCESSED_PATH = Path("data/processed/events.csv")
AGENDA_SLUG = "que-faire-a-paris"  # used to build the public event URL
LANG = "fr"
CITY_PREFIX = "Paris"  # the agenda also lists a few suburbs; we keep Paris only

# Markdown / HTML patterns found in longDescription (see profiling in the report).
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) -> text
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)  # "## Title" -> "Title"
_MD_EMPHASIS = re.compile(r"\*\*|__")  # **bold** / __bold__ -> bold
_SPACES = re.compile(r"[ \t\xa0]+")  # runs of spaces, incl. non-breaking ones
_BLANK_LINES = re.compile(r"\n{3,}")  # 3+ newlines -> one blank line


def clean_text(text: str | None) -> str:
    """Strip markdown/HTML and normalize whitespace. Keeps line breaks and
    bullet dashes: they carry structure that helps both chunking and the LLM."""
    if not text:
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def pick_lang(field: dict | None, lang: str = LANG) -> str | None:
    """Open Agenda text fields are dicts like {"fr": "...", "en": "..."}.
    Return the requested language, or any available one as a fallback."""
    if not field:
        return None
    return field.get(lang) or next(iter(field.values()), None)


def flatten_event(event: dict) -> dict:
    """One raw event (nested JSON) -> one flat record (a future CSV row)."""
    location = event.get("location") or {}
    keywords = (event.get("keywords") or {}).get(LANG) or []
    return {
        "uid": event["uid"],
        "title": pick_lang(event.get("title")),
        "description": clean_text(pick_lang(event.get("description"))),
        "long_description": clean_text(pick_lang(event.get("longDescription"))),
        "keywords": ", ".join(keywords),
        "conditions": clean_text(pick_lang(event.get("conditions"))),  # price / access
        "date_range": pick_lang(event.get("dateRange")),  # human readable, e.g. "19 - 30 août"
        "first_begin": (event.get("firstTiming") or {}).get("begin"),
        "last_end": (event.get("lastTiming") or {}).get("end"),
        "next_begin": (event.get("nextTiming") or {}).get("begin"),
        "n_timings": len(event.get("timings") or []),  # number of sessions
        "venue": location.get("name"),
        "address": location.get("address"),
        "postal_code": location.get("postalCode"),
        "city": location.get("city"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "url": f"https://openagenda.com/fr/{AGENDA_SLUG}/events/{event['slug']}",
    }


def clean_events(events: list[dict]) -> pd.DataFrame:
    """Flatten, filter and deduplicate. Returns the table ready for indexing."""
    df = pd.DataFrame([flatten_event(e) for e in events])
    n_raw = len(df)

    df = df.drop_duplicates(subset="uid")
    # Keep Paris only. "Paris, 8e" style values are normalized to "Paris";
    # the arrondissement stays available through postal_code.
    in_paris = df["city"].fillna("").str.startswith(CITY_PREFIX)
    df = df[in_paris].assign(city=CITY_PREFIX)
    # No text means nothing to embed: drop.
    df = df[df["title"].fillna("").str.strip().ne("") & df["long_description"].str.strip().ne("")]

    print(f"Cleaned {len(df)} events (from {n_raw} raw, {n_raw - len(df)} dropped)")
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    df = clean_events(raw["events"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
