"""Fetch cultural events from the Open Agenda API and save them as raw JSON.

The raw dump is kept as-is (no cleaning) so the next step can be re-run
without hitting the API again. Cleaning happens in a separate script.

Usage:
    uv run python scripts/fetch_events.py                       # 500 events, today to +60 days
    uv run python scripts/fetch_events.py --max-events 50
    uv run python scripts/fetch_events.py --since-days 30 --until-days 90
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.openagenda.com/v2"
AGENDA_UID = 648405  # "Que faire à Paris", the official City of Paris agenda
PAGE_SIZE = 100  # API max is 300; 100 keeps each response light
RAW_PATH = Path("data/raw/events_raw.json")


def fetch_events(
    api_key: str,
    agenda_uid: int,
    since: date,
    max_events: int,
    page_size: int = PAGE_SIZE,
    until: date | None = None,
) -> list[dict]:
    """Return up to `max_events` events with at least one session between
    `since` and `until` (both inclusive; no upper bound if `until` is None).

    Open Agenda paginates with a cursor: each response carries an `after`
    value that must be sent back to get the next page. We loop until we have
    enough events or the API stops returning any.
    """
    url = f"{BASE_URL}/agendas/{agenda_uid}/events"
    events: list[dict] = []
    after: list[str] | None = None

    while len(events) < max_events:
        params: dict = {
            "key": api_key,
            "size": min(page_size, max_events - len(events)),
            "detailed": 1,  # include longDescription and other full fields
            "timings[gte]": since.isoformat(),  # window start
        }
        if until:
            params["timings[lte]"] = until.isoformat()  # window end
        if after:
            params["after[]"] = after

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", False):
            raise RuntimeError(f"Open Agenda API error: {payload}")

        page = payload.get("events", [])
        if not page:
            break  # no more events in the window
        events.extend(page)
        after = payload.get("after")
        print(f"  fetched {len(events)} / {payload.get('total')} available")
        if not after:
            break

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=500)
    parser.add_argument("--since-days", type=int, default=0, help="window start: today minus N days")
    parser.add_argument("--until-days", type=int, default=60, help="window end: today plus N days")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--output", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAGENDA_API_KEY")
    if not api_key:
        raise SystemExit("OPENAGENDA_API_KEY is missing. Add it to your .env file.")

    since = date.today() - timedelta(days=args.since_days)
    until = date.today() + timedelta(days=args.until_days)
    print(f"Fetching events from agenda {AGENDA_UID} between {since} and {until} ...")
    events = fetch_events(api_key, AGENDA_UID, since, args.max_events, args.page_size, until=until)

    # Wrap the events with a bit of provenance: useful for the report and for tests.
    dump = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "agenda_uid": AGENDA_UID,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "count": len(events),
        "events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
