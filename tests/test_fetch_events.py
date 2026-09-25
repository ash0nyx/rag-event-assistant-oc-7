"""Unit tests for scripts/fetch_events.py.

We never call the real Open Agenda API here: it is slow, needs a key, and its
content changes every day, so a test could pass today and fail tomorrow.
Instead we replace `requests.get` with a fake that serves pages we control.
That lets us check the *logic* (pagination, cursor, stop condition) in
milliseconds, offline.
"""

from datetime import date

import pytest

from scripts import fetch_events as fe


class FakeResponse:
    """Minimal stand-in for `requests.Response`: only what the script uses."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:  # real one raises on HTTP 4xx/5xx
        pass

    def json(self) -> dict:
        return self._payload


class FakeAPI:
    """Serves a scripted list of pages, one per call, and records the params
    it was called with so tests can inspect what the script sent."""

    def __init__(self, pages: list[dict]):
        self.pages = pages
        self.calls: list[dict] = []  # params of each request, in order

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResponse(self.pages[len(self.calls) - 1])


def make_page(uids: list[int], after: list[str] | None) -> dict:
    """Build a payload shaped like a real Open Agenda response."""
    return {
        "success": True,
        "total": 999,
        "events": [{"uid": uid} for uid in uids],
        "after": after,
    }


def test_concatenates_pages_and_passes_cursor_back(monkeypatch):
    # Two pages of 2 events, then the API says "no more" (empty page, no cursor).
    api = FakeAPI([
        make_page([1, 2], after=["cursor-A"]),
        make_page([3, 4], after=["cursor-B"]),
        make_page([], after=None),
    ])
    # monkeypatch swaps requests.get inside the module for our fake, and
    # restores the real one automatically when the test ends.
    monkeypatch.setattr(fe.requests, "get", api.get)

    events = fe.fetch_events("key", 123, date(2025, 1, 1), max_events=10, page_size=2)

    assert [e["uid"] for e in events] == [1, 2, 3, 4]
    assert "after[]" not in api.calls[0]  # first request starts from the top
    assert api.calls[1]["after[]"] == ["cursor-A"]  # cursor from page 1 sent on request 2
    assert api.calls[2]["after[]"] == ["cursor-B"]


def test_stops_at_max_events_and_shrinks_last_page(monkeypatch):
    # Only one page is needed: max_events=3 with page_size=100 must request size=3.
    api = FakeAPI([make_page([1, 2, 3], after=["cursor-A"])])
    monkeypatch.setattr(fe.requests, "get", api.get)

    events = fe.fetch_events("key", 123, date(2025, 1, 1), max_events=3, page_size=100)

    assert len(events) == 3
    assert len(api.calls) == 1  # no useless extra request
    assert api.calls[0]["size"] == 3


def test_sends_since_date_and_detailed_flag(monkeypatch):
    api = FakeAPI([make_page([1], after=None)])
    monkeypatch.setattr(fe.requests, "get", api.get)

    fe.fetch_events("my-key", 123, date(2026, 9, 25), max_events=1, until=date(2026, 11, 24))

    params = api.calls[0]
    assert params["key"] == "my-key"
    assert params["timings[gte]"] == "2026-09-25"  # window start
    assert params["timings[lte]"] == "2026-11-24"  # window end
    assert params["detailed"] == 1  # needed to get longDescription


def test_no_upper_bound_when_until_is_omitted(monkeypatch):
    api = FakeAPI([make_page([1], after=None)])
    monkeypatch.setattr(fe.requests, "get", api.get)

    fe.fetch_events("my-key", 123, date(2026, 9, 25), max_events=1)

    assert "timings[lte]" not in api.calls[0]


def test_raises_when_api_reports_failure(monkeypatch):
    api = FakeAPI([{"success": False, "message": "invalid key"}])
    monkeypatch.setattr(fe.requests, "get", api.get)

    with pytest.raises(RuntimeError, match="invalid key"):
        fe.fetch_events("bad-key", 123, date(2025, 1, 1), max_events=5)
