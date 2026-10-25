"""End-to-end tests of the data pipeline and the command-line entry points,
fully offline: Open Agenda is faked, embeddings are FakeEmbeddings, and every
file is written in a temporary folder (monkeypatch.chdir).
"""

import json
import sys
from datetime import date

import pytest
from fastapi import HTTPException

from api import main as api_main
from scripts import build_index as bi
from scripts import clean_events as ce
from scripts import fetch_events as fe
from tests.test_build_index import FakeEmbeddings
from tests.test_clean_events import make_event
from tests.test_fetch_events import FakeAPI, make_page

EVENTS = [make_event(1), make_event(2, city="Paris, 8e"), make_event(3, city="Bobigny")]  # 1 suburb dropped


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in an empty folder: the pipeline writes to relative data/ paths."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_rebuild_index_runs_fetch_clean_and_index(workdir, monkeypatch):
    """What POST /rebuild does, minus HTTP: three steps, files on disk, counts back."""
    monkeypatch.setenv("OPENAGENDA_API_KEY", "key")
    calls = []
    monkeypatch.setattr(fe, "fetch_events", lambda key, uid, since, n, until=None: calls.append((since, until, n)) or EVENTS)
    monkeypatch.setattr(bi, "get_embeddings", FakeEmbeddings)

    counts = api_main.rebuild_index(max_events=50, since=date(2026, 10, 1), until=date(2026, 12, 31))

    assert calls == [(date(2026, 10, 1), date(2026, 12, 31), 50)]
    assert counts == {"fetched": 3, "cleaned": 2, "chunks": 2, "vectors": 2}
    assert (workdir / "data/raw/events_raw.json").exists()
    assert (workdir / "data/processed/events.csv").exists()
    assert (workdir / "data/index/index.faiss").exists()


def test_rebuild_index_without_openagenda_key_is_a_server_error(workdir, monkeypatch):
    monkeypatch.delenv("OPENAGENDA_API_KEY", raising=False)
    with pytest.raises(HTTPException) as err:
        api_main.rebuild_index(max_events=5, since=date(2026, 10, 1), until=date(2026, 10, 2))
    assert err.value.status_code == 500


def test_fetch_events_cli_writes_the_raw_dump_with_its_window(workdir, monkeypatch):
    monkeypatch.setenv("OPENAGENDA_API_KEY", "key")
    monkeypatch.setattr(fe, "load_dotenv", lambda: None)  # do not read a real .env
    api = FakeAPI([make_page([1, 2], after=None)])
    monkeypatch.setattr(fe.requests, "get", api.get)
    monkeypatch.setattr(sys, "argv", ["fetch_events.py", "--from-date", "2026-10-01", "--to-date", "2026-10-31",
                                      "--max-events", "10", "--output", "raw.json"])

    fe.main()

    dump = json.loads((workdir / "raw.json").read_text(encoding="utf-8"))
    assert dump["count"] == 2
    assert (dump["since"], dump["until"]) == ("2026-10-01", "2026-10-31")
    assert api.calls[0]["timings[lte]"] == "2026-10-31"


def test_fetch_events_cli_stops_without_api_key(workdir, monkeypatch):
    monkeypatch.delenv("OPENAGENDA_API_KEY", raising=False)
    monkeypatch.setattr(fe, "load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fetch_events.py"])
    with pytest.raises(SystemExit, match="OPENAGENDA_API_KEY"):
        fe.main()


def test_clean_then_build_index_clis(workdir, monkeypatch):
    (workdir / "raw.json").write_text(json.dumps({"events": EVENTS}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clean_events.py", "--input", "raw.json", "--output", "out/events.csv"])
    ce.main()
    assert (workdir / "out/events.csv").exists()

    monkeypatch.setattr(bi, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(sys, "argv", ["build_index.py", "--input", "out/events.csv", "--output", "idx", "--chunk-size", "300"])
    bi.main()
    assert (workdir / "idx/index.faiss").exists() and (workdir / "idx/index.pkl").exists()


def test_api_startup_loads_the_assistant_and_the_database(workdir, monkeypatch):
    """The lifespan: build the assistant once and open the interaction store."""
    from fastapi.testclient import TestClient

    from tests.test_api import FakeAssistant

    monkeypatch.setenv("INTERACTIONS_DB_URL", f"sqlite:///{workdir}/startup.db")
    monkeypatch.setattr(api_main, "build_assistant", FakeAssistant)
    monkeypatch.setattr(api_main, "load_dotenv", lambda: None)
    with TestClient(api_main.app) as client:  # entering the client runs the lifespan
        body = client.post("/ask", json={"question": "Une expo ?"}).json()
    assert body["answer"] == "Je vous recommande Expo A."
    assert (workdir / "startup.db").exists()
