"""Unit tests for scripts/clean_events.py.

Pure functions in, so no mocking needed: we hand-build tiny fake events and
check the transformation. Each test targets one rule of the cleaning step."""

from scripts.clean_events import clean_events, clean_text, flatten_event, pick_lang


def make_event(uid: int, city: str = "Paris", long_desc: str = "Une expo.") -> dict:
    """Smallest raw event that flatten_event() accepts."""
    return {
        "uid": uid,
        "slug": f"event-{uid}",
        "title": {"fr": f"Événement {uid}"},
        "description": {"fr": "Courte description"},
        "longDescription": {"fr": long_desc},
        "keywords": {"fr": ["Expo", "Tout public"]},
        "conditions": {"fr": "Gratuit"},
        "dateRange": {"fr": "19 - 30 août"},
        "firstTiming": {"begin": "2026-08-19T08:00:00+02:00", "end": "2026-08-19T18:00:00+02:00"},
        "lastTiming": {"begin": "2026-08-30T08:00:00+02:00", "end": "2026-08-30T18:00:00+02:00"},
        "nextTiming": {"begin": "2026-08-28T08:00:00+02:00", "end": "2026-08-28T18:00:00+02:00"},
        "timings": [{}, {}, {}],
        "location": {
            "name": "Lieu test",
            "address": "1 rue Test, 75004, Paris",
            "postalCode": "75004",
            "city": city,
            "latitude": 48.85,
            "longitude": 2.35,
        },
    }


def test_clean_text_strips_markdown_and_normalizes_spaces():
    raw = "## Titre\n\n**Gras** et [un lien](https://example.org)\xa0!  <br>Fin"
    assert clean_text(raw) == "Titre\n\nGras et un lien ! Fin"


def test_clean_text_handles_missing_text():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_pick_lang_prefers_french_then_falls_back():
    assert pick_lang({"fr": "Bonjour", "en": "Hello"}) == "Bonjour"
    assert pick_lang({"en": "Hello"}) == "Hello"  # no French: take what exists
    assert pick_lang(None) is None


def test_flatten_event_produces_flat_record():
    row = flatten_event(make_event(42))
    assert row["title"] == "Événement 42"
    assert row["keywords"] == "Expo, Tout public"  # list -> "a, b" string
    assert row["n_timings"] == 3
    assert row["venue"] == "Lieu test"
    assert row["url"] == "https://openagenda.com/fr/que-faire-a-paris/events/event-42"
    # Every value is a scalar: nothing nested survives (CSV-ready).
    assert all(not isinstance(v, (dict, list)) for v in row.values())


def test_clean_events_filters_city_and_duplicates():
    events = [
        make_event(1, city="Paris"),
        make_event(2, city="Paris, 8e"),  # kept and normalized to "Paris"
        make_event(3, city="Bobigny"),  # outside the chosen zone: dropped
        make_event(1, city="Paris"),  # duplicate uid: dropped
    ]
    df = clean_events(events)
    assert sorted(df["uid"]) == [1, 2]
    assert set(df["city"]) == {"Paris"}


def test_clean_events_drops_events_without_description():
    events = [make_event(1), make_event(2, long_desc="")]
    df = clean_events(events)
    assert list(df["uid"]) == [1]
