from pathlib import Path

from nhl_research.data.nhl import games_from_score_payload
import json


def test_nhl_fixture_future_games_are_unsettled():
    raw = json.loads(Path("tests/fixtures/nhl_score_2026-10-02.json").read_text())
    payload = raw["payload"]
    payload["_meta"] = {
        "endpoint": raw["endpoint"],
        "retrieved_at_utc": raw["retrieved_at_utc"],
        "origin": raw["dataset_origin"],
    }
    rows = games_from_score_payload(payload)
    assert raw["dataset_origin"] == "REAL_API_SNAPSHOT"
    assert all(r["home_win"] is None for r in rows)
    assert all("FUTURE_GAME" in r["quality_flags"] for r in rows)


def test_nhl_opening_night_has_settled_scores():
    raw = json.loads(Path("tests/fixtures/nhl_score_2026-09-29.json").read_text())
    payload = raw["payload"]
    payload["_meta"] = {
        "endpoint": raw["endpoint"],
        "retrieved_at_utc": raw["retrieved_at_utc"],
        "origin": raw["dataset_origin"],
    }
    rows = games_from_score_payload(payload)
    settled = [r for r in rows if r["game_state"] in {"OFF", "FINAL"}]
    assert settled
    assert all(r["home_win"] in (0, 1) for r in settled)
