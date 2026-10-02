"""NHL Web API client. Only documented public endpoints are used."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nhl_research.entities import DatasetOrigin
from nhl_research.exceptions import DataUnavailableError
from nhl_research.timeutil import parse_utc

UTC = timezone.utc
BASE = "https://api-web.nhle.com/v1"
USER_AGENT = "nhl-research/0.1 (non-commercial research; +https://github.com/nhempstead/nhl-model)"


def fetch_score_date(game_date: str, timeout: int = 20) -> dict[str, Any]:
    """GET /v1/score/{YYYY-MM-DD}. Verified 2026-10-02 against api-web.nhle.com."""
    url = f"{BASE}/score/{game_date}"
    payload = _get_json(url, timeout=timeout)
    payload["_meta"] = {
        "endpoint": url,
        "retrieved_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "origin": DatasetOrigin.REAL_API_SNAPSHOT.value,
    }
    return payload


def games_from_score_payload(payload: dict[str, Any], ingested_at: datetime | None = None) -> list[dict]:
    ingested_at = ingested_at or datetime.now(tz=UTC)
    retrieved = payload.get("_meta", {}).get("retrieved_at_utc")
    available_at = parse_utc(retrieved) if retrieved else ingested_at
    rows: list[dict] = []
    for game in payload.get("games", []):
        state = game.get("gameState") or "UNKNOWN"
        start = game.get("startTimeUTC")
        scheduled = parse_utc(start) if start else None
        home = game.get("homeTeam") or {}
        away = game.get("awayTeam") or {}
        home_score = home.get("score")
        away_score = away.get("score")
        flags = []
        home_win = None
        if state in {"OFF", "FINAL"} and home_score is not None and away_score is not None:
            if home_score == away_score:
                flags.append("TIED_SCORE_UNSETTLED")
            else:
                home_win = int(home_score > away_score)
        elif state in {"FUT", "PRE", "LIVE"}:
            flags.append("NOT_YET_SETTLED")
            if state == "FUT":
                flags.append("FUTURE_GAME")
        rows.append(
            {
                "game_id": str(game.get("id")),
                "season": int(str(game.get("season") or "20262027")[:4]),
                "scheduled_start_utc": scheduled,
                "scheduled_start_known_at": scheduled,
                "start_time_precision": "exact" if scheduled else "unknown",
                "home_team": home.get("abbrev"),
                "away_team": away.get("abbrev"),
                "home_goals": home_score,
                "away_goals": away_score,
                "home_win": home_win,
                "game_state": state,
                "period_type": (game.get("periodDescriptor") or {}).get("periodType"),
                "venue": (game.get("venue") or {}).get("default"),
                "source": payload.get("_meta", {}).get("endpoint", BASE),
                "available_at": available_at,
                "ingested_at": ingested_at,
                "origin": DatasetOrigin.REAL_API_SNAPSHOT.value,
                "quality_flags": "|".join(flags) if flags else "",
                "game_date": payload.get("currentDate") or payload.get("date") or "",
            }
        )
    return rows


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise DataUnavailableError(f"NHL API HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise DataUnavailableError(f"NHL API unreachable: {url}") from exc
    return json.loads(raw)
