"""Source inventory with verified access status. No invented endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from nhl_research.entities import SourceStatus

VERIFIED_AT = "2026-10-02T15:10:00Z"

INVENTORY = [
    {
        "provider": "NHL",
        "dataset": "Web API score and schedule",
        "documentation": "https://api-web.nhle.com/v1/score/{YYYY-MM-DD} (public, undocumented stability)",
        "access_method": "HTTPS GET, no key observed on 2026-10-02",
        "license_or_terms": "Public website/API; reuse constrained by NHL terms of use. Not a licensed commercial feed.",
        "verified_at": VERIFIED_AT,
        "historical_coverage": "Date score endpoint returned 2026-09-29 opening-night games and 2026-10-02 slate during this build.",
        "update_frequency": "Live during game days (cache headers observed ~14s).",
        "publication_timing": "startTimeUTC present; future games have gameState FUT and null scores.",
        "revision_behavior": "Unknown; treat each GET as a snapshot with retrieved_at.",
        "cost_status": "No fee observed for these GETs.",
        "known_limitations": "No official SLA. 307 redirect from /schedule/now to a dated week. Not a point-in-time archive.",
        "status": SourceStatus.VERIFIED_AVAILABLE.value,
    },
    {
        "provider": "MoneyPuck",
        "dataset": "Listed CSV downloads (season summaries, all_teams game file, shots)",
        "documentation": "https://moneypuck.com/data.htm",
        "access_method": "HTTPS GET of files linked on data.htm",
        "license_or_terms": "Free for non-commercial purposes and journalists for ad-hoc use; credit MoneyPuck.com. Other use requires email permission. Non-approved scraping of unlisted pages will be blocked.",
        "verified_at": VERIFIED_AT,
        "historical_coverage": "Season files 2008-09 through 2026-27 listed. all_teams.csv served 2026-10-02. Shots 2026-27: 1,288 shots as of 2026-10-02 03:40 ET. Page last updated 2026-10-02 06:30 ET.",
        "update_frequency": "Nightly for current-season files (stated on page).",
        "publication_timing": "Current-season CSVs are cumulative through the update timestamp. They are not a historical archive of 'as of date D'.",
        "revision_behavior": "Files are replaced in place. Historical xG model version is not published per row.",
        "cost_status": "No fee for listed downloads.",
        "known_limitations": "xG is a model output (possible retrospective rescoring). Season-to-date tables leak later games if used as of an earlier cutoff. Tied 'all' situation goals leave shootout moneyline labels unavailable. Team codes use dotted abbreviations.",
        "status": SourceStatus.VERIFIED_AVAILABLE.value,
    },
    {
        "provider": "The Odds API",
        "dataset": "v4 current and historical odds",
        "documentation": "https://the-odds-api.com/liveapi/guides/v4/",
        "access_method": "HTTPS GET with apiKey. Historical: GET /v4/historical/sports/{sport}/odds",
        "license_or_terms": "Commercial API; paid plans required for historical odds.",
        "verified_at": VERIFIED_AT,
        "historical_coverage": "Docs state historical odds from 2020-06-06, 10-minute snapshots, 5-minute from 2022-09. Paid usage plans only. Cost 10 credits per region per market.",
        "update_frequency": "Current odds on request; historical is snapshot lookup by date.",
        "publication_timing": "date parameter returns closest snapshot equal to or earlier than the provided timestamp.",
        "revision_behavior": "Docs note current errors may remain in historical snapshots.",
        "cost_status": "API key not present in this environment. Historical path REQUIRES_PAID_ACCESS.",
        "known_limitations": "This environment has no ODDS_API_KEY. Historical snapshots are not vendored.",
        "status": SourceStatus.REQUIRES_PAID_ACCESS.value,
    },
    {
        "provider": "Prior project extracts",
        "dataset": "Removed odds, MoneyPuck, roster, and model dumps",
        "documentation": "Not included in this repository",
        "access_method": "none",
        "license_or_terms": "Not redistributed here",
        "verified_at": VERIFIED_AT,
        "historical_coverage": "Not shipped. Re-ingest from the provider.",
        "update_frequency": "n/a",
        "publication_timing": "n/a",
        "revision_behavior": "n/a",
        "cost_status": "Not in this repository.",
        "known_limitations": "Earlier local extracts are not part of this tree and must not be treated as current data.",
        "status": SourceStatus.UNAVAILABLE.value,
    },
    {
        "provider": "NHL EDGE / tracking",
        "dataset": "Player and puck tracking features",
        "documentation": "Not accessed in this environment",
        "access_method": "none",
        "license_or_terms": "unknown / likely restricted",
        "verified_at": VERIFIED_AT,
        "historical_coverage": "Not verified.",
        "update_frequency": "unknown",
        "publication_timing": "unknown",
        "revision_behavior": "unknown",
        "cost_status": "No access configured.",
        "known_limitations": "No endpoint, dump, or license was verified. Tracking features are excluded.",
        "status": SourceStatus.UNAVAILABLE.value,
    },
]


def inventory_payload() -> dict:
    return {
        "verified_at": VERIFIED_AT,
        "execution_date_utc": "2026-10-02",
        "season_status": {
            "target": "2026-2027",
            "regular_season_start": "2026-09-29",
            "regular_season_end": "2027-04-10",
            "as_of": "Season is underway. Opening night is complete. 2026-10-02 games were FUT at retrieval (~15:07 UTC).",
            "source": "https://www.nhl.com/news/nhl-announces-2026-27-regular-season-schedule and NHL Web API",
        },
        "sources": INVENTORY,
        "odds_api_key_present": False,
    }
