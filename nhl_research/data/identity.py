"""Date-valid team identity. Keep provider codes distinct until mapped."""

from __future__ import annotations

from datetime import date

# MoneyPuck dotted codes vs NHL abbrev
MONEYPUCK_TO_NHL = {
    "L.A": "LAK",
    "T.B": "TBL",
    "N.J": "NJD",
    "S.J": "SJS",
    "LA": "LAK",
    "TB": "TBL",
    "NJ": "NJD",
    "SJ": "SJS",
}

ODDS_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Atlanta Thrashers": "ATL",
    "Phoenix Coyotes": "PHX",
}


def normalize_team(code: str, as_of: date | None = None) -> str:
    raw = (code or "").strip()
    mapped = MONEYPUCK_TO_NHL.get(raw, raw).upper()
    if as_of is None:
        return mapped
    # Franchise relocations. These are identity rules, not strength carryover.
    if mapped == "ATL" and as_of >= date(2011, 6, 1):
        return "WPG"
    if mapped in {"PHX", "ARI"} and as_of >= date(2024, 7, 1):
        return "UTA"
    return mapped


def odds_name_to_abbrev(name: str) -> str | None:
    if name in ODDS_NAME_TO_ABBREV:
        return ODDS_NAME_TO_ABBREV[name]
    return ODDS_NAME_TO_ABBREV.get(name.replace("é", "e"))
