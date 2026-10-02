# Data-access inventory

Verified at **2026-10-02T15:10:00Z** during this build. Machine-readable copy: `nhl-research inventory`.

The 2026–27 NHL regular season is **underway**. NHL.com schedule release: season begins 2026-09-29 and ends 2026-04-10 (84 games / 1,344 games). NHL Web API on 2026-10-02 returned completed 2026-09-29 games and `FUT` games for 2026-10-02.

| provider | dataset | documentation | access_method | license_or_terms | historical_coverage | update_frequency | publication_timing | revision_behavior | cost_status | known_limitations | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NHL | Web API score/schedule | https://api-web.nhle.com/v1/score/{date} | HTTPS GET | Public site/API; NHL terms of use | Opening night 2026-09-29 and current week verified | Live (short cache) | `startTimeUTC`; future games `FUT` | Treat each GET as a snapshot | No fee observed | No SLA; `/schedule/now` 307-redirects | VERIFIED_AVAILABLE |
| MoneyPuck | Listed CSVs | https://moneypuck.com/data.htm | HTTPS GET of linked files | Non-commercial / journalist ad-hoc; credit MoneyPuck.com; ask for other use | 2008–09 through 2026–27 listed; shots 2026-27 = 1,288 as of 2026-10-02 03:40 ET; page updated 2026-10-02 06:30 ET | Nightly (stated) | Cumulative current-season files | In-place replace; xG model version unpublished | Free listed downloads | xG may be rescored; season tables leak later games; tied goals ≠ SO winner | VERIFIED_AVAILABLE |
| The Odds API | v4 current + historical | https://the-odds-api.com/liveapi/guides/v4/ | HTTPS + apiKey | Commercial | Historical from 2020-06-06; 10-min then 5-min snapshots; **paid plans only**; 10 credits / region / market | On request | Closest snapshot ≤ `date` | Errors may remain in history | **No key in this environment** | Historical endpoint not usable here | REQUIRES_PAID_ACCESS |
| Prior project extracts | odds and MoneyPuck CSV dumps | not in this repository | none | not vendored | Removed when this repository was reset | n/a | n/a | n/a | Not shipped | Re-download from the provider; do not treat an old dump as current | UNAVAILABLE |
| Daily Faceoff | rosters/injuries | not re-verified | none | site terms; scraping not approved | not in this repository | n/a | n/a | n/a | Unused | Not a 2026–27 lineup source | UNVERIFIED |
| NHL EDGE | tracking | none verified | none | unknown | none | n/a | n/a | n/a | No access | Excluded from features | UNAVAILABLE |

**Credit:** MoneyPuck.com is the source of xG and on-ice rate data used in repository files and optional downloads.

**Odds API historical (docs, paid):** `GET /v4/historical/sports/{sport}/odds` from 2020-06-06; not called in this environment because no key is present.
