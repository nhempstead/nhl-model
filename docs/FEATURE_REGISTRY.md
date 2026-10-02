# Feature registry

Only core features are used in the default models. Families were researched; most are **not** auto-included.

| name | family | hypothesis | formula | units | denominator | lag | source | missing | leakage risks | core | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elo_diff | team_strength | Opponent-adjusted results carry quality | pregame Elo home − away; p=1/(1+10^(−(diff+HA)/400)) | Elo | prior games | after settlement | canonical games | 1500 prior; LOW_EXPOSURE if <10 games | current-game update | yes | IMPLEMENTED |
| rest_days_diff | schedule | Rest associated with win probability | days since each team's previous known start, home−away | days | calendar time | previous start | schedule | **null, not 0** | revised start times | yes | IMPLEMENTED |
| home_indicator | team_strength | Home teams win more often | Elo HA default +50 (~historical ~54% vs equal) | Elo/prob | settled games | training window | games | n/a | estimating HA on test season | yes | IMPLEMENTED |
| xg_pct_L20_diff | shot quality | 5v5 xG share is more stable than goals | rolling mean of prior 20 games, home−away | proportion | xGF+xGA | shift(1) | MoneyPuck / featured | **null, not 0** | unknown xG model version; season-to-date files | yes | IMPLEMENTED |
| market_home_prob_tminus | market | Same-time de-vig is a strong baseline | proportional de-vig, one book, snapshot≤T | probability | two-way implied | snapshot time | odds | **null; returns blocked** | close, best-price stitch, post-start | yes | IMPLEMENTED |
| realized_starting_goalie | goaltending | Starter quality matters if known pregame | excluded from core | n/a | n/a | unknown | shots-derived starts | unknown stays unknown | inferring from who played | no | NOT_YET_IMPLEMENTED |
| special teams rates | special teams | PP/PK skill beyond 5v5 | not in core pending exposure-weighted reconstruction | rates | ice time | shift(1) | MoneyPuck situations | null | using `all` situation that includes current PP | no | NOT_YET_IMPLEMENTED |
| tracking / NHL EDGE | tracking | micro-stats add skill | excluded; no verified access | n/a | n/a | n/a | none | n/a | n/a | no | BLOCKED |
| coaching/officials | context | observable changes shift p | excluded; not verified at cutoff | n/a | n/a | n/a | none | n/a | narrative weighting | no | NOT_YET_IMPLEMENTED |

Ablations: data-only models omit `market_home_prob_tminus`. Market-informed comparison uses that quote only when `snapshot_time <= prediction_time`.
