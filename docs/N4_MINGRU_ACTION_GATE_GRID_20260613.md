# N=4 minGRU Action-Gate Threshold Grid - 2026-06-13

## Goal

Evaluate whether the oversampled minGRU action gate has a useful held-out operating point at stricter decision thresholds. The previous single run showed that the gate could produce overrides, but slightly hurt mean/CVaR while preserving success and p10.

Fixed base checkpoint:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k/checkpoint_iter_002.pt`

Fixed gate:

`reports/n4_mingru_action_gate_promoted_option_oversample_20260613_seed9252k/mingru_action_gate.pt`

## Implementation Update

Added `scripts/evaluate_mingru_action_gate_grid.py`, a reusable held-out evaluator for trained minGRU action gates. It sweeps:

- gate confidence;
- gate margin;
- apply threshold.

It reports base metrics, candidate deltas, override counts, and best candidate. It also supports `--max-candidates` so bounded probes can complete cleanly instead of being manually interrupted.

## Bounded Held-Out Grid

Run:

`reports/n4_mingru_action_gate_grid_oversample_top2_20260613`

Evaluation block:

- starts: `1900000`, `2000000`, `2100000`, `2200000`;
- 20 episodes each;
- total: 80 held-out mixed episodes.

Base promoted minGRU:

| Mean | P10 | CVaR | Success |
| ---: | ---: | ---: | ---: |
| 486.600 | 452.6 | 418.5 | 0.6875 |

Grid candidates:

| Confidence | Margin | Apply | Mean | P10 | CVaR | Success | Overrides |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 0.05 | 0.65 | 486.488 | 452.6 | 417.5 | 0.6875 | 295 |
| 0.75 | 0.05 | 0.85 | 486.575 | 452.6 | 418.25 | 0.6875 | 168 |

The stricter apply threshold reduces overrides and recovers most of the CVaR loss, but still does not beat the no-gate base. The best candidate by the configured success/p10/CVaR/mean ordering is still below base CVaR.

## Interpretation

The trained gate has no useful threshold among the tested early operating points. Threshold tightening can make it less harmful, but not helpful. This supports the previous conclusion: the current labels are not good enough. More classifier pressure or threshold tweaking is unlikely to crack N=4.

## Next Step

Improve label quality before training another gate:

- collect from held-out-like near misses rather than only mined hard seeds;
- use two-phase option sequences instead of repeated single-action holds;
- rank/choose candidate windows by recovery pressure and motif score;
- add matched high-risk preservation negatives from solved episodes.

Current N=4 status remains unsolved. Best held-out success is still `0.6875`; best p10 remains `452.6` from the promoted scout-selected minGRU checkpoint.
