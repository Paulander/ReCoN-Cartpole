# N=4 minGRU PPO Mixed-Fresh-Starts Probe - 2026-06-13

## Goal

Bring the mixed-block discipline into minGRU PPO rollout collection, not just supervised curriculum data. The runner previously mixed a hard-seed list with one contiguous fresh seed block. This update lets the fresh component come from multiple seed starts.

## Implementation

Updated `scripts/train_mingru_ppo.py`:

- added `--seed-starts` for the fresh rollout component;
- added `mixed_fresh_seed_values`, using round-robin block expansion such as `[100, 200] x 5 -> 100, 200, 101, 201, 102`;
- kept existing hard-seed mixing semantics, so `--hard-seed-probability` now mixes hard seeds with a mixed fresh grid.

Focused tests cover pure mixed fresh seeds and hard+mixed-fresh composition.

## PPO Probe

Report:

`reports/n4_mingru_ppo_mixed_fresh_starts_20260613_seed9290k`

Start checkpoint:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k/checkpoint_iter_002.pt`

Training setup:

- 48 rollout episodes;
- fresh seed starts: `9290000`, `9300000`, `9310000`, `9320000`;
- hard seed list: `reports/n4_mingru_fresh_option_aux_hardseed_mine_20260613_seed9140k/hard_seeds.txt`;
- hard probability: `0.35` (`17` hard, `31` fresh episodes);
- 3 PPO iterations, 16 rollout episodes each;
- scout starts: `1900000`, `2000000`, `2100000`, `2200000`, 4 episodes each;
- final held-out starts: same four starts, 10 episodes each.

Scout selected checkpoint:

`reports/n4_mingru_ppo_mixed_fresh_starts_20260613_seed9290k/checkpoint_iter_003.pt`

Scout metrics for the selected checkpoint:

| mean | p10 | success | episodes |
| ---: | ---: | ---: | ---: |
| 490.0 | 466.5 | 0.750 | 16 |

40-episode final comparison:

| evaluator | mean | p10 | success | episodes |
| --- | ---: | ---: | ---: | ---: |
| start checkpoint | 485.05 | 438.7 | 0.700 | 40 |
| mixed-fresh PPO candidate | 485.075 | 439.6 | 0.700 | 40 |

Promotion score:

- start: `1187.2050`
- candidate: `1188.1075`
- promoted: `true`

## Wider 80-Episode Check

Because the 40-episode promotion was small, the selected checkpoint was checked on the usual 80-seed mixed block via:

`reports/n4_mingru_routing_compare_mixed80_mixedfresh_20260613`

| mode | mean | p10 | success | episodes |
| --- | ---: | ---: | ---: | ---: |
| pure minGRU | 486.575 | 452.7 | 0.6875 | 80 |
| ReCoN-routed minGRU | 486.575 | 452.7 | 0.6875 | 80 |

There were zero action differences between pure and ReCoN-routed behavior in this comparison.

## Interpretation

Mixed fresh starts produced a small same-success promotion on the 40-episode block and a tiny p10 nudge on the wider 80-episode check, but did not break the `0.6875` success plateau. This is useful but not a solve.

The current N=4 state remains unsolved. The recurrent branch now has better rollout-seed plumbing; the next useful move is probably not another tiny PPO continuation with the same authority, but either a stronger curriculum/on-policy stage or a policy architecture/objective change that can alter the failure seeds without eroding preserved successes.
