# N=4 Mixed minGRU PPO Update - 2026-06-13

## Goal

The guarded minGRU PPO runs had become too hard-tail focused: training batches sampled only mined failure seeds, so rollout success during PPO stayed at zero and the optimizer mostly learned from failure-avoidance pressure. This update adds a mixed seed path so PPO can train on both:

- mined hard-tail seeds that expose the remaining failures;
- fresh current-distribution seeds that preserve normal successful behavior.

## Code change

`scripts/train_mingru_ppo.py` now supports:

- `--hard-seed-probability`, default `1.0` to preserve the old hard-only behavior when `--seed-list` is supplied;
- JSON or text seed-list parsing local to the PPO script;
- deterministic hard/fresh mixing from `--train-seed`;
- report fields for `hard_seed_probability`, `hard_seed_count`, and `fresh_seed_count`.

The tests in `tests/test_policy_terminal_training.py` cover hard-only repetition, deterministic mixed hard/fresh selection, and the reported mix counts.

## Probe Run

Run directory:

`reports/n4_mingru_ppo_mixed_guarded_20260613_seed9210k`

Starting checkpoint:

`reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt`

Training setup:

- N=4, serial Lagrange dynamics, `dt=0.0005`;
- 5 discrete force bins, force magnitude `10`;
- observation mode `normalized_raw4_subchains_prev_force`;
- hard seed probability `0.5`;
- 96 train episodes: 48 hard-tail seeds and 48 fresh seeds;
- guarded promotion against the start checkpoint;
- held-out mixed validation starts `1900000`, `2000000`, `2100000`, `2200000`, 20 episodes each.

## Result

The mixed batches fixed the worst signal problem: PPO rollout batches again contained successful episodes. However, the trained candidate did not beat the incumbent on guarded held-out validation and was not promoted.

| Model | Mean | P10 | Success |
| --- | ---: | ---: | ---: |
| start ReCoN minGRU terminal | 487.1375 | 443.8 | 0.6875 |
| candidate ReCoN minGRU terminal | 487.1125 | 443.8 | 0.675 |
| start pure minGRU policy | 486.725 | 451.9 | 0.6875 |
| candidate pure minGRU policy | 486.75 | 451.9 | 0.6875 |

Promotion score:

- incumbent: `1180.01375`;
- candidate: `1167.51125`;
- promoted: `false`.

## Interpretation

This is useful negative evidence. Mixed hard/current PPO no longer collapses into all-failure training, and it preserves the pure minGRU policy metric, but the ReCoN-routed terminal still regresses slightly. The next bottleneck is likely the interaction between the learned terminal and ReCoN arbitration/confidence/routing rather than raw recurrent policy capacity alone.

Current N=4 status remains unsolved:

- best held-out success is still about `0.6875` on the 80-episode mixed block;
- no new solve claim should be made;
- the best minGRU incumbent remains `reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt`;
- the best feedforward PPO terminal remains `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.

## Next Most Plausible Step

Investigate and tune the ReCoN routing interface around the minGRU terminal:

- compare pure minGRU actions against ReCoN-routed actions on the exact failure seeds;
- inspect whether proposal confidence, scope, or arbitration suppresses good recurrent actions;
- try a selective residual/gate that only changes high-risk moments while preserving the incumbent action elsewhere;
- keep guarded promotion based on held-out mixed blocks, not train seed success.
