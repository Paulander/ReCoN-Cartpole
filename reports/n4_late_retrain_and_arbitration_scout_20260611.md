# N=4 Late-Retrain and Arbitration Scout - 2026-06-11

## Fresh Late-Survival Retrain

Run: `reports/n4_late_survival_retrain_20260611_seed2210k`.

This trained a fresh 5-bin PPO terminal with `--late-survival-bonus 0.25` from the beginning, using the combined 600 near-miss hard-seed pool and the same mixed validation grid. It underperformed the current frontier.

| checkpoint | mean | p10 | cvar | success | promoted |
|---|---:|---:|---:|---:|---:|
| chunk_1 50k | 474.6 | 415.5 | 385.0 | 0.588 | true |
| chunk_2 100k | 471.8 | 408.6 | 382.9 | 0.558 | false |
| chunk_3 150k | 473.3 | 409.9 | 385.7 | 0.579 | false |

The run was stopped after chunk 3; chunk 4 checkpoint existed, but another full mixed-grid validation/final eval was not justified.

## Blend/Scope Grid On Current Best Checkpoint

Run: `reports/n4_best_blend_scope_mixed_grid_20260611`.

Checkpoint: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.

| scope | blend | mean | p10 | cvar | success |
|---|---:|---:|---:|---:|---:|
| stabilize_chain | 0.90 | 484.9 | 434.9 | 414.8 | 0.696 |
| stabilize_chain | 1.00 | 484.9 | 434.9 | 414.8 | 0.696 |
| stabilize_chain | 0.75 | 479.3 | 410.0 | 385.7 | 0.679 |
| selected | 1.00 | 444.3 | 334.9 | 315.9 | 0.508 |
| all | 1.00 | 444.3 | 334.9 | 315.9 | 0.508 |

## Interpretation

The simple knobs tested so far do not close the N=4 robustness gap:

- More action resolution: 9-bin and continuous scouts underperformed the 5-bin frontier.
- Late-survival microfit/retrain: did not improve mixed-grid success above 0.696.
- Blend/scope arbitration: current `stabilize_chain` scope is already the right route; `selected`/`all` are much worse.

The current frontier still looks like a learned-policy quality ceiling rather than a small arbitration or reward-shaping knob. Next useful work should be a more structural policy-learning change: e.g. a larger architecture/training schedule with held-out multi-block validation, recurrent transfer with a proper shape/sequence curriculum, or a different RL loop/objective.
