# N=4 Residual and PPO Sweep Update - 2026-06-12

## Status

N=4 is still **not solved**. This pass improved the training/evaluation infrastructure and ran two new probes:

1. proposal-diagnostic learned residual terminal
2. repaired PPO sweep slice with mixed-grid validation and held-out action comparison

Neither produced a held-out improvement over the current best policy terminal.

## Residual Terminal Changes

Added shared residual feature construction in `src/recon_cartpole/control/residual_features.py`.

Residual feature modes:

- `basic`: previous behavior, base force + risk gate + previous force
- `proposal_diagnostics`: adds ReCoN-style proposal forces, proposal disagreement with base policy, goal pressures, worst-pole index, and episode fraction

The ReCoN controller now supports `residual_policy_terminal_feature_mode`, so training and deployed inference can use the same residual observation shape. The residual trainer now also reports actual ReCoN-integrated evaluation, not only standalone residual-env metrics.

## Residual Run

Run: `reports/n4_residual_proposal_diag_20260612_seed2660k`

Config summary:

- frozen base: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`
- mode: `bin_delta`
- feature mode: `proposal_diagnostics`
- residual gate threshold: `0.62`
- hard seed probability: `0.70`
- timesteps: `40k`
- eval: 8 mixed seed starts x 15 episodes = 120 episodes

| evaluator | mean | p10 | CVaR | success | episodes |
|---|---:|---:|---:|---:|---:|
| ReCoN frozen base | 486.1 | 437.7 | 420.8 | 0.675 | 120 |
| ReCoN residual specialist | 486.0 | 437.7 | 420.8 | 0.675 | 120 |

The residual changed actions (`mean_abs_residual_delta ~= 0.89`) but did not improve outcomes.

## Residual Gate Sweep

Run: `reports/n4_residual_proposal_diag_gate_sweep_20260612`

| threshold | mean | p10 | CVaR | success |
|---:|---:|---:|---:|---:|
| 0.20 | 485.1 | 437.6 | 418.2 | 0.667 |
| 0.35 | 485.7 | 437.6 | 420.2 | 0.667 |
| 0.50 | 485.9 | 437.6 | 420.5 | 0.675 |
| 0.62 | 486.0 | 437.7 | 420.8 | 0.675 |
| 0.75 | 486.0 | 437.7 | 420.8 | 0.675 |
| 0.90 | 486.1 | 437.7 | 420.8 | 0.675 |

Lower thresholds hurt. Strict thresholds protect the baseline but do not improve it.

## PPO Sweep Repair

`run_ppo_sweep.py` now passes the current tail-curriculum arguments, including:

- `max_cvar_regression`
- `promotion_mode`
- `final_seed_starts`
- `target_kl`
- teacher-anchor defaults

A tiny smoke sweep completed successfully: `reports/smoke_ppo_sweep_current_args`.

## PPO Sweep Slice

Run: `reports/n4_ppo_sweep_slice_20260612_seed2670k`

Best validation candidate:

- candidate `01`
- LR `5e-7`
- clip `0.005`
- n_steps `512`
- n_epochs `1`
- entropy `0.001`
- net `128,128`
- late survival bonus `0.01`

Mixed validation slice, 8 starts x 15 episodes:

| model | mean | p10 | CVaR | success |
|---|---:|---:|---:|---:|
| start/base | 486.1 | 437.7 | 420.8 | 0.675 |
| candidate 01 | 486.2 | 437.7 | 420.7 | 0.683 |

Held-out 1.5M/1.6M x 120 action comparison:

| model | success | changed seeds | success gains | success losses |
|---|---:|---:|---:|---:|
| base_best | 0.692 | n/a | n/a | n/a |
| candidate 01 | 0.692 | 114 | 0 | 0 |

The validation bump did not generalize.

## Interpretation

Both residual specialization and small PPO continuation currently produce many behavior changes but little or no held-out outcome movement. The next higher-probability move is still a more explicitly state-conditioned intervention: collect near-failure states, learn a gate that predicts when an intervention is likely to help, and train/evaluate the specialist only under that gate.

## Verification

`uv run pytest -q -s` -> `60 passed`.
