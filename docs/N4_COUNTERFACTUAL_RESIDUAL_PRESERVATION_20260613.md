# N=4 Counterfactual Residual Preservation Update - 2026-06-13

## Goal

Improve the learned residual-terminal path without letting it damage already-successful behavior. The target use case is still the frozen best PPO terminal routed through ReCoN; the residual should only intervene when counterfactual probes show a credible local recovery benefit.

## Implementation

Added two data-selection upgrades to `scripts/train_counterfactual_residual_terminal.py`:

- `--collect-seed-starts` mixes collection seeds round-robin across held-out-like seed blocks instead of using only one contiguous range.
- High-risk success negatives select solved-episode states with high recovery pressure via `--success-risk-negative-count`, `--success-risk-window-*`, and `--success-risk-stride`.

Reports now record success-negative counts, high-risk success-negative counts, mixed collection starts, and preservation settings. Tests cover mixed seed starts, high-risk success-negative selection, and label-summary accounting.

## Probe 1: Conservative Option Labels

Report: `reports/n4_counterfactual_residual_preserve_risk_20260613_seed9260k`

Setup: frozen PPO terminal `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`, mixed collection starts `980000`, `1500000`, `1600000`, `2100000`, residual option hold `3`, tail `2`, high-risk success negatives enabled.

Dataset:

| rows | success negatives | high-risk success negatives | non-noop labels |
| ---: | ---: | ---: | ---: |
| 179 | 91 | 52 | 0 |

Held-out eval was unchanged because the residual learned no-op everywhere:

| evaluator | mean | p10 | cvar | success |
| --- | ---: | ---: | ---: | ---: |
| frozen base | 485.65 | 432.9 | 415.75 | 0.700 |
| residual | 485.65 | 432.9 | 415.75 | 0.700 |

## Probe 2: Pressure-Sensitive Diagnostic

Report: `reports/n4_counterfactual_residual_pressure_diag_20260613_seed9261k`

Setup: smaller diagnostic run with pressure-final/max penalties and looser label gates.

Dataset:

| rows | success negatives | high-risk success negatives | non-noop labels |
| ---: | ---: | ---: | ---: |
| 58 | 42 | 24 | 0 |

Tiny held-out eval was also unchanged:

| evaluator | mean | p10 | cvar | success |
| --- | ---: | ---: | ---: | ---: |
| frozen base | 472.5 | 423.0 | 405.0 | 0.500 |
| residual | 472.5 | 423.0 | 405.0 | 0.500 |

## Interpretation

The preservation sampler is useful infrastructure, but the current counterfactual residual-bin search is locally no-op optimal around the frozen PPO terminal under both conservative and pressure-sensitive scoring. This explains why previous residual/gate attempts mostly preserved or damaged behavior rather than improving the tail.

Do not spend more time only tightening residual thresholds. The next useful residual attempt needs a different intervention space or label source: longer option sequences, explicit state rewinds with recurrent context preservation, trajectory-level recovery labels, or a separate specialist policy trained on recoverable windows and gated by stronger success-preservation negatives.

Current N=4 status remains unsolved; these probes do not change the best held-out metrics.
