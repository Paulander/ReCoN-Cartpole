# Recovery-Window Residual PPO

Status: `completed`
Residual model: `reports/n4_recovery_window_residual_currentbest_narrow_20260613_seed9160/recovery_window_residual_policy.zip`
Windows: `1018` from `120` collection episodes
Window types: `{'preserve_success': 728, 'recovery': 290}`
Window horizon: `120`; residual feature mode: `subchain_diagnostics`
Reward config: `{'pressure_drop_weight': 3.0, 'pressure_after_weight': 0.35, 'shift_penalty': 0.1, 'low_risk_change_penalty': 0.8, 'failure_penalty': 4.0, 'window_success_bonus': 1.0, 'preserve_success_shift_penalty': 1.0, 'preserve_success_noop_bonus': 0.08}`

| evaluator | mean | p10 | cvar | success | mean abs delta | episodes |
|---|---:|---:|---:|---:|---:|---:|
| recon_frozen_base | 487.7 | 442.9 | 417.1 | 0.688 | 0.000 | 80 |
| recon_recovery_window_residual | 487.7 | 442.9 | 417.1 | 0.688 | 0.000 | 80 |

## Claim Discipline

Training starts from non-held-out recovery windows; evaluation uses separate held-out seeds through normal ReCoN residual-terminal integration. No train-seed solve claim is made.
