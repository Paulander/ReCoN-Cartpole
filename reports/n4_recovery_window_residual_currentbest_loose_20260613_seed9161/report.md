# Recovery-Window Residual PPO

Status: `completed`
Residual model: `reports/n4_recovery_window_residual_currentbest_loose_20260613_seed9161/recovery_window_residual_policy.zip`
Windows: `563` from `120` collection episodes
Window types: `{'preserve_success': 273, 'recovery': 290}`
Window horizon: `120`; residual feature mode: `subchain_diagnostics`
Reward config: `{'pressure_drop_weight': 4.0, 'pressure_after_weight': 0.25, 'shift_penalty': 0.04, 'low_risk_change_penalty': 0.2, 'failure_penalty': 3.0, 'window_success_bonus': 2.0, 'preserve_success_shift_penalty': 0.3, 'preserve_success_noop_bonus': 0.02}`

| evaluator | mean | p10 | cvar | success | mean abs delta | episodes |
|---|---:|---:|---:|---:|---:|---:|
| recon_frozen_base | 487.7 | 442.9 | 417.1 | 0.688 | 0.000 | 80 |
| recon_recovery_window_residual | 487.7 | 442.9 | 416.6 | 0.688 | 0.390 | 80 |

## Claim Discipline

Training starts from non-held-out recovery windows; evaluation uses separate held-out seeds through normal ReCoN residual-terminal integration. No train-seed solve claim is made.
