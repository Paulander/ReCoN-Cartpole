# ReCoN Residual Gate Sweep

Status: `completed`
Base model: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`
Residual model: `reports/n4_residual_proposal_diag_20260612_seed2660k/residual_policy_terminal.zip`
Residual feature mode: `proposal_diagnostics`

| threshold | max force | mean | p10 | cvar | success | episodes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.200 | 4.00 | 485.1 | 437.6 | 418.2 | 0.667 | 120 |
| 0.350 | 4.00 | 485.7 | 437.6 | 420.2 | 0.667 | 120 |
| 0.500 | 4.00 | 485.9 | 437.6 | 420.5 | 0.675 | 120 |
| 0.620 | 4.00 | 486.0 | 437.7 | 420.8 | 0.675 | 120 |
| 0.750 | 4.00 | 486.0 | 437.7 | 420.8 | 0.675 | 120 |
| 0.900 | 4.00 | 486.1 | 437.7 | 420.8 | 0.675 | 120 |

Best candidate: threshold `0.900`, max force `4.00`.

## Claim Discipline

This is a held-out/evaluation sweep over a fixed learned residual. No train-seed solve claims are made from this table.
