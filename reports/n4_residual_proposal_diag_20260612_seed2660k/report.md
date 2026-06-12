# Residual Policy Terminal Training

Status: `completed`
Base model: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`
Residual model: `reports/n4_residual_proposal_diag_20260612_seed2660k/residual_policy_terminal.zip`
Residual feature mode: `proposal_diagnostics`

| evaluator | mean | p10 | cvar | success | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| residual_env_frozen_base | 447.5 | 339.6 | 0.0 | 0.542 | 500.0 | 120 |
| residual_env_specialist | 444.2 | 336.0 | 0.0 | 0.508 | 500.0 | 120 |
| recon_frozen_base | 486.1 | 437.7 | 420.8 | 0.675 | 500.0 | 120 |
| recon_residual_specialist | 486.0 | 437.7 | 420.8 | 0.675 | 500.0 | 120 |

## Mechanisms

The base PPO terminal is frozen. The residual learner sees base force, previous force, and a risk gate; low-risk changes are penalized so the specialist focuses on late/tail failures rather than rewriting successful behavior.
