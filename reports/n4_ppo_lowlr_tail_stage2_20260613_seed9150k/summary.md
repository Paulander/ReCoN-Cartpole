# PPO Terminal Hyperparameter Sweep

Status: `stopped_early_plateau`
N poles: `4`
Validation seed starts: `900000, 930000, 970000, 1010000, 1500000, 1600000, 1900000, 2000000`
Final seed starts: `1900000, 2000000, 2100000, 2200000`

| idx | grid | status | lr | clip | steps | epochs | gae | ent | net | vecnorm | late bonus | mean | p10 | cvar | success | score |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | completed_not_solved | 1e-07 | 0.001 | 1024 | 1 | 0.95 | 0.0 | 128,128 | False | 0.0 | 487.7 | 442.9 | 417.1 | 0.688 | 905.1 |
| 1 | 1 | completed_not_solved | 1e-07 | 0.001 | 1024 | 1 | 0.95 | 0.0 | 128,128 | False | 0.005 | 487.7 | 442.9 | 417.1 | 0.688 | 905.1 |
| 2 | 2 | completed_not_solved | 1e-07 | 0.001 | 1024 | 1 | 0.95 | 0.001 | 128,128 | False | 0.0 | 487.7 | 443.8 | 417.1 | 0.688 | 905.8 |

Best checkpoint: `reports/n4_ppo_lowlr_tail_stage2_20260613_seed9150k/candidate_02/checkpoint_016000.zip`

## Claim Discipline

This sweep varies PPO training knobs for the learned terminal. It records whether VecNormalize and late-survival reward shaping were active, and it only reports N=4 solved if the held-out final block clears the configured threshold.
## Stop Reason

Stopped manually after three completed candidates all preserved the `0.6875` held-out success plateau; candidate 2 only nudged p10 from `442.9` to `443.8`.

