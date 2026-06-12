# PPO Terminal Hyperparameter Sweep

Status: `completed_not_solved`
N poles: `4`
Validation seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Final seed start: `1040000`

| idx | status | lr | clip | steps | epochs | gae | ent | net | vecnorm | late bonus | mean | p10 | cvar | success | score |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | completed_not_solved | 5e-07 | 0.005 | 512 | 1 | 0.95 | 0.0 | 64,64 | False | 0.0 | 486.1 | 437.7 | 420.8 | 0.675 | 901.7 |
| 1 | completed_not_solved | 5e-07 | 0.005 | 512 | 1 | 0.95 | 0.001 | 128,128 | False | 0.01 | 486.2 | 437.7 | 420.7 | 0.683 | 902.8 |
| 2 | completed_not_solved | 5e-07 | 0.005 | 1024 | 1 | 0.95 | 0.001 | 128,128 | False | 0.0 | 486.1 | 437.7 | 420.4 | 0.683 | 902.6 |

Best checkpoint: `reports/n4_ppo_sweep_slice_20260612_seed2670k/candidate_01/checkpoint_005000.zip`

## Claim Discipline

This sweep varies PPO training knobs for the learned terminal. It records whether VecNormalize and late-survival reward shaping were active, and it only reports N=4 solved if the held-out final block clears the configured threshold.
