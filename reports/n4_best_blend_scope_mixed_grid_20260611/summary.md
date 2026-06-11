# N=4 Best Checkpoint Blend/Scope Mixed-Grid Evaluation

Checkpoint: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`
Seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Episodes per start: `30`

| scope | blend | mean | p10 | cvar | success |
|---|---:|---:|---:|---:|---:|
| stabilize_chain | 0.90 | 484.9 | 434.9 | 414.8 | 0.696 |
| stabilize_chain | 1.00 | 484.9 | 434.9 | 414.8 | 0.696 |
| stabilize_chain | 0.75 | 479.3 | 410.0 | 385.7 | 0.679 |
| selected | 0.90 | 444.3 | 334.9 | 315.9 | 0.508 |
| selected | 1.00 | 444.3 | 334.9 | 315.9 | 0.508 |
| all | 0.90 | 444.3 | 334.9 | 315.9 | 0.508 |
| all | 1.00 | 444.3 | 334.9 | 315.9 | 0.508 |
| selected | 0.75 | 460.3 | 383.9 | 361.9 | 0.487 |
| all | 0.75 | 460.3 | 383.9 | 361.9 | 0.487 |
