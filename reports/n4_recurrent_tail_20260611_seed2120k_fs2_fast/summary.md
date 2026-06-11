# Tail-First Policy Terminal Curriculum

Status: `completed_not_solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.2`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `1150000, 1160000, 1170000`
Validation episodes per start: `30`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `140.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk_1 | 25000 | 843.7 | 471.0 | 396.9 | 367.4 | 0.544 | 40 | True |
| chunk_2 | 50000 | 872.7 | 474.7 | 404.2 | 380.4 | 0.622 | 34 | True |
| chunk_3 | 75000 | 844.8 | 471.2 | 396.9 | 368.7 | 0.544 | 40 | False |
| chunk_4 | 100000 | 844.7 | 471.2 | 396.9 | 368.6 | 0.544 | 40 | False |

Best validation checkpoint: `reports/n4_recurrent_tail_20260611_seed2120k_fs2_fast/checkpoint_050000.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | cvar | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure_recurrent_ppo | 367.8 | 274.6 | 258.2 | 0.080 | 100 |
| recon_policy_terminal | 481.1 | 428.4 | 395.3 | 0.630 | 100 |

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
