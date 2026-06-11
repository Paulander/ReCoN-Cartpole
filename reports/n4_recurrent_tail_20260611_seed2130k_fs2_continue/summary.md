# Tail-First Policy Terminal Curriculum

Status: `running`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.3`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `1190000, 1200000, 1210000`
Validation episodes per start: `30`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `140.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0 | 913.8 | 479.8 | 432.4 | 404.6 | 0.589 | 37 | True |
| chunk_1 | 15000 | 843.2 | 468.7 | 398.2 | 371.8 | 0.511 | 40 | False |
| chunk_2 | 30000 | 840.9 | 468.6 | 396.2 | 371.1 | 0.511 | 40 | False |
| chunk_3 | 45000 | 838.8 | 468.4 | 395.3 | 369.6 | 0.511 | 40 | False |

Best validation checkpoint: `reports/n4_recurrent_tail_20260611_seed2130k_fs2_continue/checkpoint_000000_start.zip`

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
