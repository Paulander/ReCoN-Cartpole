# Tail-First Policy Terminal Curriculum

Status: `completed_not_solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.25`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Validation episodes per start: `20`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `180.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk_1 | 50000 | 888.7 | 474.4 | 400.9 | 378.1 | 0.600 | 40 | True |
| chunk_2 | 100000 | 926.5 | 478.8 | 424.0 | 391.5 | 0.631 | 40 | True |
| chunk_3 | 150000 | 921.7 | 479.2 | 422.8 | 390.9 | 0.613 | 40 | False |

Best validation checkpoint: `reports/n4_9bin_tail_scout_20260611_seed2180k/checkpoint_100000.zip`

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
