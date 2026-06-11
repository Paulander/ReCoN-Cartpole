# Tail-First Policy Terminal Curriculum

Status: `stopped_underperformed`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.125`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Validation episodes per start: `30`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `180.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk_1 | 50000 | 904.8 | 474.6 | 415.5 | 385.0 | 0.588 | 40 | True |
| chunk_2 | 100000 | 891.2 | 471.8 | 408.6 | 382.9 | 0.558 | 40 | False |
| chunk_3 | 150000 | 898.8 | 473.3 | 409.9 | 385.7 | 0.579 | 40 | False |

Best validation checkpoint: `reports/n4_late_survival_retrain_20260611_seed2210k/checkpoint_050000.zip`

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.

## Stop Reason

Stopped after `chunk_3` remained far below the mixed-grid frontier; `checkpoint_200000.zip` existed but was not worth another full validation/final eval.
