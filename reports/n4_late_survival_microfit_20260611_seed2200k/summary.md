# Tail-First Policy Terminal Curriculum

Status: `stopped_tail_regressed`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.25`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Validation episodes per start: `30`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `180.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0 | 968.7 | 484.9 | 434.9 | 414.8 | 0.696 | 40 | True |
| chunk_1 | 5000 | 968.7 | 484.8 | 434.9 | 414.8 | 0.696 | 40 | True |
| chunk_2 | 10000 | 966.1 | 484.4 | 434.0 | 412.7 | 0.696 | 40 | False |

Best validation checkpoint: `reports/n4_late_survival_microfit_20260611_seed2200k/checkpoint_005000.zip`

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.

## Stop Reason

Stopped after `chunk_2`: success stayed `0.696` but p10/CVaR regressed from the start checkpoint.
