# Tail-First Policy Terminal Curriculum

Status: `running`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.35`
Teacher action penalty: `0.01` until `0.8` horizon fraction below risk `0.85`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Validation episodes per start: `30`
Promotion mode: `lexicographic_success`
Score weights: mean `0.35`, p10 `0.75`, CVaR `0.75`, success `130.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0 | 897.4 | 484.9 | 434.9 | 414.8 | 0.696 | 40 | True |
| chunk_1 | 5000 | 894.7 | 484.4 | 433.9 | 413.1 | 0.692 | 40 | False |

Best validation checkpoint: `reports/n4_teacher_anchor_tail_20260612_seed2640k/checkpoint_000000_start.zip`

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
