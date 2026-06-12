# Tail-First Policy Terminal Curriculum

Status: `completed_not_solved`
Reward mode: `survival`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.25`
Teacher action penalty: `0.0` until `1.0` horizon fraction below risk `1.0`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`
Validation episodes per start: `30`
Promotion mode: `lexicographic_success`
Score weights: mean `0.35`, p10 `0.75`, CVaR `0.75`, success `130.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0 | 897.4 | 484.9 | 434.9 | 414.8 | 0.696 | 40 | True |
| chunk_1 | 5000 | 897.9 | 484.9 | 434.9 | 414.7 | 0.700 | 40 | True |
| chunk_2 | 10000 | 897.4 | 484.5 | 434.9 | 414.2 | 0.700 | 40 | False |
| chunk_3 | 15000 | 896.3 | 484.4 | 434.9 | 414.2 | 0.692 | 40 | False |
| chunk_4 | 20000 | 847.0 | 474.7 | 413.9 | 392.8 | 0.583 | 40 | False |

Best validation checkpoint: `reports/n4_targetkl_survival_tail_20260612_seed2650k/checkpoint_005000.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | cvar | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure_ppo | 444.3 | 332.8 | n/a | 0.533 | 240 |
| recon_policy_terminal | 484.4 | 441.9 | 413.2 | 0.688 | 240 |

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
