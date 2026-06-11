# Tail-First Policy Terminal Curriculum

Status: `solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.45`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `1010000, 1020000, 1030000`
Validation episodes per start: `40`
Score weights: mean `0.35`, p10 `0.75`, CVaR `0.75`, success `130.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0 | 927.4 | 488.9 | 453.0 | 429.8 | 0.725 | 33 | True |
| chunk_1 | 15000 | 925.3 | 488.8 | 453.0 | 429.8 | 0.708 | 35 | False |
| chunk_2 | 30000 | 881.3 | 482.2 | 432.8 | 403.1 | 0.658 | 40 | False |
| chunk_3 | 45000 | 914.4 | 486.9 | 446.6 | 426.9 | 0.683 | 38 | False |
| chunk_4 | 60000 | 927.4 | 488.9 | 453.0 | 429.8 | 0.725 | 33 | False |

Best validation checkpoint: `reports/n4_tail_curriculum_20260611/checkpoint_000000_start.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | cvar | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure_ppo | 448.4 | 331.0 | n/a | 0.557 | 300 |
| recon_policy_terminal | 485.4 | 442.8 | 410.4 | 0.703 | 300 |

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
