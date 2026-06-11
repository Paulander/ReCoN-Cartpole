# Tail-First Policy Terminal Curriculum

Status: `completed_not_solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.4`
Adaptive tail seed refresh: `40` seeds/chunk
Validation seed starts: `1110000, 1120000, 1130000`
Validation episodes per start: `40`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `140.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk_1 | 25000 | 847.6 | 470.7 | 396.9 | 368.5 | 0.567 | 40 | True |
| chunk_2 | 50000 | 847.6 | 470.7 | 396.9 | 368.5 | 0.567 | 40 | False |
| chunk_3 | 75000 | 847.6 | 470.7 | 396.9 | 368.5 | 0.567 | 40 | False |
| chunk_4 | 100000 | 847.2 | 470.7 | 396.8 | 368.1 | 0.567 | 40 | False |

Best validation checkpoint: `reports/n4_recurrent_tail_20260611_seed2110k/checkpoint_025000.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | cvar | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure_recurrent_ppo | 281.6 | 235.9 | 229.3 | 0.000 | 300 |
| recon_policy_terminal | 470.1 | 400.8 | 368.6 | 0.573 | 300 |

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
