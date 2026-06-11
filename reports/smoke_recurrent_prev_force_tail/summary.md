# Tail-First Policy Terminal Curriculum

Status: `completed_not_solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw_prev_force`
Hard seed probability: `0.2`
Adaptive tail seed refresh: `2` seeds/chunk
Validation seed starts: `1230000, 1240000`
Validation episodes per start: `2`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `140.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk_1 | 64 | 821.4 | 441.0 | 421.6 | 415.0 | 0.000 | 2 | True |

Best validation checkpoint: `reports/smoke_recurrent_prev_force_tail/checkpoint_000064.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | cvar | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure_recurrent_ppo | 420.5 | 402.5 | 398.0 | 0.000 | 2 |
| recon_policy_terminal | 427.0 | 403.8 | 398.0 | 0.000 | 2 |

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
