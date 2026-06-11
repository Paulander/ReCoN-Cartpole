# Tail-First Policy Terminal Curriculum

Status: `completed_not_solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy observation mode: `normalized_raw`
Hard seed probability: `0.4`
Adaptive tail seed refresh: `2` seeds/chunk
Validation seed starts: `1110000, 1120000`
Validation episodes per start: `2`
Score weights: mean `0.25`, p10 `0.85`, CVaR `0.85`, success `140.0`

| checkpoint | timesteps | score | mean | p10 | cvar | success | tail seeds | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk_1 | 64 | 714.4 | 428.5 | 356.3 | 317.0 | 0.250 | 2 | True |

Best validation checkpoint: `reports/smoke_recurrent_policy_tail_v2/checkpoint_000064.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | cvar | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure_recurrent_ppo | 379.0 | 371.0 | 369.0 | 0.000 | 2 |
| recon_policy_terminal | 379.0 | 371.0 | 369.0 | 0.000 | 2 |

## Claim Discipline

This runner optimizes lower-tail validation behavior for a learned PPO terminal inside ReCoN. Tail seeds may enter the training pool after validation, but final solve claims require separate held-out seed blocks.
