# Iterative Policy Terminal Training

Status: `completed_not_solved`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy terminal blend: `1.0`
Policy terminal scope: `stabilize_chain`
Frame stack: `1`
Policy observation mode: `normalized_raw`
Success bonus: `25.0`
Failure penalty: `2.0`
Hard train seeds: `600` at probability `0.6`
Validation seed starts: `950000`
Validation seed count per start: `1`
Score weights: mean `1.0`, p10 `0.35`, success `90.0`

| checkpoint | timesteps | score | mean | p10 | success |
|---|---:|---:|---:|---:|---:|
| start | 0 | 599.4 | 444.0 | 444.0 | 0.00 |

Best validation checkpoint: `reports/n4_stronger_iterative_start_confirm_1000k_20260610/checkpoint_000000_start.zip`

## Final Held-Out Eval

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure_ppo | 442.4 | 333.0 | 0.52 | 300 |
| recon_policy_terminal | 485.3 | 433.9 | 0.68 | 300 |

## Claim Discipline

This runner promotes checkpoints by ReCoN-routed validation survival. It is still a learned PPO terminal inside ReCoN, not pure symbolic ReCoN. N=4 is solved only if the final held-out block meets the configured threshold.
