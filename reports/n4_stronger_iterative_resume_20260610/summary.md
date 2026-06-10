# Iterative Policy Terminal Training

Status: `running`
Reward mode: `upright_shaping`
Selection mode: `hard_select`
Policy terminal blend: `1.0`
Policy terminal scope: `stabilize_chain`
Frame stack: `1`
Policy observation mode: `normalized_raw`
Success bonus: `25.0`
Failure penalty: `2.0`
Hard train seeds: `600` at probability `0.6`
Validation seed starts: `950000, 960000, 970000`
Validation seed count per start: `80`
Score weights: mean `1.0`, p10 `0.35`, success `90.0`

| checkpoint | timesteps | score | mean | p10 | success |
|---|---:|---:|---:|---:|---:|
| start | 0 | 707.7 | 486.7 | 446.0 | 0.72 |
| chunk_1 | 25000 | 704.1 | 485.8 | 443.9 | 0.70 |
| chunk_2 | 50000 | 700.3 | 483.8 | 438.5 | 0.70 |

Best validation checkpoint: `reports/n4_stronger_iterative_resume_20260610/checkpoint_000000_start.zip`

## Claim Discipline

This runner promotes checkpoints by ReCoN-routed validation survival. It is still a learned PPO terminal inside ReCoN, not pure symbolic ReCoN. N=4 is solved only if the final held-out block meets the configured threshold.
