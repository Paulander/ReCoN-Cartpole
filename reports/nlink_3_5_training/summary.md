# N-link 3/4/5 Training Sweep

This run carries bandit priors across training episodes, freezes learning for held-out evaluation, and exports one replay per link count.
The custom N-link dynamics are still a benchmark scaffold, so treat these as iteration metrics rather than solved-control claims.

| links | mode | train episodes | eval episodes | eval mean steps | eval p10 | eval max | success@horizon | replay |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 3 | recon_fast_bandit | 80 | 25 | 41.3 | 34.0 | 68.0 | 0.00 | [N=3](nlink_3_replay.html) |
| 4 | recon_fast_bandit | 80 | 25 | 37.8 | 32.0 | 55.0 | 0.00 | [N=4](nlink_4_replay.html) |
| 5 | recon_fast_bandit | 80 | 25 | 34.8 | 30.0 | 47.0 | 0.00 | [N=5](nlink_5_replay.html) |

## Next Control Work

1. Validate or replace the approximate coupled N-link dynamics before making high-N claims.
2. Promote slow consolidation after training/eval split is stable.
3. Add failure taxonomy by rail exit, base-pole angle, and outer-link divergence.