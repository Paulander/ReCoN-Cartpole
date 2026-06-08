# Curriculum Run

This run trains shared proposal gains plus persistent bandit state per stage, then freezes learning for held-out evaluation.

| stage | N | trials | eval mean | eval p10 | eval max | passed | report |
|---|---:|---:|---:|---:|---:|---|---|
| nlink_3_warm | 3 | 6 | 235.4 | 176.9 | 415.0 | yes | [replay](00_nlink_3_warm/best_replay.html) |
| nlink_3_stable | 3 | 20 | 500.0 | 500.0 | 500.0 | yes | [replay](01_nlink_3_stable/best_replay.html) |
| nlink_3_robust | 3 | 30 | 499.0 | 500.0 | 500.0 | yes | [replay](02_nlink_3_robust/best_replay.html) |