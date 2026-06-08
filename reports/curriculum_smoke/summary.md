# Curriculum Run

This run trains shared proposal gains plus persistent bandit state per stage, then freezes learning for held-out evaluation.

| stage | N | trials | eval mean | eval p10 | eval max | passed | report |
|---|---:|---:|---:|---:|---:|---|---|
| smoke_n3 | 3 | 3 | 54.0 | 47.1 | 62.0 | yes | [replay](00_smoke_n3/best_replay.html) |