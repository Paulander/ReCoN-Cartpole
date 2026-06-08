# ReCoN CartPole Ablations

All rows use identical environment parameters and held-out seeds. Mechanisms are reported separately so gain-search performance is not mislabeled as ReCoN learning.

| mode | mechanisms | mean survival | p10 survival | success rate | max survival |
|---|---|---:|---:|---:|---:|
| baseline_heuristic | none | 50.0 | 50.0 | 1.00 | 50.0 |
| static_recon | none | 50.0 | 50.0 | 1.00 | 50.0 |
| recon_fast | edge_plasticity | 50.0 | 50.0 | 1.00 | 50.0 |
| recon_bandit | bandit_persistence | 50.0 | 50.0 | 1.00 | 50.0 |
| recon_fast_bandit | edge_plasticity, bandit_persistence | 50.0 | 50.0 | 1.00 | 50.0 |
| recon_slow | edge_plasticity, bandit_persistence, slow_consolidation | 50.0 | 50.0 | 1.00 | 50.0 |