# ReCoN vs PPO Same-Seed Comparison

Rows use identical held-out seeds and environment parameters within each N. PPO is explicit: if optional RL dependencies are missing, the PPO row is marked `unavailable` rather than omitted.

Horizon: `120`
Eval episodes per N: `5`
ReCoN train episodes per mode: `5`
PPO train timesteps: `1000`
PPO device: `cpu`

| N | family | mode | status | mechanisms | mean | p10 | success | max |
|---:|---|---|---|---|---:|---:|---:|---:|
| 2 | recon | baseline_heuristic | completed | none | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | static_recon | completed | none | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | recon_fast | completed | edge_plasticity | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | recon_bandit | completed | bandit_persistence | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | recon_fast_bandit | completed | edge_plasticity, bandit_persistence | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | recon_slow | completed | edge_plasticity, bandit_persistence, slow_consolidation | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | recon_learn_only | completed | edge_plasticity, bandit_persistence, slow_consolidation, node_param_learning | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | recon_slow_no_gain_search | completed | edge_plasticity, bandit_persistence, slow_consolidation, node_param_learning | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | gain_search_only | completed | gain_mutation | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | recon | gain_search_recon_fast_bandit | completed | edge_plasticity, bandit_persistence, gain_mutation | 120.0 | 120.0 | 1.00 | 120.0 |
| 2 | ppo | ppo | completed | ppo_policy_gradient | 101.6 | 81.6 | 0.40 | 120.0 |
| 3 | recon | baseline_heuristic | completed | none | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | static_recon | completed | none | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | recon_fast | completed | edge_plasticity | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | recon_bandit | completed | bandit_persistence | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | recon_fast_bandit | completed | edge_plasticity, bandit_persistence | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | recon_slow | completed | edge_plasticity, bandit_persistence, slow_consolidation | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | recon_learn_only | completed | edge_plasticity, bandit_persistence, slow_consolidation, node_param_learning | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | recon_slow_no_gain_search | completed | edge_plasticity, bandit_persistence, slow_consolidation, node_param_learning | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | gain_search_only | completed | gain_mutation | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | recon | gain_search_recon_fast_bandit | completed | edge_plasticity, bandit_persistence, gain_mutation | 120.0 | 120.0 | 1.00 | 120.0 |
| 3 | ppo | ppo | completed | ppo_policy_gradient | 120.0 | 120.0 | 1.00 | 120.0 |
| 4 | recon | baseline_heuristic | completed | none | 76.4 | 53.2 | 0.20 | 120.0 |
| 4 | recon | static_recon | completed | none | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | recon_fast | completed | edge_plasticity | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | recon_bandit | completed | bandit_persistence | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | recon_fast_bandit | completed | edge_plasticity, bandit_persistence | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | recon_slow | completed | edge_plasticity, bandit_persistence, slow_consolidation | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | recon_learn_only | completed | edge_plasticity, bandit_persistence, slow_consolidation, node_param_learning | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | recon_slow_no_gain_search | completed | edge_plasticity, bandit_persistence, slow_consolidation, node_param_learning | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | gain_search_only | completed | gain_mutation | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | recon | gain_search_recon_fast_bandit | completed | edge_plasticity, bandit_persistence, gain_mutation | 77.8 | 54.8 | 0.20 | 120.0 |
| 4 | ppo | ppo | completed | ppo_policy_gradient | 73.6 | 49.6 | 0.20 | 120.0 |

## Claim Discipline

This is a comparison artifact, not a solved claim. A mode is solved only if it meets the configured held-out threshold with the required evaluation episode count.