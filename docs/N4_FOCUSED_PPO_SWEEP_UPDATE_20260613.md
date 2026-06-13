# N=4 Focused PPO Sweep Update - 2026-06-13

This note records a focused feedforward PPO terminal sweep from the current best PPO terminal. The run used the current N=4 5-bin serial-Lagrange setup, mixed validation, and held-out final blocks only for reporting. It was stopped early after four completed candidates because the success plateau did not move.

## Start Checkpoint

```text
reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip
```

## Sweep Setup

Report directory:

```text
reports/n4_ppo_focused_currentbest_20260613_seed9200k
```

Common setup:

- N=4, `serial_lagrange`, `dt=0.0005`
- 5 discrete actions, `force_mag=10`
- `initial_angle_range=0.05`, `force_noise=0.02`, `link_coupling=12`
- ReCoN `hard_select`, policy terminal scope `stabilize_chain`
- observation mode `normalized_raw`, frame stack `1`
- mixed validation starts: `900000`, `930000`, `970000`, `1010000`, `1500000`, `1600000`, `1900000`, `2000000`
- final held-out starts: `1900000`, `2000000`, `2100000`, `2200000`, 20 episodes each

Focused grid slice:

- `learning_rate=2.5e-7`
- `clip_range=0.001`
- `n_steps=512`
- `n_epochs=1`
- `gae_lambda=0.95`
- `net_arch=128,128`
- `vec_normalize=false`
- swept `ent_coef in {0.0, 0.001}` and `late_survival_bonus in {0.0, 0.005}`

## Results

| idx | ent | late bonus | best checkpoint | mean | p10 | cvar | success | score |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | start | 487.725 | 442.9 | 417.1 | 0.6875 | 905.0975 |
| 1 | 0.0 | 0.005 | start | 487.725 | 442.9 | 417.1 | 0.6875 | 905.0975 |
| 2 | 0.001 | 0.0 | chunk 1 | 487.725 | 443.8 | 417.1 | 0.6875 | 905.7725 |
| 3 | 0.001 | 0.005 | chunk 1 | 487.700 | 443.8 | 417.1 | 0.6875 | 905.76375 |

Best local checkpoint by this slice:

```text
reports/n4_ppo_focused_currentbest_20260613_seed9200k/candidate_02/checkpoint_008000.zip
```

## Interpretation

This slice reproduced the familiar low-learning-rate behavior: PPO can nudge p10 slightly (`442.9 -> 443.8`) but did not change success (`0.6875`). The late-survival bonus did not help in this corner; entropy `0.001` accounted for the small p10 improvement.

The run was stopped after four completed candidates because the remaining candidates in this focused low-LR/low-clip family were unlikely to crack the N=4 success gap. This is useful evidence, not a solve.

## Next Direction

Do not spend more time on nearly identical low-LR feedforward PPO continuation. The next higher-signal options are:

1. Mixed-distribution PPO for the minGRU terminal, not hard-seed-only rollouts.
2. A more selective learned residual/gate that abstains by default and only overrides on reliable failure precursors.
3. A compositional/shared subchain terminal rather than flat subchain features into one monolithic learner.

Claim discipline remains unchanged: N=4 is near-solved, not robustly solved.
