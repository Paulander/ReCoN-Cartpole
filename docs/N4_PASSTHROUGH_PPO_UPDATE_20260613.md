# N=4 Passthrough minGRU PPO Update - 2026-06-13

## Goal

After the minGRU routing diagnostic showed that `hard_select` can suppress the learned terminal, the next PPO attempt used passthrough-enabled guarded validation. This makes ReCoN-routed evaluation match the pure terminal action when confidence gates pass, so the held-out metric reflects the recurrent terminal update rather than a routing artifact.

## Run

Run directory:

`reports/n4_mingru_ppo_passthrough_mixed_explore_20260613_seed9220k`

Starting checkpoint:

`reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt`

Important settings:

- N=4, serial Lagrange dynamics, `dt=0.0005`;
- 5 discrete force bins, force magnitude `10`;
- observation mode `normalized_raw4_subchains_prev_force`;
- motif score enabled from `reports/n4_subchain_motif_diag_recon_20260612_seed2420k/prototype_model.json`;
- passthrough enabled with confidence floor `0.05` and logit-margin floor `0.0`;
- 160 training episodes: 56 hard-tail seeds and 104 fresh seeds;
- hard seed probability `0.35`;
- 8 PPO iterations, 20 rollout episodes per iteration;
- learning rate `1e-5`, clip range `0.12`, entropy coefficient `0.01`, reference KL coefficient `0.01`;
- held-out mixed validation starts `1900000`, `2000000`, `2100000`, `2200000`, 20 episodes each.

## Result

The candidate did not promote.

| Model | Mean | P10 | Success |
| --- | ---: | ---: | ---: |
| start pure minGRU | 486.725 | 451.9 | 0.6875 |
| start ReCoN minGRU + passthrough | 486.725 | 451.9 | 0.6875 |
| candidate pure minGRU | 486.950 | 451.1 | 0.6750 |
| candidate ReCoN minGRU + passthrough | 486.950 | 451.1 | 0.6750 |

Promotion score:

- incumbent: `1188.0725`;
- candidate: `1174.7950`;
- promoted: `false`.

Training batches were no longer all-failure batches, but the signal was volatile:

| Iteration | Train mean steps | Train success | ref KL |
| ---: | ---: | ---: | ---: |
| 1 | 478.25 | 0.55 | 0.000142 |
| 2 | 463.10 | 0.45 | 0.000881 |
| 3 | 463.80 | 0.35 | 0.001094 |
| 4 | 461.85 | 0.25 | 0.001172 |
| 5 | 485.25 | 0.55 | 0.000574 |
| 6 | 465.95 | 0.40 | 0.000181 |
| 7 | 456.15 | 0.40 | 0.000239 |
| 8 | 466.25 | 0.45 | 0.000629 |

## Interpretation

This run cleanly separates two issues:

1. The earlier pure-vs-ReCoN metric mismatch was a routing issue. Passthrough removes it: start pure and start ReCoN metrics are identical.
2. The PPO update itself still does not improve the recurrent terminal. Even with passthrough and a healthier mixed train distribution, the candidate lost one held-out success.

The update was not KL-unstable; reference KL stayed very small. The problem is more likely that the reward/data mix gives noisy, low-leverage gradients around the incumbent plateau. Aggressive PPO can move the policy slightly, but the movement does not target the specific held-out tail failures reliably enough.

## Next Step

Use passthrough-enabled evaluation as the default for minGRU terminal claims, then change the recurrent improvement strategy rather than simply increasing PPO pressure. The most likely next useful step is one of:

- add per-iteration scout validation/checkpoint selection so PPO does not keep only the final policy;
- train a smaller gated residual/update head against failure windows while freezing the incumbent minGRU body;
- return to curriculum dataset generation and oversample states immediately before held-out tail failures, with explicit action-preservation loss on incumbent-success windows.

Current N=4 status remains unsolved. The best held-out success is still `0.6875` on the 80-seed mixed block, and no solve claim should be made.
