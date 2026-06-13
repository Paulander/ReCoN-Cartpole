# N=4 Conservative minGRU PPO Continuation - 2026-06-13

## Goal

Continue from the newly promoted scout-selected minGRU checkpoint and test whether a more conservative PPO pass can improve the held-out tail without losing the p10 gain.

Starting checkpoint:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k/checkpoint_iter_002.pt`

Run directory:

`reports/n4_mingru_ppo_scout_continue_conservative_20260613_seed9240k`

## Setup

Important differences from the previous promoted run:

- lower learning rate: `3e-6`;
- tighter clip range: `0.05`;
- stronger reference KL: `0.04`;
- 120 training episodes, with 36 hard-tail and 84 fresh seeds;
- 5 PPO iterations, 24 rollout episodes each;
- larger scout validation: 40 held-out mixed episodes per iteration;
- passthrough-enabled final validation against the promoted checkpoint.

## Scout Results

| Iteration | Scout success | Scout p10 | Scout mean | Scout score |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.70 | 438.7 | 485.100 | 1187.2100 |
| 2 | 0.70 | 437.8 | 484.975 | 1186.2975 |
| 3 | 0.70 | 437.8 | 485.025 | 1186.3025 |
| 4 | 0.70 | 438.7 | 485.050 | 1187.2050 |
| 5 | 0.70 | 438.7 | 485.075 | 1187.2075 |

The selector chose iteration 1:

`reports/n4_mingru_ppo_scout_continue_conservative_20260613_seed9240k/checkpoint_iter_001.pt`

## Final Held-Out Result

| Model | Mean | P10 | Success |
| --- | ---: | ---: | ---: |
| promoted incumbent | 486.600 | 452.6 | 0.6875 |
| conservative continuation | 486.600 | 452.6 | 0.6875 |

Promotion score:

- incumbent: `1188.7600`;
- candidate: `1188.7600`;
- promoted: `false`, because the candidate did not exceed `min_promotion_delta=0.1`.

## Interpretation

This run preserved the promoted checkpoint exactly on the 80-episode held-out mixed block, but did not improve it. The larger scout block showed a misleading-looking `0.70` success rate, but with much lower p10. Full validation correctly rejected the continuation.

The useful lesson is that conservative PPO around the promoted checkpoint is now mostly behavior-preserving. It does not appear to generate new tail-recovery behavior by itself.

## Next Step

Move away from plain PPO continuation as the primary lever. The next most promising route is to train a gated residual/update head or failure-window specialist while freezing most of the promoted minGRU behavior. That better matches the observed problem: the incumbent already solves many seeds, and broad PPO updates mostly risk swapping which tail cases fail rather than learning a targeted correction.

Current N=4 status remains unsolved. Best held-out success is still `0.6875`; best p10 is `452.6` from the promoted scout-selected checkpoint.
