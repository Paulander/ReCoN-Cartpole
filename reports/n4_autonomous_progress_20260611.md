# N=4 Autonomous Progress Note - 2026-06-11

## Current Frontier

The best learned feedforward policy terminal inside ReCoN remains `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.

Evidence so far:

| Evaluation | Mean | P10 | Success | Notes |
|---|---:|---:|---:|---|
| Tail curriculum held-out 300 (`reports/n4_tail_curriculum_20260611`) | 485.4 | 442.8 | 0.703 | Passes the configured >=0.70 solve threshold on that held-out block. |
| Fixed 930000..930299 benchmark | 483.2 | 432.9 | 0.667 | Repeatedly inspected benchmark; not solved. |
| 8-block scout, 100 episodes/block | 479.5-486.5 | 420.2-449.4 | 0.620-0.760 | Robustness is block-dependent; mean and p10 stay strong, success is fragile. |

## Recurrent Policy Terminal

A true RecurrentPPO terminal path was added and verified. Best recurrent run so far is `reports/n4_recurrent_tail_20260611_seed2120k_fs2_fast`:

- ReCoN-routed recurrent terminal final 100: mean 481.1, p10 428.4, success 0.630.
- Pure recurrent PPO final 100: mean 367.8, p10 274.6, success 0.080.

Interpretation: recurrent learning is real and ReCoN routing helps it substantially, but this recurrent configuration does not beat the best feedforward policy terminal.

## Weak-Block Hard Seeds

Collected 176 late-failure hard seeds from weak scout blocks: `reports/hard_seeds_n4_best_weak_blocks_20260611`.

Aggregate over the 500 weak-block scout episodes:

- mean 482.8
- p10 431.9
- success 0.648
- failures: pole_1_angle 92, pole_2_angle 70, pole_0_angle 14

## Guarded Microfit Result

Run: `reports/n4_robust_tail_microfit_20260611_seed2160k`.

Started from the best feedforward checkpoint and trained tiny 5k PPO chunks against the weak-block hard-seed pool, validating on eight mixed seed starts. It did not improve the start checkpoint:

| Checkpoint | Mean | P10 | CVaR | Success | Promoted |
|---|---:|---:|---:|---:|---:|
| start | 484.9 | 434.9 | 414.8 | 0.696 | true |
| chunk_1 5k | 484.7 | 434.0 | 414.8 | 0.696 | false |
| chunk_2 10k | 484.8 | 434.0 | 414.5 | 0.696 | false |

The run was stopped after two flat chunks.

## Practical Next Moves

1. Try a different action formulation before more fine-tuning: continuous action or more than 5 discrete force bins. The current failures are late near-misses where coarse action resolution may matter.
2. Add a policy objective that explicitly values finishing the final 50-100 ticks, not just survival-shaped reward. The current model gets many failures in the 400-499 range and p10 stays high.
3. If continuing recurrent work, use transfer/architecture changes rather than more identical N=4 RecurrentPPO seeds. Direct recurrent runs learned but plateaued below the feedforward terminal.

Claim discipline: N=4 has one held-out >=0.70 pass, but robustness across seed blocks is not settled. Do not claim a strong/perfect N=4 solve until multi-block held-out success is consistently >=0.70.
