# N=4 Autonomous Training Update - 2026-06-11

## Status

N=4 is **not solved yet**. The current robust frontier remains essentially the original feedforward policy terminal:

| model/run | eval block | mean | p10 | CVaR | success |
|---|---|---:|---:|---:|---:|
| original best ReCoN policy terminal | mixed 8x30 seeds | 484.9 | 434.9 | 414.8 | 0.696 |
| residual bin-delta inside ReCoN | mixed 8x30 seeds | 484.8 | 434.0 | 414.6 | 0.696 |
| ultra-low-LR microfit 15k | mixed 8x30 seeds | 484.9 | 434.0 | 414.8 | 0.700 |
| original best ReCoN policy terminal | held-out 1,500,000 x300 | 485.4 | 441.9 | 417.0 | 0.683 |
| ultra-low-LR microfit 15k | held-out 1,500,000 x300 | 485.4 | 441.9 | 417.0 | 0.683 |
| ultra-low-LR microfit 20k | held-out 1,500,000 x300 | 485.5 | 441.9 | 417.0 | 0.683 |

The 15k/20k microfit checkpoints touched 0.700 on the mixed validation grid but did **not** generalize to the held-out 1.5M block, so I am not treating them as solves.

## What Changed In Code

- Added fixed-width `normalized_raw4` / `normalized_raw4_prev_force` observations so recurrent N=3 -> N=4 transfer has stable input shape.
- Added policy-terminal normalizer loading/export plumbing for VecNormalize-trained candidates.
- Added `scripts/run_ppo_sweep.py` for systematic PPO sweeps with lr/clip/steps/epochs/GAE/entropy/net/VecNormalize/late-bonus axes.
- Added `scripts/train_residual_policy_terminal.py` with force and discrete `bin_delta` residual modes.
- Wired residual policy terminals into `ReConCartPoleController`, so residual learning can causally alter the policy-terminal force inside ReCoN.
- Added `scripts/train_recurrent_policy_terminal_curriculum.py` for N=3 -> N=4-low-angle -> N=4-current -> hard-tail recurrent training.

All committed code passed the full test suite: `54 passed`.

## Training Findings

- From-best PPO sweep candidate with lr `2.5e-6` regressed from 0.696 to 0.558 by 50k, so it was stopped.
- Direct residual learning improved frozen PPO direct-control metrics slightly, but when integrated into ReCoN it did not improve the actual ReCoN frontier.
- Recurrent curriculum transferred very well to easy N=4: N=3 warmup reached 1.000 success and N=4 low-angle/no-noise started at 0.981 success.
- The recurrent policy failed on current noisy N=4: current stage stayed around 0.562 success and hard-tail chunks did not improve it.
- Ultra-low-LR from-best microfit produced a validation-only 0.700 success checkpoint, but held-out 300-seed validation showed no improvement over base.

## Current Best Interpretation

The last gap is not being closed by generic PPO continuation. The base policy is already near a local optimum; most updates either do nothing or damage the lower tail. The recurrent ladder proves easier distributions are learnable, but the noisy current N=4 distribution needs either better recurrent training curriculum/architecture or a more targeted tail objective.

## Best Next Directions

1. Train/evaluate a success-prioritized microfit that promotes by `(success, p10, CVaR)` lexicographically instead of scalar score, then validate on multiple held-out 300 blocks.
2. Improve recurrent current-stage training: lower LR after transfer, freeze feature layers briefly, and add a current-distribution distillation loss from the feedforward best so recurrence does not start below the frontier.
3. Build a hard-seed classifier or failure-mode gate that decides when to defer to the base policy vs a specialist; residual bin-delta helped direct PPO but needs better gating to help ReCoN.
