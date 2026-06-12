# N=4 Conservative Continuation Update - 2026-06-12

## Status

N=4 is still **not solved**. The current robust frontier remains the original ReCoN policy-terminal checkpoint:

`reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`

The best known held-out result remains approximately `0.692` success on the 1.5M/1.6M seed blocks. New continuation attempts can reproduce `0.700` on the mixed validation block, but the gain is still not robust out-of-block.

## New Diagnostic

Added `scripts/compare_policy_actions.py`, which runs two ReCoN policy-terminal checkpoints on identical seeds and reports:

- per-seed survival deltas
- first action-difference step
- action-difference fraction
- success gains/losses
- selected regime and force at the first divergence

This makes it easier to separate real behavioral improvements from broad neutral action drift.

## Action-Difference Findings

Both promoted validation candidates show the same pattern: many action changes, but only one validation success flip.

| comparison | validation success | changed seeds | success gains | success losses | first-diff median |
|---|---:|---:|---:|---:|---:|
| base -> lexicographic 15k | 0.696 -> 0.700 | 107 / 240 | 1 | 0 | 254 |
| base -> target-KL 5k | 0.696 -> 0.700 | 114 / 240 | 1 | 0 | 240 |

In both cases the sole success gain is seed `930013`, where the first action flips at step `0` from full left to full right. This does not generalize to held-out blocks.

## New Training Knobs

Added conservative continuation controls:

- `--target-kl` for PPO policy step limiting.
- `--teacher-anchor-model-path`, `--teacher-action-penalty`, `--teacher-anchor-until-fraction`, and `--teacher-anchor-risk-threshold` for training-only teacher-action anchoring.

The teacher anchor penalizes action deviations from a teacher policy only during training and only while the episode is before the configured horizon fraction and below the risk threshold.

## Runs

Teacher-anchored attempt:

`reports/n4_teacher_anchor_tail_20260612_seed2640k`

- First checkpoint regressed: success `0.696 -> 0.692`, p10 `434.9 -> 433.9`, CVaR `414.8 -> 413.1`.
- Stopped early. Strong anchoring plus shaping was too restrictive/misaligned.

Target-KL survival attempt:

`reports/n4_targetkl_survival_tail_20260612_seed2650k`

| checkpoint | mean | p10 | CVaR | success | promoted |
|---|---:|---:|---:|---:|---|
| start | 484.9 | 434.9 | 414.8 | 0.696 | yes |
| 5k | 484.9 | 434.9 | 414.7 | 0.700 | yes |
| 10k | 484.5 | 434.9 | 414.2 | 0.700 | no |
| 15k | 484.4 | 434.9 | 414.2 | 0.692 | no |
| 20k | 474.7 | 413.9 | 392.8 | 0.583 | no |

Final held-out eval for the promoted 5k checkpoint:

| evaluator | mean | p10 | CVaR | success | episodes |
|---|---:|---:|---:|---:|---:|
| pure PPO | 444.3 | 332.8 | n/a | 0.533 | 240 |
| ReCoN policy terminal | 484.4 | 441.9 | 413.2 | 0.688 | 240 |

## Interpretation

Whole-policy continuation is now the weak link. Even very small PPO updates can produce a validation-local one-seed gain, then quickly damage the policy with additional chunks. The learned policy is near a brittle local optimum: broad action drift is easy, robust tail improvement is not.

The next higher-probability route is a state-conditioned specialist/gate trained from failure-mode data, with strict held-out evaluation. The gate should only intervene in diagnosed near-failure states instead of updating the whole policy everywhere.

## Verification

`uv run pytest -q -s` -> `58 passed`.
