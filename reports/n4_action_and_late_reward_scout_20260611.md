# N=4 Action and Late-Reward Scout - 2026-06-11

All rows use the same mixed validation grid: seed starts `900000, 930000, 970000, 1010000, 1040000, 1070000, 1140000, 1300000`, `20` or `30` episodes per start as noted.

| Run | Validation episodes/start | Best checkpoint | Mean | P10 | CVaR | Success | Verdict |
|---|---:|---|---:|---:|---:|---:|---|
| 5-bin frontier start (`n4_robust_tail_microfit_20260611_seed2160k`) | 30 | `checkpoint_000000_start.zip` | 484.9 | 434.9 | 414.8 | 0.696 | Baseline mixed-grid frontier. |
| 9-bin scout (`n4_9bin_tail_scout_20260611_seed2180k`) | 20 | `checkpoint_100000.zip` | 478.8 | 424.0 | 391.5 | 0.631 | Learns, but trails the 5-bin frontier. |
| continuous scout (`n4_continuous_tail_scout_20260611_seed2190k`) | 20 | `checkpoint_050000.zip` | 464.7 | 384.9 | 362.1 | 0.512 | Worse in this PPO setup. |
| 5-bin late-survival microfit (`n4_late_survival_microfit_20260611_seed2200k`) | 30 | `checkpoint_005000.zip` | 484.8 | 434.9 | 414.8 | 0.696 | Preserves frontier for one chunk, then regresses; not enough. |

## Code Change

Added a training-only `LateSurvivalBonusWrapper` with CLI flags:

- `--late-survival-bonus`
- `--late-survival-start-fraction`

Evaluation calls still disable training bonuses via `use_success_bonus=False`, so held-out metrics remain survival-only.

## Interpretation

The N=4 gap does not look like a simple action-resolution issue under short PPO scouts: 9 bins and continuous action both underperform the existing 5-bin learned terminal. A modest final-100-tick reward gives a less destructive update than stronger terminal reward, but still does not move mixed-grid success above `0.696`.

Next promising direction: either a larger retrain with the late-survival objective from scratch, or a more structural policy objective/model change rather than micro fine-tuning the current checkpoint.
