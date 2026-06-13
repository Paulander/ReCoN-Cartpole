# N=4 minGRU Action-Gate Probe - 2026-06-13

## Goal

Test a targeted learned correction on top of the promoted minGRU checkpoint, while freezing the base recurrent policy. This addresses the evidence that broad PPO continuation mostly preserves or swaps tail failures rather than learning a specific correction.

Frozen base checkpoint:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k/checkpoint_iter_002.pt`

## Implementation Update

`train_mingru_action_gate.py` now supports positive oversampling for rare counterfactual override labels:

- new flag: `--positive-oversample-factor`;
- metadata records `original_row_count`, `expanded_row_count`, and `positive_oversample_factor`;
- evaluation behavior is unchanged.

This was needed because the first option-style gate found a few positive labels, but the classifier learned the all-noop solution.

## Probe 1: One-Tick Late-Failure Gate

Run:

`reports/n4_mingru_action_gate_promoted_hardseed_20260613_seed9250k`

Setup:

- collect from mined hard seeds;
- target failures: `pole_1_angle`, `pole_2_angle`;
- failure offsets: `0, 10, 20, 40`;
- forced action hold: `1` tick;
- conservative counterfactual score gap.

Result:

- rows: `83`;
- positives: `0`;
- status: `completed_no_positive_labels`;
- max score gap: `0.0019`.

Interpretation: one-tick action overrides at/near the failure edge do not reveal useful local rescues for the promoted checkpoint.

## Probe 2: Option-Style Gate Without Oversampling

Run:

`reports/n4_mingru_action_gate_promoted_option_probe_20260613_seed9251k`

Setup changes:

- earlier failure offsets: `20, 40, 80, 120`;
- forced action hold: `3` ticks;
- longer probe horizon: `120`;
- looser score gaps.

Result:

- rows: `71`;
- positives: `4`;
- apply positives: `4`;
- training positive recall: `0.0`;
- held-out overrides: `0`;
- held-out success: unchanged at `0.6875`.

Interpretation: earlier multi-tick options can find sparse corrective labels, but the unbalanced classifier ignores them.

## Probe 3: Option-Style Gate With Positive Oversampling

Run:

`reports/n4_mingru_action_gate_promoted_option_oversample_20260613_seed9252k`

Setup changes:

- `--positive-oversample-factor 12`;
- gate confidence `0.75`;
- gate margin `0.05`;
- apply threshold `0.65`.

Training result:

- original rows: `71`;
- expanded rows: `115`;
- positives: `4` expanded to `48` positive training rows;
- training positive recall: `1.0`;
- apply accuracy: `0.9826`.

Held-out mixed result:

| Evaluator | Mean | P10 | CVaR | Success | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: |
| base promoted minGRU | 486.600 | 452.6 | 418.5 | 0.6875 | 0 |
| oversampled action gate | 486.488 | 452.6 | 417.5 | 0.6875 | 295 |

Per-seed changes were small but net negative:

- no success gains;
- no success losses;
- mean survival decreased by `0.1125`;
- CVaR decreased by `1.0`;
- largest observed held-out losses included seed `2000019` at `-6` steps and seed `1900012` at `-2` steps.

## Interpretation

The targeted gate path is now technically working: it can find sparse multi-tick counterfactual labels, learn them with oversampling, and produce held-out overrides. However, the current labels are not good enough to improve the promoted checkpoint. The gate mostly churns behavior without increasing success or p10.

This is useful evidence against simply pushing classifier pressure higher. The next improvement needs better labels/window selection, not merely a stronger classifier.

## Next Step

The most plausible next iteration is to collect failure-window labels from held-out-like near misses and require stronger counterfactual evidence before labeling an override:

- use longer option sequences or two-phase options rather than a single repeated action;
- rank candidate windows by recovery pressure/motif score before labeling;
- include matched preservation rows from solved episodes at similar risk levels;
- evaluate gate thresholds with a separate threshold sweep before full application.

Current N=4 status remains unsolved. Best held-out success is still `0.6875`; best p10 remains `452.6` from the promoted scout-selected minGRU checkpoint.
