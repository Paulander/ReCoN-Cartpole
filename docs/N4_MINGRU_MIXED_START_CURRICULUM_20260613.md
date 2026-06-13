# N=4 minGRU Mixed-Start Curriculum Probe - 2026-06-13

## Goal

Move the recurrent branch closer to the requested curriculum setup: N=3 stable -> N=4 low-angle/no-noise -> N=4 current -> hard-tail, with previous force/history and held-out mixed-block evaluation.

The previous curriculum runner already had the stage sequence, but each stage used one contiguous seed start unless a manual seed list was supplied. This update adds mixed stage starts so training data can cover multiple blocks without hand-writing seed files.

## Implementation

Updated `scripts/train_mingru_curriculum.py`:

- added `--n3-seed-starts`, `--low-angle-seed-starts`, `--current-seed-starts`, and `--tail-seed-starts`;
- added round-robin seed expansion, e.g. `[100, 200] x 5 -> 100, 200, 101, 201, 102`;
- materializes a per-stage `collection_seeds.txt` for `build_policy_dataset.py`;
- records `seed_starts` and the generated `seed_list` in each stage metadata file.

Focused tests cover round-robin seed expansion, generated seed-list materialization, default stage construction, and existing passthrough forwarding.

## Probe

Report:

`reports/n4_mingru_curriculum_mixedstarts_probe_20260613_seed9280k`

Started from incumbent:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k/checkpoint_iter_002.pt`

Training configuration:

| stage | episodes | seed starts | samples | weight |
| --- | ---: | --- | ---: | ---: |
| N=3 stable | 4 | `2810000`, `2820000` | 2000 | 0.25 |
| N=4 low-angle/no-noise | 6 | `2910000`, `2920000` | 3000 | 0.50 |
| N=4 current | 8 | `3010000`, `3020000`, `3030000`, `3040000` | 3850 | 1.00 |
| N=4 hard-tail | 8 | `3110000`, `3120000`, `3130000`, `3140000` | 3807 | 1.50 |

Total samples: `12657`.

Evaluation used held-out mixed starts `1900000`, `2000000`, `2100000`, `2200000`, 10 episodes each.

## Result

| evaluator | mean | p10 | success | episodes |
| --- | ---: | ---: | ---: | ---: |
| incumbent ReCoN minGRU | 485.05 | 438.7 | 0.700 | 40 |
| mixed-start curriculum candidate | 484.775 | 437.2 | 0.700 | 40 |

Promotion score:

- incumbent: `1187.2050`
- candidate: `1185.6775`
- promoted: `false`

The candidate preserved success but slightly regressed mean and p10, so the incumbent remains the best checkpoint.

## Interpretation

The mixed-start curriculum plumbing is now in place and verified. The small supervised continuation did not crack the N=4 tail; it behaved like the PPO continuation attempts, preserving broad behavior while slightly softening the tail.

Likely next recurrent move: use mixed-start stage collection with a stronger training change, not just a small supervised continuation. Options include a larger mixed curriculum run from scratch/earlier DAgger checkpoint, on-policy minGRU PPO after mixed curriculum, or stage-specific hard-tail weighting with scout selection. This probe is not solve evidence.

Current N=4 status remains unsolved.
