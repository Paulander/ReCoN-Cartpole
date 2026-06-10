# N=4 Stronger Learning Pass

Status: `borderline_not_robustly_solved`

This pass tested whether the current N=4 plateau could be moved by a stronger learned-policy route rather than another hand-coded ReCoN patch. The experiment resumed the current best PPO policy terminal only for measurement and conservative continuation. No ReCoN script nodes or controller thresholds were added.

## Checkpoints

- Starting checkpoint: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`
- Eval-only fresh block report: `reports/n4_stronger_iterative_start_final_20260610/summary.md`
- Confirmation block report: `reports/n4_stronger_iterative_start_confirm_1000k_20260610/summary.md`
- Conservative continuation report: `reports/n4_stronger_iterative_resume_20260610/summary.md`

## Results

| split | evaluator | mean | p10 | success | episodes | status |
|---|---|---:|---:|---:|---:|---|
| validation `950000,960000,970000`, 80 each | recon_policy_terminal | 486.7 | 446.0 | 0.721 | 240 | pass |
| final `990000..990299` | pure_ppo | 442.9 | 324.9 | 0.533 | 300 | fail |
| final `990000..990299` | recon_policy_terminal | 485.6 | 443.8 | 0.703 | 300 | pass |
| confirmation `1000000..1000299` | pure_ppo | 442.4 | 333.0 | 0.517 | 300 | fail |
| confirmation `1000000..1000299` | recon_policy_terminal | 485.3 | 433.9 | 0.683 | 300 | fail |

The `990000..990299` block clears the operational N=4 threshold (`mean >= 475`, `p10 >= 350`, `success >= 0.70`). The independent `1000000..1000299` confirmation block misses success by about five episodes (`0.683` vs `0.700`) while preserving strong mean and p10. Therefore this should not be called robustly solved yet.

## Continuation Attempt

Conservative PPO continuation from the same checkpoint degraded validation:

| checkpoint | timesteps | mean | p10 | success |
|---|---:|---:|---:|---:|
| start | 0 | 486.7 | 446.0 | 0.721 |
| chunk_1 | 25000 | 485.8 | 443.9 | 0.700 |
| chunk_2 | 50000 | 483.8 | 438.5 | 0.700 |

The continuation was stopped after two chunks because it was drifting downward. This suggests the current policy is near a fragile local optimum under the existing PPO objective and hard-seed mix.

## Interpretation

The project is not stuck at the original `0.667` fixed 930k benchmark in a general sense: on fresher seed blocks, the same learned terminal plus ReCoN arbitration reaches `0.683-0.703` success with strong p10. However, it is still borderline. The difference between pure PPO and ReCoN-routed policy is large on both fresh blocks, so ReCoN arbitration is materially helping behavior:

- `990k`: ReCoN `0.703` vs pure PPO `0.533`
- `1000k`: ReCoN `0.683` vs pure PPO `0.517`

The next performance move should not be another tiny fine-tune of this same MLP objective. Better candidates are:

- train a fresh policy with a less brittle curriculum and validation promotion across several fresh blocks;
- add recurrence or frame history from the start rather than trying to bolt it on after a feedforward checkpoint;
- revise the reward/curriculum so extra PPO updates improve the fragile tail instead of reducing it.

## Claim Discipline

Claim supported: `N=4 is borderline and ReCoN-routed learned control is clearly ahead of pure PPO on these evaluations.`

Claim not supported: `N=4 is robustly solved.`
