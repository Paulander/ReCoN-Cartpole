# minGRU Recurrent Curriculum

Status: `completed`
Observation mode: `normalized_raw4_subchains_prev_force`
Sequence length: `16`

| stage | n | teacher | rollout | angle | noise | weight | samples |
|---|---:|---|---|---:|---:|---:|---:|
| n3_stable | 3 | static_recon | teacher | 0.030 | 0.010 | 1.000 | 2500 |
| n4_low_angle_no_noise | 4 | recon_policy_terminal | teacher | 0.020 | 0.000 | 1.000 | 2500 |
| n4_current | 4 | recon_policy_terminal | teacher | 0.050 | 0.020 | 1.000 | 9465 |
| n4_hard_tail | 4 | recon_policy_terminal | mingru_terminal | 0.050 | 0.020 | 2.000 | 71474 |

## Held-Out N=4 Eval

Eval status: `completed`

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure_mingru_policy | 487.5 | 442.7 | 0.688 | 80 |
| recon_mingru_terminal | 486.7 | 441.8 | 0.675 | 80 |

## Claim Discipline

This is a recurrent curriculum experiment. N=3 and low-angle stages are training curriculum only; solve claims require held-out N=4 metrics.
