# N=4 Guarded Curriculum Update - 2026-06-13

This note records two guarded minGRU curriculum runs launched after adding candidate-vs-incumbent promotion checks. Both runs used the current N=4 5-bin serial-Lagrange setup and held-out mixed blocks only for promotion/evaluation. No solve claim is supported by these runs.

## Incumbent

The protected incumbent remains:

```text
reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt
```

Its tracked held-out reference on the 80-episode mixed block is:

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure minGRU policy | 486.7 | 451.9 | 0.6875 | 80 |
| ReCoN minGRU terminal | 487.1 | 443.8 | 0.6875 | 80 |

Promotion score is `1000*success_rate + p10_survival + 0.1*mean_survival`; the incumbent score in these comparisons was `1180.01375`.

## Tail-Heavy Guarded Curriculum

Report directory:

```text
reports/n4_mingru_curriculum_tailheavy_guarded_20260613_seed9182k
```

Curriculum mix:

| stage | episodes | samples | sample weight |
|---|---:|---:|---:|
| N=3 stable | 8 | 3992 | 0.1 |
| N=4 low-angle/no-noise | 16 | 8000 | 0.15 |
| N=4 current | 40 | 19130 | 0.35 |
| N=4 hard-tail | 120 | 54070 | 6.0 |

Held-out result:

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure minGRU policy | 486.9 | 442.5 | 0.675 | 80 |
| ReCoN minGRU terminal | 486.3 | 441.5 | 0.675 | 80 |

Candidate score: `1165.13`. Promoted: `false`.

Interpretation: the hard-tail block dominated the dataset and hurt the mixed held-out score. This looks like tail overfitting or distribution shift, not progress.

## Balanced Guarded Curriculum

Report directory:

```text
reports/n4_mingru_curriculum_balanced_guarded_20260613_seed9185k
```

Curriculum mix:

| stage | episodes | samples | sample weight |
|---|---:|---:|---:|
| N=3 stable | 8 | 4000 | 0.2 |
| N=4 low-angle/no-noise | 24 | 12000 | 0.3 |
| N=4 current | 80 | 38801 | 1.0 |
| N=4 hard-tail | 80 | 35963 | 2.5 |

Held-out result:

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure minGRU policy | 487.6 | 444.0 | 0.6875 | 80 |
| ReCoN minGRU terminal | 486.8 | 442.7 | 0.675 | 80 |

Candidate score: `1166.38375`. Promoted: `false`.

Interpretation: reducing tail dominance preserved the pure-policy success rate but still damaged the ReCoN-routed terminal success and did not beat the incumbent. This argues against more supervised curriculum reweighting of the same form as the next main path.

## Current Read

The guarded promotion checks worked: both candidates were rejected and the incumbent checkpoint stayed protected. The plateau is now clearer: supervised minGRU curriculum variants can imitate the teacher around `65-68%` action accuracy but have not improved held-out N=4 robustness.

Recommended next move is to pivot away from more of the same supervised reweighting and toward one of:

1. PPO-side exploration/fine-tuning with the same promotion guard, preferably from the incumbent and with stricter mixed-block validation.
2. A different residual/gating formulation that abstains by default and only acts on reliably classified failure precursors.
3. A true shared subchain/recurrent module rather than flat subchain features feeding one monolithic terminal.

Claim discipline remains unchanged: N=4 is near-solved, not robustly solved.
