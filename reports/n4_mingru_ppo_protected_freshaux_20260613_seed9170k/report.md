# Protected minGRU PPO Fresh Option-Aux

Status: `completed`
Promoted: `False`
Best checkpoint: `reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt`
Candidate checkpoint: `reports/n4_mingru_ppo_protected_freshaux_20260613_seed9170k/mingru_ppo.pt`

Promotion metric: `1000*success_rate + p10_survival + 0.1*mean_survival`
Incumbent score: `1180.013750`
Candidate score: `1180.012500`
Minimum delta: `0.1`

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| start pure | 486.7 | 451.9 | 0.6875 | 80 |
| candidate pure | 486.8 | 451.9 | 0.6875 | 80 |
| start ReCoN | 487.1 | 443.8 | 0.6875 | 80 |
| candidate ReCoN | 487.1 | 443.8 | 0.6875 | 80 |

## Training History

| iter | episodes | mean steps | success | ref KL | approx KL |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 452.0 | 0.000 | 5.28e-08 | -4.41e-06 |
| 2 | 16 | 452.3 | 0.000 | 8.15e-08 | -5.30e-07 |
| 3 | 16 | 434.8 | 0.000 | 3.32e-07 | -1.48e-05 |

## Claim Discipline

This run compares the PPO-updated candidate against the starting checkpoint on the same held-out seeds. The candidate was not promoted; no solve claim is made.
