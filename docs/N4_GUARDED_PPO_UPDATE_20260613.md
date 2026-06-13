# N=4 Guarded minGRU PPO Update - 2026-06-13

This note records two guarded PPO fine-tuning runs from the current best minGRU incumbent. Both used the current N=4 5-bin serial-Lagrange setup and evaluated candidate-vs-incumbent on the same 80 held-out mixed-block episodes (`1900000`, `2000000`, `2100000`, `2200000`, 20 each). No train-seed solve claims are used.

## Incumbent

```text
reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt
```

Held-out ReCoN minGRU reference:

| mean | p10 | success | promotion score |
|---:|---:|---:|---:|
| 487.1375 | 443.8 | 0.6875 | 1180.01375 |

Promotion score is `1000*success_rate + p10_survival + 0.1*mean_survival`.

## Conservative Late-Survival PPO

Report directory:

```text
reports/n4_mingru_ppo_guarded_latesurvival_20260613_seed9190k
```

Key settings: `lr=1e-6`, `clip=0.03`, `gamma=0.997`, `late_survival_bonus=0.02`, `ref_kl_coef=0.08`, six iterations of 16 hard-tail episodes each.

Training signal:

| final train mean steps | hard-tail train success | final ref KL |
|---:|---:|---:|
| 444.625 | 0.0 | 0.0000101 |

Held-out result:

| evaluator | mean | p10 | success | score | promoted |
|---|---:|---:|---:|---:|---|
| ReCoN minGRU candidate | 487.125 | 443.8 | 0.6875 | 1180.0125 | false |

Interpretation: this run was effectively a no-op. It preserved success but failed to improve the incumbent.

## Stronger Late-Survival PPO

Report directory:

```text
reports/n4_mingru_ppo_guarded_stronger_20260613_seed9191k
```

Key settings: `lr=5e-6`, `clip=0.08`, `gamma=0.997`, `late_survival_bonus=0.05`, `ref_kl_coef=0.02`, six iterations of 16 hard-tail episodes each.

Training signal:

| final train mean steps | hard-tail train success | final ref KL |
|---:|---:|---:|
| 444.3125 | 0.0 | 0.0003122 |

Held-out result:

| evaluator | mean | p10 | success | score | promoted |
|---|---:|---:|---:|---:|---|
| ReCoN minGRU candidate | 487.0375 | 443.8 | 0.6875 | 1180.00375 | false |

Interpretation: relaxing the anchor made the model move more, but it still did not create hard-tail successes and slightly reduced held-out mean/score.

## Current Read

Hard-seed-only minGRU PPO is not currently breaking the N=4 plateau. The guarded promotion mechanism worked: both PPO candidates were rejected and the incumbent stayed protected.

The next useful direction should change the data or control surface, not merely repeat this recipe. Higher-signal options are:

1. Mixed-distribution PPO fine-tuning rather than hard-seed-only rollouts, so the gradient sees preservation and recovery together.
2. A more selective learned residual/gate that abstains by default and only acts on reliable late-failure precursors.
3. A feedforward PPO sweep slice from the best PPO terminal with the systematic grid machinery, especially if using broader mixed validation and explicit promotion discipline.

Claim discipline remains unchanged: N=4 is near-solved, not robustly solved.
