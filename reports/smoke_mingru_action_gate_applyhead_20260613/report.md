# minGRU Counterfactual Action Gate

Status: `completed`
Checkpoint: `reports/n4_mingru_curriculum_subchain_motif_dagger9_hardtail_20260613_seed9099k/supervised_mingru/mingru_terminal.pt`
Gate path: `reports/smoke_mingru_action_gate_applyhead_20260613/mingru_action_gate.pt`
Training rows: `50`, positives: `9`, apply positives: `1`
Label counts: `{'0': 41, '1': 2, '2': 0, '3': 0, '4': 0, '5': 7}`
Failure classes: `['pole_1_angle', 'pole_2_angle']`
Gate confidence: `0.55`
Gate margin: `0.3`
Gate apply threshold: `0.6`
Forced action hold steps: `20`

| evaluator | mean | p10 | cvar | success | overrides | override rate | episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_recon_mingru | 484.4 | 432.2 | 414.5 | 0.650 | 0 | 0.0000 | 20 |
| gated_recon_mingru | 484.4 | 432.2 | 414.5 | 0.650 | 0 | 0.0000 | 20 |

## Claim Discipline

The gate is trained from recurrent-prefix counterfactual probes near selected failure classes and evaluated on separately requested held-out seeds. It is not a solve claim unless held-out metrics clear the configured solve threshold.
