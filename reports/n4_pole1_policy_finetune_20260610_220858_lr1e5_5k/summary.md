# N=4 Pole_1 Policy Fine-Tune Report

Status: `not solved`
Benchmark split: `930000..930299` fixed benchmark seeds, not pristine held-out
Best fixed-benchmark candidate: `recon_feedforward_terminal_finetuned`
Best mean/p10/success: `483.3` / `432.9` / `0.670`
Best checkpoint path: `reports/n4_pole1_policy_finetune_20260610_220858_lr1e5_5k/finetune_training/ppo_policy_terminal.zip`

## Answers
- Failure-focused dataset exists? `yes`, `15000` samples at `reports/n4_pole1_policy_finetune_20260610_220858_lr1e5_5k/pole1_failure_dataset.npz`.
- Targeted fine-tune trained? `yes`, `5000` timesteps.
- Did fixed benchmark pass solve threshold? `no`.
- Fail -> success / success -> fail: `1` / `0`.
- Dominant failure after best candidate: `pole_1_angle`.
- Confirmation split run? `no`; fixed benchmark did not pass, so no solved confirmation was warranted.
## Candidate Table
- `recon_feedforward_terminal_finetuned`: mean `483.3`, p10 `432.9`, success `0.670`, net success `+1`
- `baseline_best_frozen`: mean `483.2`, p10 `432.9`, success `0.667`, net success `+0`
- `recon_feedforward_terminal_finetuned_25k_rejected`: mean `481.3`, p10 `425.9`, success `0.647`, net success `-6`
- `recon_mingru_terminal_frozen_reference`: mean `459.2`, p10 `379.9`, success `0.443`, net success `n/a`

## Reproduce
```bash
uv run python scripts/run_n4_pole1_policy_finetune.py --out reports/n4_pole1_policy_finetune_20260610_220858_lr1e5_5k --finetune-timesteps 5000
```

## Interpretation
The 5k conservative fine-tune rescued one benchmark seed without regressions, but the 25k fine-tune regressed six successes and neither closed the gap. The blocker appears to require stronger policy learning, better failure-focused training distribution, or reward reformulation rather than another narrow routing gate.
