# N=4 Pole_1 Robustness Report

Status: `not solved`
Report directory: `reports/n4_pole1_robustness_20260610_171635`
Best candidate: `recon_feedforward_terminal_frozen`
Best mean/p10/success: `483.2` / `432.9` / `0.67`

## Answers

- Does the best feedforward policy terminal solve N=4 on 300 held-out seeds? `no`.
- Is pole_1_angle still dominant? `yes`; dominant non-success failure for the best row is `pole_1_angle`.
- Does ReCoN arbitration help feedforward? `helped (+43.9 mean steps, +0.147 success)`.
- Does ReCoN learning around feedforward help? `hurt (-17.3 mean steps, -0.173 success)`.
- Did the pole_1 fix improve robustness? `hurt (-0.0 mean steps, -0.003 success)`.
- Remaining problem: compare `trace_comparison.md`; if feedforward is solved, remaining recurrent gap is policy learning. If not, inspect pole_1 timing and environment/control edge cases before N=5.

## Key Rows

- `recon_feedforward_terminal_frozen`: mean `483.2`, p10 `432.9`, success `0.67`
- `recon_feedforward_terminal_with_pole1_fix`: mean `483.2`, p10 `432.9`, success `0.66`
- `recon_feedforward_terminal_plus_recon_learning`: mean `465.9`, p10 `391.0`, success `0.49`
- `static_recon`: mean `460.6`, p10 `382.8`, success `0.45`
- `recon_mingru_terminal_frozen`: mean `459.2`, p10 `379.9`, success `0.44`
- `baseline_heuristic`: mean `456.5`, p10 `375.0`, success `0.41`
- `pure_feedforward_policy_terminal`: mean `439.3`, p10 `324.0`, success `0.52`

## Reproduce Best Eval

```bash
uv run python scripts/run_n4_pole1_robustness.py --out reports/n4_pole1_robustness_20260610_171635 --feedforward-checkpoint reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip --recurrent-checkpoint reports/n4_autonomous_recurrent_20260610_160353/candidate_logs/b0021ba0d0bb/supervised/mingru_terminal.pt --validation-episodes 300 --validation-seed-start 930000
```

## Resume/Extend

```bash
uv run python scripts/run_n4_pole1_robustness.py --out reports/n4_pole1_robustness_20260610_171635_extended --feedforward-checkpoint reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip --recurrent-checkpoint reports/n4_autonomous_recurrent_20260610_160353/candidate_logs/b0021ba0d0bb/supervised/mingru_terminal.pt --validation-episodes 300 --validation-seed-start 930000 --plus-train-episodes 120
```
