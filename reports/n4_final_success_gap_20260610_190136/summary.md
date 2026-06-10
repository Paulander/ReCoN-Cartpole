# N=4 Final Success Gap Report

Status: `not solved`
Best candidate: `rail_vs_pole_priority_gate`
Best mean/p10/success: `483.2` / `432.9` / `0.667`
Baseline mean/p10/success: `483.2` / `432.9` / `0.667`

## Answers
- Did any patch reach success >=0.70? `no`.
- Did it preserve mean/p10 thresholds? `yes`.
- Fail -> success seeds: `0`.
- Success -> fail seeds: `0`.
- Mechanism: `none; no candidate produced fail->success rescues`.
- Is pole_1_angle still dominant? `yes`; dominant failure is `pole_1_angle`.
- Enough to claim solved under fixed threshold? `no`.

## Candidate Table
- `rail_vs_pole_priority_gate`: mean `483.2`, p10 `432.9`, success `0.667`, net success `+0`
- `anti_oscillation_damper`: mean `483.3`, p10 `432.9`, success `0.663`, net success `-1`
- `terminal_force_passthrough_high_confidence`: mean `483.1`, p10 `432.9`, success `0.657`, net success `-3`
- `rail_gate_terminal_passthrough_combo`: mean `483.1`, p10 `432.9`, success `0.657`, net success `-3`

## Reproduce
```bash
uv run python scripts/run_n4_final_success_gap.py --out reports/n4_final_success_gap_20260610_190136 --validation-episodes 300 --validation-seed-start 930000
```
