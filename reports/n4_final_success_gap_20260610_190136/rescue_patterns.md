# Rescue Patterns

Baseline successes/failures: `200` / `100`
Late failures (>=450): `53`
Very late failures (>=475): `27`
Failure distribution: `{'pole_1_angle': 53, 'pole_2_angle': 44, 'pole_0_angle': 3}`
Traced arbitration effect vs pure PPO: `{'helped': 95, 'neutral': 1, 'hurt': 4}`
Selected regime counts in traced failures: `{'recover_worst_pole': 13630, 'stabilize_chain': 31341}`

## Easiest To Save
- `930039`: step `497`, `pole_1_angle`
- `930127`: step `497`, `pole_1_angle`
- `930179`: step `496`, `pole_2_angle`
- `930115`: step `495`, `pole_2_angle`
- `930205`: step `494`, `pole_2_angle`
- `930291`: step `494`, `pole_2_angle`
- `930083`: step `493`, `pole_1_angle`
- `930122`: step `491`, `pole_0_angle`
- `930147`: step `491`, `pole_2_angle`
- `930058`: step `490`, `pole_2_angle`
- `930066`: step `490`, `pole_0_angle`
- `930245`: step `490`, `pole_2_angle`
- `930021`: step `489`, `pole_0_angle`
- `930092`: step `489`, `pole_1_angle`
- `930263`: step `489`, `pole_1_angle`
- `930013`: step `488`, `pole_1_angle`
- `930118`: step `485`, `pole_2_angle`
- `930204`: step `485`, `pole_1_angle`
- `930137`: step `484`, `pole_2_angle`
- `930009`: step `483`, `pole_1_angle`
- `930052`: step `481`, `pole_1_angle`
- `930161`: step `481`, `pole_1_angle`
- `930044`: step `480`, `pole_2_angle`
- `930084`: step `480`, `pole_1_angle`
- `930212`: step `479`, `pole_2_angle`
- `930123`: step `477`, `pole_2_angle`
- `930260`: step `475`, `pole_2_angle`
- `930185`: step `474`, `pole_2_angle`
- `930139`: step `473`, `pole_1_angle`
- `930172`: step `473`, `pole_1_angle`

## Trace Diagnostics
- `last100_pole1_max_abs_angle` avg `0.175`, p90 `0.210`
- `last100_pole1_max_abs_velocity` avg `1.229`, p90 `1.593`
- `last100_force_oscillation_score` avg `0.711`, p90 `0.889`
- `last100_max_rail_abs_x` avg `0.121`, p90 `0.173`
- `policy_final_force_abs_diff_mean` avg `5.486`, p90 `6.880`
