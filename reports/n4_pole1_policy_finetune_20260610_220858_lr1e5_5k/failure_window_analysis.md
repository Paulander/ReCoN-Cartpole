# Failure Window Analysis

Dataset samples: `15000`
Failure seeds: `60`; matched success seeds: `60`
Failure-window labels: 25 `1500`, 50 `3000`, 100 `6000`, 150 `9000`

## Failure Windows
- Avg |theta1|: `0.1292`; p90 `0.1885`
- Avg |theta1_dot|: `1.0037`; p90 `1.4057`
- Avg |ReCoN force - PPO force|: `6.867`; p90 `20.000`
- PPO/ReCoN force sign disagreement rate: `0.357`

## Matched Success Windows
- Avg |theta1|: `0.0466`
- Avg |theta1_dot|: `0.3270`

## Interpretation
This dataset is diagnostic and fine-tune support data. The fixed 930000..930299 seeds are labelled as a fixed benchmark split, not pristine held-out data, because previous iterations have inspected them repeatedly.
