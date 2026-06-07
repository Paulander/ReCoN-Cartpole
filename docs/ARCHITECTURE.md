# Architecture

ReCoN is used as an interpretable executive graph, not as a PPO-shaped black box.

The generated graph has this high-level flow:

```text
root_balance
  observe_state
  estimate_risk
  select_control_regime
    avoid_rail
    damp_energy
    recover_worst_pole
    recover_base_pole
    stabilize_chain
    center_cart
  arbitrate_force
  apply_force
```

For each pole, the graph factory also adds repeated monitor/proposal nodes. The
runner uses the pragmatic `ReConEngine` for rollouts and keeps the formal engine
available for future explanatory traces.

The custom N-link environment uses exact Gymnasium-style CartPole equations for
`n_poles=1` and a deterministic coupled approximation for higher pole counts. That
approximation is good enough for controller iteration and visualization, but the
next research step is replacing it with a validated multibody solver before making
claims about high-N performance.

