from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_policy_terminal.py"
    spec = spec_from_file_location("train_policy_terminal", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_iterative_trainer():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    path = scripts_dir / "train_policy_terminal_iterative.py"
    sys.path.insert(0, str(scripts_dir))
    spec = spec_from_file_location("train_policy_terminal_iterative", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer()
make_env = trainer.make_env
iterative_trainer = _load_iterative_trainer()


def _args(success_bonus: float):
    return SimpleNamespace(
        n_poles=1,
        horizon=1,
        dt=0.02,
        dynamics_mode="parallel",
        action_mode="discrete",
        discrete_action_bins=2,
        force_mag=10.0,
        initial_angle_range=0.0,
        force_noise=0.0,
        link_coupling=0.35,
        hard_train_seeds="",
        hard_train_seed_probability=1.0,
        frame_stack=1,
        policy_observation_mode="env",
        success_bonus=success_bonus,
        failure_penalty=0.0,
    )


def test_success_bonus_is_training_only():
    train_env = make_env(_args(7.0), reward_mode="survival", use_success_bonus=True)
    train_env.reset(seed=1)
    _obs, reward, terminated, truncated, info = train_env.step(1)
    assert not terminated
    assert truncated
    assert reward == 8.0
    assert info["success_bonus"] == 7.0

    eval_env = make_env(_args(7.0), reward_mode="survival", use_success_bonus=False)
    eval_env.reset(seed=1)
    _obs, reward, terminated, truncated, info = eval_env.step(1)
    assert not terminated
    assert truncated
    assert reward == 1.0
    assert "success_bonus" not in info


def test_residual_env_can_use_recon_base_controller(monkeypatch):
    residual = _load_script("train_residual_policy_terminal")
    captured = {"started": 0}

    class FakeController:
        def __init__(self, config):
            captured["mode"] = config.mode
            captured["policy_terminal_path"] = config.policy_terminal_path

        def start_episode(self):
            captured["started"] += 1

        def act(self, env_obs, raw):
            return 4, {"force": 10.0}

    monkeypatch.setattr(residual, "ReConCartPoleController", FakeController)
    args = SimpleNamespace(
        n_poles=1,
        horizon=4,
        dt=0.02,
        dynamics_mode="parallel",
        env_action_mode="discrete",
        discrete_action_bins=5,
        force_mag=10.0,
        initial_angle_range=0.0,
        force_noise=0.0,
        link_coupling=0.35,
        base_model_path="base.zip",
        device="cpu",
        residual_base_controller="recon_policy_terminal",
        selection_mode="hard_select",
        policy_terminal_blend=1.0,
        policy_terminal_scope="stabilize_chain",
        base_observation_mode="normalized_raw",
        base_normalizer_path="",
        residual_feature_mode="basic",
        residual_action_bins=5,
        hard_seed_probability=0.0,
    )

    env = residual.ResidualCorrectionEnv(args)
    obs, info = env.reset(seed=3)
    action, force = env._base_action_and_force(obs, info["raw_state"])

    assert captured["mode"] == "recon_policy_terminal"
    assert captured["policy_terminal_path"] == "base.zip"
    assert captured["started"] == 1
    assert action == 4
    assert force == 10.0


def test_counterfactual_residual_collect_seed_values_reads_hard_seed_json(tmp_path):
    residual = _load_script("train_counterfactual_residual_terminal")
    path = tmp_path / "hard_seeds.json"
    path.write_text('{"hard_seeds": [10, 20, 30]}', encoding="utf-8")
    args = SimpleNamespace(collect_seed_list=str(path), collect_seed_start=1, collect_episodes=2)

    assert residual.collect_seed_values(args) == [10, 20]


def test_policy_dataset_explicit_seeds_reads_hard_seed_json(tmp_path):
    dataset_builder = _load_script("build_policy_dataset")
    path = tmp_path / "hard_seeds.json"
    path.write_text('{"hard_seeds": [11, 22, 33]}', encoding="utf-8")
    args = SimpleNamespace(seed_list=str(path), seed_start=1, episodes=2)

    assert dataset_builder.explicit_seeds(args) == [11, 22, 33]
    assert dataset_builder.collection_seeds(args) == [11, 22]


def test_counterfactual_residual_label_summary_counts_non_noop():
    residual = _load_script("train_counterfactual_residual_terminal")
    rows = [
        {"label": 2, "score_gap": 0.0},
        {"label": 1, "score_gap": 0.5},
        {"label": 3, "score_gap": 0.25},
    ]

    summary = residual.label_summary(rows, classes=5)

    assert summary["row_count"] == 3
    assert summary["label_counts"]["2"] == 1
    assert summary["non_noop_count"] == 2
    assert summary["max_score_gap"] == 0.5


def test_counterfactual_residual_failure_window_keeps_high_pressure_states():
    residual = _load_script("train_counterfactual_residual_terminal")

    states = []
    for step in range(30):
        raw = [0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]
        if step == 14:
            raw[3] = 0.20
        states.append({"step": step, "raw_before": raw, "force": 0.0})
    args = SimpleNamespace(
        use_failure_window=True,
        failure_window_start=0,
        failure_window_end=20,
        failure_window_stride=5,
        failure_window_target_offset=15,
        max_window_states=5,
        failure_offsets=[1],
        max_failure_states=3,
        n_poles=4,
    )

    selected = residual.select_failure_states(args, {"seed": 123, "states": states})

    assert len(selected) == 3
    assert any(item["step"] == 14 for item in selected)
    assert all(item["seed"] == 123 for item in selected)
    assert all("failure_offset" in item for item in selected)
    assert all("recovery_pressure" in item for item in selected)


def test_counterfactual_residual_train_model_oversamples_non_noop_labels():
    residual = _load_script("train_counterfactual_residual_terminal")
    rows = [
        {"feature": [0.0, 0.0], "label": 2},
        {"feature": [1.0, 0.0], "label": 2},
        {"feature": [0.0, 1.0], "label": 1},
    ]
    args = SimpleNamespace(
        residual_action_bins=5,
        hidden_size=4,
        max_class_weight=8.0,
        noop_class_weight=1.0,
        learning_rate=1e-3,
        train_seed=7,
        epochs=1,
        batch_size=2,
        non_noop_oversample_factor=4,
    )

    _model, meta = residual.train_model(rows, args)

    assert meta["original_row_count"] == 3
    assert meta["expanded_row_count"] == 6
    assert meta["non_noop_oversample_factor"] == 4
    assert meta["label_counts"]["1"] == 4





def test_counterfactual_residual_train_model_can_emit_apply_gate(tmp_path):
    import torch

    residual = _load_script("train_counterfactual_residual_terminal")
    rows = [
        {"feature": [0.0, 0.0], "label": 2, "apply_label": 0},
        {"feature": [1.0, 0.0], "label": 2, "apply_label": 0},
        {"feature": [0.0, 1.0], "label": 1, "apply_label": 1},
    ]
    args = SimpleNamespace(
        residual_action_bins=5,
        hidden_size=4,
        max_class_weight=8.0,
        noop_class_weight=1.0,
        learning_rate=1e-3,
        train_seed=7,
        epochs=1,
        batch_size=2,
        non_noop_oversample_factor=1,
        train_apply_gate=True,
        apply_epochs=1,
        apply_positive_weight=1.0,
        max_apply_positive_weight=4.0,
        residual_apply_threshold=0.5,
    )

    model, meta = residual.train_model(rows, args)
    path = tmp_path / "gated.pt"
    residual.save_model(model, meta, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert meta["apply_gate_enabled"] is True
    assert meta["apply_label_counts"] == {"0": 2, "1": 1}
    assert meta["format"] == "counterfactual_gated_residual_terminal_v2"
    assert "apply_state_dict" in payload


def test_counterfactual_residual_builds_two_phase_option_sequences():
    residual = _load_script("train_counterfactual_residual_terminal")
    args = SimpleNamespace(residual_action_bins=5, option_hold_steps=3, option_tail_steps=2)

    center = residual.candidate_residual_sequences(args, 2)
    shifted = residual.candidate_residual_sequences(args, 4)

    assert center == [([2], [1])]
    assert len(shifted) == 5
    assert shifted[0] == ([4, 0], [3, 2])
    assert shifted[-1] == ([4, 4], [3, 2])


def test_counterfactual_residual_label_state_respects_advantage_gates(monkeypatch):
    residual = _load_script("train_counterfactual_residual_terminal")

    def fake_score(_args, _raw_state, _step, _base_force, residual_class):
        options = {
            0: {"survived": 8, "margin": -0.2, "pressure_final": 0.9, "score": 7.8},
            1: {"survived": 10, "margin": 0.1, "pressure_final": 0.4, "score": 10.1},
            2: {"survived": 10, "margin": 0.0, "pressure_final": 0.5, "score": 10.0},
            3: {"survived": 11, "margin": 0.0, "pressure_final": 0.6, "score": 11.0},
            4: {"survived": 9, "margin": 0.0, "pressure_final": 0.8, "score": 9.0},
        }
        item = dict(options[int(residual_class)])
        item.update({"class": int(residual_class), "shift": int(residual_class) - 2, "first_action": 2, "forced_steps": 1})
        return item

    monkeypatch.setattr(residual, "counterfactual_score", fake_score)
    monkeypatch.setattr(residual, "residual_observation", lambda *_args, **_kwargs: residual.np.zeros(3, dtype=residual.np.float32))
    state = {"raw_before": [0.0] * 10, "step": 100, "force": 0.0, "seed": 1}
    args = SimpleNamespace(
        residual_action_bins=5,
        score_tolerance=1e-6,
        min_score_gap=0.1,
        min_survival_gain=2,
        min_margin_gain=0.0,
        min_pressure_gain=-999.0,
    )

    gated = residual.label_state(args, state)
    args.min_survival_gain = 1
    allowed = residual.label_state(args, state)

    assert gated["label"] == 2
    assert gated["best_survival_gain"] == 1
    assert allowed["label"] == 3
    assert allowed["chosen_survival_gain"] == 1


def test_counterfactual_residual_apply_gate_can_be_stricter_than_action_label(monkeypatch):
    residual = _load_script("train_counterfactual_residual_terminal")

    def fake_score(_args, _raw_state, _step, _base_force, residual_class):
        options = {
            0: {"survived": 8, "margin": -0.2, "pressure_final": 0.9, "score": 7.8},
            1: {"survived": 10, "margin": 0.0, "pressure_final": 0.5, "score": 10.0},
            2: {"survived": 10, "margin": 0.0, "pressure_final": 0.5, "score": 10.0},
            3: {"survived": 11, "margin": 0.0, "pressure_final": 0.45, "score": 11.0},
            4: {"survived": 9, "margin": 0.0, "pressure_final": 0.8, "score": 9.0},
        }
        item = dict(options[int(residual_class)])
        item.update({"class": int(residual_class), "shift": int(residual_class) - 2, "first_action": 2, "forced_steps": 1})
        return item

    monkeypatch.setattr(residual, "counterfactual_score", fake_score)
    monkeypatch.setattr(residual, "residual_observation", lambda *_args, **_kwargs: residual.np.zeros(3, dtype=residual.np.float32))
    args = SimpleNamespace(
        residual_action_bins=5,
        score_tolerance=1e-6,
        min_score_gap=0.1,
        min_survival_gain=1,
        min_margin_gain=0.0,
        min_pressure_gain=-999.0,
        apply_min_score_gap=0.1,
        apply_min_survival_gain=2,
        apply_min_margin_gain=0.0,
        apply_min_pressure_gain=-999.0,
    )

    row = residual.label_state(args, {"raw_before": [0.0] * 10, "step": 100, "force": 0.0, "seed": 1})

    assert row["label"] == 3
    assert row["chosen_survival_gain"] == 1
    assert row["apply_label"] == 0
    assert row["chosen_score_gap"] == pytest.approx(1.0)


def test_counterfactual_residual_label_state_can_use_pressure_advantage(monkeypatch):
    residual = _load_script("train_counterfactual_residual_terminal")

    def fake_score(_args, _raw_state, _step, _base_force, residual_class):
        options = {
            0: {"survived": 10, "margin": 0.0, "pressure_final": 0.8, "score": 9.8},
            1: {"survived": 10, "margin": 0.0, "pressure_final": 0.4, "score": 10.2},
            2: {"survived": 10, "margin": 0.0, "pressure_final": 0.6, "score": 10.0},
            3: {"survived": 10, "margin": 0.0, "pressure_final": 0.55, "score": 10.05},
            4: {"survived": 10, "margin": 0.0, "pressure_final": 0.7, "score": 9.9},
        }
        item = dict(options[int(residual_class)])
        item.update({"class": int(residual_class), "shift": int(residual_class) - 2, "first_action": 2, "forced_steps": 1})
        return item

    monkeypatch.setattr(residual, "counterfactual_score", fake_score)
    monkeypatch.setattr(residual, "residual_observation", lambda *_args, **_kwargs: residual.np.zeros(3, dtype=residual.np.float32))
    args = SimpleNamespace(
        residual_action_bins=5,
        score_tolerance=1e-6,
        min_score_gap=0.1,
        min_survival_gain=0,
        min_margin_gain=0.0,
        min_pressure_gain=0.1,
    )

    row = residual.label_state(args, {"raw_before": [0.0] * 10, "step": 100, "force": 0.0, "seed": 1})

    assert row["label"] == 1
    assert row["chosen_survival_gain"] == 0
    assert row["chosen_pressure_gain"] == pytest.approx(0.2)
    assert row["best_pressure_gain"] == pytest.approx(0.2)


def test_recovery_window_residual_extracts_failed_episode_windows():
    recovery = _load_script("train_recovery_window_residual_policy")
    states = []
    for step in range(12):
        raw = [0.0, 0.0, 0.01, 0.02]
        if step == 9:
            raw[2] = 0.20
        states.append({"step": step, "raw_before": raw, "force": 0.0})
    args = SimpleNamespace(
        use_failure_window=True,
        failure_window_start=0,
        failure_window_end=8,
        failure_window_stride=2,
        failure_window_target_offset=3,
        max_window_states=5,
        failure_offsets=[0, 1],
        max_failure_states=3,
        n_poles=1,
    )

    windows = recovery.window_rows_from_episode(
        args, {"seed": 44, "steps": 12, "return": 11.0, "success": False, "states": states}
    )

    assert len(windows) == 3
    assert all(item["seed"] == 44 for item in windows)
    assert all("raw_state" in item for item in windows)
    assert any(item["step"] == 9 for item in windows)
    assert max(item["recovery_pressure"] for item in windows) > 0.4


def test_recovery_window_residual_env_resets_to_window_and_steps(monkeypatch):
    recovery = _load_script("train_recovery_window_residual_policy")

    class FakeController:
        def __init__(self, _config):
            self.starts = 0

        def start_episode(self):
            self.starts += 1

        def act(self, _obs, _raw):
            return 2, {"force": 0.0}

    monkeypatch.setattr(recovery, "ReConCartPoleController", FakeController)
    args = SimpleNamespace(
        n_poles=1,
        horizon=50,
        window_horizon=3,
        dt=0.02,
        dynamics_mode="parallel",
        discrete_action_bins=5,
        force_mag=10.0,
        initial_angle_range=0.0,
        force_noise=0.0,
        link_coupling=0.35,
        base_model_path="base.zip",
        base_normalizer_path="",
        base_observation_mode="normalized_raw",
        selection_mode="hard_select",
        policy_terminal_blend=1.0,
        policy_terminal_scope="stabilize_chain",
        residual_feature_mode="basic",
        residual_action_bins=5,
        residual_gate_threshold=0.0,
        residual_hold_steps=1,
        cycle_windows=True,
        shift_penalty=0.0,
        low_risk_change_penalty=0.5,
        pressure_drop_weight=1.0,
        pressure_after_weight=0.0,
        failure_penalty=0.0,
        window_success_bonus=0.0,
    )
    windows = [{"seed": 7, "step": 10, "raw_state": [0.0, 0.0, 0.02, 0.0], "base_force": 0.0, "recovery_pressure": 0.1}]

    env = recovery.RecoveryWindowResidualEnv(args, windows)
    obs, info = env.reset(seed=1)
    next_obs, reward, terminated, truncated, step_info = env.step(2)

    assert obs.shape == env.observation_space.shape
    assert next_obs.shape == env.observation_space.shape
    assert info["window_seed"] == 7
    assert info["window_step"] == 10
    assert step_info["base_action"] == 2
    assert step_info["applied_shift"] == 0
    assert step_info["low_risk_change_penalty"] == 0.0
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated


def test_residual_env_penalizes_changes_on_base_solved_episode(monkeypatch):
    residual = _load_script("train_residual_policy_terminal")

    class FakeController:
        def __init__(self, _config):
            self.starts = 0

        def start_episode(self):
            self.starts += 1

        def act(self, env_obs, raw):
            return 2, {"force": 0.0}

    monkeypatch.setattr(residual, "ReConCartPoleController", FakeController)
    args = SimpleNamespace(
        n_poles=1,
        horizon=1,
        dt=0.02,
        dynamics_mode="parallel",
        env_action_mode="discrete",
        discrete_action_bins=5,
        force_mag=10.0,
        initial_angle_range=0.0,
        force_noise=0.0,
        link_coupling=0.35,
        base_model_path="base.zip",
        device="cpu",
        residual_base_controller="recon_policy_terminal",
        selection_mode="hard_select",
        policy_terminal_blend=1.0,
        policy_terminal_scope="stabilize_chain",
        base_observation_mode="normalized_raw",
        base_normalizer_path="",
        residual_feature_mode="basic",
        residual_mode="bin_delta",
        residual_action_bins=5,
        residual_gate_threshold=0.0,
        max_residual_force=4.0,
        hard_seed_probability=0.0,
        preserve_base_success_penalty=0.5,
        low_risk_change_penalty=0.0,
        late_survival_bonus=0.0,
        late_survival_start_fraction=0.8,
        recovery_progress_weight=0.0,
        failure_penalty=0.0,
        success_bonus=0.0,
    )

    env = residual.ResidualCorrectionEnv(args)
    _obs, info = env.reset(seed=7)
    _next_obs, reward, _terminated, truncated, step_info = env.step(4)

    assert info["base_episode_success"] is True
    assert truncated is True
    assert step_info["base_episode_success"] is True
    assert step_info["preserve_base_success_penalty"] == pytest.approx(0.5)
    assert reward == pytest.approx(0.5)


def test_residual_recovery_pressure_increases_with_state_risk():
    import numpy as np

    residual = _load_script("train_residual_policy_terminal")
    calm = np.asarray([0.0, 0.0, 0.01, -0.01, 0.1, -0.1], dtype=np.float32)
    risky = np.asarray([1.8, 0.0, 0.16, -0.12, 3.0, -2.5], dtype=np.float32)

    assert residual.recovery_pressure(risky, 2) > residual.recovery_pressure(calm, 2)
    assert residual.recovery_pressure(calm, 2) >= 0.0


def test_residual_proposal_diagnostic_features_are_stable_size():
    from recon_cartpole.control.residual_features import residual_aux_feature_size, residual_aux_features
    import numpy as np

    raw = np.asarray([0.0, 0.0, 0.05, -0.04, 0.1, -0.1], dtype=np.float32)
    basic = residual_aux_features(raw, n_poles=2, force_mag=10.0, base_force=5.0, mode="basic")
    diag = residual_aux_features(raw, n_poles=2, force_mag=10.0, base_force=5.0, mode="proposal_diagnostics")
    subchain = residual_aux_features(raw, n_poles=2, force_mag=10.0, base_force=5.0, mode="subchain_diagnostics")

    assert basic.shape == (residual_aux_feature_size("basic"),)
    assert diag.shape == (residual_aux_feature_size("proposal_diagnostics"),)
    assert subchain.shape == (residual_aux_feature_size("subchain_diagnostics"),)
    assert diag.shape[0] > basic.shape[0]
    assert subchain.shape[0] == diag.shape[0] + 12
    assert np.isfinite(diag).all()
    assert np.isfinite(subchain).all()


def test_teacher_action_anchor_penalizes_early_mismatch():
    import gymnasium as gym
    import numpy as np

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(5)

        def __init__(self):
            self.config = SimpleNamespace(
                n_poles=1,
                horizon=10,
                x_threshold=2.4,
                theta_threshold_radians=0.2,
                action_mode="discrete",
                discrete_action_bins=5,
                force_mag=10.0,
            )
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            self.steps = 0
            return np.zeros(4, dtype=np.float32), {"raw_state": [0.0, 0.0, 0.0, 0.0]}

        def step(self, action):
            self.steps += 1
            return np.zeros(4, dtype=np.float32), 1.0, False, self.steps >= 10, {"raw_state": [0.0, 0.0, 0.0, 0.0]}

    class Teacher:
        def predict(self, obs, deterministic=True):
            return 2, None

    env = trainer.TeacherActionAnchorWrapper(
        TinyEnv(),
        model_path="unused.zip",
        penalty=0.25,
        observation_mode="env",
        until_fraction=0.5,
        risk_threshold=1.0,
        teacher_model=Teacher(),
    )
    env.reset(seed=1)
    _obs, reward, _terminated, _truncated, info = env.step(4)
    assert reward == 0.75
    assert info["teacher_action_penalty"] == 0.25
    _obs, reward, _terminated, _truncated, info = env.step(2)
    assert reward == 1.0
    assert "teacher_action_penalty" not in info


def test_hard_seed_wrapper_offsets_sampling_by_worker_seed():
    import gymnasium as gym
    import numpy as np

    class SeedRecordingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def __init__(self):
            self.reset_seeds = []

        def reset(self, *, seed=None, options=None):
            self.reset_seeds.append(seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0.0, False, True, {}

    env_a = SeedRecordingEnv()
    env_b = SeedRecordingEnv()
    wrapper_a = trainer.HardSeedResetWrapper(env_a, [100, 200, 300], probability=1.0)
    wrapper_b = trainer.HardSeedResetWrapper(env_b, [100, 200, 300], probability=1.0)

    wrapper_a.reset(seed=1)
    wrapper_b.reset(seed=2)

    assert env_a.reset_seeds == [200]
    assert env_b.reset_seeds == [300]


def test_failure_penalty_subtracts_only_on_termination():
    import gymnasium as gym
    import numpy as np

    class TerminatingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.0, True, False, {}

    env = trainer.FailurePenaltyWrapper(TerminatingEnv(), 3.5)
    _obs, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -2.5
    assert info["failure_penalty"] == 3.5


def test_validation_seed_starts_expand_to_disjoint_blocks():
    args = SimpleNamespace(
        validation_seed_start=10,
        validation_seed_starts=[100, 200],
        validation_episodes=3,
    )

    assert iterative_trainer.validation_seeds(args) == [100, 101, 102, 200, 201, 202]


def test_validation_seed_start_falls_back_to_single_block():
    args = SimpleNamespace(
        validation_seed_start=10,
        validation_seed_starts=None,
        validation_episodes=3,
    )

    assert iterative_trainer.validation_seeds(args) == [10, 11, 12]


def test_score_weights_are_configurable():
    summary = {"mean_survival": 100.0, "p10_survival": 40.0, "success_rate": 0.5}

    assert iterative_trainer.score(summary) == 135.0
    assert iterative_trainer.score(
        summary, mean_weight=0.0, p10_weight=0.0, success_weight=200.0
    ) == 100.0


def test_tail_curriculum_scores_lower_tail_and_success():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    summary = tail.tail_metrics([500, 500, 400, 300, 250], horizon=500, cvar_fraction=0.4)

    assert summary["success_rate"] == 0.4
    assert summary["cvar_survival"] == 275.0
    assert tail.tail_score(
        summary,
        mean_weight=0.0,
        p10_weight=0.0,
        cvar_weight=1.0,
        success_weight=100.0,
    ) == 315.0


def test_tail_curriculum_selects_near_miss_tail_seeds():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    summary = {
        "per_seed": [
            {"seed": 1, "steps": 499, "success": False},
            {"seed": 2, "steps": 200, "success": False},
            {"seed": 3, "steps": 500, "success": True},
            {"seed": 4, "steps": 450, "success": False},
        ]
    }

    assert tail.tail_seed_pool(summary, limit=2, min_steps=300) == [1, 4]


def test_tail_curriculum_promotion_rejects_tail_regression():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    args = SimpleNamespace(max_success_regression=0.01, max_p10_regression=5.0, max_cvar_regression=8.0, promotion_mode="score")
    best = {
        "score": 1000.0,
        "validation": {"success_rate": 0.72, "p10_survival": 440.0},
    }
    row = {
        "score": 1200.0,
        "validation": {"success_rate": 0.70, "p10_survival": 450.0},
    }

    assert tail.should_promote(row, best, args) is False


def test_iterative_training_env_enables_hard_seed_wrapper(monkeypatch):
    calls = {}

    def fake_make_env(args, *, reward_mode, use_hard_seeds):
        calls["reward_mode"] = reward_mode
        calls["use_hard_seeds"] = use_hard_seeds
        return "env"

    monkeypatch.setattr(iterative_trainer, "make_env", fake_make_env)
    args = SimpleNamespace(reward_mode="upright_shaping")

    assert iterative_trainer.make_training_env(args) == "env"
    assert calls == {"reward_mode": "upright_shaping", "use_hard_seeds": True}



def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recurrent_terminal_scripts_import_and_hash_configs():
    dataset_builder = _load_script("build_policy_dataset")
    supervised = _load_script("train_mingru_supervised")
    ladder = _load_script("train_recurrent_terminal_ladder")
    autonomous = _load_script("run_n4_autonomous_recurrent")
    pole1 = _load_script("run_n4_pole1_robustness")
    final_gap = _load_script("run_n4_final_success_gap")
    pole1_finetune = _load_script("run_n4_pole1_policy_finetune")
    tail_curriculum = _load_script("train_policy_terminal_tail_curriculum")
    recurrent_tail = _load_script("train_recurrent_policy_terminal_tail_curriculum")
    ppo_sweep = _load_script("run_ppo_sweep")
    residual = _load_script("train_residual_policy_terminal")
    counterfactual_residual = _load_script("train_counterfactual_residual_terminal")
    recurrent_curriculum = _load_script("train_recurrent_policy_terminal_curriculum")
    action_compare = _load_script("compare_policy_actions")
    residual_grid = _load_script("evaluate_recon_residual_grid")
    counterfactual_gate = _load_script("train_counterfactual_action_gate")
    mingru_curriculum = _load_script("train_mingru_curriculum")
    mingru_hard_seeds = _load_script("collect_mingru_hard_seeds")
    mingru_onpolicy = _load_script("train_mingru_onpolicy")
    mingru_ppo = _load_script("train_mingru_ppo")
    subchain_pair = _load_script("train_subchain_pair_terminal")

    assert callable(dataset_builder.collect)
    assert callable(supervised.train)
    assert ladder.config_hash({"a": 1}) == ladder.config_hash({"a": 1})
    assert ladder.config_hash({"a": 1}) != ladder.config_hash({"a": 2})
    assert autonomous.mechanism_flags("recon_mingru_terminal_plus_recon_learning")["edge_plasticity"] is True
    assert callable(pole1.compare_outcomes)
    assert callable(final_gap.candidate_configs)
    assert callable(pole1_finetune.collect_failure_dataset)
    assert callable(tail_curriculum.run_tail_curriculum)
    assert callable(recurrent_tail.run_recurrent_tail_curriculum)
    assert callable(ppo_sweep.run_sweep)
    assert callable(residual.train_residual)
    assert callable(counterfactual_residual.run)
    assert callable(recurrent_curriculum.run_curriculum)
    assert callable(action_compare.run_comparison)
    assert callable(residual_grid.run_sweep)
    assert callable(counterfactual_gate.run)
    assert callable(mingru_curriculum.run)
    assert callable(mingru_hard_seeds.run_collect)
    assert callable(mingru_onpolicy.run)
    assert callable(mingru_ppo.run)
    assert callable(subchain_pair.run)



def test_subchain_pair_counterfactual_expands_tail_options(monkeypatch):
    subchain_pair = _load_script("train_subchain_pair_terminal")
    args = SimpleNamespace(
        option_hold_steps=3,
        option_tail_steps=5,
        force_mag=10.0,
        discrete_action_bins=5,
    )

    sequences = subchain_pair.candidate_force_sequences(args, first_force=0.0)

    assert len(sequences) == 5
    assert sequences[0]["forces"] == [0.0, -10.0]
    assert sequences[-1]["forces"] == [0.0, 10.0]
    assert all(item["steps"] == [3, 5] for item in sequences)
    assert subchain_pair.baseline_force_sequence(args, base_force=5.0) == {
        "forces": [5.0, 5.0],
        "steps": [3, 5],
    }

    def fake_sequence_score(_args, _raw_state, _step, _base_force, sequence):
        tail_force = float(sequence["forces"][-1])
        return {
            "force": float(sequence["forces"][0]),
            "tail_force": tail_force,
            "forces": list(sequence["forces"]),
            "score": tail_force,
        }

    monkeypatch.setattr(subchain_pair, "counterfactual_sequence_score", fake_sequence_score)

    best = subchain_pair.counterfactual_score(args, raw_state=[0.0], step=0, base_force=0.0, candidate_force=0.0)

    assert best["force"] == 0.0
    assert best["tail_force"] == 10.0
    assert best["score"] == 10.0


def test_subchain_option_trace_exports_primary_policy_rows():
    subchain_pair = _load_script("train_subchain_pair_terminal")
    args = SimpleNamespace(
        n_poles=4,
        horizon=500,
        force_mag=10.0,
        discrete_action_bins=5,
        option_policy_observation_mode="normalized_raw4_prev_force",
    )
    rows = subchain_pair.empty_policy_rows()
    raw = [0.1, -0.2, 0.01, -0.02, 0.03, -0.04, 0.5, -0.4, 0.3, -0.2]

    subchain_pair.append_policy_option_row(
        args,
        rows,
        raw,
        target_force=10.0,
        prev_force=-5.0,
        confidence=0.5,
        weight=2.0,
        seed=123,
        episode=7,
        step=42,
        source="counterfactual_recovery_option",
    )
    data = subchain_pair.finalize_policy_dataset(rows)

    assert data["observations"].shape == (1, 11)
    assert data["prev_forces"].tolist() == [-5.0]
    assert data["teacher_forces"].tolist() == [10.0]
    assert data["teacher_actions"].tolist() == [4]
    assert data["failure_within_k"].tolist() == [0.0]
    assert data["sample_weights"].tolist() == [1.0]
    assert data["episodes"].tolist() == [7]
    assert data["step_indices"].tolist() == [42]



def test_mingru_onpolicy_discounted_returns():
    onpolicy = _load_script("train_mingru_onpolicy")

    returns = onpolicy.discounted_returns([1.0, 2.0, 3.0], 0.5)

    assert returns.tolist() == pytest.approx([2.75, 3.5, 3.0])



def test_mingru_ppo_clipped_policy_loss_prefers_clipped_surrogate():
    torch = pytest.importorskip("torch")
    mingru_ppo = _load_script("train_mingru_ppo")
    ratio = torch.tensor([1.5, 0.5], dtype=torch.float32)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    loss = mingru_ppo.ppo_clipped_policy_loss(ratio, advantages, 0.2)

    assert float(loss) == pytest.approx(-0.2)


def test_mingru_ppo_normalize_can_be_disabled():
    mingru_ppo = _load_script("train_mingru_ppo")
    raw = mingru_ppo.normalize([1.0, 2.0, 3.0], enabled=False)
    normalized = mingru_ppo.normalize([1.0, 2.0, 3.0], enabled=True)

    assert raw.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert float(normalized.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(normalized.std()) == pytest.approx(1.0, rel=1e-5)

def test_action_comparison_summarizes_seed_deltas():
    action_compare = _load_script("compare_policy_actions")
    rows = [
        {
            "seed": 1,
            "a_steps": 499,
            "b_steps": 500,
            "delta_steps": 1,
            "a_success": False,
            "b_success": True,
            "a_failure": "pole_1_angle",
            "b_failure": "success",
            "same_outcome": False,
            "action_diff_count": 1,
            "first_action_diff": {"step": 498},
        },
        {
            "seed": 2,
            "a_steps": 500,
            "b_steps": 498,
            "delta_steps": -2,
            "a_success": True,
            "b_success": False,
            "a_failure": "success",
            "b_failure": "rail_right",
            "same_outcome": False,
            "action_diff_count": 2,
            "first_action_diff": {"step": 100},
        },
    ]

    summary = action_compare.summarize_comparison(rows, horizon=500)

    assert summary["success_gain_count"] == 1
    assert summary["success_loss_count"] == 1
    assert summary["changed_seed_count"] == 2
    assert summary["first_diff_step_median"] == 299.0


def test_late_survival_bonus_starts_at_threshold():
    import gymnasium as gym
    import numpy as np
    from types import SimpleNamespace

    class CountingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def __init__(self):
            self.config = SimpleNamespace(horizon=4)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.0, False, False, {}

    env = trainer.LateSurvivalBonusWrapper(CountingEnv(), bonus=0.5, start_fraction=0.75)
    env.reset(seed=1)
    _obs, reward_1, *_rest, info_1 = env.step(0)
    _obs, reward_2, *_rest, info_2 = env.step(0)
    _obs, reward_3, *_rest, info_3 = env.step(0)

    assert reward_1 == 1.0
    assert reward_2 == 1.0
    assert reward_3 == 1.5
    assert "late_survival_bonus" not in info_1
    assert "late_survival_bonus" not in info_2
    assert info_3["late_survival_bonus"] == 0.5


def test_policy_observation_wrapper_pads_n3_to_raw4_prev_force():
    args = _args(0.0)
    args.n_poles = 3
    args.horizon = 2
    args.policy_observation_mode = "normalized_raw4_prev_force"
    env = make_env(args, reward_mode="survival", use_success_bonus=False)
    obs, _info = env.reset(seed=4)

    assert obs.shape == (11,)
    assert obs[5] == 0.0
    assert obs[9] == 0.0
    assert obs[-1] == 0.0


def test_observation_normalizer_wrapper_applies_stats(tmp_path):
    import gymnasium as gym
    import json
    import numpy as np

    class ToyEnv(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return np.asarray([3.0, 6.0], dtype=np.float32), {}

        def step(self, action):
            return np.asarray([3.0, 6.0], dtype=np.float32), 0.0, False, True, {}

    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"mean": [1.0, 0.0], "var": [4.0, 9.0], "epsilon": 0.0, "clip_obs": 10.0}), encoding="utf-8")
    env = trainer.ObservationNormalizerWrapper(ToyEnv(), str(path))
    obs, _info = env.reset(seed=0)

    assert np.allclose(obs, [1.0, 2.0])


def test_tail_curriculum_lexicographic_promotion_prefers_success_over_score():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    args = SimpleNamespace(
        max_success_regression=0.0,
        max_p10_regression=2.0,
        max_cvar_regression=2.0,
        promotion_mode="lexicographic_success",
    )
    best = {
        "score": 900.0,
        "validation": {
            "success_rate": 0.696,
            "p10_survival": 434.9,
            "cvar_survival": 414.8,
            "mean_survival": 484.9,
        },
    }
    row = {
        "score": 897.0,
        "validation": {
            "success_rate": 0.700,
            "p10_survival": 434.0,
            "cvar_survival": 414.8,
            "mean_survival": 484.9,
        },
    }

    assert tail.should_promote(row, best, args) is True


def test_tail_curriculum_final_seed_starts_expand_blocks():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    args = SimpleNamespace(final_seed_start=10, final_seed_starts=[100, 200], final_eval_episodes=2)

    assert tail.final_eval_seeds(args) == [100, 101, 200, 201]


def test_counterfactual_gate_summarizes_positive_labels():
    counterfactual_gate = _load_script("train_counterfactual_action_gate")
    rows = [
        {"label": 0, "chosen_survived": 10, "best_survived": 10, "chosen_score": 9.9, "best_score": 9.9},
        {"label": 3, "chosen_survived": 8, "best_survived": 11, "chosen_score": 7.0, "best_score": 10.5},
    ]

    summary = counterfactual_gate.dataset_label_summary(rows, classes=6)

    assert summary["row_count"] == 2
    assert summary["positive_count"] == 1
    assert summary["label_counts"]["0"] == 1
    assert summary["label_counts"]["3"] == 1
    assert summary["max_survival_gap"] == 3.0
    assert summary["max_score_gap"] == 3.5


def test_counterfactual_gate_noop_eval_resets_override_counts():
    counterfactual_gate = _load_script("train_counterfactual_action_gate")
    base = {"mean_survival": 500.0, "override_count": 9, "checked_steps": 100, "override_rate": 0.09}

    gate = counterfactual_gate.gate_eval_from_base(base)

    assert gate["mean_survival"] == 500.0
    assert gate["override_count"] == 0
    assert gate["checked_steps"] == 0
    assert gate["override_rate"] == 0.0
    assert base["override_count"] == 9


def test_ppo_sweep_candidate_grid_can_select_exact_indices():
    ppo_sweep = _load_script("run_ppo_sweep")
    args = SimpleNamespace(
        learning_rates="1e-6,2e-6",
        clip_ranges="0.01",
        n_steps_values="512",
        n_epochs_values="2",
        gae_lambdas="0.9",
        ent_coefs="0.0",
        net_arch_values="64,64;128,128",
        vec_normalize_values="false,true",
        late_survival_bonus_values="0.0,0.02",
        candidate_indices="0,15,13",
        shuffle_stride=1,
        candidate_offset=0,
        max_candidates=0,
    )

    rows = ppo_sweep.candidate_grid(args)

    assert [row["grid_index"] for row in rows] == [0, 15, 13]
    assert rows[0]["learning_rate"] == pytest.approx(1e-6)
    assert rows[1]["learning_rate"] == pytest.approx(2e-6)
    assert rows[1]["net_arch"] == "128,128"
    assert rows[1]["vec_normalize"] is True
    assert rows[1]["late_survival_bonus"] == pytest.approx(0.02)


def test_ppo_sweep_candidate_grid_keeps_grid_index_under_stride():
    ppo_sweep = _load_script("run_ppo_sweep")
    args = SimpleNamespace(
        learning_rates="1e-6,2e-6,3e-6",
        clip_ranges="0.01",
        n_steps_values="512",
        n_epochs_values="2",
        gae_lambdas="0.9",
        ent_coefs="0.0",
        net_arch_values="64,64",
        vec_normalize_values="false",
        late_survival_bonus_values="0.0",
        candidate_indices="",
        shuffle_stride=2,
        candidate_offset=0,
        max_candidates=2,
    )

    rows = ppo_sweep.candidate_grid(args)

    assert [row["grid_index"] for row in rows] == [0, 2]


def test_recurrent_tail_final_seed_starts_expand_blocks():
    recurrent_tail = _load_script("train_recurrent_policy_terminal_tail_curriculum")
    args = SimpleNamespace(final_seed_start=10, final_seed_starts=[100, 200], final_eval_episodes=2)

    assert recurrent_tail.recurrent_final_seeds(args) == [100, 101, 200, 201]


def test_recurrent_tail_final_seed_start_fallback():
    recurrent_tail = _load_script("train_recurrent_policy_terminal_tail_curriculum")
    args = SimpleNamespace(final_seed_start=10, final_seed_starts=None, final_eval_episodes=2)

    assert recurrent_tail.recurrent_final_seeds(args) == [10, 11]


def test_mingru_curriculum_default_stages_progress_n3_to_n4():
    curriculum = _load_script("train_mingru_curriculum")
    args = SimpleNamespace(
        n3_episodes=1,
        low_angle_episodes=2,
        current_episodes=3,
        tail_episodes=4,
        n3_seed_start=10,
        low_angle_seed_start=20,
        current_seed_start=30,
        tail_seed_start=40,
        tail_seed_list="seeds.txt",
        behavior_checkpoint_path="behavior.pt",
        initial_angle_range=0.05,
        force_noise=0.02,
        n3_sample_weight=0.1,
        low_angle_sample_weight=0.2,
        current_sample_weight=0.5,
        tail_sample_weight=2.0,
    )

    stages = curriculum.default_stages(args)

    assert [stage["name"] for stage in stages] == ["n3_stable", "n4_low_angle_no_noise", "n4_current", "n4_hard_tail"]
    assert [stage["n_poles"] for stage in stages] == [3, 4, 4, 4]
    assert [stage["sample_weight"] for stage in stages] == [0.1, 0.2, 0.5, 2.0]
    assert stages[-1]["rollout_policy"] == "mingru_terminal"


def test_mingru_curriculum_aggregate_offsets_episode_ids():
    curriculum = _load_script("train_mingru_curriculum")

    def payload(values):
        return {key: curriculum.np.asarray(value) for key, value in values.items()}

    first = {
        "data": payload({
            "observations": [[1.0], [2.0]],
            "prev_forces": [0.0, 1.0],
            "teacher_forces": [0.0, 1.0],
            "teacher_actions": [2, 3],
            "returns_to_go": [2.0, 1.0],
            "failure_within_k": [0.0, 1.0],
            "seeds": [10, 10],
            "sources": ["teacher", "teacher"],
            "rollout_sources": ["teacher", "teacher"],
            "rollout_forces": [0.0, 1.0],
            "rollout_actions": [2, 3],
            "episodes": [0, 0],
            "step_indices": [0, 1],
            "sample_weights": [0.25, 0.25],
        })
    }
    second = {
        "data": payload({
            "observations": [[3.0]],
            "prev_forces": [0.0],
            "teacher_forces": [0.0],
            "teacher_actions": [2],
            "returns_to_go": [1.0],
            "failure_within_k": [0.0],
            "seeds": [20],
            "sources": ["teacher"],
            "rollout_sources": ["teacher"],
            "rollout_forces": [0.0],
            "rollout_actions": [2],
            "episodes": [0],
            "step_indices": [0],
            "sample_weights": [2.0],
        })
    }

    merged = curriculum.aggregate_stage_data([first, second])

    assert merged["observations"].shape == (3, 1)
    assert merged["episodes"].tolist() == [0, 0, 1]
    assert merged["sample_weights"].tolist() == [0.25, 0.25, 2.0]


def test_merge_policy_datasets_offsets_episodes_and_weights(tmp_path):
    merger = _load_script("merge_policy_datasets")

    def save(path, values):
        payload = {key: merger.np.asarray(value) for key, value in values.items()}
        merger.np.savez_compressed(path, **payload)

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save(
        first,
        {
            "observations": [[1.0], [2.0]],
            "prev_forces": [0.0, 1.0],
            "teacher_forces": [0.0, 1.0],
            "teacher_actions": [2, 3],
            "returns_to_go": [2.0, 1.0],
            "failure_within_k": [0.0, 1.0],
            "seeds": [10, 10],
            "sources": ["teacher", "teacher"],
            "rollout_sources": ["teacher", "teacher"],
            "rollout_forces": [0.0, 1.0],
            "rollout_actions": [2, 3],
            "episodes": [0, 0],
            "step_indices": [0, 1],
            "sample_weights": [1.0, 1.0],
        },
    )
    save(
        second,
        {
            "observations": [[3.0]],
            "prev_forces": [0.0],
            "teacher_forces": [0.0],
            "teacher_actions": [2],
            "returns_to_go": [1.0],
            "failure_within_k": [0.0],
            "seeds": [20],
            "sources": ["counterfactual_recovery"],
            "rollout_sources": ["counterfactual_option"],
            "rollout_forces": [0.0],
            "rollout_actions": [2],
            "episodes": [0],
            "step_indices": [0],
            "sample_weights": [0.5],
        },
    )

    merged, metadata = merger.merge_datasets(
        [
            merger.DatasetSpec(first, weight=1.0, source_prefix="base"),
            merger.DatasetSpec(second, weight=4.0, source_prefix="option"),
        ]
    )

    assert merged["observations"].shape == (3, 1)
    assert merged["episodes"].tolist() == [0, 0, 1]
    assert merged["sample_weights"].tolist() == [1.0, 1.0, 2.0]
    assert merged["motif_scores"].tolist() == [0.0, 0.0, 0.0]
    assert merged["sources"].tolist() == ["base:teacher", "base:teacher", "option:counterfactual_recovery"]
    assert metadata["samples"] == 3

    filtered, filtered_meta = merger.merge_datasets(
        [
            merger.DatasetSpec(first, weight=1.0, source_prefix="base"),
            merger.DatasetSpec(second, weight=4.0, source_prefix="option"),
        ],
        include_sources={"teacher", "counterfactual_recovery"},
    )

    assert filtered["observations"].shape == (3, 1)
    assert filtered_meta["include_sources"] == ["counterfactual_recovery", "teacher"]


def test_subchain_observation_mode_adds_adjacent_pair_features():
    from recon_cartpole.control.policy_observation import (
        adjacent_subchain_features,
        policy_observation_from_state,
        policy_observation_size,
    )
    import numpy as np

    raw = np.asarray([0.0, 1.0, 0.01, -0.02, 0.03, -0.04, 0.5, -0.6, 0.7, -0.8], dtype=np.float32)

    obs = policy_observation_from_state(
        raw,
        raw,
        4,
        "normalized_raw4_subchains_prev_force",
        previous_force=5.0,
        force_mag=10.0,
    )

    assert obs.shape == (policy_observation_size(4, "normalized_raw4_subchains_prev_force"),)
    assert obs.shape == (23,)
    assert obs[-1] == 0.5
    pair = adjacent_subchain_features(obs[2:6], obs[6:10], 4)
    assert np.allclose(obs[10:22], pair)


def test_mingru_build_inputs_avoids_duplicate_prev_force_column():
    import numpy as np

    supervised = _load_script("train_mingru_supervised")
    data = {
        "observations": np.zeros((2, 11), dtype=np.float32),
        "prev_forces": np.asarray([0.0, 5.0], dtype=np.float32),
    }
    args = SimpleNamespace(
        include_prev_force=True,
        include_context=False,
        observation_mode="normalized_raw4_prev_force",
        force_mag=10.0,
    )

    inputs = supervised.build_inputs(data, args)

    assert inputs.shape == (2, 11)


def test_mingru_build_inputs_appends_scaled_motif_score():
    import numpy as np

    supervised = _load_script("train_mingru_supervised")
    data = {
        "observations": np.zeros((2, 3), dtype=np.float32),
        "prev_forces": np.asarray([0.0, 5.0], dtype=np.float32),
        "motif_scores": np.asarray([5.0, -10.0], dtype=np.float32),
    }
    args = SimpleNamespace(
        include_prev_force=False,
        include_context=False,
        include_motif_score=True,
        motif_score_scale=5.0,
        observation_mode="env",
        force_mag=10.0,
    )

    inputs = supervised.build_inputs(data, args)

    assert inputs.shape == (2, 4)
    assert inputs[:, -1].tolist() == [1.0, -2.0]


def test_mingru_terminal_observation_appends_motif_score(tmp_path):
    import json
    import numpy as np
    from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig

    model_path = tmp_path / "motif.json"
    model_path.write_text(
        json.dumps(
            {
                "positive_mean": [0.0] * 14,
                "negative_mean": [10.0] * 14,
                "scale": [1.0] * 14,
            }
        ),
        encoding="utf-8",
    )
    terminal = MinGRUTerminal(
        4,
        10.0,
        5,
        MinGRUTerminalConfig(
            enabled=True,
            observation_mode="normalized_raw4_prev_force",
            include_prev_force=True,
            include_context=False,
            include_motif_score=True,
            motif_model_path=str(model_path),
            motif_score_scale=10.0,
        ),
    )
    raw = np.zeros(10, dtype=np.float32)

    vector = terminal.observation_vector(raw, raw, {})

    assert vector.shape == (12,)
    assert terminal.input_size == 12
    assert vector[-1] == pytest.approx(10.0)


def test_mingru_supervised_filter_training_data_by_episode_survival():
    import numpy as np

    supervised = _load_script("train_mingru_supervised")
    data = {
        "observations": np.zeros((3, 2), dtype=np.float32),
        "teacher_actions": np.asarray([0, 1, 2], dtype=np.int64),
        "returns_to_go": np.asarray([500.0, 100.0, 20.0], dtype=np.float32),
        "step_indices": np.asarray([0, 300, 470], dtype=np.int64),
    }
    args = SimpleNamespace(min_sample_episode_survival=490.0, max_sample_episode_survival=0.0)

    filtered, report = supervised.filter_training_data(data, args)

    assert report["enabled"] is True
    assert report["input_samples"] == 3
    assert report["kept_samples"] == 2
    assert filtered["teacher_actions"].tolist() == [0, 2]


def test_mingru_supervised_sample_weights_emphasize_tail_states():
    import numpy as np

    supervised = _load_script("train_mingru_supervised")
    data = {
        "teacher_actions": np.asarray([0, 1, 2], dtype=np.int64),
        "failure_within_k": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "step_indices": np.asarray([0, 250, 500], dtype=np.int64),
        "returns_to_go": np.asarray([500.0, 100.0, 0.0], dtype=np.float32),
    }
    args = SimpleNamespace(
        horizon=500,
        failure_sample_weight=2.0,
        late_sample_weight=1.0,
        low_return_sample_weight=1.0,
    )

    weights = supervised.sample_weights(data, args)

    assert weights.shape == (3,)
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]
    assert float(np.mean(weights)) == pytest.approx(1.0)


def test_mingru_supervised_dataset_sample_weights_multiply_tail_weights():
    import numpy as np

    supervised = _load_script("train_mingru_supervised")
    data = {
        "teacher_actions": np.asarray([0, 1, 2], dtype=np.int64),
        "failure_within_k": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "step_indices": np.asarray([0, 250, 500], dtype=np.int64),
        "returns_to_go": np.asarray([500.0, 100.0, 0.0], dtype=np.float32),
        "sample_weights": np.asarray([1.0, 0.5, 4.0], dtype=np.float32),
    }
    args = SimpleNamespace(
        horizon=500,
        failure_sample_weight=2.0,
        late_sample_weight=1.0,
        low_return_sample_weight=1.0,
    )

    weights = supervised.sample_weights(data, args)

    assert weights.shape == (3,)
    assert weights[2] > weights[1]
    assert weights[2] > weights[0]
    assert float(np.mean(weights)) == pytest.approx(1.0)


def test_mingru_partial_input_resume_copies_shared_columns():
    import torch

    supervised = _load_script("train_mingru_supervised")
    source = {
        "input_proj.weight": torch.ones((3, 4)),
        "input_proj.bias": torch.ones(3),
        "other.weight": torch.ones((2, 2)),
    }
    target = {
        "input_proj.weight": torch.zeros((3, 6)),
        "input_proj.bias": torch.zeros(3),
        "other.weight": torch.zeros((2, 3)),
    }

    adapted, report = supervised.adapt_state_dict_for_input_expansion(target, source)

    assert torch.all(adapted["input_proj.weight"][:, :4] == 1.0)
    assert torch.all(adapted["input_proj.weight"][:, 4:] == 0.0)
    assert torch.all(adapted["input_proj.bias"] == 1.0)
    assert report["partial"][0]["key"] == "input_proj.weight"
    assert report["partial"][0]["copied_columns"] == 4
    assert report["partial"][1]["key"] == "other.weight"


def test_mingru_supervised_resume_checkpoint_records_source(tmp_path):
    import numpy as np
    from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig

    supervised = _load_script("train_mingru_supervised")
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        observations=np.zeros((4, 5), dtype=np.float32),
        prev_forces=np.zeros(4, dtype=np.float32),
        teacher_actions=np.asarray([0, 1, 0, 1], dtype=np.int64),
        returns_to_go=np.ones(4, dtype=np.float32),
        failure_within_k=np.zeros(4, dtype=np.float32),
        episodes=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    resume = tmp_path / "resume.pt"
    MinGRUTerminal(
        1,
        10.0,
        2,
        MinGRUTerminalConfig(
            enabled=True,
            hidden_size=4,
            sequence_length=2,
            observation_mode="env",
            include_prev_force=False,
            include_context=False,
        ),
    ).save_checkpoint(str(resume))

    report = supervised.train(
        SimpleNamespace(
            dataset=str(dataset),
            resume_checkpoint=str(resume),
            out=str(tmp_path / "out"),
            n_poles=1,
            horizon=2,
            force_mag=10.0,
            discrete_action_bins=2,
            observation_mode="env",
            hidden_size=4,
            sequence_length=2,
            include_prev_force=False,
            include_context=False,
            scope="stabilize_chain",
            blend=1.0,
            confidence_floor=0.05,
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            validation_fraction=0.25,
            value_weight=0.05,
            failure_weight=0.10,
            confidence_weight=0.05,
            max_grad_norm=1.0,
            seed=123,
        )
    )

    assert report["resume_checkpoint"] == str(resume)
    assert Path(report["checkpoint_path"]).exists()


def test_mingru_checkpoint_load_preserves_runtime_arbitration_overrides(tmp_path):
    from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig

    checkpoint = tmp_path / "mingru.pt"
    MinGRUTerminal(
        1,
        10.0,
        2,
        MinGRUTerminalConfig(
            enabled=True,
            hidden_size=4,
            sequence_length=2,
            observation_mode="env",
            include_prev_force=False,
            include_context=False,
            blend=0.25,
            scope="all",
            confidence_floor=0.8,
            passthrough_enabled=True,
            passthrough_confidence_floor=0.9,
            passthrough_logit_margin_floor=0.4,
        ),
    ).save_checkpoint(str(checkpoint))

    loaded = MinGRUTerminal(
        1,
        10.0,
        2,
        MinGRUTerminalConfig(
            enabled=True,
            hidden_size=4,
            sequence_length=2,
            observation_mode="env",
            include_prev_force=False,
            include_context=False,
            blend=1.0,
            scope="stabilize_chain",
            confidence_floor=0.05,
            passthrough_enabled=False,
            passthrough_confidence_floor=0.1,
            passthrough_logit_margin_floor=0.0,
            checkpoint_path=str(checkpoint),
        ),
    )

    assert loaded.config.blend == 1.0
    assert loaded.config.scope == "stabilize_chain"
    assert loaded.config.confidence_floor == 0.05
    assert loaded.config.passthrough_enabled is False
    assert loaded.config.passthrough_confidence_floor == 0.1
    assert loaded.config.passthrough_logit_margin_floor == 0.0
    assert loaded.config.hidden_size == 4


def test_mingru_terminal_padded_prev_force_observation_uses_state_force():
    import numpy as np
    from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig

    terminal = MinGRUTerminal(
        3,
        10.0,
        5,
        MinGRUTerminalConfig(
            enabled=True,
            observation_mode="normalized_raw4_prev_force",
            include_prev_force=True,
            include_context=False,
        ),
    )
    terminal.prev_force = 5.0
    raw = np.asarray([0.24, 0.5, 0.01, -0.02, -0.03, 0.1, -0.2, 0.3], dtype=np.float32)

    vector = terminal.observation_vector(raw, raw, {})

    assert vector.shape == (11,)
    assert terminal.input_size == 11
    assert vector[5] == 0.0
    assert vector[9] == 0.0
    assert vector[-1] == 0.5

    prediction = terminal.predict(raw, raw, {})

    assert prediction.valid is True
    assert len(prediction.logits) == 5


def test_policy_dataset_teacher_observation_mode_is_separate(monkeypatch):
    builder = _load_script("build_policy_dataset")
    captured = {}

    class FakeController:
        def __init__(self, config):
            captured["observation_mode"] = config.policy_terminal_observation_mode

    monkeypatch.setattr(builder, "ReConCartPoleController", FakeController)
    args = SimpleNamespace(
        teacher="recon_policy_terminal",
        n_poles=4,
        discrete_action_bins=5,
        force_mag=10.0,
        selection_mode="hard_select",
        policy_terminal_path="teacher.zip",
        policy_terminal_blend=1.0,
        policy_terminal_scope="stabilize_chain",
        observation_mode="normalized_raw4_prev_force",
        teacher_observation_mode="normalized_raw",
    )

    builder.make_teacher(args)

    assert captured["observation_mode"] == "normalized_raw"


def test_policy_dataset_mingru_behavior_uses_rollout_config(monkeypatch):
    builder = _load_script("build_policy_dataset")
    captured = {}

    class FakeMinGRU:
        def __init__(self, n_poles, force_mag, discrete_action_bins, config):
            captured["n_poles"] = n_poles
            captured["force_mag"] = force_mag
            captured["discrete_action_bins"] = discrete_action_bins
            captured["config"] = config

    monkeypatch.setattr(builder, "MinGRUTerminal", FakeMinGRU)
    args = SimpleNamespace(
        rollout_policy="mingru_terminal",
        behavior_checkpoint_path="student.pt",
        behavior_hidden_size=128,
        behavior_sequence_length=32,
        behavior_observation_mode="normalized_raw4_prev_force",
        behavior_include_prev_force=True,
        behavior_include_context=False,
        behavior_confidence_floor=0.05,
        policy_terminal_scope="stabilize_chain",
        n_poles=4,
        force_mag=10.0,
        discrete_action_bins=5,
    )

    behavior = builder.make_behavior(args)

    assert isinstance(behavior, FakeMinGRU)
    assert captured["n_poles"] == 4
    assert captured["config"].hidden_size == 128
    assert captured["config"].sequence_length == 32
    assert captured["config"].include_context is False
    assert captured["config"].checkpoint_path == "student.pt"


def test_policy_dataset_default_behavior_is_teacher_rollout():
    builder = _load_script("build_policy_dataset")

    assert builder.make_behavior(SimpleNamespace()) is None


def test_policy_dataset_label_source_can_use_rollout_action():
    builder = _load_script("build_policy_dataset")
    args = SimpleNamespace(label_source="rollout")

    action, force, source = builder.label_action_and_force(
        args, teacher_action=1, teacher_force=-5.0, rollout_action=4, rollout_force=10.0
    )

    assert action == 4
    assert force == 10.0
    assert source == "rollout"


def test_policy_dataset_explicit_seed_list(tmp_path):
    builder = _load_script("build_policy_dataset")
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("101\n202,303\n", encoding="utf-8")
    args = SimpleNamespace(seed_list=str(seed_file))

    assert builder.explicit_seeds(args) == [101, 202, 303]


def test_recurrent_ladder_validation_seed_starts_expand_blocks():
    ladder = _load_script("train_recurrent_terminal_ladder")
    args = SimpleNamespace(validation_seed_start=10, validation_seed_starts=[100, 200], validation_episodes=2)

    assert ladder.ladder_validation_seeds(args) == [100, 101, 200, 201]


def test_recurrent_ladder_terminal_config_exposes_passthrough():
    ladder = _load_script("train_recurrent_terminal_ladder")
    args = SimpleNamespace(
        observation_mode="normalized_raw4_prev_force",
        include_prev_force=True,
        include_context=False,
        blend=1.0,
        scope="stabilize_chain",
        confidence_floor=0.05,
        passthrough_enabled=True,
        passthrough_confidence_floor=0.90,
        passthrough_logit_margin_floor=0.10,
    )

    config = ladder.terminal_config(args, "candidate.pt", hidden=256, seq_len=32)

    assert config.passthrough_enabled is True
    assert config.passthrough_confidence_floor == 0.90
    assert config.passthrough_logit_margin_floor == 0.10
    assert config.checkpoint_path == "candidate.pt"

def test_subchain_motif_prototype_scores_separate_classes():
    motif = _load_script("train_subchain_motif_gate")
    import numpy as np

    x = np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)

    model = motif.fit_prototypes(x, y)
    scores = motif.motif_scores(model, x)

    assert scores[y == 1].mean() > scores[y == 0].mean()
    assert motif.roc_auc(y, scores) == 1.0


def test_subchain_motif_vector_uses_adjacent_pairs():
    motif = _load_script("train_subchain_motif_gate")
    import numpy as np

    args = SimpleNamespace(
        n_poles=4,
        theta_threshold=1.0,
        pole_velocity_scale=1.0,
        x_threshold=2.0,
        cart_velocity_scale=5.0,
    )
    raw = np.asarray([1.0, 2.5, 0.0, 1.0, 3.0, 6.0, 0.0, 2.0, 4.0, 8.0], dtype=np.float32)

    vector = motif.subchain_vector(raw, args)

    assert vector.shape == (14,)
    assert vector[0] == 0.5
    assert vector[1] == 0.5
    assert vector[2:6].tolist() == [1.0, 2.0, 0.5, 1.0]

def test_recovery_window_rows_preserve_motif_selection_metadata(monkeypatch):
    recovery = _load_script("train_recovery_window_residual_policy")

    def fake_select(_args, _episode):
        return [
            {
                "step": 7,
                "raw_before": [0.0] * 10,
                "force": 5.0,
                "failure_offset": 3,
                "recovery_pressure": 1.25,
                "motif_score": 4.5,
                "candidate_rank": 5.75,
            }
        ]

    monkeypatch.setattr(recovery, "select_failure_states", fake_select)
    rows = recovery.window_rows_from_episode(SimpleNamespace(), {"seed": 42, "success": False})

    assert rows == [
        {
            "seed": 42,
            "step": 7,
            "raw_state": [0.0] * 10,
            "base_force": 5.0,
            "failure_offset": 3,
            "recovery_pressure": 1.25,
            "motif_score": 4.5,
            "candidate_rank": 5.75,
            "preserve_success": False,
        }
    ]


def test_recovery_window_rows_can_add_success_preservation_windows():
    recovery = _load_script("train_recovery_window_residual_policy")
    args = SimpleNamespace(
        n_poles=4,
        preserve_success_stride=2,
        max_success_preservation_windows=2,
    )
    states = [
        {"step": idx, "raw_before": [0.01 * idx] * 10, "force": float(idx)}
        for idx in range(6)
    ]

    rows = recovery.window_rows_from_episode(args, {"seed": 99, "success": True, "states": states})

    assert [row["step"] for row in rows] == [2, 4]
    assert all(row["preserve_success"] is True for row in rows)
    assert all(row["failure_offset"] == -1 for row in rows)
    assert rows[0]["base_force"] == 2.0


def test_counterfactual_residual_can_rank_failure_states_by_motif():
    residual = _load_script("train_counterfactual_residual_terminal")

    model = {
        "positive_mean": [0.0] * 14,
        "negative_mean": [10.0] * 14,
        "scale": [1.0] * 14,
    }
    args = SimpleNamespace(
        n_poles=4,
        force_mag=10.0,
        x_threshold=2.4,
        theta_threshold=1.0,
        cart_velocity_scale=5.0,
        pole_velocity_scale=5.0,
        motif_model_path="cached",
        _motif_model_cache=model,
        motif_score_min=float("-inf"),
        motif_top_k=0,
        use_failure_window=True,
        failure_window_start=0,
        failure_window_end=5,
        failure_window_stride=5,
        failure_window_target_offset=0,
        max_window_states=2,
        max_failure_states=1,
        failure_offsets=[],
    )
    motif_match = [0.0] * 10
    pressure_match = [1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]
    episode = {
        "seed": 123,
        "states": [
            {"step": idx, "raw_before": motif_match if idx == 0 else pressure_match, "force": 0.0}
            for idx in range(6)
        ],
    }

    selected = residual.select_failure_states(args, episode)

    assert [item["step"] for item in selected] == [0]
    assert "motif_score" in selected[0]


def test_motif_gated_passthrough_can_suppress_to_base_force():
    gate = _load_script("evaluate_motif_gated_passthrough")
    args = SimpleNamespace(force_mag=10.0, discrete_action_bins=5)
    diagnostics = {
        "force": 10.0,
        "proposal": {"source_node": "mingru_terminal"},
        "mingru_terminal": {"passthrough_applied": True, "passthrough_base_force": -5.0},
    }

    action, info = gate.gated_action(4, diagnostics, score=2.0, args=args, gate_mode="suppress_passthrough", threshold=1.0)

    assert action == 1
    assert info["changed"] is True
    assert info["reason"] == "suppressed_passthrough"


def test_motif_gated_passthrough_can_force_terminal_force():
    gate = _load_script("evaluate_motif_gated_passthrough")
    args = SimpleNamespace(force_mag=10.0, discrete_action_bins=5)
    diagnostics = {
        "force": -5.0,
        "proposal": {"source_node": "recover_worst_pole"},
        "mingru_terminal": {"force": 10.0},
    }

    action, info = gate.gated_action(1, diagnostics, score=2.0, args=args, gate_mode="force_passthrough", threshold=1.0)

    assert action == 4
    assert info["changed"] is True
    assert info["reason"] == "forced_passthrough"

def test_residual_grid_controller_forwards_hold_steps(monkeypatch):
    grid = _load_script("evaluate_recon_residual_grid")
    captured = {}

    class FakeController:
        def __init__(self, config):
            captured["hold_steps"] = config.residual_policy_terminal_hold_steps
            captured["feature_mode"] = config.residual_policy_terminal_feature_mode

    monkeypatch.setattr(grid, "ReConCartPoleController", FakeController)
    args = SimpleNamespace(
        n_poles=4,
        discrete_action_bins=5,
        force_mag=10.0,
        selection_mode="hard_select",
        base_model_path="base.zip",
        policy_terminal_blend=1.0,
        policy_terminal_scope="stabilize_chain",
        base_observation_mode="normalized_raw",
        base_normalizer_path="",
        residual_model_path="residual.pt",
        residual_mode="bin_delta",
        residual_action_bins=5,
        residual_feature_mode="subchain_diagnostics",
        residual_hold_steps=4,
    )

    controller = grid.controller_for(args, threshold=0.7, max_force=4.0)

    assert isinstance(controller, FakeController)
    assert captured["hold_steps"] == 4
    assert captured["feature_mode"] == "subchain_diagnostics"

