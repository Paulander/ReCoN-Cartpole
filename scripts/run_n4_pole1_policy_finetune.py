
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from recon_cartpole.control.policy_observation import policy_observation_from_state
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_n4_pole1_robustness import FEEDFORWARD_CHECKPOINT, classify, make_env  # noqa: E402
from train_policy_terminal import train_policy_terminal  # noqa: E402

BASELINE_EVAL = 'reports/n4_pole1_robustness_20260610_171635/300_seed_eval.json'
BASELINE_MODE = 'recon_feedforward_terminal_frozen'
DEFAULT_HARD_SEEDS = 'reports/hard_seeds_n4_combined_nearmiss_600/hard_seeds.json'


def load_seed_file(path: str) -> list[int]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding='utf-8')
    if p.suffix == '.json':
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ('hard_seeds', 'failed_seeds', 'seeds'):
                if key in data:
                    return [int(item) for item in data[key]]
        if isinstance(data, list):
            return [int(item) for item in data]
    return [int(part) for part in text.replace('\n', ',').split(',') if part.strip()]


def controller_for(model_path: str, args: argparse.Namespace) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode='recon_policy_terminal',
            action_mode='discrete',
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode='hard_select',
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=model_path,
            policy_terminal_blend=1.0,
            policy_terminal_scope='stabilize_chain',
            policy_terminal_observation_mode=args.observation_mode,
        )
    )


def baseline_result(args: argparse.Namespace) -> dict[str, Any]:
    p = Path(args.baseline_eval)
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        result = dict(data[BASELINE_MODE])
        result['candidate'] = 'baseline_best_frozen'
        result['success_at_500'] = result.get('success_rate', result.get('success_at_500', 0.0))
        return result
    seeds = [args.benchmark_seed_start + idx for idx in range(args.benchmark_episodes)]
    return evaluate_model_path('baseline_best_frozen', args.feedforward_checkpoint, seeds, args)


def evaluate_model_path(candidate: str, model_path: str, seeds: list[int], args: argparse.Namespace) -> dict[str, Any]:
    controller = controller_for(model_path, args)
    per_seed = []
    steps = []
    returns = []
    failures: Counter[str] = Counter()
    started = time.perf_counter()
    for idx, seed in enumerate(seeds, 1):
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=False)
        raw = result.get('trace', [])[-1].get('raw_state', []) if result.get('trace') else []
        # Trace is disabled; classify from a one-step trace rerun only for failures is too costly here.
        failure = 'success' if int(result['steps']) >= args.horizon else 'failure'
        steps.append(float(result['steps']))
        returns.append(float(result['return']))
        failures[failure] += 1
        per_seed.append(
            {
                'seed': int(seed),
                'candidate': candidate,
                'steps': int(result['steps']),
                'return': float(result['return']),
                'success': int(result['steps']) >= args.horizon,
                'failure': failure,
                'final_raw_state': raw,
            }
        )
        if idx % args.progress_every == 0:
            print(f'[n4-pole1-finetune] eval {candidate}: {idx}/{len(seeds)}', flush=True)
    summary = summarize_steps(steps, args.horizon)
    return {
        'candidate': candidate,
        'model_path': model_path,
        'episodes': len(seeds),
        'mean_survival': summary['mean_survival'],
        'median_survival': float(np.median(steps)) if steps else 0.0,
        'p10_survival': summary['p10_survival'],
        'p90_survival': float(np.percentile(steps, 90)) if steps else 0.0,
        'success_at_500': summary['success_rate'],
        'max_survival': summary['max_survival'],
        'returns_mean': float(np.mean(returns)) if returns else 0.0,
        'failure_distribution': dict(failures),
        'wall_clock_seconds': time.perf_counter() - started,
        'per_seed': per_seed,
    }


def evaluate_model_path_trace(candidate: str, model_path: str, seeds: list[int], args: argparse.Namespace) -> dict[str, Any]:
    controller = controller_for(model_path, args)
    per_seed = []
    steps = []
    returns = []
    failures: Counter[str] = Counter()
    for idx, seed in enumerate(seeds, 1):
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=True)
        trace = result['trace']
        raw = trace[-1].get('raw_state', []) if trace else []
        failure = classify(raw, int(result['steps']), args)
        steps.append(float(result['steps']))
        returns.append(float(result['return']))
        failures[failure] += 1
        per_seed.append(
            {
                'seed': int(seed),
                'candidate': candidate,
                'steps': int(result['steps']),
                'return': float(result['return']),
                'success': int(result['steps']) >= args.horizon,
                'failure': failure,
                'final_raw_state': raw,
            }
        )
        if idx % args.progress_every == 0:
            print(f'[n4-pole1-finetune] trace-eval {candidate}: {idx}/{len(seeds)}', flush=True)
    summary = summarize_steps(steps, args.horizon)
    return {
        'candidate': candidate,
        'model_path': model_path,
        'episodes': len(seeds),
        'mean_survival': summary['mean_survival'],
        'median_survival': float(np.median(steps)) if steps else 0.0,
        'p10_survival': summary['p10_survival'],
        'p90_survival': float(np.percentile(steps, 90)) if steps else 0.0,
        'success_at_500': summary['success_rate'],
        'max_survival': summary['max_survival'],
        'returns_mean': float(np.mean(returns)) if returns else 0.0,
        'failure_distribution': dict(failures),
        'per_seed': per_seed,
    }


def force_from_action(action: Any, args: argparse.Namespace) -> float:
    idx = int(np.asarray(action).reshape(-1)[0])
    return float(np.linspace(-args.force_mag, args.force_mag, args.discrete_action_bins)[idx])


def collect_failure_dataset(args: argparse.Namespace, baseline: dict[str, Any], out: Path) -> dict[str, Any]:
    from stable_baselines3 import PPO

    model = PPO.load(args.feedforward_checkpoint, device='cpu')
    failures = [item for item in baseline['per_seed'] if not item['success']]
    successes = [item for item in baseline['per_seed'] if item['success']]
    failure_seeds = [int(item['seed']) for item in sorted(failures, key=lambda row: row['steps'], reverse=True)[: args.dataset_failures]]
    success_seeds = [int(item['seed']) for item in successes[: args.dataset_successes]]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[Any]] = {
        'observations': [],
        'raw_states': [],
        'ppo_forces': [],
        'recon_forces': [],
        'actions': [],
        'failure_within_25': [],
        'failure_within_50': [],
        'failure_within_100': [],
        'failure_within_150': [],
        'returns_to_go': [],
        'seeds': [],
        'steps': [],
        'sources': [],
    }
    for source, seeds in [('fixed_benchmark_failure', failure_seeds), ('fixed_benchmark_success_match', success_seeds)]:
        for seed in seeds:
            controller = controller_for(args.feedforward_checkpoint, args)
            env = make_env(args)
            obs, info = env.reset(seed=seed)
            controller.start_episode()
            episode: list[dict[str, Any]] = []
            total_rewards: list[float] = []
            for step in range(args.horizon):
                raw = info.get('raw_state')
                policy_obs = policy_observation_from_state(obs, raw, args.n_poles, args.observation_mode)
                ppo_action, _ = model.predict(policy_obs, deterministic=True)
                ppo_force = force_from_action(ppo_action, args)
                action, diagnostics = controller.act(obs, raw)
                recon_force = float(diagnostics.get('force', 0.0))
                obs, reward, terminated, truncated, info = env.step(action)
                total_rewards.append(float(reward))
                episode.append(
                    {
                        'observation': policy_obs.astype(np.float32),
                        'raw_state': np.asarray(info.get('raw_state', []), dtype=np.float32),
                        'ppo_force': ppo_force,
                        'recon_force': recon_force,
                        'action': int(np.asarray(action).reshape(-1)[0]),
                        'seed': seed,
                        'step': step,
                        'source': source,
                        'selected_regime': diagnostics.get('selected_regime', ''),
                    }
                )
                if terminated or truncated:
                    break
            rewards = np.asarray(total_rewards, dtype=np.float32)
            rtg = np.cumsum(rewards[::-1])[::-1] if rewards.size else np.asarray([], dtype=np.float32)
            terminal = len(episode) < args.horizon
            selected_indices: set[int] = set()
            if terminal and source == 'fixed_benchmark_failure':
                last = len(episode) - 1
                for window in (25, 50, 100, 150):
                    selected_indices.update(range(max(0, last - window + 1), last + 1))
            else:
                if len(episode) <= args.success_window_stride:
                    selected_indices.update(range(len(episode)))
                else:
                    selected_indices.update(range(0, len(episode), args.success_window_stride))
            for idx in sorted(selected_indices):
                row = episode[idx]
                steps_to_end = len(episode) - 1 - idx
                raw = row['raw_state']
                arrays['observations'].append(row['observation'])
                arrays['raw_states'].append(raw)
                arrays['ppo_forces'].append(row['ppo_force'])
                arrays['recon_forces'].append(row['recon_force'])
                arrays['actions'].append(row['action'])
                arrays['failure_within_25'].append(float(terminal and steps_to_end < 25))
                arrays['failure_within_50'].append(float(terminal and steps_to_end < 50))
                arrays['failure_within_100'].append(float(terminal and steps_to_end < 100))
                arrays['failure_within_150'].append(float(terminal and steps_to_end < 150))
                arrays['returns_to_go'].append(float(rtg[idx]) if idx < len(rtg) else 0.0)
                arrays['seeds'].append(seed)
                arrays['steps'].append(row['step'])
                arrays['sources'].append(source)
                theta1 = float(raw[3]) if raw.size >= 4 else 0.0
                theta1_dot = float(raw[3 + args.n_poles]) if raw.size >= 3 + args.n_poles else 0.0
                rows.append(
                    {
                        'seed': seed,
                        'step': int(row['step']),
                        'source': source,
                        'theta1': theta1,
                        'theta1_dot': theta1_dot,
                        'ppo_force': float(row['ppo_force']),
                        'recon_force': float(row['recon_force']),
                        'force_delta': float(row['recon_force'] - row['ppo_force']),
                        'selected_regime': row['selected_regime'],
                        'failure_within_25': bool(terminal and steps_to_end < 25),
                        'failure_within_50': bool(terminal and steps_to_end < 50),
                        'failure_within_100': bool(terminal and steps_to_end < 100),
                        'failure_within_150': bool(terminal and steps_to_end < 150),
                        'return_to_go': float(rtg[idx]) if idx < len(rtg) else 0.0,
                    }
                )
    dataset_path = out / 'pole1_failure_dataset.npz'
    dataset = {
        'observations': np.stack(arrays['observations']).astype(np.float32),
        'raw_states': np.stack(arrays['raw_states']).astype(np.float32),
        'ppo_forces': np.asarray(arrays['ppo_forces'], dtype=np.float32),
        'recon_forces': np.asarray(arrays['recon_forces'], dtype=np.float32),
        'actions': np.asarray(arrays['actions'], dtype=np.int64),
        'failure_within_25': np.asarray(arrays['failure_within_25'], dtype=np.float32),
        'failure_within_50': np.asarray(arrays['failure_within_50'], dtype=np.float32),
        'failure_within_100': np.asarray(arrays['failure_within_100'], dtype=np.float32),
        'failure_within_150': np.asarray(arrays['failure_within_150'], dtype=np.float32),
        'returns_to_go': np.asarray(arrays['returns_to_go'], dtype=np.float32),
        'seeds': np.asarray(arrays['seeds'], dtype=np.int64),
        'steps': np.asarray(arrays['steps'], dtype=np.int64),
        'sources': np.asarray(arrays['sources']),
    }
    np.savez_compressed(dataset_path, **dataset)
    summary = {
        'dataset_path': str(dataset_path),
        'samples': int(dataset['observations'].shape[0]),
        'failure_seed_count': len(failure_seeds),
        'success_seed_count': len(success_seeds),
        'source_counts': dict(Counter(arrays['sources'])),
        'failure_within_25': int(np.sum(dataset['failure_within_25'])),
        'failure_within_50': int(np.sum(dataset['failure_within_50'])),
        'failure_within_100': int(np.sum(dataset['failure_within_100'])),
        'failure_within_150': int(np.sum(dataset['failure_within_150'])),
    }
    (out / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'failure_windows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    write_failure_window_analysis(rows, summary, out)
    return summary


def write_failure_window_analysis(rows: list[dict[str, Any]], summary: dict[str, Any], out: Path) -> None:
    failure_rows = [row for row in rows if row['source'] == 'fixed_benchmark_failure']
    success_rows = [row for row in rows if row['source'] != 'fixed_benchmark_failure']
    lines = ['# Failure Window Analysis', '']
    lines.append(f"Dataset samples: `{summary['samples']}`")
    lines.append(f"Failure seeds: `{summary['failure_seed_count']}`; matched success seeds: `{summary['success_seed_count']}`")
    lines.append(f"Failure-window labels: 25 `{summary['failure_within_25']}`, 50 `{summary['failure_within_50']}`, 100 `{summary['failure_within_100']}`, 150 `{summary['failure_within_150']}`")
    lines.append('')
    if failure_rows:
        force_deltas = [abs(row['force_delta']) for row in failure_rows]
        wrong_sign = [np.sign(row['ppo_force']) != np.sign(row['recon_force']) for row in failure_rows if abs(row['ppo_force']) > 1e-6 and abs(row['recon_force']) > 1e-6]
        theta1 = [abs(row['theta1']) for row in failure_rows]
        theta1_dot = [abs(row['theta1_dot']) for row in failure_rows]
        lines.extend([
            '## Failure Windows',
            f"- Avg |theta1|: `{np.mean(theta1):.4f}`; p90 `{np.percentile(theta1, 90):.4f}`",
            f"- Avg |theta1_dot|: `{np.mean(theta1_dot):.4f}`; p90 `{np.percentile(theta1_dot, 90):.4f}`",
            f"- Avg |ReCoN force - PPO force|: `{np.mean(force_deltas):.3f}`; p90 `{np.percentile(force_deltas, 90):.3f}`",
            f"- PPO/ReCoN force sign disagreement rate: `{np.mean(wrong_sign) if wrong_sign else 0.0:.3f}`",
        ])
    if success_rows:
        lines.extend(['', '## Matched Success Windows'])
        lines.append(f"- Avg |theta1|: `{np.mean([abs(row['theta1']) for row in success_rows]):.4f}`")
        lines.append(f"- Avg |theta1_dot|: `{np.mean([abs(row['theta1_dot']) for row in success_rows]):.4f}`")
    lines.extend([
        '',
        '## Interpretation',
        'This dataset is diagnostic and fine-tune support data. The fixed 930000..930299 seeds are labelled as a fixed benchmark split, not pristine held-out data, because previous iterations have inspected them repeatedly.',
    ])
    (out / 'failure_window_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def training_args(args: argparse.Namespace, out: Path) -> SimpleNamespace:
    return SimpleNamespace(
        n_poles=args.n_poles,
        horizon=args.horizon,
        dt=args.dt,
        dynamics_mode=args.dynamics_mode,
        action_mode='discrete',
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        timesteps=args.finetune_timesteps,
        model_path='',
        resume_model_path=args.feedforward_checkpoint,
        train_seed=args.train_seed,
        hard_train_seeds=args.hard_train_seeds,
        hard_train_seed_probability=args.hard_train_seed_probability,
        eval_seed_start=args.validation_seed_start,
        eval_episodes=args.validation_episodes,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        device=args.device,
        policy='MlpPolicy',
        net_arch=args.net_arch,
        activation=args.activation,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        reward_mode=args.reward_mode,
        selection_mode='hard_select',
        policy_terminal_blend=1.0,
        policy_terminal_scope='stabilize_chain',
        policy_observation_mode=args.observation_mode,
        frame_stack=1,
        verbose=args.verbose,
        out=str(out / 'finetune_training'),
    )


def compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    base = {int(item['seed']): item for item in baseline['per_seed']}
    changed = []
    fail_to_success = []
    success_to_fail = []
    for item in candidate['per_seed']:
        seed = int(item['seed'])
        b = base.get(seed)
        if not b:
            continue
        delta = int(item['steps']) - int(b['steps'])
        if delta != 0 or bool(item['success']) != bool(b['success']):
            changed.append({'seed': seed, 'baseline': b, 'candidate': item, 'delta_steps': delta})
        if not b['success'] and item['success']:
            fail_to_success.append(seed)
        if b['success'] and not item['success']:
            success_to_fail.append(seed)
    return {
        'changed_count': len(changed),
        'fail_to_success': fail_to_success,
        'success_to_fail': success_to_fail,
        'net_success_change': len(fail_to_success) - len(success_to_fail),
        'changed': changed,
    }


def candidate_row(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    comp = compare_to_baseline(result, baseline)
    failures = {k: v for k, v in result['failure_distribution'].items() if k != 'success'}
    return {
        'candidate': result['candidate'],
        'episodes': result['episodes'],
        'mean_survival': result['mean_survival'],
        'median_survival': result['median_survival'],
        'p10_survival': result['p10_survival'],
        'success_at_500': result['success_at_500'],
        'max_survival': result['max_survival'],
        'fail_to_success': len(comp['fail_to_success']),
        'success_to_fail': len(comp['success_to_fail']),
        'net_success_change': comp['net_success_change'],
        'changed_count': comp['changed_count'],
        'dominant_failure': max(failures.items(), key=lambda item: item[1])[0] if failures else 'none',
        'model_path': result.get('model_path', ''),
    }


def write_tables(rows: list[dict[str, Any]], out: Path) -> None:
    (out / 'candidate_table.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    fields = list(rows[0]) if rows else []
    with (out / 'candidate_table.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_changed_seed_analysis(comparison: dict[str, Any], out: Path) -> None:
    lines = ['# Changed Seed Analysis', '']
    lines.append(f"Changed seeds: `{comparison['changed_count']}`")
    lines.append(f"Fail -> success: `{len(comparison['fail_to_success'])}` {comparison['fail_to_success'][:80]}")
    lines.append(f"Success -> fail: `{len(comparison['success_to_fail'])}` {comparison['success_to_fail'][:80]}")
    lines.extend(['', '## Largest Improvements'])
    for item in sorted(comparison['changed'], key=lambda row: row['delta_steps'], reverse=True)[:30]:
        lines.append(f"- `{item['seed']}`: {item['baseline']['steps']} -> {item['candidate']['steps']} ({item['delta_steps']:+d})")
    lines.extend(['', '## Largest Regressions'])
    for item in sorted(comparison['changed'], key=lambda row: row['delta_steps'])[:30]:
        lines.append(f"- `{item['seed']}`: {item['baseline']['steps']} -> {item['candidate']['steps']} ({item['delta_steps']:+d})")
    (out / 'changed_seed_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'changed_seed_analysis.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')


def solved(result: dict[str, Any]) -> bool:
    return result['episodes'] >= 300 and result['mean_survival'] >= 475.0 and result['p10_survival'] >= 350.0 and result['success_at_500'] >= 0.70


def write_summary(out: Path, args: argparse.Namespace, dataset: dict[str, Any], train_report: dict[str, Any], rows: list[dict[str, Any]], benchmark: dict[str, Any], confirmation: dict[str, Any] | None) -> None:
    best = max(rows, key=lambda row: (row['success_at_500'], row['mean_survival'], row['p10_survival']))
    lines = ['# N=4 Pole_1 Policy Fine-Tune Report', '']
    lines.append(f"Status: `{'solved' if solved(benchmark) else 'not solved'}`")
    lines.append("Benchmark split: `930000..930299` fixed benchmark seeds, not pristine held-out")
    lines.append(f"Best fixed-benchmark candidate: `{best['candidate']}`")
    lines.append(f"Best mean/p10/success: `{best['mean_survival']:.1f}` / `{best['p10_survival']:.1f}` / `{best['success_at_500']:.3f}`")
    lines.append(f"Best checkpoint path: `{best['model_path']}`")
    lines.extend(['', '## Answers'])
    lines.append(f"- Failure-focused dataset exists? `yes`, `{dataset['samples']}` samples at `{dataset['dataset_path']}`.")
    lines.append(f"- Targeted fine-tune trained? `yes`, `{train_report.get('train_timesteps', 0)}` timesteps.")
    lines.append(f"- Did fixed benchmark pass solve threshold? `{'yes' if solved(benchmark) else 'no'}`.")
    lines.append(f"- Fail -> success / success -> fail: `{best['fail_to_success']}` / `{best['success_to_fail']}`.")
    lines.append(f"- Dominant failure after best candidate: `{best['dominant_failure']}`.")
    if confirmation:
        lines.append(f"- Confirmation 940000..940299 success: `{confirmation['success_at_500']:.3f}`.")
    else:
        lines.append('- Confirmation split run? `no`; fixed benchmark did not pass, so no solved confirmation was warranted.')
    lines.extend(['', '## Candidate Table'])
    for row in rows:
        lines.append(f"- `{row['candidate']}`: mean `{row['mean_survival']:.1f}`, p10 `{row['p10_survival']:.1f}`, success `{row['success_at_500']:.3f}`, net success `{row['net_success_change']:+d}`")
    lines.extend(['', '## Reproduce'])
    lines.append('```bash')
    lines.append(f"uv run python scripts/run_n4_pole1_policy_finetune.py --out {out} --finetune-timesteps {args.finetune_timesteps}")
    lines.append('```')
    lines.extend(['', '## Interpretation'])
    if solved(benchmark):
        lines.append('The fine-tuned policy terminal passed the fixed benchmark. Treat confirmation split status above as the overfit check.')
    else:
        lines.append('The fine-tuned policy terminal did not close the success gap. The blocker appears to require stronger policy learning or reward/data reformulation rather than another narrow routing gate.')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.out or f'reports/n4_pole1_policy_finetune_{timestamp}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config_resolved.yaml').write_text(yaml.safe_dump(vars(args), sort_keys=False), encoding='utf-8')
    baseline = baseline_result(args)
    (out / 'baseline_300_eval.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    dataset = collect_failure_dataset(args, baseline, out)

    hard_seeds = load_seed_file(args.hard_train_seeds)
    (out / 'hard_train_seeds.txt').write_text('\n'.join(str(seed) for seed in hard_seeds) + '\n', encoding='utf-8')
    print(f'[n4-pole1-finetune] training fine-tune on {len(hard_seeds)} hard seeds', flush=True)
    train_report = train_policy_terminal(training_args(args, out))
    (out / 'training_report.json').write_text(json.dumps(train_report, indent=2), encoding='utf-8')

    benchmark_seeds = [args.benchmark_seed_start + idx for idx in range(args.benchmark_episodes)]
    baseline_trace = evaluate_model_path_trace('baseline_best_frozen', args.feedforward_checkpoint, benchmark_seeds, args)
    finetuned = evaluate_model_path_trace('recon_feedforward_terminal_finetuned', train_report['model_path'], benchmark_seeds, args)
    results = {'baseline_best_frozen': baseline_trace, 'recon_feedforward_terminal_finetuned': finetuned}
    (out / 'final_300_eval.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    rows = [candidate_row(baseline_trace, baseline_trace), candidate_row(finetuned, baseline_trace)]
    write_tables(rows, out)
    comparison = compare_to_baseline(finetuned, baseline_trace)
    write_changed_seed_analysis(comparison, out)

    confirmation = None
    if solved(finetuned) and args.confirmation_episodes > 0:
        confirmation_seeds = [args.confirmation_seed_start + idx for idx in range(args.confirmation_episodes)]
        confirmation = evaluate_model_path_trace('recon_feedforward_terminal_finetuned_confirmation', train_report['model_path'], confirmation_seeds, args)
        (out / 'confirmation_300_eval.json').write_text(json.dumps(confirmation, indent=2), encoding='utf-8')
    write_summary(out, args, dataset, train_report, rows, finetuned, confirmation)
    return {'out': str(out), 'rows': rows, 'best_model_path': train_report['model_path']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='')
    parser.add_argument('--baseline-eval', default=BASELINE_EVAL)
    parser.add_argument('--feedforward-checkpoint', default=FEEDFORWARD_CHECKPOINT)
    parser.add_argument('--hard-train-seeds', default=DEFAULT_HARD_SEEDS)
    parser.add_argument('--n-poles', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--dt', type=float, default=0.0005)
    parser.add_argument('--dynamics-mode', choices=['parallel', 'serial_lagrange'], default='serial_lagrange')
    parser.add_argument('--discrete-action-bins', type=int, default=5)
    parser.add_argument('--force-mag', type=float, default=10.0)
    parser.add_argument('--initial-angle-range', type=float, default=0.05)
    parser.add_argument('--force-noise', type=float, default=0.02)
    parser.add_argument('--link-coupling', type=float, default=12.0)
    parser.add_argument('--observation-mode', choices=['env', 'normalized_raw'], default='normalized_raw')
    parser.add_argument('--dataset-failures', type=int, default=100)
    parser.add_argument('--dataset-successes', type=int, default=100)
    parser.add_argument('--success-window-stride', type=int, default=5)
    parser.add_argument('--finetune-timesteps', type=int, default=25_000)
    parser.add_argument('--train-seed', type=int, default=1810000)
    parser.add_argument('--hard-train-seed-probability', type=float, default=0.80)
    parser.add_argument('--validation-seed-start', type=int, default=920000)
    parser.add_argument('--validation-episodes', type=int, default=100)
    parser.add_argument('--benchmark-seed-start', type=int, default=930000)
    parser.add_argument('--benchmark-episodes', type=int, default=300)
    parser.add_argument('--confirmation-seed-start', type=int, default=940000)
    parser.add_argument('--confirmation-episodes', type=int, default=300)
    parser.add_argument('--success-bonus', type=float, default=25.0)
    parser.add_argument('--failure-penalty', type=float, default=2.0)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--vec-env', choices=['dummy', 'subproc'], default='dummy')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--net-arch', default='64,64')
    parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
    parser.add_argument('--learning-rate', type=float, default=2.5e-5)
    parser.add_argument('--n-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.08)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--reward-mode', choices=['survival', 'upright_shaping'], default='upright_shaping')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--x-threshold', type=float, default=2.4)
    parser.add_argument('--theta-threshold', type=float, default=12.0 * 2.0 * np.pi / 360.0)
    parser.add_argument('--velocity-failure-threshold', type=float, default=8.0)
    parser.add_argument('--rail-conflict-x', type=float, default=1.5)
    parser.add_argument('--pole1-velocity-mix', type=float, default=0.30)
    parser.add_argument('--low-confidence-threshold', type=float, default=0.2)
    parser.add_argument('--high-confidence-threshold', type=float, default=0.7)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
