"""NC/CA/TOO/TFOO/Proposed paired, closed-loop evaluation without learning.

NC, TOO and TFOO each load their independently trained upper policy. Shared
Proposed-upper controllers are supplementary ablations only. Every moving
controller uses the deterministic upper mean on its own observation.
"""
import json
from pathlib import Path

import torch

from .baseline_protocol import PAPER_COMPARISONS, PAPER_SUPPLEMENTS, is_paper_caliber, labeled_results
from .checkpoint import atomic_json, json_digest, digest
from .collector import collect_metrics
from .evaluate import gate_manifest, manifest
from .metrics import PooledMetrics
from .physics_adapter import PhysicsAdapter
from .policy import Policy
from .randomness import Streams
from .timing import ClockConfig
from .upper_baselines import UpperLearner, fixed_upper_policy, load_typed


def load_checkpoints(paths, binding, config, device, *, final_test=False):
    """Validate all four types/identities before constructing any learner."""
    from .runtime import Learner
    if set(paths) != {'Proposed', 'NC', 'TOO', 'TFOO'}:
        raise ValueError('Paper panel requires Proposed, NC, TOO and TFOO checkpoints')
    saved = {name: load_typed(path, binding, name) for name, path in paths.items()}
    seeds = {s['seed'] for s in saved.values()}
    if len(seeds) != 1: raise ValueError('Paper checkpoints must use the same master seed')
    if final_test:
        for name, s in saved.items():
            endpoint = 600 if name == 'Proposed' else 200
            if s['completed_episode'] != endpoint or s['pending_evaluation'] is not None:
                raise ValueError('Final test requires completed checkpoint endpoints')
            if name != 'Proposed' and s['learner']['ppo']['counts']['batches'] != 40:
                raise ValueError('Independent baseline final checkpoint needs 40 PPO batches')
    learners = {}
    for name, s in saved.items():
        learner = (Learner if name == 'Proposed' else UpperLearner)(s['seed'], device, config)
        learner.load_state_dict(s['learner'])
        learners[name] = learner
    return learners, {name: digest(path) for name, path in paths.items()}, next(iter(seeds))


def comparison_policy(learner, scene, streams, internal_name, *, stochastic=True):
    """Override routing exactly as the frozen equal/CVX references do."""
    if not is_paper_caliber(scene['base_scene']['objective']):
        raise ValueError('Paper comparison requires the v2.3 physical objective')
    choices = {name: (mode, route) for _, name, mode, route in PAPER_COMPARISONS + PAPER_SUPPLEMENTS}
    if internal_name not in choices:
        raise ValueError('Unknown paper comparison')
    mode, route = choices[internal_name]
    if route is not None:
        return fixed_upper_policy(learner.ppo, scene, streams, route, stochastic=False, mode=mode)
    return Policy(learner.sac.actor, learner.ppo.actor, None, streams,
                  mode=mode, stochastic=stochastic, upper_stochastic=False,
                  deployment=scene['deployment'])


def _write_table(output, summary):
    lines = [
        '# Paired closed-loop comparison', '',
        'NC/TOO/TFOO each use their independently trained upper checkpoint.',
        'Upper actions are deterministic; Proposed lower: '+summary['identity']['lower_policy']+'.', '',
        '| Paper | Internal | Truncated delay (s) | Upload (J/window) | CPU (J/window) | U2U (J/window) | Propulsion (J/window) | Cost/window | Cost/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary['main_comparison']:
        m = row['metrics']
        delay = 'n/a' if m['truncated_delay_s'] is None else format(m['truncated_delay_s'], '.9g')
        energy = [format(m['energy_j'][k]/m['windows'], '.9g') for k in ('upload', 'cpu', 'u2u', 'propulsion')]
        cells = [row['paper_name'], row['internal_name'], delay] + energy + [
            format(m['cost_sum']/m['windows'], '.9g'), format(m['cost_per_s'], '.9g')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Supplementary ablations', '',
              'shared_upper_equal/shared_upper_cvx reuse Proposed upper weights; hover_local remains the G1 reference.', '',
              '| Internal | Truncated delay (s) | Cost/s |', '|---|---:|---:|']
    for row in summary['supplementary']:
        m = row['metrics']
        lines.append(f'| {row["internal_name"]} | {m["truncated_delay_s"]} | {m["cost_per_s"]} |')
    (Path(output)/('comparison_'+summary['identity']['lower_policy']+'.md')).write_bytes(
        ('\n'.join(lines)+'\n').encode('utf-8'))


def panel(learners, scene, master, panel_manifest, output, checkpoint_ids, *,
          stochastic=True, timing=None, reward_scale=100.):
    """Evaluate five main controllers and three supplementary controls."""
    if not is_paper_caliber(scene['base_scene']['objective']):
        raise ValueError('Paper comparison requires the v2.3 physical objective')
    split = panel_manifest['split']
    if panel_manifest != manifest(split) and panel_manifest != gate_manifest(20):
        raise ValueError('Evaluation manifest differs from sealed seed contract')
    if set(learners) != {'Proposed', 'NC', 'TOO', 'TFOO'} or set(checkpoint_ids) != set(learners):
        raise ValueError('Paper panel requires four separately typed checkpoint learners')
    timing = ClockConfig.from_scene(scene) if timing is None else timing
    lower_policy = 'stochastic' if stochastic else 'deterministic'
    key = dict(protocol='paper_comparison_independent_upper_2', checkpoints=checkpoint_ids,
               manifest=json_digest(panel_manifest), upper_policy='deterministic',
               lower_policy=lower_policy, upper_source='independent_NC_TOO_TFOO; shared_upper_ablations_use_Proposed',
               rollout_seconds=timing.rollout_seconds, reward_scale=reward_scale)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    combinations = PAPER_COMPARISONS + PAPER_SUPPLEMENTS
    totals = {name: PooledMetrics() for _, name, _, _ in combinations}
    scenarios = {name: {} for name in totals}
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for scenario in panel_manifest['scenarios']:
            # Keep the exact manifest stream list for every controller,
            # including deterministic execution, to retain paired references.
            for stream in panel_manifest['policy_streams']:
                for paper, name, _, _ in combinations:
                    owner = paper if paper in ('NC', 'TOO', 'TFOO') else 'Proposed'
                    item = dict(key, scenario=scenario, stream=stream, master=master,
                                paper_name=paper, internal_name=name, checkpoint_owner=owner,
                                checkpoint=checkpoint_ids[owner])
                    path = output/f'{lower_policy}_{name}_{scenario["scenario"]:02d}_{stream}.json'
                    if path.exists():
                        saved = json.loads(path.read_text(encoding='utf-8'))
                        if saved['identity'] != item:
                            raise ValueError('Existing evaluation belongs to another contract')
                        metrics = saved['metrics']
                    else:
                        adapter = PhysicsAdapter(scene, timing=timing)
                        adapter.reset(**{k: v for k, v in scenario.items() if k.endswith('_seed')})
                        streams = Streams(master, episode=scenario['scenario'], split=split, policy_stream=stream)
                        policy = comparison_policy(learners[owner], scene, streams, name, stochastic=stochastic)
                        metrics = collect_metrics(adapter, policy, identity=item, reward_scale=reward_scale)
                        diagnostics = {}
                        if name in ('moving_cvx', 'shared_upper_cvx'):
                            records = policy.lower_override.rule.records
                            diagnostics = dict(solver_calls=len(records),
                                               fallback_calls=sum(r.get('fallback', False) for r in records))
                        atomic_json(path, dict(identity=item, metrics=metrics, diagnostics=diagnostics,
                                               learning_updates=0, replay_writes=0))
                    totals[name].add(metrics)
                    scenarios[name].setdefault(str(scenario['scenario']), PooledMetrics()).add(metrics)
    summary = dict(identity=key, manifest=panel_manifest, master=master,
                   by_mode={k: v.result() for k, v in totals.items()},
                   per_scenario={k: {sid: p.result() for sid, p in rows.items()} for k, rows in scenarios.items()},
                   learning_updates=0, replay_writes=0)
    summary.update(labeled_results(summary['by_mode']))
    atomic_json(output/f'summary_{lower_policy}.json', summary)
    _write_table(output, summary)
    return summary
