"""Frozen-checkpoint comparison with observed geometry and additive task clocks."""

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.bandwidth import MeteredMDP, summarize_links, summarize_meter
from hrl_uav.experiment import isolated_rng, rollout, write_json
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP


JOINT = ROOT / 'results/hrl_pretrain200_joint600_finalonly_seed1'
PURE = ROOT / 'results/split_mdp_actor_lr3e5_resume600_to800_seed1'
OUT = JOINT / 'paired_motion_audit'
ROLES = ('heavy', 'medium', 'light')
COMPONENTS = ('upload_ms', 'routing_wait_ms', 'transfer_stage_ms', 'compute_stage_ms')


class ClockAuditMDP(MeteredMDP):
    def __init__(self, config):
        super().__init__(config, record_links=True)
        env = self.env
        self.task_clocks, traces, parents = [], {}, {}
        create, compute, finish, record = (env._create_chunk, env._enqueue_compute_chunk,
                                           env._mark_chunk_completed, env._record_completed_parent_task)

        def traced_create(parent, size, target, stage):
            chunk = create(parent, size, target, stage)
            traces[chunk.chunk_id] = {'route': env._current_time(), 'chunk': chunk.chunk_id}
            parents.setdefault(parent.task_id, []).append(chunk.chunk_id)
            return chunk

        def traced_compute(chunk):
            traces[chunk.chunk_id]['compute'] = env._current_time()
            return compute(chunk)

        def traced_finish(chunk, finish_time):
            traces[chunk.chunk_id]['finish'] = finish_time
            return finish(chunk, finish_time)

        def traced_record(parent):
            ids = parents.pop(parent.task_id)
            # For tied last-finishing chunks use the lowest id, not a favorable path.
            critical = max((traces[i] for i in ids), key=lambda t: (t['finish'], -t['chunk']))
            values = np.diff([parent.birth_time, parent.upload_finish_time, critical['route'],
                              critical['compute'], parent.finish_time]) * 1000
            assert values.min() > -1e-6, values
            self.task_clocks.append({'task_id': parent.task_id, 'source': parent.source_uav,
                                    'priority': parent.priority,
                                    'delay_ms': (parent.finish_time-parent.birth_time)*1000,
                                    **dict(zip(COMPONENTS, values))})
            for i in ids:
                traces.pop(i)
            return record(parent)

        env._create_chunk, env._enqueue_compute_chunk = traced_create, traced_compute
        env._mark_chunk_completed, env._record_completed_parent_task = traced_finish, traced_record


def geometry_and_routing(root, label, seed, env):
    with np.load(root / f'{label}_{seed}_trajectory.npz') as archive:
        uav, users, centroid, actions, routes = [archive[k].copy() for k in
            ('uav', 'users', 'centroid', 'macro_action', 'executed_route')]
    meter = pd.read_csv(root / f'{label}_{seed}_uavs.csv')
    service = meter[meter.service_steps > 0].sort_values(['time_s', 'uav'])
    roles = service.role.to_numpy().reshape(-1, 3)
    order = np.argsort(np.vectorize(ROLES.index)(roles), axis=1)
    role_routes = np.stack([routes[i][:, o][:, :, o] for i, o in enumerate(order)])
    macro_roles = roles[::env.service_resource_frames]
    actions = actions.reshape(-1, 3, 2)
    role_actions = {str(u): {role: actions[macro_roles[:, u] == role, u].mean(0).tolist()
                           for role in ROLES} for u in range(3)}
    distances = np.linalg.norm(uav[:, :, None, :] - uav[:, None, :, :], axis=-1)
    gt_distance = np.sqrt(((users-uav[:, env.user_allocations])**2).sum(-1)+env.H**2)
    gt_rates = np.stack([env.compute_u2g_rate_bps(user_coords=us, uav_coords=uv)/1e6
                         for us, uv in zip(users, uav)])
    path = np.linalg.norm(np.diff(uav, axis=0), axis=-1).sum(0)
    role_loads = {}
    for role in ROLES:
        r = meter[meter.role == role]
        role_loads[role] = {k: float(r[k].sum()/(len(r)*0.5)) for k in
                           ('assigned_source_mbit', 'assigned_destination_mbit', 'assigned_local_mbit',
                            'actual_tx_in_mbit', 'actual_tx_out_mbit', 'computed_mbit')}
        role_loads[role]['mean_compute_queue_mbit'] = float(r.queue_compute_mbit.mean())
    return {
        'seed': seed, 'policy': label, 'centroid_residual_m': float(np.linalg.norm(uav-centroid, axis=-1).mean()),
        'u2g_slant_m': float(gt_distance.mean()), 'u2g_mean_user_rate_mbps': float(gt_rates.mean()),
        'u2u_distances_m': {f'{i}-{j}': float(distances[:, i, j].mean()) for i in range(3) for j in range(i+1, 3)},
        'path_after_first_macro_m_per_uav': path.tolist(),
        'action_mean_by_uav': actions.mean(0).tolist(), 'action_std_by_uav': actions.std(0).tolist(),
        'action_mean_by_uav_role': role_actions, 'role_routes_high_low': role_routes.mean(0).tolist(),
        'load_roles': role_loads,
        'assigned_forward_fraction': float(1-meter.assigned_local_mbit.sum()/meter.assigned_source_mbit.sum()),
    }


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    report = json.loads((JOINT / 'report.json').read_text(encoding='utf-8'))
    pure_report = json.loads((PURE / 'report.json').read_text(encoding='utf-8'))
    checkpoint_path = JOINT / 'final.pt'
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = saved['config']
    env = QueueMDP(config)
    env.reset(1, config['training']['window_macros'])
    agent = DirectHRL.from_mdp(env, config)
    agent.sac_actor.action_net.load_state_dict(saved['sac_actor'])
    agent.macro_actor.load_state_dict(saved['macro_actor'])
    ppo_stats = saved['training_state']['last_ppo_stats']
    ppo_steps = saved['updates']
    del saved
    seeds = config['test_seeds']
    geometry = [geometry_and_routing(root, label, seed, env.env)
                for root, label in ((JOINT, 'final'), (JOINT, 'centroid_same_sac'), (PURE, 'pretrain_final'))
                for seed in seeds]
    write_json(OUT / 'geometry.json', geometry)
    result = {'scope': 'Exploratory replay of one final checkpoint on three reused development scenes; no training.',
              'task_clock_semantics': 'Critical last-finishing chunk. Upload includes its queue; transfer/compute stages include queues and deployment pauses. Event clocks use the existing simulator order, not continuous-time physical service decomposition.',
              'geometry_semantics': 'Post-macro snapshots. U2G rates are equal-user geometry estimates, not task-weighted executed throughput. Path excludes initial deployment.',
              'joint_mean': report['mean'], 'pure800_mean': pure_report['mean'],
              'matched_centroid_mean': {k: float(np.mean([row[k] for row in report['matched_centroid']])) for k in report['mean']},
              'ppo_last_update': ppo_stats, 'updates': ppo_steps, 'replays': [], 'geometry': geometry}
    for label, macro, archived in (('final', True, report['final']), ('centroid_same_sac', False, report['matched_centroid'])):
        for seed in seeds:
            with isolated_rng(), patch('hrl_uav.experiment.QueueMDP', ClockAuditMDP):
                summary, _, trajectory, routes, mdp = rollout(config, agent, seed, config['test_macros'],
                                                             macro_enabled=macro, record=True)
            reference = next(row for row in archived if int(row['seed']) == seed)
            metrics = {**summary, **summarize_meter(mdp.readings, summary['time_s'])}
            errors = {key: abs(metrics[key]-reference[key]) for key in report['mean']
                      if key != 'late_queue_slope_mbit_s'}
            assert max(errors.values()) < 1e-8, errors
            with np.load(JOINT / f'{label}_{seed}_trajectory.npz') as old:
                assert np.array_equal(np.array([t[0] for t in trajectory]), old['uav'])
                assert np.array_equal(np.array([t[1] for t in trajectory]), old['users'])
                assert np.array_equal(np.array([r[1] for r in routes]), old['executed_route'])
            table = pd.DataFrame(mdp.task_clocks)
            error = np.max(np.abs(table[list(COMPONENTS)].sum(axis=1)-table.delay_ms))
            assert error < 1e-8
            table.to_csv(OUT / f'{label}_{seed}_task_clocks.csv', index=False, encoding='utf-8-sig')
            weights = np.where(table.priority == 'high', mdp.env.w_D_high, mdp.env.w_D_low)
            components = {'weighted': {key: float(np.average(table[key], weights=weights)) for key in (*COMPONENTS, 'delay_ms')}}
            for priority in ('high', 'low'):
                subset = table[table.priority == priority]
                components[priority] = {key: float(subset[key].mean()) for key in (*COMPONENTS, 'delay_ms')}
            assert abs(components['weighted']['delay_ms']-reference['task_delay_weighted_ms']) < 1e-8
            result['replays'].append({'policy': label, 'seed': seed, 'components': components,
                                      'completed_tasks': len(table), 'reproduction_max_error': max(errors.values()),
                                      'decomposition_max_error': float(error),
                                      'links': summarize_links(mdp.link_readings, summary['time_s'], seed, config['environment']['u2u_bandwidth_mhz'])})
            print(label, seed, components, flush=True)
            write_json(OUT / 'report.json', result)
    result['checkpoint_hash_unchanged'] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == digest
    assert result['checkpoint_hash_unchanged']
    write_json(OUT / 'report.json', result)


if __name__ == '__main__':
    main()
