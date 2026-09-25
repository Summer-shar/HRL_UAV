"""Paired frozen evaluation for the fresh-pretrain joint phase."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from industrial_device_sac_pretrain import experiment as pre
from industrial_device_pilot import experiment as device
from hrl_uav.industrial_deployment_checkpoint import load_checkpoint, restore_checkpoint
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers
from hrl_uav.industrial_hotspot_gate import paired_rows, pooled_metrics


SCHEMA = 'industrial_device_fresh_joint_evaluation_v1'
MAX_PHYSICAL_STEPS = 108000


def load_joint(path):
    saved = load_checkpoint(path)
    p, scene = saved['protocol'], saved['scene']
    pre.validate_protocol(p)
    if saved.get('stage') != 'joint':
        raise ValueError('Joint evaluation requires a joint checkpoint')
    agent = pre.build_fresh(p, scene)
    state = restore_checkpoint(agent, saved, protocol=p, scene=scene,
                               identity=pre.source_identity(p))
    return agent, scene, p, state


def run_row(agent, scene, p, scenario, mode):
    mdp = pre.DeviceMDP(scene, p)
    real = device.reset_scene(mdp, scenario, scenario['profile_permutation'])
    controller = DeploymentLearningControllers(agent, mdp.encoder,
        deterministic=True, macro_enabled=mode == 'moving')
    curve = []
    with torch.no_grad():
        for _ in range(180):
            action, _ = controller.macro(mdp.controller_information())
            movement = mdp.start_macro(action)
            for _ in range(mdp.frames_per_macro):
                mdp.prepare_service()
                mdp.step_frame(controller.micro(mdp.controller_information()))
            curve.append(dict(**device.snapshot(mdp), cost_sum=mdp.env.cost_sum,
                holding_task_s=mdp.env.holding, energy_j=sum(mdp.env.energy.values()),
                raw_action=np.asarray(action).tolist(),
                safety_scale=float(movement['safety_scale'])))
    metrics = mdp.metrics()
    return dict(status='complete', technical_valid=True, frozen_verified=True,
        metrics_frozen_before_priming=True, scenario_id=scenario['scenario_id'],
        scenario_sha=pre.digest(scenario), realization_sha=real['realization_sha256'],
        event_sha256=metrics['event_sha256'], canonical_event_sha256=metrics['canonical_event_sha256'],
        profile_permutation=real['profile_by_region'], mode=mode, metrics=metrics,
        low_unfinished=metrics['priority'][1]['arrived'] - metrics['priority'][1]['completed'],
        physical_return=-metrics['cost_sum'], curve=curve,
        upload_area_mbit_s_5s=float(5 * sum(x['upload_mbits'] for x in curve)))


def evaluate(checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    try:
        agent, scene, p, state = load_joint(checkpoint)
        before = device.frozen_fingerprint(agent)
        identity = pre.source_identity(p)
        checkpoint_sha = pre.file_sha(checkpoint)
        for scenario in p['development_manifest']['scenarios']:
            for mode in ('hover', 'moving'):
                row = run_row(agent, scene, p, scenario, mode)
                rows.append(row)
                pre.atomic_json(output / f"{scenario['scenario_id']}_{mode}.json", row)
                pre.atomic_json(output / 'progress.json', dict(status='running',
                    trajectories=len(rows), physical_steps=len(rows) * 9000))
        if len(rows) != 12 or device.frozen_fingerprint(agent) != before:
            raise AssertionError('Frozen paired evaluation changed learner state')
        moving = [r for r in rows if r['mode'] == 'moving']
        hover = [r for r in rows if r['mode'] == 'hover']
        pairs = paired_rows(moving, hover)
        if any(a['canonical_event_sha256'] != b['canonical_event_sha256'] for a, b in pairs):
            raise AssertionError('Moving and hover events are not paired')
        mp, hp = pooled_metrics(moving), pooled_metrics(hover)
        report = dict(schema=SCHEMA, status='complete', episode=state['completed_episodes'],
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
            protocol_sha256=pre.digest(p), identity=identity, frozen_verified=True,
            physical_steps=MAX_PHYSICAL_STEPS, learning_updates=0, replay_writes=0,
            moving_pooled=mp, hover_pooled=hp,
            cost_ratio=mp['mean_cost_per_s'] / hp['mean_cost_per_s'],
            delay_ratio=mp['weighted_delay_ms'] / hp['weighted_delay_ms'],
            rows=rows, summary=mp, elapsed_seconds=time.perf_counter() - started)
        pre.atomic_json(output / 'report.json', report)
        pre.atomic_json(output / 'progress.json', dict(status='complete',
            physical_steps=MAX_PHYSICAL_STEPS, trajectories=12))
        return report
    except BaseException as exc:
        pre.atomic_json(output / 'failure.json', dict(schema=SCHEMA, status='failed',
            error=repr(exc), completed_trajectories=len(rows), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output)
