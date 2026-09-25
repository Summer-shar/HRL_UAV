"""Frozen hover-only evaluation for fresh SAC pretraining checkpoints."""

from pathlib import Path
import time

import numpy as np
import torch

from industrial_device_sac_pretrain import experiment as e
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers
from hrl_uav.industrial_hotspot_gate import evaluate_admission, pooled_metrics


SCHEMA = 'industrial_device_fresh_hover_evaluation_v1'
MAX_PHYSICAL_STEPS = 54000
MAX_WALL_SECONDS = 900


def trajectory(agent, scene, p, scenario, *, macros=180, reserve=None):
    """Shared rollout used by the full evaluator and bounded functional check."""
    mdp = e.DeviceMDP(scene, p)
    real = e.device.reset_scene(mdp, scenario, scenario['profile_permutation'])
    positions = mdp.env.physics.uav_coords.copy()
    controllers = DeploymentLearningControllers(agent, mdp.encoder,
        deterministic=True, macro_enabled=False)
    curve, discounted, upper_return = [], 0., 0.
    with torch.no_grad():
        for macro_index in range(macros):
            action, _ = controllers.macro(mdp.controller_information())
            if np.any(action):
                raise AssertionError('Evaluation must hover')
            movement = mdp.start_macro(action)
            macro_reward = 0.
            for frame_index in range(mdp.frames_per_macro):
                if reserve:
                    reserve(mdp.env.frame_steps)
                mdp.prepare_service()
                start = mdp.env.step_index
                result = mdp.step_frame(controllers.micro(mdp.controller_information()))
                discounted += sum(mdp.gamma_step ** (start + index) * row['cost']
                                  for index, row in enumerate(result['readings']))
                macro_reward += mdp.gamma_frame ** frame_index * result['reward']
            upper_return += mdp.gamma_macro ** macro_index * macro_reward
            if not np.array_equal(positions, mdp.env.physics.uav_coords):
                raise AssertionError('Frozen hover evaluation moved UAVs')
            curve.append(dict(**e.device.snapshot(mdp), cost_sum=mdp.env.cost_sum,
                holding_task_s=mdp.env.holding, energy_j=sum(mdp.env.energy.values()),
                raw_action=np.asarray(action).tolist(),
                safety_scale=float(movement['safety_scale']), upper_reward=macro_reward))
    if mdp.env.step_index != macros * 50 or not np.isclose(upper_return, -discounted, rtol=1e-12, atol=1e-8):
        raise AssertionError('Evaluation duration or reward accounting changed')
    return _row(mdp, scenario, real, curve, discounted)


def _row(mdp, scenario, realization, curve, discounted):
    metrics = mdp.metrics()
    scale_bits = max(1., mdp.env.generated_bits)
    scale_cycles = max(1., mdp.env.generated_cycles)
    if any(abs(value) > 1e-5 + 1e-9 *
           (scale_cycles if key.endswith('cycles') else scale_bits)
           for key, value in metrics['conservation'].items()):
        raise AssertionError('Task conservation failed')
    return dict(status='complete', technical_valid=True, frozen_verified=True,
        metrics_frozen_before_priming=True, scenario_id=scenario['scenario_id'],
        scenario_sha=e.digest(scenario), realization_sha=realization['realization_sha256'],
        event_sha256=metrics['event_sha256'],
        canonical_event_sha256=metrics['canonical_event_sha256'],
        profile_permutation=realization['profile_by_region'], variant='sac_hover',
        metrics=metrics,
        low_unfinished=metrics['priority'][1]['arrived'] - metrics['priority'][1]['completed'],
        discounted_cost=discounted, physical_return=-metrics['cost_sum'], curve=curve,
        upload_area_mbit_s_5s=float(5 * sum(point['upload_mbits'] for point in curve)),
        learning_updates=0, replay_writes=0)


def evaluate(checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, attempted = [], 0
    before = None
    checkpoint = Path(checkpoint)
    e.atomic_json(output / 'budget.json', dict(schema=SCHEMA,
        maximum_physical_steps=MAX_PHYSICAL_STEPS, maximum_wall_seconds=MAX_WALL_SECONDS,
        trajectories=6, seconds_per_trajectory=900, learning_updates=0,
        replay_writes=0, automatic_retry=False))
    try:
        agent, scene, p, state = e.load_candidate(checkpoint)
        if p['purpose'] != 'research' or state.get('functional_validation_only'):
            raise ValueError('Evaluation requires a research pretraining checkpoint')
        if state['completed_episodes'] not in p['budget']['evaluation_nodes']:
            raise ValueError('Not a registered evaluation node')
        scenarios = p['development_manifest']['scenarios']
        if len(scenarios) != 6 or p['development_manifest']['seconds'] != 900:
            raise ValueError('Evaluation requires six sealed 900 s scenarios')
        before = e.device.frozen_fingerprint(agent)
        identity = e.source_identity(p)
        checkpoint_sha = e.file_sha(checkpoint)
        e.atomic_json(output / 'manifest.json', dict(schema=SCHEMA, status='running',
            episode=state['completed_episodes'], checkpoint_sha256=checkpoint_sha,
            protocol_sha256=e.digest(p), identity=identity))
        def reserve(steps):
            nonlocal attempted
            if time.perf_counter() - started >= MAX_WALL_SECONDS:
                raise TimeoutError('Evaluation wall-clock budget reached')
            if attempted + steps > MAX_PHYSICAL_STEPS:
                raise TimeoutError('Evaluation physical-step budget reached')
            attempted += steps
        for scenario in scenarios:
            row = trajectory(agent, scene, p, scenario, reserve=reserve)
            rows.append(row)
            e.atomic_json(output / f"{scenario['scenario_id']}.json", row)
            e.atomic_json(output / 'progress.json', dict(status='running',
                trajectories=len(rows), physical_steps=attempted))
        if attempted != MAX_PHYSICAL_STEPS:
            raise AssertionError('Incomplete hover evaluation budget')
        after = e.device.frozen_fingerprint(agent)
        if before != after or e.source_identity(p) != identity or e.file_sha(checkpoint) != checkpoint_sha:
            raise AssertionError('Frozen evaluation changed learner state')
        baseline = e.read(e.BASELINE_PATH)
        if (e.file_sha(e.BASELINE_PATH) != p['baseline_report']['sha256'] or
                baseline['scene_sha256'] != p['scene_sha256'] or
                baseline['dev_manifest_sha'] != e.digest(p['development_manifest'])):
            raise ValueError('Cached baseline belongs to another scene or panel')
        gate = evaluate_admission(rows, baseline['local_rows'], baseline['cvx_rows'],
                                  p['admission_rules'])
        pooled = pooled_metrics(rows)
        report = dict(schema=SCHEMA, status='complete', episode=state['completed_episodes'],
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
            protocol_sha256=e.digest(p), identity=identity, frozen_verified=True,
            frozen_fingerprint_before=before, frozen_fingerprint_after=after,
            physical_steps=attempted, learning_updates=0, replay_writes=0,
            summary=pooled, mean_cost_per_s=pooled['mean_cost_per_s'],
            weighted_delay_ms=pooled['weighted_delay_ms'], quality_passed=bool(gate['quality_passed']),
            gate_result=gate, rows=rows, elapsed_seconds=time.perf_counter() - started)
        e.atomic_json(output / 'report.json', report)
        e.atomic_json(output / 'manifest.json', dict(schema=SCHEMA, status='complete',
            physical_steps=attempted, checkpoint_sha256=checkpoint_sha,
            protocol_sha256=e.digest(p), identity=identity,
            frozen_verified=True, report_sha256=e.file_sha(output / 'report.json')))
        e.atomic_json(output / 'progress.json', dict(status='complete', physical_steps=attempted))
        return report
    except BaseException as exc:
        e.atomic_json(output / 'failure.json', dict(schema=SCHEMA, status='failed',
            error=repr(exc), attempted_physical_steps=attempted,
            completed_trajectories=len(rows), automatic_retry=False))
        raise
