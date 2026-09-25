"""Fresh, bounded hover SAC pretraining. No inherited weights or replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from industrial_device_pilot import experiment as device
from hrl_uav.industrial_deployment_checkpoint import (
    make_checkpoint, restore_checkpoint, save_checkpoint, load_checkpoint, capture_rng)
from hrl_uav.industrial_deployment_experiment import ppo_fingerprint
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers, DeploymentTrainingSchedule
from hrl_uav.industrial_scene import workload_summary

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCHEMA = 'industrial_device_fresh_hover_sac_v1'
SOURCE_RUN = ROOT / 'industrial_hotspot_records/load24725_seed1_20260923/training'
REFERENCE_PATH = ROOT / 'industrial_hotspot_records/load24725_device_lr_continue_20260924/phase75/protocol.json'
BASELINE_PATH = SOURCE_RUN / 'evaluations/pretrain-0200/report.json'
NODES = [50, 100, 150, 200, 250, 300]
read, digest, hashed, file_sha = device.read, device.digest, device.hashed, device.file_sha
now, counters = device.now, device.counters


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def atomic_json(path, value):
    """Retry only the same atomic file publication after transient Windows locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(11):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 10:
                raise
            time.sleep(0.05 * (attempt + 1))


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + '\n')


def make_protocol():
    reference = read(REFERENCE_PATH)['reference_device_protocol']
    original = read(SOURCE_RUN / 'protocol.json')
    return contract(reference, original)


def contract(reference, original):
    device.validate_protocol(reference)
    learning = copy.deepcopy(reference['learning'])
    learning['pretrain_lr'] = [1e-5, 3e-4, 3e-5]
    return dict(schema=SCHEMA, purpose='research', training_seed=1,
        reference_device_protocol=copy.deepcopy(reference),
        original_v4_protocol=copy.deepcopy(original),
        scene_sha256=reference['scene_sha256'], scene_path=reference['scene_path'],
        observation_schema=reference['observation_schema'], macro_dim=523, micro_dim=86,
        learning=learning, runtime=dict(device='cpu', torch_threads=1, max_wall_seconds_per_job=14400),
        budget=dict(episode_seconds=900, pretrain_episodes=300, evaluation_nodes=NODES,
            max_training_physical_steps=2700000, max_evaluation_physical_steps=324000,
            joint_episodes=0, automatic_extension=False, automatic_retry=False),
        randomness=copy.deepcopy(original['randomness']['pretrain']),
        training_stride=original['randomness']['training_stride'],
        development_manifest=copy.deepcopy(original['development_manifest']),
        admission_rules=copy.deepcopy(original['admission']),
        baseline_report=dict(path=str(BASELINE_PATH), sha256=file_sha(BASELINE_PATH)),
        initialization='Fresh SAC networks, targets, temperature, empty Adam and replay; fresh inactive 523D PPO.',
        selection=dict(eligibility='Original single-checkpoint Local usability rules; no convergence claim.',
            primary='minimum pooled mean physical cost per second among eligible evaluated nodes',
            tie_break='earliest episode', final_test_used=False, auto_joint=False),
        episode_semantics=copy.deepcopy(original['episode_semantics']))


def validate_protocol(p):
    if p.get('schema') != SCHEMA or p != contract(p['reference_device_protocol'], p['original_v4_protocol']):
        raise ValueError('Unregistered pretraining contract change')


class DeviceMDP(device.DeviceMDP):
    def __init__(self, scene, p):
        # Reuse the current observation and physical transition only. The
        # reference protocol's transfer entry point is never called.
        super().__init__(scene, p['reference_device_protocol'])


def actual_lrs(agent):
    return [[float(g['lr']) for g in opt.param_groups] for opt in (
        agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer,
        agent.ppo_actor_optimizer, agent.ppo_critic_optimizer)]


def assert_config(agent, p):
    if (agent.cfg != agent.full_config['training'] or
            agent.cfg['pretrain_lr'] != p['learning']['pretrain_lr'] or
            actual_lrs(agent) != [[1e-5], [3e-4], [3e-5], [3e-4], [1e-3]] or
            agent.macro_state_dim != 523 or agent.micro_state_dim != 86):
        raise AssertionError('Effective configuration, optimizer rates or dimensions changed')


def build_fresh(p, scene):
    validate_protocol(p)
    if digest(scene) != p['scene_sha256']:
        raise ValueError('Incorrect scene for fresh pretraining')
    load = workload_summary(scene['base_scene'])['total_compute_gcycles_s']
    if not np.isclose(load, 24.725, rtol=0, atol=1e-9):
        raise ValueError('Expected the current 24.725 Gcycles/s scene')
    agent = device.build_fresh(p['reference_device_protocol'], scene)
    # This constructor has loaded no weights, optimizer, replay or training RNG.
    agent.cfg['pretrain_lr'] = copy.deepcopy(p['learning']['pretrain_lr'])
    agent.full_config['training']['pretrain_lr'] = copy.deepcopy(p['learning']['pretrain_lr'])
    agent.full_config['schema'] = SCHEMA
    for group in agent.sac_actor.optimizer.param_groups:
        group['lr'] = 1e-5
    if any(counters(agent).values()) or agent.macro_rollout or any(agent.guard_state().values()):
        raise AssertionError('Fresh initialization inherited learning counters')
    if any(opt.state for opt in (agent.sac_actor.optimizer, agent.sac_critic.optimizer,
            agent.entropy.optimizer, agent.ppo_actor_optimizer, agent.ppo_critic_optimizer)):
        raise AssertionError('Fresh initialization inherited optimizer state')
    if agent.stage != 'pretrain':
        raise AssertionError('Fresh learner must start in pretrain')
    assert_config(agent, p)
    return agent


def source_identity(p):
    identity = device.source_identity(p['reference_device_protocol'])
    for name in ('__init__.py', 'experiment.py', 'evaluation.py', 'validate.py'):
        identity['source_sha256']['industrial_device_sac_pretrain/' + name] = file_sha(HERE / name)
    return identity


def payload(agent, p, scene, state):
    assert_config(agent, p)
    return make_checkpoint(agent, p, scene, source_identity(p), state)


def load_candidate(path):
    saved = load_checkpoint(path)
    p, scene = saved['protocol'], saved['scene']
    validate_protocol(p)
    if saved['identity'] != source_identity(p) or saved['stage'] != 'pretrain':
        raise ValueError('Pretraining source identity or stage differs')
    agent = build_fresh(p, scene)
    state = restore_checkpoint(agent, saved, protocol=p, scene=scene, identity=source_identity(p))
    assert_config(agent, p)
    return agent, scene, p, state


def seeds_for(p, index):
    return {name + '_seed': p['randomness'][name + '_base'] +
        p['training_seed'] * p['training_stride'] + index
        for name in ('arrival', 'layout', 'load', 'routing')}


def train_episode(agent, mdp, p, scheduler, index, *, macros=180, heartbeat=None, wall_check=None):
    seeds = seeds_for(p, index)
    realization = device.reset_scene(mdp, seeds)
    positions = mdp.env.physics.uav_coords.copy()
    scheduler.set_stage('pretrain', 300 * 180)
    control = DeploymentLearningControllers(agent, mdp.encoder, deterministic=False, macro_enabled=False)
    curve = []
    def boundary(transition):
        if np.any(transition.action) or not np.array_equal(positions, mdp.env.physics.uav_coords):
            raise AssertionError('Hover pretraining moved a UAV')
        curve.append(dict(**device.snapshot(mdp), reward=float(transition.reward)))
        scheduler.macro_boundary(transition)
    rollout = device.pilot.PilotRollout(mdp, control.macro, control.micro, on_macro_boundary=boundary)
    total_return = 0.
    for macro in range(macros):
        if wall_check:
            wall_check()
        collected = rollout.run(1, truncate=macro == macros - 1, collect_metrics=macro == macros - 1)
        scheduler.completed_micro(collected)
        total_return += mdp.gamma_macro ** macro * collected['macro'][0].reward
        if heartbeat and macro % 60 == 59:
            heartbeat(macro + 1)
    assert_pretrain_counters(agent, scheduler, p, mdp.scene_wrapper)
    assert_config(agent, p)
    metrics = collected['metrics']
    return dict(stage='pretrain', episode=index + 1, seconds=macros * 5, seeds=seeds,
        realization_sha256=realization['realization_sha256'], profile_by_region=realization['profile_by_region'],
        metrics=metrics, physical_return=-metrics['cost_sum'], discounted_return=total_return,
        trajectory=curve, updates=counters(agent), truncated=True, terminated=False)


def assert_pretrain_counters(agent, scheduler, p, scene):
    """Check counters for the fresh hover-only contract."""
    frames_per_macro = int(round(scene['base_scene']['time']['motion_s'] /
                                 scene['base_scene']['time']['routing_s']))
    physical_per_frame = int(round(scene['base_scene']['time']['routing_s'] /
                                   scene['base_scene']['time']['physical_s']))
    if scheduler.frames != scheduler.macros * frames_per_macro:
        raise AssertionError('Hover policy and service clocks disagree')
    if scheduler.physical_steps != scheduler.frames * physical_per_frame:
        raise AssertionError('Hover physical-step count mismatch')
    if scheduler.stage_macros['joint'] or scheduler.ppo_samples or scheduler.consumed_ppo_samples:
        raise AssertionError('PPO was sampled during SAC pretraining')
    settings = p['learning']
    first = int(np.ceil(max(settings['batch_size'], settings['replay_warmup']) /
                       settings['sac_update_frames'])) * settings['sac_update_frames']
    expected = max(0, (scheduler.frames - first) // settings['sac_update_frames'] + 1)
    expected_actor = max(0, expected // settings['sac_policy_delay'] -
                         settings['sac_actor_warmup_updates'] // settings['sac_policy_delay'])
    if (agent.sac_updates != expected or
            agent.sac_critic_gradient_steps != expected * settings['sac_critic_steps_per_update'] or
            agent.sac_actor_updates != expected_actor or
            agent.replay.memory_counter != scheduler.frames):
        raise AssertionError('SAC pretraining lifecycle counter mismatch')


def choose_best(evaluations):
    eligible = [r for r in evaluations if r['quality_passed']]
    return min(eligible, key=lambda r: (r['mean_cost_per_s'], r['episode'])) if eligible else None


def train(output, acceptance_path, resume=None):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('New output directory required; existing attempts are retained')
    p = make_protocol() if resume is None else load_checkpoint(resume)['protocol']
    validate_protocol(p)
    identity = source_identity(p)
    acceptance_path = Path(acceptance_path).absolute()
    acceptance = read(acceptance_path)
    if (acceptance.get('passed') is not True or acceptance.get('identity') != identity or
            acceptance.get('protocol_sha256') != digest(p)):
        raise ValueError('Matching functional acceptance required')
    if resume is None:
        scene = read(p['scene_path'])
        agent = build_fresh(p, scene)
        state = dict(at_stream_boundary=True, completed_episodes=0, completed_seconds=0.,
            evaluations=[], pending_evaluation=None, ppo_initial_sha=ppo_fingerprint(agent),
            initial_counters=counters(agent), initial_replay_size=0, parent_checkpoint=None,
            acceptance_path=str(acceptance_path), acceptance_sha256=file_sha(acceptance_path),
            deadline_epoch=time.time() + p['runtime']['max_wall_seconds_per_job'])
    else:
        agent, scene, p, state = load_candidate(resume)
        if state.get('functional_validation_only') or state['completed_episodes'] >= 300 and not state['pending_evaluation']:
            raise ValueError('Functional or already completed checkpoint cannot resume research')
        state['resumed_from'] = str(Path(resume).absolute())
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for name, value in (('protocol', p), ('scene', scene), ('initialization', state)):
        atomic_json(output / (name + '.json'), value)
    manifest = dict(schema=SCHEMA, status='running', identity=identity, started_utc=now(),
        protocol_sha256=digest(p), scene_sha256=digest(scene), actual_optimizer_lrs=actual_lrs(agent),
        expected_workload=workload_summary(scene['base_scene']), budget=p['budget'])
    mdp = DeviceMDP(scene, p)
    scheduler = DeploymentTrainingSchedule(agent, 'ppo_sac', p,
        on_update=lambda row: append(output / 'learning.jsonl', row))
    if resume:
        scheduler.load_state_dict(state['schedule'])

    def remaining():
        seconds = state['deadline_epoch'] - time.time()
        if seconds <= 0:
            raise TimeoutError('Registered wall limit reached; no automatic extension')
        return seconds

    def publish():
        if source_identity(p) != identity:
            raise RuntimeError('Sources changed while training')
        state.update(schedule=scheduler.state_dict(), at_stream_boundary=True)
        result = save_checkpoint(output / 'checkpoints/rolling', payload(agent, p, scene, state))
        manifest.update(committed_episodes=state['completed_episodes'], latest_checkpoint=result,
            updates=counters(agent), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / 'manifest.json', manifest)
        return result

    def evaluate_pending():
        pending = state['pending_evaluation']
        if not pending:
            return
        episode = pending['episode']
        node = Path(pending['checkpoint_directory'])
        checkpoint = node / 'pretrain_final.pt'
        if not checkpoint.exists():
            save_checkpoint(node, payload(agent, p, scene, state), persistent_name='pretrain_final.pt')
        destination = Path(pending['output_directory'])
        report_path = destination / 'report.json'
        before = device.frozen_fingerprint(agent)
        if not report_path.exists():
            if destination.exists():
                raise RuntimeError('Partial evaluation retained; automatic simulation retry disabled')
            atomic_json(output / 'progress.json', dict(status='evaluating', committed_episodes=episode,
                target_episodes=300, updates=counters(agent)))
            with (output / f'eval-{episode:04d}.stdout.txt').open('x') as stdout, \
                    (output / f'eval-{episode:04d}.stderr.txt').open('x') as stderr:
                subprocess.run([sys.executable, '-B', '-m', 'industrial_device_sac_pretrain.experiment',
                    'evaluate', '--checkpoint', str(checkpoint), '--output', str(destination)],
                    cwd=ROOT, stdout=stdout, stderr=stderr, check=True, timeout=min(remaining(), 1000),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        report = read(report_path)
        if (report['status'] != 'complete' or report['identity'] != identity or
                report['protocol_sha256'] != digest(p) or report['checkpoint_sha256'] != file_sha(checkpoint) or
                report['episode'] != episode or report['physical_steps'] != 54000 or
                report['learning_updates'] or report['replay_writes'] or before != device.frozen_fingerprint(agent)):
            raise AssertionError('Frozen evaluation did not satisfy the sealed contract')
        if episode in [r['episode'] for r in state['evaluations']]:
            raise AssertionError('Duplicate development evaluation')
        state['evaluations'].append(dict(episode=episode, report_path=str(report_path),
            report_sha256=file_sha(report_path), checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint),
            quality_passed=report['quality_passed'], **report['summary']))
        state['pending_evaluation'] = None
        atomic_json(output / 'selection.json', dict(status='provisional' if episode < 300 else 'complete',
            rule=p['selection'], evaluations=state['evaluations'], selected=choose_best(state['evaluations'])))
        publish()

    try:
        if resume is None:
            publish()
        evaluate_pending()
        for index in range(state['completed_episodes'], 300):
            remaining()
            state['at_stream_boundary'] = False
            ep_start = time.perf_counter()
            def heartbeat(macros):
                atomic_json(output / 'progress.json', dict(status='running', committed_episodes=index,
                    active_episode=index + 1, episode_seconds=macros * 5, target_episodes=300,
                    updates=counters(agent), actual_optimizer_lrs=actual_lrs(agent),
                    elapsed_seconds=time.perf_counter() - started))
            row = train_episode(agent, mdp, p, scheduler, index, heartbeat=heartbeat, wall_check=remaining)
            if (ppo_fingerprint(agent) != state['ppo_initial_sha'] or any(agent.guard_state().values())
                    or scheduler.ppo_samples or scheduler.stage_macros['joint'] or agent.macro_rollout):
                raise AssertionError('Pretraining changed PPO')
            if scheduler.physical_steps != (index + 1) * 9000 or scheduler.physical_steps > 2700000:
                raise AssertionError('Training budget or episode length changed')
            row['wall_seconds'] = time.perf_counter() - ep_start
            state.update(completed_episodes=index + 1, completed_seconds=(index + 1) * 900, last_episode=row)
            if index + 1 in NODES:
                state['pending_evaluation'] = dict(episode=index + 1,
                    checkpoint_directory=str(output / f'checkpoints/episode-{index+1:04d}'),
                    output_directory=str(output / f'evaluations/episode-{index+1:04d}'))
            receipt = publish()
            append(output / 'episodes.jsonl', row)
            atomic_json(output / 'progress.json', dict(status='running', committed_episodes=index + 1,
                target_episodes=300, updates=counters(agent), latest_checkpoint=receipt,
                actual_optimizer_lrs=actual_lrs(agent), elapsed_seconds=time.perf_counter() - started))
            evaluate_pending()
        if [r['episode'] for r in state['evaluations']] != NODES:
            raise AssertionError('Missing development nodes')
        manifest['status'] = 'complete'
        final = publish()
        atomic_json(output / 'progress.json', dict(status='complete', committed_episodes=300,
            target_episodes=300, updates=counters(agent), latest_checkpoint=final,
            selected=choose_best(state['evaluations']), joint_started=False,
            elapsed_seconds=time.perf_counter() - started))
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed', error=repr(exc))
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'progress.json', dict(status=manifest['status'], error=repr(exc),
            committed_episodes=state['completed_episodes'], automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    run = sub.add_parser('train')
    run.add_argument('--output', required=True)
    run.add_argument('--acceptance', required=True)
    run.add_argument('--checkpoint')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.phase == 'train':
        train(args.output, args.acceptance, args.checkpoint)
    else:
        from .evaluation import evaluate
        evaluate(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
