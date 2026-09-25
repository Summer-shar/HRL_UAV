"""Bounded joint PPO/SAC training from a fresh SAC hover checkpoint."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

from industrial_device_sac_pretrain import experiment as pre
from industrial_device_pilot import experiment as device
from hrl_uav.industrial_deployment_checkpoint import make_checkpoint, restore_checkpoint, save_checkpoint, load_checkpoint
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers, DeploymentTrainingSchedule

ROOT = Path(__file__).resolve().parents[1]
MAX_EPISODES = 25


def load_joint(path):
    saved = load_checkpoint(path)
    p, scene = saved['protocol'], saved['scene']
    pre.validate_protocol(p)
    if saved.get('stage') != 'pretrain':
        raise ValueError('Joint start must come from a completed hover-pretraining checkpoint')
    agent = pre.build_fresh(p, scene)
    state = restore_checkpoint(agent, saved, protocol=p, scene=scene,
                               identity=pre.source_identity(p))
    if state.get('completed_episodes', 0) not in p['budget']['evaluation_nodes']:
        raise ValueError('Source checkpoint is not a completed registered pretraining node')
    agent.set_stage('joint')
    return agent, scene, p, state


def payload(agent, p, scene, state):
    state = copy.deepcopy(state)
    state['at_stream_boundary'] = True
    return make_checkpoint(agent, p, scene, pre.source_identity(p), state)


def train_episode(agent, mdp, p, scheduler, index, *, heartbeat=None, wall_check=None):
    seeds = pre.seeds_for(p, index)
    realization = device.reset_scene(mdp, seeds)
    control = DeploymentLearningControllers(agent, mdp.encoder, deterministic=False, macro_enabled=True)
    trajectory = []

    def boundary(transition):
        trajectory.append(dict(**device.snapshot(mdp), action=transition.action.tolist(),
            upper_reward=float(transition.reward), cost_sum=mdp.env.cost_sum,
            holding_task_s=mdp.env.holding, energy_j=sum(mdp.env.energy.values())))
        scheduler.macro_boundary(transition)

    rollout = device.pilot.PilotRollout(mdp, control.macro, control.micro,
                                         on_macro_boundary=boundary)
    upper_return = 0.
    for macro in range(180):
        if wall_check:
            wall_check()
        collected = rollout.run(1, truncate=macro == 179, collect_metrics=macro == 179)
        scheduler.completed_micro(collected)
        upper_return += mdp.gamma_macro ** macro * collected['macro'][0].reward
        if heartbeat and macro % 60 == 59:
            heartbeat(macro + 1)
    return dict(stage='joint', episode=index + 1, seconds=900, seeds=seeds,
        realization_sha256=realization['realization_sha256'],
        profile_by_region=realization['profile_by_region'], metrics=collected['metrics'],
        physical_return=-collected['metrics']['cost_sum'], discounted_return=upper_return,
        trajectory=trajectory, updates=pre.counters(agent), truncated=True, terminated=False)


def assert_counts(agent, scheduler, completed_episodes):
    if scheduler.stage != 'joint' or scheduler.stage_macros['pretrain']:
        raise AssertionError('Joint stage clock is invalid')
    if scheduler.frames != scheduler.macros * 10 or scheduler.physical_steps != scheduler.frames * 5:
        raise AssertionError('Joint physical clock mismatch')
    if scheduler.ppo_samples != scheduler.consumed_ppo_samples + len(agent.macro_rollout):
        raise AssertionError('PPO behavior batch accounting mismatch')
    if scheduler.macros != completed_episodes * 180:
        raise AssertionError('Joint episode accounting mismatch')


def train(output, source, acceptance):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Use a new joint output directory')
    acceptance = pre.read(acceptance)
    agent, scene, p, source_state = load_joint(source)
    identity = pre.source_identity(p)
    if acceptance.get('passed') is not True or acceptance.get('identity') != identity:
        raise ValueError('Pretraining acceptance does not match the source checkpoint')
    output.mkdir(parents=True)
    started = time.perf_counter()
    state = dict(at_stream_boundary=True, stage='joint', completed_episodes=0,
        completed_seconds=0., evaluations=[], pending_evaluation=None,
        source_checkpoint=str(Path(source).absolute()),
        source_checkpoint_sha256=pre.file_sha(source), pretrain_state=source_state)
    for name, value in (('protocol', p), ('scene', scene), ('initialization', state)):
        pre.atomic_json(output / f'{name}.json', value)
    manifest = dict(schema='industrial_device_fresh_joint_v1', status='running',
        source_checkpoint=str(Path(source).absolute()), source_checkpoint_sha256=pre.file_sha(source),
        identity=identity, protocol_sha256=pre.digest(p), target_episodes=MAX_EPISODES)
    pre.atomic_json(output / 'manifest.json', manifest)
    mdp = pre.DeviceMDP(scene, p)
    scheduler = DeploymentTrainingSchedule(agent, 'ppo_sac', p,
        on_update=lambda row: pre.append(output / 'learning.jsonl', row))
    scheduler.set_stage('joint', MAX_EPISODES * 180)
    deadline = time.time() + p['runtime']['max_wall_seconds_per_job']

    def check():
        if time.time() >= deadline:
            raise TimeoutError('Joint phase wall-clock budget reached')

    def publish():
        state['schedule'] = scheduler.state_dict()
        state['stage'] = 'joint'
        record = save_checkpoint(output / 'checkpoints' / 'rolling', payload(agent, p, scene, state))
        manifest.update(committed_episodes=state['completed_episodes'], updates=pre.counters(agent),
                        actual_optimizer_lrs=pre.actual_lrs(agent), latest_checkpoint=record,
                        elapsed_seconds=time.perf_counter() - started)
        pre.atomic_json(output / 'manifest.json', manifest)
        return record

    try:
        for index in range(MAX_EPISODES):
            check()

            def heartbeat(macros):
                pre.atomic_json(output / 'progress.json', dict(status='running',
                    committed_episodes=index, active_episode=index + 1,
                    episode_seconds=macros * 5, target_episodes=MAX_EPISODES,
                    updates=pre.counters(agent), actual_optimizer_lrs=pre.actual_lrs(agent),
                    elapsed_seconds=time.perf_counter() - started))

            row = train_episode(agent, mdp, p, scheduler, index,
                                heartbeat=heartbeat, wall_check=check)
            state.update(completed_episodes=index + 1, completed_seconds=(index + 1) * 900,
                         last_episode=row)
            assert_counts(agent, scheduler, index + 1)
            record = publish()
            row['checkpoint_sha256'] = record['sha256']
            pre.append(output / 'episodes.jsonl', row)
            pre.atomic_json(output / 'progress.json', dict(status='running',
                committed_episodes=index + 1, target_episodes=MAX_EPISODES,
                updates=pre.counters(agent), latest_checkpoint=record,
                actual_optimizer_lrs=pre.actual_lrs(agent), elapsed_seconds=time.perf_counter() - started))

        state['pending_evaluation'] = dict(
            checkpoint_directory=str(output / 'checkpoints' / 'episode-0025'),
            output_directory=str(output / 'evaluations' / 'episode-0025'))
        endpoint = save_checkpoint(output / 'checkpoints' / 'episode-0025',
                                   payload(agent, p, scene, state), persistent_name='final.pt')
        pre.atomic_json(output / 'progress.json', dict(status='evaluating', committed_episodes=25,
            latest_checkpoint=endpoint, updates=pre.counters(agent)))
        destination = Path(state['pending_evaluation']['output_directory'])
        subprocess.run([sys.executable, '-B', '-m', 'industrial_device_joint_fresh.evaluation',
            '--checkpoint', endpoint['persistent_path'], '--output', str(destination)],
            cwd=ROOT, check=True, timeout=min(1000, max(1, int(deadline - time.time()))),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        state['pending_evaluation'] = None
        report = pre.read(destination / 'report.json')
        state['evaluations'] = [dict(episode=25, report_path=str(destination / 'report.json'),
            report_sha256=pre.file_sha(destination / 'report.json'), **report['summary'])]
        final = save_checkpoint(output / 'checkpoints' / 'rolling', payload(agent, p, scene, state))
        manifest.update(status='complete', committed_episodes=25, latest_checkpoint=final,
                        evaluation=report['summary'], elapsed_seconds=time.perf_counter() - started)
        pre.atomic_json(output / 'manifest.json', manifest)
        pre.atomic_json(output / 'selection.json', dict(status='complete',
            selected=state['evaluations'][0], evaluation=report))
        pre.atomic_json(output / 'progress.json', dict(status='complete', committed_episodes=25,
            target_episodes=25, updates=pre.counters(agent), latest_checkpoint=final,
            actual_optimizer_lrs=pre.actual_lrs(agent), elapsed_seconds=time.perf_counter() - started))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc), elapsed_seconds=time.perf_counter() - started)
        pre.atomic_json(output / 'manifest.json', manifest)
        pre.atomic_json(output / 'progress.json', dict(status='failed', error=repr(exc),
            committed_episodes=state['completed_episodes']))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--acceptance', required=True)
    args = parser.parse_args()
    train(args.output, args.source, args.acceptance)
