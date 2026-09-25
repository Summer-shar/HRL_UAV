"""Continue completed fresh-SAC joint training from episode 25 to episode 50.

The parent stored the pretraining protocol even during joint training. Preserve
that legacy learner contract for exact restoration; an explicit joint_run_contract
in the run state, manifest and contract.json records the actual extended budget.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

from industrial_device_joint_fresh import experiment as joint
from industrial_device_joint_fresh.evaluation import load_joint
from industrial_device_sac_pretrain import experiment as pre
from hrl_uav.industrial_deployment_checkpoint import _same
from hrl_uav.industrial_deployment_training import DeploymentTrainingSchedule

TARGET = 50


def runner_hashes():
    return {name: pre.file_sha(Path(__file__).parent / name)
            for name in ('__init__.py', 'experiment.py', 'evaluation.py', 'continue50.py')}


def prepare(source, output):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Existing continuation output is preserved')
    agent, scene, p, state = load_joint(source)
    saved = pre.load_checkpoint(source)
    if (state.get('completed_episodes') != 25 or state.get('pending_evaluation') or
            [r['episode'] for r in state['evaluations']] != [25] or
            state.get('functional_validation_only')):
        raise ValueError('A completed research joint-25 checkpoint with evaluated state is required')
    report_entry = state['evaluations'][0]
    report_path = Path(report_entry['report_path'])
    report = pre.read(report_path)
    if (pre.file_sha(report_path) != report_entry['report_sha256'] or
            report['status'] != 'complete' or report['episode'] != 25 or
            report['physical_steps'] != 108000):
        raise ValueError('Parent evaluation is incomplete or changed')
    pre.assert_config(agent, p)
    # Compare every persisted learner field; temporary gradients are not saved.
    restored = joint.payload(agent, p, scene, state)
    if not _same(restored, {k: v for k, v in saved.items() if k != '_load_metadata'}):
        raise AssertionError('Parent checkpoint restoration changed persisted state')
    schedule = DeploymentTrainingSchedule(agent, 'ppo_sac', p)
    schedule.load_state_dict(state['schedule'])
    joint.assert_counts(agent, schedule, 25)
    if schedule.physical_steps != 225000 or agent.ppo_updates != 15:
        raise AssertionError('Incorrect joint-25 clocks')
    fingerprint = pre.device.frozen_fingerprint(agent)
    schedule.set_stage('joint', TARGET * 180)
    if pre.device.frozen_fingerprint(agent) != fingerprint:
        raise AssertionError('Budget extension changed learning state')
    contract = dict(schema='fresh_joint_continuation_25_to_50_v1',
        source_checkpoint=saved['_load_metadata'], start_episode=25, target_episode=TARGET,
        extra_training_physical_steps=225000, extra_evaluation_physical_steps=108000,
        episode_seconds=900, evaluation_nodes=[50], wall_seconds=3600,
        task_flow='Continue existing joint runner seed indices 25..49 without reseeding learner RNG',
        automatic_extension=False, automatic_retry=False,
        inherited_protocol_sha256=pre.digest(p), runner_sha256=runner_hashes())
    output.mkdir(parents=True, exist_ok=False)
    state.update(joint_run_contract=contract, resumed_from=saved['_load_metadata'],
                 base_counters=pre.counters(agent), base_sac_target_updates=agent.sac_target_updates)
    state['schedule'] = schedule.state_dict()
    receipt = pre.save_checkpoint(output / 'prepared', joint.payload(agent, p, scene, state),
                                  persistent_name='final.pt')
    again, _, _, again_state = load_joint(receipt['persistent_path'])
    if pre.device.frozen_fingerprint(again) != fingerprint or not _same(state, again_state):
        raise AssertionError('Prepared checkpoint round trip failed')
    for name, value in (('contract', contract), ('protocol', p), ('scene', scene)):
        pre.atomic_json(output / (name + '.json'), value)
    pre.atomic_json(output / 'preflight.json', dict(passed=True, additional_service_steps=0,
        additional_learning_updates=0, parent_exact_restore=True, prepared_exact_restore=True,
        actual_optimizer_lrs=pre.actual_lrs(again), counters=pre.counters(again),
        prepared_checkpoint=receipt, identity=pre.source_identity(p), contract=contract))
    return receipt


def check_counts(agent, schedule, state, p):
    joint.assert_counts(agent, schedule, state['completed_episodes'])
    frames = (state['completed_episodes'] - 25) * 1800
    events = frames // p['learning']['sac_update_frames']
    base = state['base_counters']
    delay = p['learning']['sac_policy_delay']
    if (agent.replay.memory_counter != base['replay_writes'] + frames or
            agent.sac_updates != base['sac'] + events or
            agent.sac_critic_gradient_steps != base['sac_critic_fits'] +
                events * p['learning']['sac_critic_steps_per_update'] or
            agent.sac_target_updates != state['base_sac_target_updates'] + events or
            agent.sac_actor_updates != base['sac_actor'] +
                (base['sac'] + events) // delay - base['sac'] // delay or
            agent.ppo_updates != schedule.macros // p['learning']['ppo_update_macros']):
        raise AssertionError('Continuation learning clocks disagree')
    pre.assert_config(agent, p)


def train(output):
    output = Path(output).absolute()
    preflight = pre.read(output / 'preflight.json')
    if (not preflight['passed'] or preflight['contract']['runner_sha256'] != runner_hashes() or
            (output / 'manifest.json').exists()):
        raise ValueError('Requires unconsumed matching preparation')
    agent, scene, p, state = load_joint(preflight['prepared_checkpoint']['persistent_path'])
    identity, hashes = pre.source_identity(p), runner_hashes()
    schedule = DeploymentTrainingSchedule(agent, 'ppo_sac', p,
        on_update=lambda row: pre.append(output / 'learning.jsonl', row))
    schedule.load_state_dict(state['schedule'])
    check_counts(agent, schedule, state, p)
    started = time.perf_counter()
    manifest = dict(status='running', contract=state['joint_run_contract'],
        identity=identity, started_utc=pre.now())
    pre.atomic_json(output / 'manifest.json', manifest)
    mdp = pre.DeviceMDP(scene, p)

    def check():
        if time.perf_counter() - started > 3600:
            raise TimeoutError('Continuation wall budget reached')

    def publish():
        if pre.source_identity(p) != identity or runner_hashes() != hashes:
            raise RuntimeError('Sources changed during continuation')
        state.update(schedule=schedule.state_dict(), at_stream_boundary=True)
        receipt = pre.save_checkpoint(output / 'checkpoints/rolling',
                                       joint.payload(agent, p, scene, state))
        manifest.update(committed_episodes=state['completed_episodes'],
            latest_checkpoint=receipt, updates=pre.counters(agent),
            actual_optimizer_lrs=pre.actual_lrs(agent), elapsed_seconds=time.perf_counter() - started)
        pre.atomic_json(output / 'manifest.json', manifest)
        return receipt

    try:
        for index in range(25, TARGET):
            state['at_stream_boundary'] = False
            def heartbeat(macros):
                pre.atomic_json(output / 'progress.json', dict(status='running',
                    committed_episodes=index, active_episode=index + 1, episode_seconds=macros * 5,
                    target_episodes=TARGET, updates=pre.counters(agent),
                    actual_optimizer_lrs=pre.actual_lrs(agent), elapsed_seconds=time.perf_counter() - started))
            row = joint.train_episode(agent, mdp, p, schedule, index, heartbeat=heartbeat, wall_check=check)
            state.update(completed_episodes=index + 1, completed_seconds=(index + 1) * 900,
                         last_episode=row)
            check_counts(agent, schedule, state, p)
            receipt = publish()
            pre.append(output / 'episodes.jsonl', row)
            pre.atomic_json(output / 'progress.json', dict(status='running',
                committed_episodes=index + 1, target_episodes=TARGET, updates=pre.counters(agent),
                latest_checkpoint=receipt, elapsed_seconds=time.perf_counter() - started))
        destination = output / 'evaluations/episode-0050'
        state['pending_evaluation'] = dict(episode=50, output_directory=str(destination))
        endpoint = pre.save_checkpoint(output / 'checkpoints/episode-0050',
            joint.payload(agent, p, scene, state), persistent_name='final.pt')
        pre.atomic_json(output / 'progress.json', dict(status='evaluating',
            committed_episodes=50, target_episodes=TARGET, updates=pre.counters(agent)))
        before = pre.device.frozen_fingerprint(agent)
        with (output / 'evaluation.stdout.txt').open('x') as stdout, (output / 'evaluation.stderr.txt').open('x') as stderr:
            subprocess.run([sys.executable, '-B', '-m', 'industrial_device_joint_fresh.evaluation',
                '--checkpoint', endpoint['persistent_path'], '--output', str(destination)],
                cwd=joint.ROOT, stdout=stdout, stderr=stderr, check=True,
                timeout=min(1000, max(1, 3600 - (time.perf_counter() - started))),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        report = pre.read(destination / 'report.json')
        if (report['status'] != 'complete' or report['episode'] != 50 or
                report['identity'] != identity or report['protocol_sha256'] != pre.digest(p) or
                report['checkpoint_sha256'] != endpoint['sha256'] or
                report['physical_steps'] != 108000 or report['learning_updates'] or report['replay_writes'] or
                before != pre.device.frozen_fingerprint(agent)):
            raise AssertionError('Frozen evaluation integrity check failed')
        state['evaluations'].append(dict(episode=50, report_path=str(destination / 'report.json'),
            report_sha256=pre.file_sha(destination / 'report.json'), **report['summary']))
        state['pending_evaluation'] = None
        manifest['status'] = 'complete'
        receipt = publish()
        pre.atomic_json(output / 'progress.json', dict(status='complete', committed_episodes=50,
            target_episodes=TARGET, updates=pre.counters(agent), latest_checkpoint=receipt,
            evaluation=dict(moving=report['moving_pooled'], hover=report['hover_pooled']),
            elapsed_seconds=time.perf_counter() - started))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc), automatic_retry=False)
        pre.atomic_json(output / 'manifest.json', manifest)
        pre.atomic_json(output / 'progress.json', dict(status='failed', error=repr(exc),
            committed_episodes=state['completed_episodes'], at_stream_boundary=state['at_stream_boundary']))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'train'])
    parser.add_argument('--source')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        print(prepare(args.source, args.output))
    else:
        train(args.output)
