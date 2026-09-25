"""Explicit device-observation candidate; legacy production remains loadable.

Transfer the complete existing SAC learner, reset PPO, and continue joint
learning. The only research intervention is PPO's observation/representation.
Use the module entry point from the repository root. No work starts on import.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from industrial_hotspot_analysis.load24725_pilot_20260923 import pilot
from industrial_device_pilot.observation import DeviceObservationEncoder, OBSERVATION_SCHEMA, MACRO_DIM
from industrial_device_pilot.networks import SetActorNet, SetCriticNet, NETWORK_ARCHITECTURE
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers, DeploymentTrainingSchedule
from hrl_uav.industrial_deployment_checkpoint import (make_checkpoint, restore_checkpoint,
    save_checkpoint, load_checkpoint, capture_rng)

SCHEMA = 'industrial_device_joint_pilot_v1'
read, atomic_json, file_sha = pilot.read, pilot.atomic_json, pilot.file_sha
digest, hashed, now, append = pilot.digest, pilot.hashed, pilot.now, pilot.append
frozen_fingerprint, counters = pilot.frozen_fingerprint, pilot.counters
snapshot, reset_scene, seeds_for = pilot.snapshot, pilot.reset_scene, pilot.seeds_for
paired_rows, pooled_metrics = pilot.paired_rows, pilot.pooled_metrics
assert_counts = pilot.assert_counts


def make_protocol(purpose='pilot'):
    baseline = pilot.make_protocol('control')
    p = copy.deepcopy(baseline)
    p.update(schema=SCHEMA, purpose=purpose, arm='device', baseline_protocol=baseline,
        observation_schema=OBSERVATION_SCHEMA, ppo_architecture=NETWORK_ARCHITECTURE,
        ppo_initialization_seed=baseline['training_seed'] + 8700000,
        transfer='Full unchanged SAC learner; fresh PPO device encoders and optimizers; direct joint learning.',
        intervention='PPO per-device raw workload/position observation and shared device encoding only.',
        first_stop_episode=25, automatic_extension=False)
    validate_protocol(p)
    return p


def reference_protocol(p):
    reference = copy.deepcopy(p['baseline_protocol'])
    for key in ('purpose', 'learning', 'budget'):
        reference[key] = copy.deepcopy(p[key])
    return reference


def validate_protocol(p):
    if p.get('schema') != SCHEMA or p.get('arm') != 'device':
        raise ValueError('An explicit device-observation protocol is required')
    if (p.get('observation_schema') != OBSERVATION_SCHEMA or
            p.get('ppo_architecture') != NETWORK_ARCHITECTURE):
        raise ValueError('Observation/architecture contract mismatch')
    baseline = p.get('baseline_protocol', {})
    pilot.validate_protocol(baseline)
    if baseline['arm'] != 'control' or baseline['purpose'] != 'pilot':
        raise ValueError('Candidate must preserve the original control discount')
    pilot.validate_protocol(reference_protocol(p))
    for key in ('parent', 'scene_path', 'scene_sha256', 'randomness', 'runtime',
                'development_manifest', 'episode_semantics', 'training_seed', 'upper_half_life_s'):
        if p.get(key) != baseline[key]:
            raise ValueError('Unregistered change outside PPO representation: ' + key)
    if (p.get('ppo_initialization_seed') != p['training_seed'] + 8700000 or
            p.get('first_stop_episode') != 25 or p.get('automatic_extension') is not False):
        raise ValueError('Initialization/bounded extension contract mismatch')
    if p['purpose'] == 'pilot' and p['budget'] != baseline['budget']:
        raise ValueError('Research budget changed')


class DeviceMDP(pilot.PilotMDP):
    def __init__(self, scene, protocol):
        validate_protocol(protocol)
        super().__init__(scene, reference_protocol(protocol))
        self.encoder = DeviceObservationEncoder(self.config, self.deployment_config)

    def reset(self, *args, **kwargs):
        # Hotspot reset installs its legacy encoder; replace it after realization.
        super().reset(*args, **kwargs)
        self.encoder = DeviceObservationEncoder(self.config, self.deployment_config)
        return self.macro_observation()


def _install_device_ppo(agent, scene, p):
    rng_before = hashed(capture_rng())
    sac_before = hashed(pilot.sac_state(agent))
    device = next(agent.macro_actor.parameters()).device
    if device.type != 'cpu':
        raise ValueError('This paired pilot is registered for CPU only')
    # Architecture initialization must not consume the paired rollout RNG.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(p['ppo_initialization_seed'])
        agent.macro_actor = SetActorNet().to(device)
        agent.macro_critic = SetCriticNet().to(device)
        with torch.no_grad():
            agent.macro_actor.out.weight.zero_()
            agent.macro_actor.out.bias.zero_()
            agent.macro_actor.log_std.fill_(math.log(agent.cfg['ppo_initial_std']))
        agent.macro_actor.log_std.requires_grad_(False)
        agent.ppo_actor_optimizer = torch.optim.Adam(
            agent.macro_actor.parameters(), lr=agent.cfg['ppo_actor_lr'], eps=1e-5)
        agent.ppo_critic_optimizer = torch.optim.Adam(
            agent.macro_critic.parameters(), lr=agent.cfg['ppo_critic_lr'], eps=1e-5)
    mdp = DeviceMDP(scene, p)
    agent.macro_state_dim = MACRO_DIM
    agent.observation_spec = mdp.observation_spec()
    agent.full_config = copy.deepcopy(agent.full_config)
    agent.full_config['schema'] = SCHEMA
    agent.full_config['observation'] = dict(layout=OBSERVATION_SCHEMA,
        ppo_architecture=NETWORK_ARCHITECTURE, macro_dim=MACRO_DIM, micro_dim=86)
    if hashed(capture_rng()) != rng_before or hashed(pilot.sac_state(agent)) != sac_before:
        raise AssertionError('PPO installation changed SAC or paired global RNG')
    return agent


def make_initial_agent(p):
    validate_protocol(p)
    agent, scene, receipt = pilot.make_initial_agent(reference_protocol(p))
    _install_device_ppo(agent, scene, p)
    receipt = copy.deepcopy(receipt)
    receipt['reference_initial_ppo_sha256'] = receipt['initial_ppo_sha256']
    receipt.update(initial_ppo_sha256=hashed([agent.macro_actor.state_dict(), agent.macro_critic.state_dict()]),
        new_protocol_sha256=digest(p), observation_schema=OBSERVATION_SCHEMA,
        ppo_architecture=NETWORK_ARCHITECTURE, ppo_initialization_seed=p['ppo_initialization_seed'],
        paired_global_rng_preserved=True, macro_dim=MACRO_DIM, micro_dim=86)
    return agent, scene, receipt


def build_fresh(p, scene):
    validate_protocol(p)
    return _install_device_ppo(pilot.build_fresh(reference_protocol(p), scene), scene, p)


def source_identity(p):
    identity = pilot.source_identity(reference_protocol(p))
    for name in ('__init__.py', 'observation.py', 'networks.py', 'experiment.py', 'evaluation.py'):
        path = HERE / name
        identity['source_sha256']['industrial_device_pilot/' + name] = file_sha(path)
    return identity


def saved_payload(agent, p, scene, state):
    validate_protocol(p)
    result = make_checkpoint(agent, p, scene, source_identity(p), state)
    result['device_pilot_schema'] = SCHEMA
    return result


def load_candidate(path):
    saved = load_checkpoint(path)
    if saved.get('device_pilot_schema') != SCHEMA:
        raise ValueError('Not a device-observation candidate checkpoint')
    p, scene = saved['protocol'], saved['scene']
    validate_protocol(p)
    if digest(scene) != p['scene_sha256']:
        raise ValueError('Scene identity mismatch')
    agent = build_fresh(p, scene)
    state = restore_checkpoint(agent, saved, protocol=p, scene=scene, identity=source_identity(p))
    return agent, scene, p, state


def describe():
    p = make_protocol()
    mdp = DeviceMDP(read(p['scene_path']), p)
    return dict(schema=SCHEMA, observation=mdp.observation_spec(),
        architecture=NETWORK_ARCHITECTURE, gamma_macro=mdp.gamma_macro,
        gamma_frame=mdp.gamma_frame, first_stop_episode=25, maximum_episodes=50,
        automatic_extension=False, environment_service_steps=0, learning_updates=0,
        source_identity=source_identity(p))


def train(output, acceptance_path, stop_after=25, resume=None):
    """One explicit bounded phase; checkpoint after every complete 900s stream.

    Repeated invocations never silently increase the horizon. A 25->50 request
    requires the completed 25-round paired evaluation and starts a new bounded
    wall-time phase, while all learning/step counters continue unchanged.
    """
    if stop_after not in (25, 50):
        raise ValueError('Only the registered 25/50 evaluation nodes are allowed')
    output = Path(output).absolute()
    acceptance_path = Path(acceptance_path).absolute()
    if resume is None and stop_after != 25:
        raise ValueError('The first phase must stop at 25 for review')
    if resume is None:
        p = make_protocol()
    else:
        p = load_checkpoint(resume)['protocol']
    validate_protocol(p)
    if p['purpose'] != 'pilot':
        raise ValueError('Functional-test weights cannot start research training')
    identity = source_identity(p)
    acceptance = read(acceptance_path)
    if acceptance.get('passed') is not True or acceptance.get('source_identity') != identity:
        raise ValueError('Matching bounded functional verification is required')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing attempts are retained')
    started = time.perf_counter()
    if resume is None:
        agent, scene, receipt = make_initial_agent(p)
        state = dict(at_stream_boundary=True, completed_episodes=0, completed_seconds=0.,
            evaluations=[], transfer=receipt, phase_target=25,
            wall_deadline_utc_epoch=time.time()+p['runtime']['max_wall_seconds_per_job'],
            acceptance_path=str(acceptance_path), acceptance_sha256=file_sha(acceptance_path),
            automatic_extension=False)
    else:
        agent, scene, p, state = load_candidate(resume)
        receipt = state['transfer']
        if stop_after != state['phase_target']:
            if not (state['phase_target'] == 25 and stop_after == 50 and
                    state['completed_episodes'] == 25 and not state.get('pending_evaluation') and
                    [row['episode'] for row in state['evaluations']] == [25]):
                raise ValueError('Extension requires completed 25-round evaluation')
            state['phase_target'] = 50
            state['wall_deadline_utc_epoch'] = time.time()+p['runtime']['max_wall_seconds_per_job']
        if state['completed_episodes'] > stop_after or (
                state['completed_episodes'] == stop_after and not state.get('pending_evaluation')):
            raise ValueError('This phase is already complete')
        state['resumed_from'] = str(Path(resume).absolute())
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('protocol', p), ('scene', scene), ('transfer', receipt)):
        atomic_json(output/(name+'.json'), value)
    manifest = dict(schema=SCHEMA, status='running', arm='device', identity=identity,
        protocol_sha256=digest(p), started_utc=now(), source_directory=str(HERE),
        phase_target=stop_after, training_seed=p['training_seed'], automatic_extension=False)
    mdp = DeviceMDP(scene, p)
    scheduler = DeploymentTrainingSchedule(agent, 'ppo_sac', p,
        on_update=lambda row: append(output/'learning.jsonl', row))
    if resume is not None:
        scheduler.load_state_dict(state['schedule'])

    def remaining():
        value = min(p['runtime']['max_wall_seconds_per_job']-(time.perf_counter()-started),
                    state['wall_deadline_utc_epoch']-time.time())
        if value <= 0:
            raise TimeoutError('Phase wall budget reached; latest completed stream retained')
        return value

    def publish():
        if source_identity(p) != identity:
            raise RuntimeError('Candidate or production source changed during training')
        state.update(schedule=scheduler.state_dict(), at_stream_boundary=True)
        payload = saved_payload(agent, p, scene, state)
        record = save_checkpoint(output/'checkpoints/rolling', payload)
        manifest.update(committed_episodes=state['completed_episodes'], completed_seconds=state['completed_seconds'],
            updates=counters(agent), latest_checkpoint=record, elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/'manifest.json', manifest)
        return record

    def finish_pending_evaluation():
        pending = state.get('pending_evaluation')
        if pending is None:
            return
        episode = pending['episode']
        node_dir = Path(pending['checkpoint_directory'])
        destination = Path(pending['output_directory'])
        checkpoint = node_dir/'final.pt'
        if not checkpoint.exists():
            save_checkpoint(node_dir, saved_payload(agent, p, scene, state), persistent_name='final.pt')
        result_path = destination/'report.json'
        before = frozen_fingerprint(agent)
        if not result_path.exists():
            if destination.exists():
                raise RuntimeError('Partial evaluation retained; no automatic repeated budget')
            atomic_json(output/'progress.json', dict(status='evaluating', committed_episodes=episode,
                phase_target=stop_after, updates=counters(agent)))
            command = [sys.executable, '-B', '-m', 'industrial_device_pilot.experiment', 'evaluate',
                '--checkpoint', str(checkpoint), '--output', str(destination)]
            with (output/f'evaluation-{episode:04d}.stdout.txt').open('x') as stdout, \
                 (output/f'evaluation-{episode:04d}.stderr.txt').open('x') as stderr:
                subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True,
                    timeout=min(remaining(), 1000), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        result = read(result_path)
        if (result['status'] != 'complete' or result['identity'] != identity or
            result['protocol_sha256'] != digest(p) or result['checkpoint_sha256'] != file_sha(checkpoint) or
            result['episode'] != episode or frozen_fingerprint(agent) != before or
            result['physical_steps'] != 108000 or result['learning_updates'] or result['replay_writes']):
            raise AssertionError('Evaluation identity, budget or frozen state mismatch')
        if any(row['episode'] == episode for row in state['evaluations']):
            raise AssertionError('Duplicate evaluation node')
        state['evaluations'].append(dict(episode=episode, path=str(result_path), sha256=file_sha(result_path),
            cost_ratio=result['cost_ratio'], delay_ratio=result['delay_ratio'],
            upload_area_ratio=result['upload_area_ratio'], bad=result['bad_performance_node']))
        state.pop('pending_evaluation')
        publish()

    try:
        finish_pending_evaluation()
        for index in range(state['completed_episodes'], stop_after):
            remaining()
            ep_started = time.perf_counter()
            # Preserve the old 50-round update contract even when pausing at 25.
            scheduler.set_stage('joint', 50*180)
            seeds = seeds_for(p, index)
            realization = reset_scene(mdp, seeds)
            state['at_stream_boundary'] = False
            control = DeploymentLearningControllers(agent, mdp.encoder, deterministic=False)
            trajectory = []
            def boundary(transition):
                trajectory.append(dict(**snapshot(mdp), action=transition.action.tolist(),
                    upper_reward=transition.reward, cost_sum=mdp.env.cost_sum,
                    holding_task_s=mdp.env.holding, energy_j=sum(mdp.env.energy.values())))
                scheduler.macro_boundary(transition)
            rollout = pilot.PilotRollout(mdp, control.macro, control.micro, on_macro_boundary=boundary)
            upper_return = 0.
            for macro in range(180):
                remaining()
                collected = rollout.run(1, truncate=macro == 179, collect_metrics=macro == 179)
                scheduler.completed_micro(collected)
                upper_return += mdp.gamma_macro**macro * collected['macro'][0].reward
                if macro % 60 == 59:
                    atomic_json(output/'progress.json', dict(status='running', committed_episodes=index,
                        active_episode=index+1, episode_simulated_seconds=(macro+1)*5,
                        phase_target=stop_after, elapsed_seconds=time.perf_counter()-started, updates=counters(agent)))
            metrics = collected['metrics']
            state.update(at_stream_boundary=True, completed_episodes=index+1, completed_seconds=(index+1)*900)
            assert_counts(agent, scheduler, receipt)
            if scheduler.physical_steps > p['budget']['max_training_physical_steps']:
                raise AssertionError('Training physical-step budget exceeded')
            row = dict(episode=index+1, arm='device', seeds=seeds,
                realization_sha256=realization['realization_sha256'], profile_by_region=realization['profile_by_region'],
                metrics=metrics, physical_return=-metrics['cost_sum'], upper_discounted_return=upper_return,
                updates=counters(agent), episode_training_seconds=time.perf_counter()-ep_started, trajectory=trajectory)
            state['last_episode'] = row
            if index+1 in (25, 50):
                state['pending_evaluation'] = dict(episode=index+1,
                    checkpoint_directory=str(output/f'checkpoints/episode-{index+1:04d}'),
                    output_directory=str(output/f'evaluations/episode-{index+1:04d}'))
            checkpoint = publish()
            row['checkpoint_sha256'] = checkpoint['sha256']
            append(output/'episodes.jsonl', row)
            if index+1 in (25, 50):
                finish_pending_evaluation()
        expected = [node for node in (25, 50) if node <= stop_after]
        if [row['episode'] for row in state['evaluations']] != expected:
            raise AssertionError('Required evaluation nodes are missing')
        manifest.update(status='phase_complete', quality_status='review_required')
        publish()
        atomic_json(output/'progress.json', dict(status='phase_complete', committed_episodes=stop_after,
            updates=counters(agent), evaluations=state['evaluations'], automatic_extension=False,
            elapsed_seconds=time.perf_counter()-started))
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed',
            error=repr(exc), elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/'manifest.json', manifest)
        atomic_json(output/'progress.json', dict(status=manifest['status'], committed_episodes=state['completed_episodes'],
            error=repr(exc), automatic_extension=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    sub.add_parser('describe')
    run = sub.add_parser('train')
    run.add_argument('--output', required=True)
    run.add_argument('--acceptance', required=True)
    run.add_argument('--stop-after', type=int, choices=(25, 50), default=25)
    run.add_argument('--checkpoint')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.phase == 'describe':
        print(json.dumps(describe(), ensure_ascii=False, allow_nan=False, indent=2))
    elif args.phase == 'train':
        train(args.output, args.acceptance, args.stop_after, args.checkpoint)
    else:
        from industrial_device_pilot.evaluation import evaluate
        evaluate(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
