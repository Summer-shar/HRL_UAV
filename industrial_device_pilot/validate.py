"""Bounded candidate acceptance; no test learner is eligible for research.

Counts include all attempts in the shared validation directory. A killed attempt
without a final accounting record conservatively consumes its full reservation.
"""
import argparse
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np


def load_helpers():
    path = ROOT / 'industrial_hotspot_analysis/load24725_pilot_20260923/validate_pilot.py'
    spec = importlib.util.spec_from_file_location('device_validation_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helpers()
require, digest, write_json = H.require, H.digest, H.write_json


def reset(mdp, index):
    return mdp.reset(941001 + index, 942001 + index, 943001 + index)


def observation_checks(e, scene, p):
    import torch
    from industrial_device_pilot.networks import SetActorNet, SetCriticNet
    mdp = e.DeviceMDP(scene, p)
    reset(mdp, 0)
    view = mdp.controller_information()
    encoder = mdp.encoder
    require(encoder.spec()['macro']['dim'] == 523, 'Wrong macro dimension')
    require(encoder.spec()['micro']['dim'] == 86, 'Wrong micro dimension')
    # Synthetic reports: legal encoder inputs, not a claim of realized history.
    terminals = list(view.base.terminals)
    rates = np.asarray(view.base.uplink_rate_bps)
    positions = np.asarray([x.position_m for x in terminals[:6]])
    matrix = np.stack([np.ones(6), 1 / rates[:6], positions[:, 0] / rates[:6], positions[:, 1] / rates[:6]])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(matrix, full_matrices=True)
    null = vt[-1] / np.max(np.abs(vt[-1]))
    views = []
    for sign in (-1, 1):
        changed = terminals.copy()
        for i in range(6):
            changed[i] = replace(changed[i], upload_bits=(200000. + sign * 100000. * null[i], 0.),
                                 pending_cycles=(1e9, 0.), pending_count=(1., 0.))
        views.append(replace(view, base=replace(view.base, terminals=tuple(changed))))
    old_a, old_b = [encoder.legacy_encoder.encode_macro(v) for v in views]
    a, b = [encoder.encode_macro(v) for v in views]
    require(np.max(np.abs(old_a - old_b)) < 1e-7, 'Synthetic old-observation alias construction failed')
    require(np.max(np.abs(a - b)) > .01, 'Device association remains aliased in new input')
    changed_rate = replace(views[0], base=replace(views[0].base,
        uplink_rate_bps=tuple(float(x * .5) for x in rates)))
    require(np.array_equal(a, encoder.encode_macro(changed_rate)), 'PPO depends on radio rate')
    require(not np.array_equal(encoder.encode_micro(views[0]), encoder.encode_micro(changed_rate)),
            'SAC pressure rate dependency unexpectedly removed')
    unavailable_rate = replace(views[0], base=replace(views[0].base,
        uplink_rate_bps=tuple(float('nan') for _ in rates)))
    require(np.array_equal(a, encoder.encode_macro(unavailable_rate)), 'PPO rejects unavailable radio rates')
    try:
        encoder.encode_micro(unavailable_rate)
    except ValueError:
        pass
    else:
        raise AssertionError('Legacy SAC unexpectedly accepts unavailable radio rates')
    reordered = tuple(t for source in range(3) for t in reversed(views[0].base.terminals)
                      if t.source == source)
    permuted = replace(views[0], base=replace(views[0].base, terminals=reordered))
    pa = encoder.encode_macro(permuted)
    require(not np.array_equal(a, pa), 'Permutation probe did not alter device slots')
    with torch.random.fork_rng():
        torch.manual_seed(91531)
        actor, critic = SetActorNet(), SetCriticNet()
        require(bool(actor.out.weight.abs().sum() > 0), 'Zero actor head would mask permutation errors')
        inputs = torch.as_tensor(np.stack([a, pa, b]), dtype=torch.float32)
        with torch.no_grad():
            mean, std = actor(inputs)
            value = critic(inputs)
            embeddings = actor.pooled_features(inputs)
        mean_error = float((mean[0] - mean[1]).abs().max())
        value_error = float((value[0] - value[1]).abs().max())
        require(mean_error < 2e-6 and value_error < 2e-6, 'Set networks depend on device ordering')
        require(torch.equal(std[0], std[1]), 'Standard deviation depends on ordering')
        association_embedding_difference = float((embeddings[0] - embeddings[2]).abs().max())
        require(association_embedding_difference > 1e-6, 'Probe set encoder lost constructed association')
    invalid_results = []
    for label, bad in (
        ('duplicate_id', replace(views[0], base=replace(views[0].base,
            terminals=(views[0].base.terminals[1],) + views[0].base.terminals[1:]))),
        ('negative_work', replace(views[0], base=replace(views[0].base,
            terminals=(replace(views[0].base.terminals[0], upload_bits=(-1., 0.)),) + views[0].base.terminals[1:]))),
        ('stale_report', replace(views[0], base=replace(views[0].base,
            terminals=(replace(views[0].base.terminals[0], sample_time_s=-1.),) + views[0].base.terminals[1:])))):
        try:
            encoder.encode_macro(bad)
        except ValueError:
            invalid_results.append(label)
        else:
            raise AssertionError('Invalid report accepted: ' + label)
    reset(mdp, 1)
    require(mdp.macro_observation().shape == (523,), 'Reset reverted to legacy encoder')
    return dict(passed=True, macro_dim=523, micro_dim=86,
        old_alias_max_abs_difference=float(np.max(np.abs(old_a - old_b))),
        new_alias_max_abs_difference=float(np.max(np.abs(a - b))),
        nonzero_actor_permutation_max_abs_error=mean_error, value_permutation_max_abs_error=value_error,
        association_embedding_max_abs_difference=association_embedding_difference,
        invalid_reports_rejected=invalid_results, synthetic_not_environment_reachability_proof=True)


def collect_script(mdp, rollout_class):
    reset(mdp, 2)
    readings = []
    original = mdp.step_frame
    def step(scores):
        result = original(scores)
        readings.extend(copy.deepcopy(result['readings']))
        return result
    mdp.step_frame = step
    rollout = rollout_class(mdp, H.script_motion, H.script_route)
    segments = [rollout.run(1, truncate=i == 3) for i in range(4)]
    return dict(segments=segments, readings=readings, view=mdp.controller_information(),
                steps=mdp.env.step_index, macro_calls=rollout.macro_calls, micro_calls=rollout.micro_calls)


def rollout_checks(e, scene, p):
    legacy = e.pilot.PilotMDP(scene, e.pilot.make_protocol('control'))
    candidate = e.DeviceMDP(scene, p)
    old = collect_script(legacy, e.pilot.PilotRollout)
    new = collect_script(candidate, e.pilot.PilotRollout)
    for key in ('readings', 'view', 'steps', 'macro_calls', 'micro_calls'):
        require(digest(old[key]) == digest(new[key]), 'Physical path changed: ' + key)
    errors = []
    for oi, ni in zip(old['segments'], new['segments']):
        require(digest(oi['micro']) == digest(ni['micro']), 'SAC tuples changed')
        require(digest(oi['metrics']) == digest(ni['metrics']), 'Metrics changed')
        x, y = oi['macro'][0], ni['macro'][0]
        for key in ('action', 'reward', 'discount', 'trace_continue', 'truncated', 'logprob'):
            require(digest(getattr(x, key)) == digest(getattr(y, key)), 'Macro non-observation field changed: ' + key)
        expected = sum(candidate.gamma_frame ** j * t.reward for j, t in enumerate(ni['micro']))
        error = abs(expected - y.reward)
        require(error < 1e-10, 'Macro reward is not discounted lower reward sum')
        errors.append(error)
        require(y.state.shape == y.next_state.shape == (523,), 'PPO tuple has wrong observation width')
    return dict(passed=True, paths=2, steps_per_path=200, total_physical_steps=400,
        physical_sha256=digest(old['readings']), reward_identity_max_abs_error=max(errors),
        gamma_step=candidate.gamma_step, gamma_frame=candidate.gamma_frame, gamma_macro=candidate.gamma_macro)


def episode(e, agent, scene, protocol, scheduler, index, macros):
    from hrl_uav.industrial_deployment_training import DeploymentLearningControllers
    mdp = e.DeviceMDP(scene, protocol)
    reset(mdp, index)
    controller = DeploymentLearningControllers(agent, mdp.encoder, deterministic=False)
    transitions, order = [], []
    def macro(view):
        order.append(('action', mdp.env.step_index, agent.ppo_behavior_version))
        return controller.macro(view)
    def boundary(t):
        scheduler.macro_boundary(t)
        order.append(('boundary', mdp.env.step_index, agent.ppo_behavior_version))
    rollout = e.pilot.PilotRollout(mdp, macro, controller.micro, on_macro_boundary=boundary)
    started = time.perf_counter()
    for i in range(macros):
        collected = rollout.run(1, truncate=i == macros - 1, collect_metrics=i == macros - 1)
        scheduler.completed_micro(collected)
        if macros <= 4:
            transitions.append(collected)
    for i, value in enumerate(order[:-1]):
        if value[0] == 'boundary':
            require(order[i + 1][0] == 'action' and order[i + 1][2] == value[2],
                    'Action sampled before latest PPO update')
    return dict(transitions=transitions, order=order, metrics=collected['metrics'],
                physical_steps=mdp.env.step_index), time.perf_counter() - started


def continuation_check(e, out):
    from hrl_uav.industrial_deployment_training import DeploymentTrainingSchedule
    from hrl_uav.industrial_deployment_checkpoint import restore_checkpoint, save_checkpoint
    p = e.make_protocol(purpose='functional')
    p['learning']['ppo_update_macros'] = 3
    p['learning']['ppo_epochs'] = 1
    agent, scene, receipt = e.make_initial_agent(p)
    scheduler = DeploymentTrainingSchedule(agent, 'ppo_sac', p)
    scheduler.set_stage('joint', 100000)
    first, _ = episode(e, agent, scene, p, scheduler, 3, 2)
    require(len(agent.macro_rollout) == 2 and agent.macro_rollout[-1][-1] == 0., 'Pending truncated PPO samples missing')
    state = dict(at_stream_boundary=True, functional=True, completed_episodes=1,
                 completed_seconds=10, schedule=scheduler.state_dict())
    payload = e.saved_payload(agent, p, scene, state)
    invalid = []
    for field in ('protocol', 'scene', 'identity', 'config', 'observations', 'guard_state'):
        bad = dict(payload)
        bad[field] = {}
        before = digest(H.learner_state(agent))
        try:
            restore_checkpoint(agent, bad, protocol=p, scene=scene, identity=payload['identity'])
        except (ValueError, AssertionError):
            invalid.append(field)
        else:
            raise AssertionError('Invalid checkpoint accepted: ' + field)
        require(before == digest(H.learner_state(agent)), 'Rejected checkpoint mutated learner: ' + field)
    bad = dict(payload, replay=dict(payload['replay']))
    bad['replay'].pop('rng_state')
    before = digest(H.learner_state(agent))
    try:
        restore_checkpoint(agent, bad, protocol=p, scene=scene, identity=payload['identity'])
    except ValueError:
        invalid.append('replay_rng')
    else:
        raise AssertionError('Incomplete replay accepted')
    require(before == digest(H.learner_state(agent)), 'Rejected replay mutated learner')
    saved = save_checkpoint(out / 'short_checkpoints', payload)
    start_hash = digest(H.learner_state(agent))
    payload_hash = digest(payload)
    reference, _ = episode(e, agent, scene, p, scheduler, 4, 2)
    end_hash = digest(H.learner_state(agent))
    expected_schedule = scheduler.state_dict()
    restored, scene2, p2, state2 = e.load_candidate(saved['path'])
    require(digest(H.learner_state(restored)) == start_hash, 'Loading changed full learner or RNG')
    require(scene2 == scene and p2 == p and state2 == state, 'Loading changed contract or cursor')
    schedule2 = DeploymentTrainingSchedule(restored, 'ppo_sac', p2)
    schedule2.load_state_dict(state2['schedule'])
    actual, _ = episode(e, restored, scene2, p2, schedule2, 4, 2)
    require(digest(actual) == digest(reference), 'Restored physical path or transitions differ')
    require(digest(H.learner_state(restored)) == end_hash, 'Actual updates after restore differ')
    require(digest(payload) == payload_hash, 'Continued optimizer mutated saved payload')
    for key, value in expected_schedule.items():
        if key not in ('sac_seconds', 'ppo_seconds'):
            require(schedule2.state_dict()[key] == value, 'Restored clock differs: ' + key)
    require(restored.ppo_updates == 1 and len(restored.macro_rollout) == 1, 'Wrong short PPO update count')
    require(restored.last_ppo_stats['accepted_actor_steps'] >= 1, 'Short PPO had no accepted action-network step')
    return dict(passed=True, transfer_receipt=receipt, saved=saved, physical_steps=300,
        first_trace=[r['macro'][0].trace_continue for r in first['transitions']],
        learner_hash_before=start_hash, learner_hash_after=end_hash,
        actual_sac_updates=restored.sac_updates, ppo_updates=restored.ppo_updates,
        pending_ppo_samples=len(restored.macro_rollout), invalid_contracts_rejected_without_mutation=invalid)


def normal_two_episodes(e, out):
    from hrl_uav.industrial_deployment_training import DeploymentTrainingSchedule
    from hrl_uav.industrial_deployment_checkpoint import save_checkpoint
    p = e.make_protocol(purpose='functional')
    agent, scene, receipt = e.make_initial_agent(p)
    require(p['learning']['ppo_update_macros'] == 300 and p['learning']['ppo_epochs'] == 10,
            'Normal-config test unexpectedly changed learning settings')
    scheduler = DeploymentTrainingSchedule(agent, 'ppo_sac', p)
    scheduler.set_stage('joint', 100000)
    initial = e.pilot.counters(agent)
    rows = []
    for index in range(2):
        result, seconds = episode(e, agent, scene, p, scheduler, 100 + index, 180)
        row = dict(episode=index + 1, elapsed_training_seconds=seconds,
            metrics=result['metrics'], physical_steps=result['physical_steps'],
            updates=e.pilot.counters(agent), pending_ppo_samples=len(agent.macro_rollout))
        rows.append(row)
        write_json(out / ('normal_episode_%02d.json' % (index + 1)), row)
        print(json.dumps(dict(stage='normal_episode_complete', **{k: row[k] for k in ('episode', 'elapsed_training_seconds', 'updates', 'pending_ppo_samples')})), flush=True)
    require(agent.ppo_updates == 1 and len(agent.macro_rollout) == 60, 'Normal 300-sample PPO contract failed')
    require(agent.last_ppo_stats['batch_size'] == 300 and agent.last_ppo_stats['accepted_actor_steps'] >= 1,
            'Normal PPO batch did not execute an accepted action-network update')
    require(scheduler.frames == 3600 and scheduler.physical_steps == 18000, 'Wrong normal physical/update clocks')
    require(agent.sac_updates - initial['sac'] == 180, 'Wrong normal SAC update count')
    state = dict(at_stream_boundary=True, functional=True, completed_episodes=2,
                 completed_seconds=1800, schedule=scheduler.state_dict())
    saved = save_checkpoint(out / 'normal_checkpoints', e.saved_payload(agent, p, scene, state), persistent_name='final.pt')
    return dict(passed=True, test_weights_and_replay_excluded_from_research=True,
                transfer_receipt=receipt, initial_updates=initial, final_updates=e.pilot.counters(agent),
                ppo_stats=agent.last_ppo_stats, pending_ppo_samples=60,
                physical_steps=18000, episodes=rows, checkpoint=saved,
                seconds_per_episode_mean=float(np.mean([r['elapsed_training_seconds'] for r in rows])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).absolute()
    out.mkdir(parents=True, exist_ok=False)
    budget = dict(physical_steps_max=22000, optimizer_step_calls_max=800,
                  all_attempts_included=True, test_weights_and_replay_excluded_from_research=True)
    prior = dict(physical_steps=0, optimizer_step_calls=0)
    for previous in out.parent.glob('*/budget.json'):
        if previous.parent == out:
            continue
        item = json.loads(previous.read_text(encoding='utf-8'))
        if item.get('status') == 'finished':
            for key in prior:
                prior[key] += item['counts'][key]
        else:
            prior['physical_steps'] += item['reservation']['physical_steps_max']
            prior['optimizer_step_calls'] += item['reservation']['optimizer_step_calls_max']
    require(prior['physical_steps'] < 22000 and prior['optimizer_step_calls'] < 800, 'Prior attempts exhaust budget')
    counts = dict(physical_steps=0, optimizer_step_calls=0)
    write_json(out / 'preregister.json', dict(**budget, intended_physical_steps=18700,
        checks=['observation', 'paired_script_4_macros', 'short_real_learning_and_resume', 'normal_two_episodes'],
        no_research_training=True))
    write_json(out / 'budget.json', dict(status='reserved', reservation=budget, prior_counts=prior, counts=counts))
    started = time.perf_counter()
    import torch
    from industrial_device_pilot import experiment as e
    from hrl_uav.industrial_env import IndustrialEnv
    physical_original, adam_original = IndustrialEnv.micro_step, torch.optim.Adam.step
    def physical(self, *a, **kw):
        require(counts['physical_steps'] + prior['physical_steps'] < 22000, 'Physical budget exhausted')
        counts['physical_steps'] += 1
        return physical_original(self, *a, **kw)
    def adam(self, *a, **kw):
        require(counts['optimizer_step_calls'] + prior['optimizer_step_calls'] < 800, 'Optimizer budget exhausted')
        counts['optimizer_step_calls'] += 1
        return adam_original(self, *a, **kw)
    IndustrialEnv.micro_step, torch.optim.Adam.step = physical, adam
    test_source_sha = e.file_sha(__file__)
    helper_source = ROOT / 'industrial_hotspot_analysis/load24725_pilot_20260923/validate_pilot.py'
    helper_source_sha = e.file_sha(helper_source)
    report = dict(passed=False, counts=counts, prior_attempt_counts=prior, budget=budget,
                  test_source_sha256=test_source_sha, validation_helper_sha256=helper_source_sha)
    try:
        p = e.make_protocol(purpose='functional')
        identity = e.source_identity(p)
        agent, scene, receipt = e.make_initial_agent(p)
        probe_mdp = e.DeviceMDP(scene, p)
        probe_obs = reset(probe_mdp, 99)
        with torch.no_grad():
            mean0, std0 = agent.macro_actor(torch.as_tensor(probe_obs).unsqueeze(0))
        require(torch.equal(mean0, torch.zeros_like(mean0)), 'Candidate initial action mean is not zero')
        require(np.allclose(std0.numpy(), p['learning']['ppo_initial_std'], atol=1e-7, rtol=0),
                'Candidate initial exploration differs')
        require(not agent.macro_actor.log_std.requires_grad, 'PPO log_std unexpectedly trainable')
        for network, optimizer in ((agent.macro_actor, agent.ppo_actor_optimizer),
                                   (agent.macro_critic, agent.ppo_critic_optimizer)):
            params = {id(x) for x in network.parameters()}
            bound = {id(x) for g in optimizer.param_groups for x in g['params']}
            require(params == bound and not optimizer.state, 'PPO optimizer is stale or inherited')
        require(agent._macro_gamma == probe_mdp.gamma_macro, 'PPO gamma is stale')
        before = digest(H.learner_state(agent))
        report['observations'] = observation_checks(e, scene, p)
        require(before == digest(H.learner_state(agent)), 'Read-only observation probes changed learner/RNG')
        report['initial_transfer'] = receipt
        del agent
        report['rollout'] = rollout_checks(e, scene, p)
        print(json.dumps(dict(stage='static_and_rollout_passed', counts=counts)), flush=True)
        report['continuation'] = continuation_check(e, out)
        print(json.dumps(dict(stage='continuation_passed', counts=counts)), flush=True)
        report['normal_two_episodes'] = normal_two_episodes(e, out)
        require(identity == e.source_identity(p), 'Source changed during acceptance')
        require(e.file_sha(__file__) == test_source_sha and e.file_sha(helper_source) == helper_source_sha,
                'Validation code changed during acceptance')
        report['source_identity'] = identity
        report['passed'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        IndustrialEnv.micro_step, torch.optim.Adam.step = physical_original, adam_original
        report['elapsed_seconds'] = time.perf_counter() - started
        write_json(out / 'verification.json', report)
        write_json(out / 'budget.json', dict(status='finished', reservation=budget, prior_counts=prior, counts=counts))
        print(json.dumps(dict(passed=report['passed'], counts=counts, elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
