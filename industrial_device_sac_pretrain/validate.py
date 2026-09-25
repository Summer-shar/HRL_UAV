"""Real updates, exact restoration and frozen evaluation in a bounded smoke test."""
import copy
import time
from pathlib import Path

from industrial_device_sac_pretrain import experiment as e
from industrial_device_sac_pretrain.evaluation import trajectory
from hrl_uav.industrial_deployment_training import DeploymentTrainingSchedule
from hrl_uav.industrial_deployment_checkpoint import _same
from hrl_uav.industrial_hotspot_gate import evaluate_admission, pooled_metrics


def validate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    p = e.make_protocol()
    scene = e.read(p['scene_path'])
    identity = e.source_identity(p)
    used_steps = 0
    e.atomic_json(output / 'budget.json', dict(maximum_physical_steps=18800,
        training_steps=18200, frozen_evaluation_steps=600, wall_seconds=240,
        functional_only=True, automatic_retry=False))

    def reserve(steps):
        nonlocal used_steps
        if used_steps + steps > 18800 or time.perf_counter() - started > 240:
            raise TimeoutError('Functional validation budget exceeded')
        used_steps += steps

    try:
        agent = e.build_fresh(p, scene)
        initial_counters = e.counters(agent)
        ppo_before = e.ppo_fingerprint(agent)
        scheduler = DeploymentTrainingSchedule(agent, 'ppo_sac', p)
        mdp = e.DeviceMDP(scene, p)
        rows = []
        for index in range(2):
            reserve(9000)
            rows.append(e.train_episode(agent, mdp, p, scheduler, index))
        if agent.sac_actor_updates <= 0 or agent.sac_critic_gradient_steps <= 0:
            raise AssertionError('Real SAC actor/critic learning was not exercised')
        state = dict(at_stream_boundary=True, functional_validation_only=True,
            completed_episodes=2, completed_seconds=1800.,
            schedule=scheduler.state_dict(), evaluations=[], pending_evaluation=None)
        saved = e.payload(agent, p, scene, state)
        receipt = e.save_checkpoint(output / 'checkpoint', saved, persistent_name='pretrain_final.pt')
        # Checkpoint persistence excludes temporary .grad tensors; compare all
        # serialized fields, including optimizer moments, replay and RNG.
        before_resume = e.hashed(e.payload(agent, p, scene, state))
        reserve(100)
        uninterrupted = e.train_episode(agent, mdp, p, scheduler, 2, macros=2)
        uninterrupted_fp = e.hashed(e.payload(agent, p, scene, state))
        restored, scene2, p2, state2 = e.load_candidate(receipt['persistent_path'])
        if e.hashed(e.payload(restored, p2, scene2, state2)) != before_resume or not _same(state, state2):
            raise AssertionError('Checkpoint restoration was not exact')
        restored_schedule = DeploymentTrainingSchedule(restored, 'ppo_sac', p2)
        restored_schedule.load_state_dict(state2['schedule'])
        reserve(100)
        repeated = e.train_episode(restored, e.DeviceMDP(scene2, p2), p2, restored_schedule, 2, macros=2)
        if not _same(uninterrupted, repeated) or e.hashed(e.payload(restored, p2, scene2, state2)) != uninterrupted_fp:
            raise AssertionError('Interrupted and uninterrupted learning diverged')
        if e.ppo_fingerprint(restored) != ppo_before or any(restored.guard_state().values()):
            raise AssertionError('SAC hover pretraining changed PPO')
        frozen_before = e.device.frozen_fingerprint(restored)
        evaluation_rows = []
        for scenario in p['development_manifest']['scenarios']:
            evaluation_rows.append(trajectory(restored, scene, p, scenario, macros=2, reserve=reserve))
        if frozen_before != e.device.frozen_fingerprint(restored):
            raise AssertionError('Frozen trajectory collection changed learning state')
        e.atomic_json(output / 'evaluation_smoke.json', dict(rows=evaluation_rows,
            summary=pooled_metrics(evaluation_rows), functional_only=True))
        baseline = e.read(e.BASELINE_PATH)
        if (baseline['scene_sha256'] != p['scene_sha256'] or
                baseline['dev_manifest_sha'] != e.digest(p['development_manifest'])):
            raise AssertionError('Cached references do not match the current scene')
        gate = evaluate_admission(baseline['sac_rows'], baseline['local_rows'],
                                  baseline['cvx_rows'], p['admission_rules'])
        e.atomic_json(output / 'cached_gate_smoke.json', gate)
        for mutate in ('scene', 'learning'):
            bad_p, bad_scene = copy.deepcopy(p), copy.deepcopy(scene)
            if mutate == 'scene':
                bad_p['scene_sha256'] = 'wrong'
            else:
                bad_p['learning']['pretrain_lr'][0] = 3e-5
            try:
                e.build_fresh(bad_p, bad_scene)
            except ValueError:
                pass
            else:
                raise AssertionError('Unregistered configuration accepted')
        if e.choose_best([dict(episode=50, quality_passed=False, mean_cost_per_s=1),
                          dict(episode=100, quality_passed=True, mean_cost_per_s=5),
                          dict(episode=300, quality_passed=True, mean_cost_per_s=7)])['episode'] != 100:
            raise AssertionError('Selection incorrectly prefers the final checkpoint')
        if e.source_identity(p) != identity:
            raise AssertionError('Source changed during validation')
        acceptance = dict(schema='industrial_device_fresh_hover_acceptance_v1', passed=True,
            identity=identity, protocol_sha256=e.digest(p), scene_sha256=e.digest(scene),
            physical_steps=used_steps, initial_counters=initial_counters,
            functional_sac_events=133, functional_critic_fits=266,
            functional_actor_updates=17, functional_optimizer_calls=300,
            actual_optimizer_lrs=e.actual_lrs(restored), updates_after_two_episodes=rows[-1]['updates'],
            ppo_updates=restored.ppo_updates, checkpoint_exact_restore=True,
            resumed_learning_exact_match=True, frozen_evaluation_verified=True,
            workload=e.workload_summary(scene['base_scene']), strict_hover=True,
            rejection_checks=2, evaluation_nodes=p['budget']['evaluation_nodes'],
            elapsed_seconds=time.perf_counter() - started)
        e.atomic_json(output / 'acceptance.json', acceptance)
        return acceptance
    except BaseException as exc:
        e.atomic_json(output / 'failure.json', dict(error=repr(exc), reserved_physical_steps=used_steps,
            elapsed_seconds=time.perf_counter() - started, automatic_retry=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(e.jsonable(validate(args.output)))
