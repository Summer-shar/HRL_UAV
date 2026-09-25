"""Short real-learning fixtures, never research training or performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hrl_uav.industrial_deployment_checkpoint import _same
from hrl_uav.industrial_experiment import digest
from hrl_uav.industrial_hotspot_gate import evaluate_admission
from hrl_uav.industrial_hotspot_experiment import (
    load_protocol, source_hashes, train_job, make_agent)
from hrl_uav.industrial_hotspot_checkpoint import load_checkpoint, restore_checkpoint
from hrl_uav.industrial_hotspot_mdp import HotspotIndustrialMDP

ROOT = Path(__file__).resolve().parents[1]


def fixture_protocol():
    protocol, scene = load_protocol('configs/industrial_hotspot_v4.json', ROOT)
    protocol['purpose'] = 'gate'
    protocol['budget'].update(episode_seconds=5, stream_seconds_alias=5,
        log_window_seconds=5, pretrain_max_episodes=2, pretrain_check_episodes=[1, 2],
        joint_episodes=2, joint_development_check_episodes=[])
    protocol['learning'].update(batch_size=8, replay_capacity=12, replay_warmup=10,
        sac_update_frames=10, sac_actor_warmup_updates=0,
        sac_critic_steps_per_update=1, ppo_update_macros=2, ppo_epochs=1)
    protocol['development_manifest']['seconds'] = 5
    protocol['final_test_manifest']['seconds'] = 5
    return protocol, scene


def synthetic_dispatch(kind, checkpoint, output, root, cache_root, budget_path):
    """Only inject engineering gate measurements under an explicit gate protocol."""
    if kind != 'admission':
        raise AssertionError('Unexpected evaluation in this learning fixture')
    saved = load_checkpoint(checkpoint)
    protocol, scene = saved['protocol'], saved['scene']
    if protocol['purpose'] != 'gate':
        raise AssertionError('Synthetic results are forbidden in research runs')
    candidate_sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

    def rows(cost, low_completed):
        return [dict(status='complete', technical_valid=True, frozen_verified=True,
            metrics_frozen_before_priming=True, scenario_id=case['scenario_id'],
            scenario_sha=digest(case), realization_sha='a' * 64,
            event_sha256='b' * 64, profile_permutation=case['profile_permutation'],
            low_unfinished=10-low_completed, metrics=dict(seconds=5, cost_sum=cost,
                holding_task_s=10, energy_j=dict(cpu=1, propulsion=2, upload=1, u2u=1),
                priority=[dict(arrived=10, completed=10, delay_sum_s=1),
                          dict(arrived=10, completed=low_completed, delay_sum_s=2)]))
            for case in protocol['development_manifest']['scenarios']]

    sac, local, cvx = rows(80, 8), rows(100, 5), rows(75, 8)
    gate = evaluate_admission(sac, local, cvx, protocol['admission'])
    hashes = source_hashes(root)
    report = dict(schema='industrial_hotspot_candidate_evaluation_v1',
        fixture_only=True, status='complete', candidate_sha=candidate_sha,
        protocol_sha256=digest(protocol), scene_sha256=digest(scene),
        source_sha256=hashes, dev_manifest_sha=digest(protocol['development_manifest']),
        identity=dict(candidate_sha=candidate_sha, scene_contract_sha=digest(scene),
            dev_manifest_sha=digest(protocol['development_manifest']),
            admission_rule_version=protocol['admission']['version'],
            evaluator_source_sha=digest(hashes)),
        sac_rows=sac, local_rows=local, cvx_rows=cvx,
        gate_result=gate, quality_passed=gate['quality_passed'])
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    path = path / 'report.json'
    path.write_text(json.dumps(report, sort_keys=True, allow_nan=False), encoding='utf-8')
    return dict(report, report_path=str(path),
        evaluation_file_sha=hashlib.sha256(path.read_bytes()).hexdigest())


class HotspotLearningTests(unittest.TestCase):
    def _run(self, output, **kwargs):
        protocol, scene = fixture_protocol()
        return train_job(protocol, scene, 'ppo_sac', 1, output, ROOT,
                         evaluation_dispatch=synthetic_dispatch, **kwargs)

    def test_true_learning_two_stages_and_exact_boundary_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            from hrl_uav.industrial_experiment import runtime_setup
            protocol, scene = fixture_protocol()
            runtime_setup(protocol, 1)
            fresh = make_agent(protocol, HotspotIndustrialMDP(scene), 1, 'ppo_sac')
            initial = copy.deepcopy(dict(sac_actor=fresh.sac_actor.action_net.state_dict(),
                sac_critic=fresh.sac_critic.critic_v.state_dict(),
                sac_target=fresh.sac_critic.target_critic_v.state_dict(),
                macro_actor=fresh.macro_actor.state_dict(),
                macro_critic=fresh.macro_critic.state_dict(),
                log_alpha=fresh.entropy.log_alpha))
            complete = self._run(directory / 'complete')
            self.assertEqual(complete['status'], 'complete')
            final = load_checkpoint(directory / 'complete/checkpoints/final.pt')
            self.assertGreater(final['updates']['sac_actor'], 0)
            self.assertGreater(final['updates']['ppo'], 0)
            self.assertEqual(final['replay']['memory_counter'], 40)
            for key, state in initial.items():
                self.assertFalse(_same(final[key], state), 'No real parameter change: ' + key)
            self.assertTrue(all(opt['state'] for opt in final['optimizers']))
            # Stop after one joint episode: the first PPO transition and a
            # wrapped SAC replay ring must both survive the reconstruction.
            partial = self._run(directory / 'partial', stop_after_episodes=3)
            self.assertEqual(partial['status'], 'interrupted')
            middle_path = directory / 'partial/checkpoints/latest.json'
            middle = load_checkpoint(middle_path)
            self.assertEqual(len(middle['training_state']['macro_rollout']), 1)
            self.assertEqual(middle['updates']['ppo'], 0)
            self.assertEqual(middle['replay']['memory_counter'], 30)
            self.assertTrue(_same(middle['macro_actor'], initial['macro_actor']))
            self.assertTrue(_same(middle['macro_critic'], initial['macro_critic']))
            continued = self._run(directory / 'continued', checkpoint=middle_path,
                                  phase='resume')
            self.assertEqual(continued['status'], 'complete')
            resumed = load_checkpoint(directory / 'continued/checkpoints/final.pt')
            for key in ('sac_actor', 'sac_critic', 'sac_target', 'macro_actor',
                        'macro_critic', 'log_alpha', 'optimizers', 'updates',
                        'replay', 'global_rng', 'guard_state'):
                self.assertTrue(_same(final[key], resumed[key]), key)
            episodes = [json.loads(line) for line in
                (directory / 'complete/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(episodes), 4)
            self.assertEqual([row['stage'] for row in episodes],
                             ['pretrain', 'pretrain', 'joint', 'joint'])
            for row in episodes:
                metrics = row['metrics']
                self.assertEqual(metrics['seconds'], 5)
                self.assertEqual(metrics['physical_steps'], 50)
                self.assertEqual(metrics['unfinished_parents'],
                    sum(p['arrived'] - p['completed'] for p in metrics['priority']))
                self.assertGreater(sum(p['arrived'] for p in metrics['priority']), 0)
                self.assertFalse(metrics['terminated'])
            self.assertEqual(episodes[0]['updates']['ppo'], 0)
            self.assertEqual(episodes[1]['updates']['ppo'], 0)

    def test_candidate_resume_reuses_committed_gate_without_double_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root / 'first', stop_after_episodes=1)
            first = load_checkpoint(root / 'first/checkpoints/latest.json')
            self.assertEqual(first['stage'], 'pretrain')
            self.assertEqual(first['updates']['ppo'], 0)
            continued = self._run(root / 'second', phase='resume',
                checkpoint=root / 'first/checkpoints/latest.json')
            self.assertEqual(continued['status'], 'complete')
            last = load_checkpoint(root / 'second/checkpoints/final.pt')
            self.assertEqual(last['replay']['memory_counter'], 40)

    def test_real_frozen_evaluation_and_exact_cache_reuse(self):
        from unittest.mock import patch
        from hrl_uav.industrial_env import IndustrialEnv
        from hrl_uav.industrial_hotspot_evaluation import candidate_evaluate, paired_evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root / 'candidate', stop_after_episodes=1)
            pointer = root / 'candidate/checkpoints/latest.json'
            before = load_checkpoint(pointer)
            receipt = candidate_evaluate(pointer, root / 'evaluation', ROOT,
                cache_root=root / 'cache', budget_path=root / 'evaluation_budget.json')
            self.assertEqual(receipt['status'], 'complete')
            for name in ('sac_rows', 'local_rows', 'cvx_rows'):
                self.assertEqual(len(receipt[name]), 6)
                self.assertTrue(all(row['frozen_verified'] for row in receipt[name]))
                self.assertTrue(all(row['metrics']['seconds'] == 5 for row in receipt[name]))
            after = load_checkpoint(pointer)
            for key in ('sac_actor', 'sac_critic', 'sac_target', 'macro_actor',
                        'macro_critic', 'log_alpha', 'optimizers', 'updates', 'replay'):
                self.assertTrue(_same(before[key], after[key]), key)
            with patch.object(IndustrialEnv, 'micro_step', side_effect=
                              AssertionError('Complete cached evaluations must not step service')):
                reused = candidate_evaluate(pointer, root / 'evaluation_reused', ROOT,
                    cache_root=root / 'cache', budget_path=root / 'evaluation_budget.json')
            self.assertEqual(receipt['sac_rows'], reused['sac_rows'])
            self._run(root / 'joint', phase='resume', checkpoint=pointer)
            paired = paired_evaluate(root / 'joint/checkpoints/final.pt', root / 'paired', ROOT,
                cache_root=root / 'cache', budget_path=root / 'evaluation_budget.json')
            self.assertEqual(paired['status'], 'complete')
            self.assertEqual(len(paired['moving_rows']), 6)
            self.assertEqual(len(paired['hover_rows']), 6)

    def test_old_checkpoint_and_research_injection_rejected(self):
        protocol, scene = fixture_protocol()
        mdp = HotspotIndustrialMDP(scene)
        agent = make_agent(protocol, mdp, 1, 'ppo_sac')
        with self.assertRaises(ValueError):
            restore_checkpoint(agent, dict(checkpoint_schema='industrial_deployment_checkpoint_v3'),
                               protocol=protocol, scene=scene)
        protocol['purpose'] = 'research'
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                train_job(protocol, scene, 'ppo_sac', 1, Path(temporary) / 'forbidden',
                          ROOT, evaluation_dispatch=synthetic_dispatch)


if __name__ == '__main__':
    unittest.main()
