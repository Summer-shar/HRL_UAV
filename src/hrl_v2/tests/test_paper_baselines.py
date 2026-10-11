"""Task 2 regression tests using temporary synthetic weights, never training."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_v2 import SCHEMA
from hrl_v2 import baseline_reference
from hrl_v2.baseline_protocol import (is_paper_caliber, reference_combinations,
                                     reference_modes_complete, labeled_results)
from hrl_v2.checkpoint import identity, load, save, digest
from hrl_v2.collector import collect_metrics
from hrl_v2.evaluate import gate_manifest, manifest
from hrl_v2.paper_evaluation import comparison_policy, panel, load_checkpoints
from hrl_v2.upper_baselines import UpperLearner, baseline_tag
from hrl_v2.paths import source_root
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.policy import Policy
from hrl_v2.randomness import Streams
from hrl_v2.runtime import configure, Learner, parameter_hash
from hrl_v2.timing import ClockConfig

ROOT = source_root().parent
CONFIG_PATH = source_root()/'hrl_v2/configs/v2_3_0_lite.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
OLD_CONFIG = json.loads((CONFIG_PATH.parent/'v2_2_1_lite.json').read_text(encoding='utf-8'))
SCENE = json.loads((CONFIG_PATH.parent/'physical_scene.json').read_text(encoding='utf-8'))


def old_module(name):
    """Execute the accepted Task 1 implementation as the legacy oracle."""
    text = subprocess.check_output(['git', 'show', '049c0da:src/hrl_v2/'+name+'.py'], cwd=ROOT).decode('utf-8')
    module = ModuleType('hrl_v2._task1_'+name)
    module.__package__ = 'hrl_v2'
    exec(compile(text, '049c0da/'+name+'.py', 'exec'), module.__dict__)
    return module


def world(scenario, seconds=10):
    adapter = PhysicsAdapter(SCENE, timing=ClockConfig.from_scene(SCENE, seconds))
    adapter.reset(**{k: v for k, v in scenario.items() if k.endswith('_seed')})
    return adapter


def arrival_bytes(adapter):
    rows = []
    for parent in adapter.env.parents.values():
        row = asdict(parent.arrival)
        # Include terminal, priority, IDs, time, bits and cycles, preserving the
        # exact IEEE float values rather than using a numerical tolerance.
        rows.append({k: v.hex() if isinstance(v, float) else v for k, v in row.items()})
    return json.dumps(rows, sort_keys=True).encode('utf-8')


class PaperBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = configure(test_cpu=True)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.checkpoint = Path(cls.temporary.name)/'synthetic.pt'
        cls.binding = identity(ROOT, CONFIG_PATH)
        learner = Learner(827, cls.device, CONFIG)
        # Nonzero, observation-dependent upper output without any optimizer
        # step. This avoids a vacuous equality check on the zero initial head.
        with torch.no_grad():
            head = learner.ppo.actor.mean
            head.weight.copy_(torch.linspace(-.02, .02, head.weight.numel()).reshape_as(head.weight))
            head.bias.copy_(torch.linspace(-.2, .2, 6))
        save(cls.checkpoint, dict(schema=SCHEMA, safe_episode_boundary=True,
             identity=cls.binding, completed_episode=0, pending_evaluation=None,
             seed=827, learner=learner.state_dict()))
        cls.learner = Learner(828, cls.device, CONFIG)
        cls.learner.load_state_dict(load(cls.checkpoint, cls.binding)['learner'])
        cls.checkpoint_paths = {'Proposed': cls.checkpoint}
        for i, name in enumerate(('NC', 'TOO', 'TFOO'), 1):
            upper = UpperLearner(827, cls.device, CONFIG)
            with torch.no_grad(): upper.ppo.actor.mean.bias.fill_(i*.1)
            path = Path(cls.temporary.name)/(name+'.pt')
            save(path, dict(schema=SCHEMA, safe_episode_boundary=True, identity=cls.binding,
                 completed_episode=0, pending_evaluation=None, seed=827,
                 learner=upper.state_dict(), **baseline_tag(name)))
            cls.checkpoint_paths[name] = path
        cls.panel_learners, cls.panel_ids, _ = load_checkpoints(
            cls.checkpoint_paths, cls.binding, CONFIG, cls.device)
        adapter = world(gate_manifest(20)['scenarios'][0], 5)
        policy = comparison_policy(cls.learner, SCENE, Streams(827), 'hover_equal')
        cls.metric = collect_metrics(adapter, policy, identity={})

    def test_v2_2_1_config_retains_exact_legacy_reference_set(self):
        self.assertFalse(is_paper_caliber(OLD_CONFIG['objective']))
        expected = {'hover_local': ('hover','local'), 'hover_equal': ('hover','equal'),
                    'hover_cvx': ('hover','cvx'), 'centroid_equal': ('centroid','equal')}
        self.assertEqual(reference_combinations(OLD_CONFIG), expected)
        self.assertTrue(reference_modes_complete(OLD_CONFIG, expected))
        self.assertFalse(reference_modes_complete(OLD_CONFIG, set(expected)-{'centroid_equal'}))
        self.assertFalse(reference_modes_complete(OLD_CONFIG, set(expected)|{'hover_greedy'}))

    def test_v2_3_0_config_requires_three_and_accepts_frozen_extras(self):
        self.assertTrue(is_paper_caliber(CONFIG['objective']))
        self.assertEqual(CONFIG['observation_version'], OLD_CONFIG['observation_version'])
        required = {'hover_local','hover_equal','hover_cvx'}
        self.assertEqual(set(reference_combinations(CONFIG)), required)
        self.assertTrue(reference_modes_complete(CONFIG, required))
        self.assertTrue(reference_modes_complete(CONFIG, required|{'centroid_equal','hover_greedy'}))
        for missing in required:
            self.assertFalse(reference_modes_complete(CONFIG, (required-{missing})|{'centroid_equal','hover_greedy'}))
        wrong = dict(CONFIG['objective'], propulsion_weight=1.)
        self.assertFalse(is_paper_caliber(wrong))

    def test_reference_legacy_serialized_output_is_byte_identical(self):
        old = old_module('baseline_reference')
        with tempfile.TemporaryDirectory() as directory:
            for greedy in (False, True):
                results, orders = [], []
                for number, module in enumerate((old, baseline_reference)):
                    output = Path(directory)/f'{greedy}_{number}'
                    with patch('hrl_v2.run.read_config', return_value=OLD_CONFIG), \
                         patch('hrl_v2.runtime.configure'), \
                         patch.object(module.time, 'perf_counter', return_value=1.), \
                         patch.object(module, 'measure_case', return_value=self.metric) as measure:
                        module.freeze(CONFIG_PATH.parent/'v2_2_1_lite.json', output, 6, 4, include_greedy=greedy)
                    results.append([(output/name).read_bytes() for name in ('contract.json','summary.json')])
                    orders.append([(c.args[4], c.args[5], c.args[6:]) for c in measure.call_args_list])
                self.assertEqual(results[0], results[1])
                self.assertEqual(orders[0], orders[1])

    def test_new_reference_report_labels_and_supplementary_only(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('hrl_v2.run.read_config', return_value=CONFIG), \
             patch('hrl_v2.runtime.configure'), \
             patch.object(baseline_reference, 'measure_case', return_value=self.metric) as measure:
            summary = baseline_reference.freeze(CONFIG_PATH, directory, 6, 4)
        self.assertEqual({c.args[6] for c in measure.call_args_list}, {'hover_local','hover_equal','hover_cvx'})
        rows = copy.deepcopy(summary['by_mode'])
        rows.update(centroid_equal=rows['hover_equal'], hover_greedy=rows['hover_local'])
        labeled = labeled_results(rows)
        self.assertEqual([(r['paper_name'], r['internal_name']) for r in labeled['main_comparison']],
                         [('CA','hover_equal')])
        self.assertEqual({r['internal_name'] for r in labeled['supplementary']},
                         {'hover_local','hover_cvx','centroid_equal','hover_greedy'})

    def test_policy_default_preserves_legacy_action_and_random_stream_bytes(self):
        old = old_module('policy').Policy
        adapter = world(gate_manifest(20)['scenarios'][0])
        report = adapter.prepare()
        for stochastic in (False, True):
            streams = [Streams(813, policy_stream=1) for _ in range(2)]
            policies = [cls(self.learner.sac.actor, self.learner.ppo.actor, None, s,
                            stochastic=stochastic) for cls, s in zip((old, Policy), streams)]
            for method in ('upper','lower','upper','lower'):
                x, y = [getattr(p, method)(report, adapter.arrivals) for p in policies]
                self.assertEqual(x[0].tobytes(), y[0].tobytes())
                if method == 'lower':
                    self.assertEqual(x[1].tobytes(), y[1].tobytes())
                else:
                    for key in x[1]:
                        np.testing.assert_array_equal(x[1][key], y[1][key])
            for key, state in streams[0].state_dict()['generators'].items():
                self.assertTrue(torch.equal(state, streams[1].state_dict()['generators'][key]))

    def test_same_checkpoint_same_observation_has_identical_upper_action(self):
        adapter = world(gate_manifest(20)['scenarios'][0])
        report = adapter.report()
        actions = []
        for name in ('moving','shared_upper_equal','shared_upper_cvx'):
            learner = Learner(824, self.device, CONFIG)
            learner.load_state_dict(load(self.checkpoint, self.binding)['learner'])
            streams = Streams(827, episode=1, split='dev', policy_stream=1)
            before = streams.generator('upper',1).get_state().clone()
            policy = comparison_policy(learner, SCENE, streams, name, stochastic=True)
            action, _ = policy.upper(report, adapter.arrivals)
            self.assertTrue(torch.equal(before, streams.generator('upper',1).get_state()))
            actions.append(action.tobytes())
            self.assertGreater(np.linalg.norm(action), .01)
        self.assertEqual(actions, [actions[0]]*3)

    def test_closed_loop_overrides_and_arrival_streams_are_exactly_paired(self):
        before = [parameter_hash(net) for net in (self.learner.sac.actor, self.learner.ppo.actor)]
        counts = (dict(self.learner.sac.counts), dict(self.learner.ppo.counts))
        for scenario in (gate_manifest(20)['scenarios'][0], gate_manifest(20)['scenarios'][8]):
            for stochastic in (False, True):
                tapes, digests = [], []
                for name in ('moving','shared_upper_equal','shared_upper_cvx'):
                    adapter = world(scenario)
                    streams = Streams(827, episode=scenario['scenario'], split='dev', policy_stream=1)
                    original_rng = streams.generator('support',1).get_state().clone()
                    policy = comparison_policy(self.learner, SCENE, streams, name, stochastic=stochastic)
                    if name == 'shared_upper_equal':
                        with patch.object(self.learner.sac.actor, 'forward', side_effect=AssertionError('TOO called SAC')), \
                             patch.object(policy, 'lower', wraps=policy.lower) as lower:
                            collect_metrics(adapter, policy, identity={})
                        for call in lower.call_args_list:
                            route, support = policy.lower_override(*call.args)
                            self.assertEqual(route.tobytes(), np.full((2,3,3),1/3,dtype=np.float64).tobytes())
                            np.testing.assert_array_equal(support, np.full(6,6))
                        self.assertEqual(lower.call_count, 20)
                    elif name == 'shared_upper_cvx':
                        solver = policy.lower_override.rule.solver
                        with patch.object(self.learner.sac.actor, 'forward', side_effect=AssertionError('TFOO called SAC')), \
                             patch.object(solver, 'solve_coefficients', wraps=solver.solve_coefficients) as solve:
                            collect_metrics(adapter, policy, identity={})
                        self.assertGreater(solve.call_count, 0)
                        self.assertFalse(any(r.get('fallback') for r in policy.lower_override.rule.records))
                    else:
                        collect_metrics(adapter, policy, identity={})
                    changed = not torch.equal(original_rng, streams.generator('support',1).get_state())
                    self.assertEqual(changed, name == 'moving' and stochastic)
                    self.assertGreater(len(adapter.env.parents), 0)
                    tapes.append(arrival_bytes(adapter))
                    digests.append(adapter.env._scene.canonical_event_digest.hexdigest())
                self.assertEqual(tapes, [tapes[0]]*3)
                self.assertEqual(digests, [digests[0]]*3)
        self.assertEqual(before, [parameter_hash(net) for net in (self.learner.sac.actor, self.learner.ppo.actor)])
        self.assertEqual(counts, (dict(self.learner.sac.counts), dict(self.learner.ppo.counts)))

    def test_paper_panel_records_five_labels_all_streams_and_strict_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            timing = ClockConfig.from_scene(SCENE, 5)
            with patch('hrl_v2.paper_evaluation.collect_metrics', return_value=self.metric) as collect:
                result = panel(self.panel_learners, SCENE, 827, manifest('dev'), directory,
                               self.panel_ids, stochastic=False, timing=timing)
            self.assertEqual(collect.call_count, 6*2*8)
            self.assertEqual([r['paper_name'] for r in result['main_comparison']], ['NC','CA','TOO','TFOO','Proposed'])
            self.assertEqual({r['internal_name'] for r in result['supplementary']},
                             {'hover_local','shared_upper_equal','shared_upper_cvx'})
            self.assertEqual(result['learning_updates'], 0)
            self.assertEqual(result['replay_writes'], 0)
            for rows in result['per_scenario'].values():
                self.assertEqual(set(rows), {str(i) for i in range(1,7)})
                for m in rows.values():
                    self.assertEqual(m['windows'], 2)
                    for key in ('truncated_delay_s','energy_j','cost_sum'): self.assertIn(key, m)
            table = (Path(directory)/'comparison_deterministic.md').read_text(encoding='utf-8')
            self.assertIn('| TOO | moving_equal |', table)
            with patch('hrl_v2.paper_evaluation.collect_metrics', side_effect=AssertionError('Cache recomputed')):
                cached = panel(self.panel_learners, SCENE, 827, manifest('dev'), directory,
                               self.panel_ids, stochastic=False, timing=timing)
            self.assertEqual(result, cached)
            with self.assertRaisesRegex(ValueError, 'another contract'):
                panel(self.panel_learners, SCENE, 827, manifest('dev'), directory,
                      dict(self.panel_ids, NC='different-checkpoint'), stochastic=False, timing=timing)

    def test_deterministic_paper_panel_metrics_are_bitwise_equal_across_streams(self):
        # Run the real closed-loop panel (no mocked metrics). Both streams
        # must reproduce every metric, including parent ledgers and energy,
        # for all main and supplementary controllers on six sealed dev scenarios.
        panel_manifest = manifest('dev')
        with tempfile.TemporaryDirectory() as directory:
            result = panel(self.panel_learners, SCENE, 827, panel_manifest, directory,
                           self.panel_ids, stochastic=False,
                           timing=ClockConfig.from_scene(SCENE, 10))
            for name in result['by_mode']:
                for scenario in panel_manifest['scenarios']:
                    records = [json.loads((Path(directory)/
                        f'deterministic_{name}_{scenario["scenario"]:02d}_{stream}.json').read_text(encoding='utf-8'))
                        for stream in panel_manifest['policy_streams']]
                    self.assertEqual([r['identity']['stream'] for r in records], [1, 2])
                    encoded = [json.dumps(r['metrics'], sort_keys=True,
                                          separators=(',', ':'), allow_nan=False).encode('utf-8')
                               for r in records]
                    self.assertEqual(encoded[0], encoded[1], (name, scenario['scenario']))

    def test_paper_panel_rejects_legacy_scene_and_changed_manifest(self):
        scene = copy.deepcopy(SCENE)
        scene['base_scene']['objective']['energy_mode'] = 'full'
        scene['base_scene']['objective']['propulsion_weight'] = 1.
        with self.assertRaisesRegex(ValueError, 'v2.3'):
            comparison_policy(self.learner, scene, Streams(827), 'moving_equal')
        changed = manifest('dev')
        changed['scenarios'][0]['arrival_seed'] += 1
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'sealed seed'):
            panel(self.panel_learners, SCENE, 827, changed, directory, self.panel_ids)

    def test_checkpoint_identity_has_no_compatibility_exception(self):
        changed = copy.deepcopy(self.binding)
        changed['files']['hrl_v2/paper_evaluation.py'] = 'old-code-digest'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            load(self.checkpoint, changed)


if __name__ == '__main__':
    unittest.main()
