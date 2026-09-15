"""Bounded continuation tests; full budget uses clock doubles, never physical training."""

from contextlib import nullcontext
import ast
import copy
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav import mission_ppo_continuation as c
from hrl_uav import mission_ppo_qualification as q
from hrl_uav import mission_ppo_stage4a as old
from hrl_uav.experiment import isolated_rng, write_json
from hrl_uav.mission_smoke import load_source, load_spec
from hrl_uav.resumption import capture_rng, state_equal
from hrl_uav.trajectory_check import ReadOnlySAC
from tests.test_mission_ppo_qualification import ClockPolicy, metric_row, restore_clock, synthetic_feed
from tests.test_mission_ppo_stage4a import ClockMeter


class ExtendedClockPolicy(ClockPolicy):
    def ppo_state(self):
        state = super().ppo_state()
        fraction = min(1., self.consumed_macros/6000)
        state["actor"]["log_std"] = torch.full((6,), (1-fraction)*np.log(1/3)+fraction*np.log(1/30))
        return state


def restore_extended_clock(policy, payload, lock):
    c.validate_boundary(payload, policy, lock)
    state = payload["ppo"]
    policy.macro_rollout = copy.deepcopy(state["rollout"])
    (policy.ppo_updates, policy.ppo_gradient_steps, policy.collected_macros, policy.consumed_macros) = state["counters"]
    policy.rng = state["rng"]
    return payload["next_scene_index"]


class ContinuationTests(unittest.TestCase):
    def test_original_collector_and_restore_bodies_are_unchanged(self):
        def body(function, start):
            tree = ast.parse(inspect.getsource(function)).body[0]
            index = next(i for i, node in enumerate(tree.body)
                         if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id == start)
            return [ast.dump(n, include_attributes=False) for n in tree.body[index:]]
        self.assertEqual(body(c.collect_stream, "meter"), body(old.collect_stream, "meter"))
        self.assertEqual(body(c.restore_boundary, "state"), body(q.restore_boundary, "state"))
        self.assertEqual(c.protocol()["training_scenes"], list(range(72040, 72200)))
        self.assertEqual(old.TRAIN_SCENES, tuple(range(72000, 72040)))
        self.assertEqual(q.protocol()["windows_per_seed"], 120)
        self.assertEqual(c.MONITORS, (240, 360, 480, 600))
        for scene in (33610, 33611, 33612, *range(33700, 33710)):
            for train in (True, False):
                with self.assertRaises(ValueError):
                    c.collect_stream({}, None, scene, train=train)
        for scene, offset in ((72040, 0), (72199, 18000), (72200, 18000)):
            with self.assertRaises(ValueError):
                c.collect_stream({}, None, scene, train=True, macro_offset=offset)

    def test_reporting_uses_actual600_endpoint_and_three_initializations(self):
        rows = [metric_row(e, "direct_ppo", (s-200)*10+(e-33600), s, 600)
                for s in q.PPO_SEEDS for e in q.VALIDATION_SCENES]
        baselines = [metric_row(e, n) for n in q.BASELINES for e in q.VALIDATION_SCENES]
        per_seed, overall, pairs = c.aggregate(rows, baselines)
        self.assertEqual([r["J"] for r in per_seed], [10.5, 20.5, 30.5])
        j = next(r for r in overall if r["policy"] == "direct_ppo" and r["metric"] == "J")
        self.assertEqual(j["std"], 10.)
        self.assertEqual(len(pairs), 24)
        self.assertTrue(all(r["window"] == 600 for r in rows + per_seed + overall + pairs))
        rows[0]["window"] = 480
        with self.assertRaises(ValueError):
            c.aggregate(rows, baselines)

    def test_archive_is_lossless_and_lease_excludes_second_writer(self):
        def persist(folder, *args):
            folder.mkdir()
            write_json(folder / "crn.json", [{"scene": 72040}])
            (folder / "binary.npz").write_bytes(bytes(range(256))*3)
            return {"duration_s": 900}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(q, "persist_case", persist):
                row = c.archive_case(root, None, 72040, 123, None)
            self.assertEqual(row["duration_s"], 900)
            self.assertEqual(c.archive_json(root, "crn.json"), [{"scene": 72040}])
            self.assertFalse(list(root.glob("case-*")))
            with c.zipfile.ZipFile(root / "stream.zip") as z:
                self.assertEqual(z.read("binary.npz"), bytes(range(256))*3)
            with c.output_lease(root):
                with self.assertRaises(OSError):
                    with c.output_lease(root):
                        pass
            with c.output_lease(root):
                pass

    def test_all_three600_window_schedules_and_interrupted_resume_with_clock(self):
        cfg = {"training": dict(old.PPO_SETTINGS, gamma_per_resource_frame=.99, joint_windows=120)}
        baselines = [metric_row(e, n) for n in q.BASELINES for e in q.VALIDATION_SCENES]
        refs = {(n, e): [{"scene": e}] for n in q.BASELINES for e in q.VALIDATION_SCENES}
        train_calls, eval_calls = [], []

        def source_tree(root):
            for seed in q.PPO_SEEDS:
                policy = ExtendedClockPolicy(cfg, seed)
                policy.collected_macros = policy.consumed_macros = 3600
                policy.ppo_updates, policy.ppo_gradient_steps = 12, 24
                with q.transaction(root / f"seed{seed}/boundary040") as attempt:
                    torch.save(q.boundary_payload(policy, 40, {}), attempt / "checkpoint.pt")
                with q.transaction(root / f"seed{seed}/monitor120") as attempt:
                    torch.save(policy.ppo_state(), attempt / "ppo.pt")

        def collector(config, policy, scene, **kwargs):
            self.assertEqual(kwargs, dict(train=True, macros=90, macro_offset=(scene-72000)*90))
            train_calls.append((policy.seed, scene))
            result = c.collect_stream(config, policy, scene, meter_factory=ClockMeter, record=False, **kwargs)
            self.assertEqual([w["queue_total_mbit"] for w in result[4]], [600, 1200, 1800])
            return result

        def evaluator(policy, scene):
            eval_calls.append((policy.seed, policy.collected_macros//30, scene))
            return (None, SimpleNamespace(scene=scene))

        def persister(folder, policy, scene, window, result):
            folder.mkdir()
            row = metric_row(scene, "direct_ppo", ppo_seed=policy.seed, window=window)
            write_json(folder / "summary.json", row)
            write_json(folder / "crn.json", [{"scene": scene}])
            return row

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(q, "restore_boundary", restore_clock), \
                patch.object(c, "restore_boundary", restore_extended_clock), \
                patch.object(c, "frozen_sac_guard", lambda _: nullcontext()), \
                patch.object(q, "crn_records", lambda meter: [{"scene": meter.scene}]), \
                patch.object(q, "persist_case", persister), patch("builtins.print"):
            root = Path(temp)
            source, output = root / "source", root / "output"
            source_tree(source)
            output.mkdir()
            runner = c.ContinuationRunner(output, {"source_lock": {}}, baselines, refs, source=source,
                                          collector=collector, evaluator=evaluator, persister=persister)
            for seed in q.PPO_SEEDS:
                policy = ExtendedClockPolicy(cfg, seed)
                runner.run_seed(policy)
                self.assertEqual((policy.ppo_updates, policy.collected_macros, policy.consumed_macros), (60, 18000, 18000))
                self.assertFalse(any(policy.flags))
                self.assertTrue(all(policy.micro_flags))
                counts = len(train_calls), len(eval_calls)
                runner.run_seed(ExtendedClockPolicy(cfg, seed))
                self.assertEqual(counts, (len(train_calls), len(eval_calls)))
            self.assertEqual(train_calls, [(s, e) for s in q.PPO_SEEDS for e in range(72040, 72200)])
            self.assertEqual(eval_calls, [(s, w, e) for s in q.PPO_SEEDS for w in c.MONITORS for e in q.VALIDATION_SCENES])
            # A separate clock run fails after work but before committing boundary042.
            interrupted = root / "interrupted"
            interrupted.mkdir()
            def failing(config, policy, scene, **kwargs):
                result = collector(config, policy, scene, **kwargs)
                if scene == 72041:
                    raise RuntimeError("synthetic interruption")
                return result
            runner = c.ContinuationRunner(interrupted, {"source_lock": {}}, baselines, refs, source=source,
                                          collector=failing, evaluator=evaluator, persister=persister)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                runner.run_seed(ExtendedClockPolicy(cfg, 201))
            self.assertTrue((interrupted / "seed201/boundary041").exists())
            self.assertFalse((interrupted / "seed201/boundary042").exists())
            self.assertEqual(len(list((interrupted / "seed201").glob(".attempt-*"))), 1)
            count = len(train_calls)
            runner.collector = collector
            runner.run_seed(ExtendedClockPolicy(cfg, 201))
            self.assertEqual(train_calls[count:], [(201, e) for e in range(72041, 72200)])


class RestoredPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = q.read_json(c.SOURCE / "manifest.json")
        cls.source_lock = cls.manifest["lock"]
        saved, _ = load_source(load_spec())
        with isolated_rng():
            cls.sac = ReadOnlySAC(saved)
        cls.cfg = cls.source_lock["configs"]["201"]

    def policy(self, seed=201):
        policy = old.MissionMacroPPO(self.source_lock["configs"][str(seed)], seed, self.sac)
        payload = torch.load(c.SOURCE / f"seed{seed}/boundary040/checkpoint.pt", map_location="cpu", weights_only=False)
        q.restore_boundary(policy, payload, self.source_lock)
        return policy

    def test_three_source_endpoints_and_deterministic_actions_match(self):
        for seed in q.PPO_SEEDS:
            policy = self.policy(seed)
            checkpoint = torch.load(c.SOURCE / f"seed{seed}/monitor120/ppo.pt", map_location="cpu", weights_only=False)
            self.assertTrue(state_equal(policy.ppo_state(), checkpoint))
            self.assertEqual(policy.ppo_updates, 12)
            self.assertFalse(policy.macro_rollout)
            self.assertTrue(policy.ppo_actor_optimizer.state_dict()["state"])
            self.assertTrue(policy.ppo_critic_optimizer.state_dict()["state"])
            for scene in q.VALIDATION_SCENES:
                with np.load(c.SOURCE / f"seed{seed}/monitor120/{scene}/macro_samples.npz") as data:
                    for state, expected in zip(data["state"], data["raw_action"]):
                        actual, _ = policy.macro_action(state, deterministic=True)
                        np.testing.assert_array_equal(actual, expected)
            self.assertTrue(all(self.sac.assert_unchanged().values()))

    def test_pending_resume_is_bitwise_and_std_uses_cumulative_windows(self):
        uninterrupted = self.policy()
        for count in (3690, 3780, 3870, 3960, 6030):
            synthetic_feed(uninterrupted, count)
            payload = c.boundary_payload(uninterrupted, count//90, {"synthetic": True})
            restored = self.policy()
            before_rng = capture_rng()
            c.restore_boundary(restored, payload, {"synthetic": True})
            self.assertTrue(state_equal(before_rng, capture_rng()))
            self.assertTrue(state_equal(uninterrupted.ppo_state(), restored.ppo_state()))
            self.assertEqual(len(restored.macro_rollout), count % 300)
            next_stop = count+90
            synthetic_feed(uninterrupted, next_stop)
            synthetic_feed(restored, next_stop)
            self.assertTrue(state_equal(uninterrupted.ppo_state(), restored.ppo_state()))
        self.assertEqual(uninterrupted.ppo_updates, 20)
        np.testing.assert_allclose(uninterrupted.macro_actor.log_std.exp().detach().cpu().numpy(), 1/30, rtol=1e-6)
        self.assertTrue(all(self.sac.assert_unchanged().values()))

    def test_corrupt_counter_config_rng_or_stage_is_not_accepted(self):
        policy = self.policy()
        payload = c.boundary_payload(policy, 40, {})
        for key, value in (("stage", q.STAGE), ("next_scene_index", 39), ("ppo_seed", 202),
                           ("replay", {}), ("lock", {"bad": True}), ("current_window", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.restore_boundary(policy, dict(payload, **{key: value}), {})
        bad = copy.deepcopy(payload)
        bad["ppo"]["actor"]["log_std"] += .1
        with self.assertRaises(AssertionError):
            c.restore_boundary(policy, bad, {})

    def test_real10s_continuation_prefix_has_no_reset_replay_or_schedule_restart(self):
        a, b = self.policy(), self.policy()
        before = a.ppo_state()
        results = []
        for policy in (a, b):
            with policy.exploration_rng(), old.frozen_sac_guard(self.sac):
                results.append(c.collect_stream(policy.config, policy, 72040, train=True, macros=1, macro_offset=3600))
        self.assertTrue(state_equal(a.ppo_state(), b.ppo_state()))
        self.assertTrue(state_equal(before["actor"], a.ppo_state()["actor"]))
        self.assertEqual((a.collected_macros, a.consumed_macros, a.ppo_updates), (3601, 3600, 12))
        self.assertEqual(len(a.macro_rollout), 1)
        summary, meter = results[0][:2]
        self.assertEqual((summary["time_s"], summary["sac_decisions"], summary["replay_writes"], summary["replay_samples"]), (10, 20, 0, 0))
        self.assertEqual(meter.reset_count, 1)
        q.assert_crn_equal(q.crn_records(meter), q.crn_records(results[1][1]))
        self.assertTrue(all(self.sac.assert_unchanged().values()))


if __name__ == "__main__":
    unittest.main()
