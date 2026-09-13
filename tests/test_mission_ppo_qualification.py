"""Runner acceptance: clock/synthetic tests; RealPrefix is a separately run10s check."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav import mission_ppo_qualification as q
from hrl_uav.experiment import append_csv, isolated_rng, write_json
from hrl_uav.mission_ppo_stage4a import MissionMacroPPO, collect_stream, frozen_sac_guard, stage_config
from hrl_uav.mission_smoke import load_source, load_spec
from hrl_uav.resumption import capture_rng, state_equal
from hrl_uav.trajectory_check import ReadOnlySAC
from tests.test_mission_ppo_stage4a import ClockMeter, FixedMacro


def synthetic_feed(policy, stop):
    """Original PPO math on synthetic samples, without an environment or SAC inference."""
    with policy.exploration_rng(), frozen_sac_guard(policy.sac):
        for index in range(policy.collected_macros, stop):
            state = np.sin(np.arange(67)*.1 + index*.01).astype(np.float32)*.1
            action, logprob = policy.macro_action(state, deterministic=False)
            reward = -float(np.square(action-.03).sum())
            policy.macro_rollout.append((state, action, logprob, reward, state*.99, float(index % 90 != 89)))
            policy.collected_macros += 1
            policy.update_ppo((index+1)/3600)


class ClockPolicy(FixedMacro):
    """No learning networks; test the real scheduler/collector, not PPO numerics."""
    def __init__(self, cfg, seed):
        super().__init__(cfg)
        self.config, self.seed = cfg, seed
        self.batch_log, self.update_log = [], []
        self.ppo_gradient_steps = 0
        self.sac.assert_unchanged = lambda: {"clock_only": True}
        self.rng = {"clock_rng": seed}

    def exploration_rng(self):
        return nullcontext()

    def update_ppo(self, progress=0., force=False):
        before = self.ppo_updates
        super().update_ppo(progress, force)
        if self.ppo_updates != before:
            self.ppo_gradient_steps += 2
            self.batch_log.append({"synthetic": np.zeros(300)})
            self.update_log.append(dict(update=self.ppo_updates, forced=force, transitions=300))

    def ppo_state(self):
        fraction = self.consumed_macros/6000
        return dict(actor={"log_std": torch.full((6,), (1-fraction)*np.log(1/3)+fraction*np.log(1/30))},
                    critic={}, optimizers=[{"state": {}}, {"state": {}}], gradients=[[], []], training_modes=[[], []],
                    counters=(self.ppo_updates, self.ppo_gradient_steps, self.collected_macros, self.consumed_macros),
                    rollout=copy.deepcopy(self.macro_rollout), rng=self.rng)


def restore_clock(policy, payload, lock):
    q.validate_boundary(payload, policy, lock)
    state = payload["ppo"]
    policy.macro_rollout = copy.deepcopy(state["rollout"])
    (policy.ppo_updates, policy.ppo_gradient_steps, policy.collected_macros, policy.consumed_macros) = state["counters"]
    policy.rng = state["rng"]
    return payload["next_scene_index"]


def metric_row(seed=33600, policy="hover", value=1., ppo_seed=None, window=120):
    return dict({k: value for k in q.METRICS}, seed=seed, policy=policy, ppo_seed=ppo_seed,
                window=window, duration_s=900)


class RunnerTests(unittest.TestCase):
    def test_clean_clone_has_no_untracked_stage3b_dependency(self):
        text = (q.ROOT / "hrl_uav/mission_ppo_stage4a.py").read_text()
        for name in ("mission_future_analysis.py", "mission_future_check.py", "mission_predictive_stage3b.md"):
            self.assertNotIn(name, text)
            self.assertTrue(all(name not in p for p in q.DEPENDENCIES))
        self.assertEqual(q.protocol()["training_scene_schedule"], list(range(72000, 72040)))
        self.assertEqual(q.protocol()["validation_scenes"], [33600, 33601])
        self.assertFalse(q.protocol()["force_updates"])

    def test_transaction_retains_interruption_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "boundary001"
            with self.assertRaises(RuntimeError):
                with q.transaction(destination) as pending:
                    write_json(pending / "sample.json", {"partial": True})
                    raise RuntimeError("simulated interruption")
            self.assertFalse(destination.exists())
            self.assertEqual(len(list(Path(temp).glob(".attempt-*"))), 1)
            with q.transaction(destination) as pending:
                write_json(pending / "sample.json", {"complete": True})
            q.verify_transaction(destination)
            with self.assertRaises(FileExistsError):
                with q.transaction(destination):
                    pass
            write_json(destination / "sample.json", {"corrupt": True})
            with self.assertRaises(ValueError):
                q.verify_transaction(destination)

    def test_aggregation_uses_three_initializations_not_six_scenes(self):
        rows = [metric_row(e, "direct_ppo", value=(s-200)*10+(e-33600), ppo_seed=s)
                for s in q.PPO_SEEDS for e in q.VALIDATION_SCENES]
        baselines = [metric_row(e, n) for n in q.BASELINES for e in q.VALIDATION_SCENES]
        seeds, overall, pairs = q.aggregate(rows, baselines)
        self.assertEqual([r["J"] for r in seeds], [10.5, 20.5, 30.5])
        j = next(r for r in overall if r["policy"] == "direct_ppo" and r["metric"] == "J")
        self.assertEqual(j["std"], 10.)
        self.assertEqual(len(pairs), 24)
        self.assertTrue(all(r["std"] is None for r in overall if r["policy"] != "direct_ppo"))
        with self.assertRaises(ValueError):
            q.aggregate(rows[:-1], baselines)
        rows[0]["window"] = 90
        with self.assertRaises(ValueError):
            q.aggregate(rows, baselines)

    def test_full_gate_checks_traces_routes_queues_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            position = (0., np.zeros((3, 2)), np.zeros((39, 2)), np.zeros((3, 2)), np.zeros((3, 2)))
            meter = SimpleNamespace(positions=[position]*9001, micro_states=np.zeros((1800, 68)),
                                    commands=[{"raw_x": 0., "raw_y": 0.}]*270,
                                    readings=[{"queue": 0.}]*1800, route_readings=[{"rho": .5}]*10800)
            observations = [(None, np.zeros(6))]*90
            result = ({"time_s": 900}, meter, [], observations, [], [])
            row = metric_row()
            np.savez(folder / "trajectory.npz", **{
                key: np.asarray([p[i] for p in meter.positions])
                for i, key in enumerate(("time_s", "uav", "users", "centroid", "groups"))},
                micro_state=meter.micro_states)
            for name, records in (("commands", meter.commands), ("queues", meter.readings), ("routes", meter.route_readings)):
                append_csv(folder / f"{name}.csv", records)
            write_json(folder / "summary.json", row)
            write_json(folder / "crn.json", [{"clock": True}])
            with patch.object(q, "crn_records", return_value=[{"clock": True}]):
                self.assertTrue(q.window0_gate(result, row, folder)["full900s"])
                bad = dict(row, J=2.)
                with self.assertRaises(AssertionError):
                    q.window0_gate(result, bad, folder)
                meter.micro_states[5, 1] = .01
                with self.assertRaises(AssertionError):
                    q.window0_gate(result, row, folder)
                meter.micro_states[5, 1] = 0
                meter.route_readings = [dict(rho=.6)] + meter.route_readings[1:]
                with self.assertRaises(AssertionError):
                    q.window0_gate(result, row, folder)

    def test_real_scheduler_clock_budget_monitors_resume_and_gate_failure(self):
        cfg = {"training": dict(q.protocol()["ppo_settings"], gamma_per_resource_frame=.99)}
        baselines = [metric_row(e, n) for n in q.BASELINES for e in q.VALIDATION_SCENES]
        refs = {(n, e): [{"scene": e}] for n in q.BASELINES for e in q.VALIDATION_SCENES}
        train_calls, eval_calls, gate_calls = [], [], []

        def collector(config, policy, scene, **kwargs):
            self.assertEqual(kwargs, dict(train=True, macros=90, macro_offset=(scene-72000)*90))
            train_calls.append((policy.seed, scene))
            result = collect_stream(config, policy, scene, meter_factory=ClockMeter, record=False, **kwargs)
            self.assertEqual([w["queue_total_mbit"] for w in result[4]], [600, 1200, 1800])
            return result

        def evaluator(policy, scene):
            eval_calls.append((policy.seed, policy.collected_macros//30, scene))
            return (None, SimpleNamespace(scene=scene))

        def persister(directory, policy, scene, window, result):
            directory.mkdir()
            row = metric_row(scene, "direct_ppo", ppo_seed=policy.seed, window=window)
            write_json(directory / "summary.json", row)
            write_json(directory / "crn.json", [{"scene": scene}])
            return row

        def gate(result, row, reference):
            gate_calls.append((row["ppo_seed"], row["seed"]))
            return {"clock_gate_only": True}

        with tempfile.TemporaryDirectory() as temp, patch.object(q, "restore_boundary", restore_clock), \
                patch.object(q, "frozen_sac_guard", lambda _: nullcontext()), \
                patch.object(q, "crn_records", lambda meter: [{"scene": meter.scene}]):
            runner = q.QualificationRunner(temp, {}, baselines, refs, collector=collector,
                                           evaluator=evaluator, persister=persister, gate=gate)
            for seed in q.PPO_SEEDS:
                policy = ClockPolicy(cfg, seed)
                runner.run_seed(policy)
                self.assertEqual((policy.ppo_updates, policy.collected_macros, policy.consumed_macros), (12, 3600, 3600))
                self.assertFalse(any(policy.flags))
                self.assertTrue(all(policy.micro_flags))
                # Completed resume does not train or evaluate again.
                counts = len(train_calls), len(eval_calls)
                runner.run_seed(ClockPolicy(cfg, seed))
                self.assertEqual(counts, (len(train_calls), len(eval_calls)))
            self.assertEqual(train_calls, [(s, e) for s in q.PPO_SEEDS for e in q.TRAIN_SCENES])
            self.assertEqual(eval_calls, [(s, w, e) for s in q.PPO_SEEDS for w in q.MONITORS for e in q.VALIDATION_SCENES])
            self.assertEqual(len(gate_calls), 6)
            runner.finalize()
            q.verify_transaction(Path(temp) / "final")
        with tempfile.TemporaryDirectory() as temp, patch.object(q, "restore_boundary", restore_clock), \
                patch.object(q, "frozen_sac_guard", lambda _: nullcontext()), \
                patch.object(q, "crn_records", lambda meter: [{"scene": meter.scene}]):
            def interrupted_collector(config, policy, scene, **kwargs):
                result = collector(config, policy, scene, **kwargs)
                if scene == 72001:
                    raise RuntimeError("after second stream, before checkpoint commit")
                return result
            runner = q.QualificationRunner(temp, {}, baselines, refs, collector=interrupted_collector,
                                           evaluator=evaluator, persister=persister, gate=gate)
            with self.assertRaisesRegex(RuntimeError, "before checkpoint"):
                runner.run_seed(ClockPolicy(cfg, 201))
            root = Path(temp) / "seed201"
            self.assertTrue((root / "boundary001").exists())
            self.assertFalse((root / "boundary002").exists())
            self.assertEqual(len(list(root.glob(".attempt-boundary002-*"))), 1)
            count = len(train_calls)
            runner.collector = collector
            runner.run_seed(ClockPolicy(cfg, 201))
            self.assertEqual(train_calls[count:], [(201, s) for s in range(72001, 72040)])
        with tempfile.TemporaryDirectory() as temp, patch.object(q, "restore_boundary", restore_clock), \
                patch.object(q, "crn_records", lambda meter: [{"scene": meter.scene}]):
            runner = q.QualificationRunner(temp, {}, baselines, refs, collector=collector,
                evaluator=evaluator, persister=persister, gate=lambda *args: (_ for _ in ()).throw(AssertionError("gate failed")))
            count = len(train_calls)
            with self.assertRaisesRegex(AssertionError, "gate failed"):
                runner.run_seed(ClockPolicy(cfg, 201))
            self.assertEqual(len(train_calls), count)
            self.assertFalse((Path(temp) / "seed201/boundary001").exists())


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        saved, _ = load_source(load_spec())
        cls.cfg = stage_config(saved["config"], 201)
        with isolated_rng():
            cls.sac = ReadOnlySAC(saved)

    def test_preflight_reads_exact_source_and_all_comparators_without_simulation(self):
        with patch.object(q, "collect_stream", side_effect=AssertionError("No simulation")):
            saved, lock, baselines, references, head = q.preflight(require_committed=False)
        self.assertEqual(lock["source_sha256"], "be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932")
        self.assertEqual(len(baselines), 8)
        self.assertEqual(len(references), 8)
        self.assertEqual(set(lock["configs"]), {"201", "202", "203"})
        self.assertEqual(saved["run_state"]["completed_windows"], 2000)
        self.assertEqual(len(head), 40)
        q.verify_sources(lock)

    def test_pending_90_180_270_and_post_update_resume_bitwise(self):
        for count in (90, 180, 270, 360, 630):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                uninterrupted = MissionMacroPPO(self.cfg, 201, self.sac)
                synthetic_feed(uninterrupted, count)
                payload = q.boundary_payload(uninterrupted, count//90, {"test": "synthetic"})
                path = Path(temp) / "checkpoint.pt"
                torch.save(payload, path)
                restored = MissionMacroPPO(self.cfg, 201, self.sac)
                before_rng = capture_rng()
                q.restore_boundary(restored, torch.load(path, map_location="cpu", weights_only=False), {"test": "synthetic"})
                self.assertTrue(state_equal(before_rng, capture_rng()))
                self.assertTrue(state_equal(uninterrupted.ppo_state(), restored.ppo_state()))
                self.assertEqual(len(restored.macro_rollout), count % 300)
                synthetic_feed(uninterrupted, ((count//300)+1)*300+60)
                synthetic_feed(restored, ((count//300)+1)*300+60)
                self.assertTrue(state_equal(uninterrupted.ppo_state(), restored.ppo_state()))
                self.assertGreater(restored.ppo_updates, 0)
                self.assertGreater(restored.ppo_gradient_steps, 0)
                self.assertTrue(all(self.sac.assert_unchanged().values()))

    def test_corrupt_or_smoke_checkpoint_rejected(self):
        policy = MissionMacroPPO(self.cfg, 201, self.sac)
        payload = q.boundary_payload(policy, 0, {})
        for key, value in (("stage", "mission_direct_ppo_smoke_only"), ("next_scene_index", 1),
                           ("ppo_seed", 202), ("replay", {}), ("lock", {"changed": True})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.restore_boundary(policy, dict(payload, **{key: value}), {})
        bad = copy.deepcopy(payload)
        bad["ppo"]["actor"]["log_std"][0] = float("nan")
        with self.assertRaises(ValueError):
            q.restore_boundary(policy, bad, {})

    def test_evaluation_does_not_mutate_pending_rollout_or_rng(self):
        policy = MissionMacroPPO(self.cfg, 201, self.sac)
        synthetic_feed(policy, 90)
        before, rng = policy.ppo_state(), capture_rng()

        def evaluate(config, agent, scene, **kwargs):
            self.assertEqual(kwargs, dict(train=False, macros=90))
            for _ in range(3):
                agent.macro_action(np.ones(67, dtype=np.float32)*.1, deterministic=True)
            np.random.random(3)
            torch.rand(3)
            return "synthetic monitor"

        with patch.object(q, "collect_stream", evaluate):
            self.assertEqual(q.evaluate_full(policy, 33600), "synthetic monitor")
        self.assertTrue(state_equal(before, policy.ppo_state()))
        self.assertTrue(state_equal(rng, capture_rng()))
        with patch.object(q, "collect_stream", side_effect=lambda *a, **kw: policy.macro_rollout.clear()):
            with self.assertRaisesRegex(AssertionError, "Monitor changed"):
                q.evaluate_full(policy, 33600)


class RealPrefix(unittest.TestCase):
    def test_one_real_10s_prefix_not_a_formal_gate(self):
        torch.set_num_threads(1)
        saved, lock, _, _, _ = q.preflight(require_committed=False)
        with isolated_rng():
            sac = ReadOnlySAC(saved)
        cfg = stage_config(saved["config"], 201)
        del saved
        policy = MissionMacroPPO(cfg, 201, sac)
        with frozen_sac_guard(sac), policy.exploration_rng():
            result = collect_stream(cfg, policy, 72000, train=True, macros=1)
        summary, meter = result[:2]
        self.assertEqual((summary["time_s"], summary["macro_decisions"], summary["sac_decisions"]), (10, 1, 20))
        self.assertEqual((policy.collected_macros, policy.ppo_updates, len(policy.macro_rollout)), (1, 0, 1))
        self.assertEqual(meter.reset_count, 1)
        self.assertEqual(summary["replay_writes"], 0)
        self.assertEqual(summary["replay_samples"], 0)
        self.assertTrue(all(sac.assert_unchanged().values()))
        with self.assertRaises(ValueError):
            q.boundary_payload(policy, 1, lock)
        with tempfile.TemporaryDirectory() as temp:
            row = q.persist_case(Path(temp) / "prefix", policy, 72000, 0, result)
            self.assertEqual(row["duration_s"], 10)
            self.assertEqual(row["ppo_updates"], 0)
            self.assertEqual(len(q.read_csv(Path(temp) / "prefix/queues.csv")), 20)
        q.verify_sources(lock)
        print("REAL_PREFIX_ONLY: scene72000,10s,20 deterministic SAC calls,0 PPO/SAC updates,no replay; full900s gate NOT run")


if __name__ == "__main__":
    unittest.main()
