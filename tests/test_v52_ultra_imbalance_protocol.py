import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(name):
    with open(
        os.path.join(ROOT, "configs", name),
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


class V52UltraImbalanceProtocolTest(unittest.TestCase):
    def setUp(self):
        self.v51 = load_config("tcom_v51_actor_autonomous_gate.json")
        self.v52 = load_config("tcom_v52_ultra_imbalance_hrl.json")

    def test_v52_changes_only_scene_identity_loads_and_method_matrix(self):
        allowed = {
            "results_version",
            "micro_pretrain_source_results_version",
            "enabled_methods",
            "zone_loads_mbps",
            "load_switch_phases_mbps",
        }
        shared_keys = set(self.v51) | set(self.v52)
        changed = {
            key
            for key in shared_keys
            if self.v51.get(key) != self.v52.get(key)
        }
        self.assertEqual(changed, allowed)

    def test_all_methods_are_independent_and_micro_remains_trainable(self):
        self.assertTrue(self.v52["independent_micro_pretraining"])
        self.assertFalse(self.v52["freeze_micro_in_joint"])
        self.assertEqual(
            self.v52["results_version"],
            self.v52["micro_pretrain_source_results_version"],
        )
        self.assertEqual(
            set(self.v52["enabled_methods"]),
            {
                "joint_ppo_gated_sac",
                "cppo_sca",
                "joint_ppo_sac",
                "centroid_gated_sac",
                "centroid_sac",
                "fixed_sac",
                "joint_td3_sac",
                "no_coop",
            },
        )

    def test_each_phase_is_locally_overloaded_but_globally_feasible(self):
        scale = float(self.v52["load_scale"])
        service_capacity_mbit = float(self.v52["cpu_ghz"]) * 9.0
        system_capacity_mbit = 3.0 * service_capacity_mbit
        for phase in self.v52["load_switch_phases_mbps"]:
            arrivals = [float(value) * scale * 10.0 for value in phase]
            self.assertEqual(sum(value > service_capacity_mbit for value in arrivals), 1)
            self.assertLess(sum(arrivals), system_capacity_mbit)
            self.assertAlmostEqual(sum(phase), 21.5)

    def test_no_hard_power_constraint_or_dual_is_reintroduced(self):
        self.assertEqual(self.v52["objective_mode"], "weighted")
        self.assertEqual(self.v52["constraint_form"], "soft_weighted")
        self.assertEqual(self.v52["power_budget_w"], 0.0)
        self.assertEqual(self.v52["energy_dual_lr"], 0.0)
        self.assertEqual(
            self.v52["macro_power_violation_penalty_weight"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
