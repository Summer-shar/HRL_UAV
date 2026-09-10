import copy
import json
import unittest

import numpy as np

from hrl_uav.mdp import QueueMDP
from hrl_uav.priority_grid import LEVELS, SOURCE, PriorityRouting, conditions, window_metrics
from hrl_uav.scene_prescreen import LocalRouting
from hrl_uav.bandwidth import UniformRouting
from hrl_uav.wireless_check import ENERGY


class PriorityGridTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((SOURCE/"manifest.json").read_text(encoding="utf-8"))["arm_configs"]["w1_uniform_b6"]

    def test_all_routes_and_anchor_equivalence(self):
        mdp = QueueMDP(self.config)
        for _, h in LEVELS:
            for _, l in LEVELS:
                agent = PriorityRouting(h, l)
                route = mdp.route(agent.choose_action(None))
                for p, a in enumerate((h,l)):
                    for i in range(3):
                        for j in range(3):
                            self.assertAlmostEqual(route[p,i,j], a if i == j else (1-a)/2, places=14)
                np.testing.assert_array_equal(agent.choose_action(np.ones(68)), agent.choose_action(np.zeros(68)))
        for value, other in ((1, LocalRouting()), (1/3, UniformRouting())):
            np.testing.assert_array_equal(mdp.route(PriorityRouting(value,value).choose_action(None)),
                                          mdp.route(other.choose_action(None)))
        with self.assertRaises(ValueError):
            PriorityRouting(0.4, 1)

    def test_budget_and_only_bandwidth_change(self):
        before = copy.deepcopy(self.config)
        arms = conditions(self.config)
        self.assertEqual(self.config, before)
        self.assertEqual(len(arms),31)
        self.assertEqual(len({a["policy"] for a in arms}),31)
        self.assertEqual(sum(a["source_policy"] is not None for a in arms)*3,9)
        self.assertEqual(sum(a["source_policy"] is None for a in arms)*3,84)
        for arm in arms:
            cfg = copy.deepcopy(arm["config"])
            cfg["environment"]["u2u_bandwidth_mhz"] = 6
            self.assertEqual(cfg,self.config)
        self.assertEqual(sum(a["alpha_high"] == a["alpha_low"] == 1 for a in arms),1)

    def test_window_difference_and_delayed_completions(self):
        service = []
        for i in range(1,1801):
            row = dict(time_s=i/2,queue_compute_mbit=1.,queue_transfer_mbit=2.,queue_upload_mbit=3.,
                service_steps=5,compute_steps=5,active_link_steps=5,active_link_bandwidth_mhz_sum=5,
                active_link_rate_mbps_sum=5,tx_service_mbit=1,tx_capacity_mbit=2,compute_service_mbit=1,
                cpu_utilization_sum=2.5)
            row.update({key:0.5 for key in ENERGY})
            for p in ("high","low"):
                row.update({f"task_arrived_{p}_count":i+2,f"task_completed_{p}_count":i,
                            f"task_unfinished_{p}_count":2,f"task_completion_{p}_rate":i/(i+2)})
            service.append(row)
        macro = []
        for end,j,delay in ((300,1,100),(600,1.5,200),(900,2,300)):
            macro.append(dict(time_s=end,J=j,power_w=4,queue_total_mbit=6,
                task_completed_high_count=end*2,task_completed_low_count=end*2,
                task_delay_high_ms=delay,task_delay_low_ms=delay))
        windows = window_metrics(service,macro,"test",33502)
        self.assertEqual([w["J"] for w in windows],[1,2,3])
        self.assertEqual([w["task_delay_high_ms"] for w in windows],[100,300,500])
        self.assertEqual([w["task_completed_high_count"] for w in windows],[600]*3)
        self.assertEqual([w["power_w"] for w in windows],[4]*3)


if __name__ == "__main__":
    unittest.main()
