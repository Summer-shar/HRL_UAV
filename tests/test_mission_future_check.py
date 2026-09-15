"""No-learning tests for saved-state conditional future validation."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import isolated_rng
from hrl_uav.mission_future_check import (
    ROOT,SOURCE,SCENES,TIMES,FUTURES,LABELS,SavedBranch,future_seed,frozen_actions,
    verify_selection,physical_fields,physical_state,without_rng,semantic_hash,
    macro_step,crn_records,mission_config,load_source,load_spec,
)
from hrl_uav.mission_future_analysis import summarize
from hrl_uav.resumption import state_equal
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.trajectory_check import ReadOnlySAC,no_learning,forbidden


class MissionFutureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = json.loads((SOURCE/"manifest.json").read_text())["config"]
        self.fields = physical_fields(self.cfg)
        for target in ("torch.autograd.backward","torch.autograd.grad","hrl_uav.learning.DirectHRL.__init__",
                       "hrl_uav.agents.ppo_agent.ActorNet.__init__","hrl_uav.agents.ppo_agent.CriticNet.__init__"):
            context = patch(target,forbidden)
            context.start()
            self.addCleanup(context.stop)

    def saved_first(self):
        folder = SOURCE/"states/33600_t100"
        state = json.loads((folder/"state.json").read_text())
        return SavedBranch(folder/"physical_snapshot.pkl.gz",state["initial_state_sha256"],self.fields)

    def test_fixed_selection_and_rng_plan(self):
        _,plan,_ = verify_selection(self.cfg)
        self.assertEqual(len(plan),16)
        self.assertEqual(sum(len(p["candidates"])*len(p["futures"]) for p in plan),320)
        seeds = [future_seed(s,t,k) for s in SCENES for t in TIMES for k in FUTURES]
        self.assertEqual(len(set(seeds)),80)
        for s,t,k in ((33610,100,1),(33700,100,1),(33600,90,1),(33600,100,6)):
            with self.assertRaises(ValueError):
                future_seed(s,t,k)
        actions = frozen_actions("e5_plus")
        self.assertEqual([a["label"] for a in actions],list(LABELS))
        self.assertEqual(actions[1]["delta"],actions[2]["delta"])
        np.testing.assert_array_equal(np.asarray(actions[1]["delta"]),-np.asarray(actions[3]["delta"]))

    def test_disk_state_preserved_except_future_rng(self):
        saved = self.saved_first()
        first = saved.restore(future_seed(33600,100,1))
        second = saved.restore(future_seed(33600,100,2))
        self.assertEqual(semantic_hash(without_rng(physical_state(first,self.fields))),
                         semantic_hash(without_rng(physical_state(second,self.fields))))
        self.assertFalse(state_equal(first.env.np_random.get_state(),second.env.np_random.get_state()))
        self.assertEqual(first.env._current_time(),100)
        self.assertEqual(first.reset_count,0)
        self.assertIs(first.feature_slices,first.micro_feature_slices)
        self.assertIsNot(first.env.parent_tasks,second.env.parent_tasks)
        np.testing.assert_array_equal(first.schedule.phase_index_by_macro,second.schedule.phase_index_by_macro)

    def test_short_real_sac_four_way_CRN_and_duplicate(self):
        saved,_ = load_source(load_spec())
        with isolated_rng():
            sac = ReadOnlySAC(saved)
        snapshot = self.saved_first()
        meters,values = [],[]
        with isolated_rng(),no_learning(sac),torch.no_grad():
            for action in frozen_actions("e5_plus"):
                meter = snapshot.restore(future_seed(33600,100,1))
                values.append(macro_step(meter,sac,action["delta"]))
                meters.append(meter)
            for meter in meters[1:]:
                assert_crn_equal(crn_records(meters[0]),crn_records(meter))
            self.assertEqual(values[1]["cost"],values[2]["cost"])
            self.assertEqual(semantic_hash(physical_state(meters[1],self.fields)),semantic_hash(physical_state(meters[2],self.fields)))
            other = snapshot.restore(future_seed(33600,100,2))
            macro_step(other,sac,np.zeros(6))
            self.assertNotEqual(crn_records(meters[0])[-1]["arrivals_sha256"],crn_records(other)[-1]["arrivals_sha256"])
            self.assertNotEqual(crn_records(meters[0])[-1]["mobility_velocity_states_sha256"],crn_records(other)[-1]["mobility_velocity_states_sha256"])
            self.assertTrue(all(sac.assert_unchanged().values()))

    def test_analysis_does_not_reselect_or_mix_original_future(self):
        plan,rows = [],[]
        for s in SCENES:
            for t in TIMES:
                candidates = frozen_actions("e2_plus")
                plan.append(dict(scene=s,start_s=t,selected_direction="e2_plus",candidates=candidates,
                                 stage3a_delta_C60=-10,stage3a_delta_holding=-10,stage3a_delta_energy=0,stage3a_relative_percent=-10))
                for k in FUTURES:
                    for a in candidates:
                        dc = dict(zero=0,stage3a_best=-1 if k==1 else 1,fixed_best=-2,opposite=-.5)[a["label"]]
                        row = dict(seed=s,start_s=t,future=k,action=a["label"],direction=a["name"],
                                   C60=100+dc,C60_holding=80+dc,C60_energy=20,ppo_return60=-100-dc,
                                   ineffective=False,power_w=800,task_delay_weighted_ms=400,task_delay_high_ms=350,task_delay_low_ms=700)
                        rows.append(row)
        overall,states,pairs,_ = summarize(rows,plan)
        selected = overall["actions"][0]
        self.assertEqual(selected["improving_state_future_pairs"],16)
        self.assertEqual(selected["new_mean_improving_states"],0)
        self.assertAlmostEqual(selected["state_mean_delta_C60_mean"],.6)
        self.assertAlmostEqual(overall["fresh_minus_selection_delta_C60"],10.6)
        self.assertEqual(len(states),16)
        self.assertEqual(len(pairs),80)
        self.assertTrue(all(s["selected_direction"]=="e2_plus" for s in states))


if __name__ == "__main__":
    unittest.main()
