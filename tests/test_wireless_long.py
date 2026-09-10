"""Continuation-specific checks; physical-model coverage already exists."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav.experiment import isolated_rng
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.resumption import capture_rng, state_equal
from hrl_uav.wireless_long import (ORIGIN_SHA, exact_restore, normalize_checkpoint,
                                  stream_plan, window_optimization_summary)


class WirelessLongTest(unittest.TestCase):
    def test_plan_keeps_live_streams_and_shortens_only_last(self):
        plan=list(stream_plan())
        self.assertEqual(plan[0],(120,3,71040))
        self.assertEqual(plan[-1],(1998,2,71666))
        self.assertEqual(sum(n for _,n,_ in plan),1880)
        self.assertTrue(all(n==3 for _,n,_ in plan[:-1]))
        self.assertIn((198,3,71066),plan)
        self.assertIn((399,3,71133),plan)

    def test_complete_restore_preserves_next_native_update_and_rng(self):
        cfg=json.loads(Path("configs/sac_only_wireless6_pilot.json").read_text())
        cfg["training"].update(batch_size=4,replay_warmup=4,replay_capacity=8)
        mdp=QueueMDP(cfg)
        with isolated_rng(), tempfile.TemporaryDirectory() as tmp:
            torch.set_num_threads(1)
            a=DirectHRL.from_mdp(mdp,cfg)
            a.sac_updates=102
            a.sac_critic_gradient_steps=102
            a.sac_target_updates=102
            for i in range(12):
                state=np.full(68,i*.001,dtype=np.float32)
                score=np.zeros(18,dtype=np.float32)
                a.store(state,score,mdp.route(score),-.3,state+.001,.99)
            a.update_sac()
            a.update_sac()
            checkpoint=copy.deepcopy(a.checkpoint(cfg))
            checkpoint["origin_sha256"]=ORIGIN_SHA
            checkpoint["run_state"]=dict(completed_windows=123,at_rollout_boundary=True,
                next_training_scene_seed=71041,rng=capture_rng())
            path=Path(tmp)/"anchor.pt"
            torch.save(checkpoint,path)
            a.update_sac()
            expected=copy.deepcopy(a.checkpoint(cfg))
            expected_rng=capture_rng()
            b=DirectHRL.from_mdp(mdp,cfg)
            b.full_config=cfg
            audit=exact_restore(b,checkpoint,path,origin=False)
            self.assertTrue(all(audit["exact_restore_checks"].values()))
            b.update_sac()
            self.assertTrue(state_equal(expected,b.checkpoint(cfg)))
            self.assertTrue(state_equal(expected_rng,capture_rng()))
            broken=copy.deepcopy(checkpoint)
            del broken["replay"]["rng_state"]
            with self.assertRaisesRegex(ValueError,"rng_state"):
                normalize_checkpoint(broken,origin=False)
            broken=copy.deepcopy(checkpoint)
            broken["run_state"]["at_rollout_boundary"]=False
            with self.assertRaisesRegex(ValueError,"resume anchor"):
                normalize_checkpoint(broken,origin=False)

    def test_optimization_summary_consumes_only_new_window_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            keys=("critic_loss","critic_loss_last_fit","critic_grad","target_std",
                  "actor_loss","actor_grad","alpha","temperature_grad")
            from hrl_uav.experiment import append_csv
            append_csv(p/"optimization_stats.csv",[dict(window=121,**dict.fromkeys(keys,2.))])
            cursor={}
            window_optimization_summary(p,121,cursor)
            append_csv(p/"optimization_stats.csv",[dict(window=122,**dict.fromkeys(keys,4.))])
            window_optimization_summary(p,122,cursor)
            from hrl_uav.wireless_long import rows
            result=rows(p/"optimization_windows.csv")
            self.assertEqual([r["events"] for r in result],[1,1])
            self.assertEqual([r["critic_loss_mean"] for r in result],[2,4])


if __name__ == "__main__":
    unittest.main()
