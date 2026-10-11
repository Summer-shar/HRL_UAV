"""Independent pure-Python execution-count audit; no models or simulation."""
from collections import Counter
import json
from pathlib import Path
import unittest

from hrl_v2.schedule import expected_counts, phase
from hrl_v2.timing import ClockConfig


def count_seed():
    config=json.loads((Path(__file__).parents[1]/"configs/v2_1_0.json").read_text())
    clock=ClockConfig.from_config(config)
    c=Counter(checkpoint_saves=1)
    replay_version=None
    replay_episodes=0
    upper_version=0
    ppo_pending=0
    for number in range(1,config["schedule"]["total_episodes_per_seed"]+1):
        p=phase(number)
        c["episodes"]+=1
        c["physics_steps"]+=clock.physical_steps
        c["lower_inference"]+=clock.lower_steps
        c["episode_start_cache_rebuilds"]+=1
        if p.stage!=1:
            # The final real lower successor commits another upper action.
            c["upper_inference"]+=clock.upper_steps+1
        if p.layer=="lower":
            version="hover" if p.stage==1 else upper_version
            if version!=replay_version:
                replay_episodes=0
                replay_version=version
                c["replay_flushes"]+=1
            for frame in range(config["sac"]["learn_every_routing_frames"],clock.lower_steps+1,
                               config["sac"]["learn_every_routing_frames"]):
                c["sac_triggers"]+=1
                c["learning_log_rows"]+=1
                if not replay_episodes:
                    c["skipped_events"]+=1
                    continue
                c["sac_events"]+=1
                c["critic_fits"]+=2
                c["replay_samples"]+=2
                if p.actor_enabled and c["sac_events"]%2==0:
                    c["sac_actor_updates"]+=1
                    c["replay_samples"]+=1
                    # The event at frame 1800 has no following online action.
                    c["interior_actor_cache_rebuilds"]+=int(frame<clock.lower_steps)
            replay_episodes=min(config["replay"]["max_completed_episodes"],replay_episodes+1)
        else:
            c["current_value_inference"]+=clock.upper_steps+1
            ppo_pending+=1
            if ppo_pending==5:
                ppo_pending=0
                upper_version+=1
                c["ppo_batches"]+=1
                c["learning_log_rows"]+=1
        c["checkpoint_saves"]+=1
        if number in config["schedule"]["development_global_episodes"]:
            panel=config["evaluation"]["development_scenarios"]*config["evaluation"]["development_policy_streams"]
            c["dev_hover_episodes"]+=panel
            c["dev_moving_episodes"]+=panel if number!=300 else 0
            c["checkpoint_saves"]+=1
    c["dev_episodes"]=c["dev_hover_episodes"]+c["dev_moving_episodes"]
    c["dev_physics_steps"]=clock.physical_steps*c["dev_episodes"]
    c["dev_lower_inference"]=clock.lower_steps*c["dev_episodes"]
    c["dev_upper_inference"]=(clock.upper_steps+1)*c["dev_moving_episodes"]
    c["ppo_value_optimizer_steps"]=c["ppo_batches"]*6*6
    c["ppo_actor_optimizer_steps_max"]=c["ppo_value_optimizer_steps"]
    c["ppo_kl_sweeps_max"]=c["ppo_batches"]*6
    return dict(c)


class ThroughputBudgetTests(unittest.TestCase):
    def test_current_research_update_budget(self):
        actual=count_seed()
        self.assertEqual({key:actual[key] for key in expected_counts()},expected_counts())
        self.assertEqual((actual["sac_triggers"],actual["skipped_events"],actual["replay_flushes"]),(36000,990,11))
        self.assertEqual((actual["replay_samples"],actual["learning_log_rows"]),(87030,36040))
        self.assertEqual((actual["ppo_value_optimizer_steps"],actual["ppo_actor_optimizer_steps_max"],actual["ppo_kl_sweeps_max"]),(1440,1440,240))

    def test_current_sampling_cutoff_cache_development_and_io_counts(self):
        actual=count_seed()
        self.assertEqual((actual["lower_inference"],actual["upper_inference"]),(1080000,54300))
        self.assertEqual(actual["current_value_inference"],36200)
        self.assertEqual((actual["interior_actor_cache_rebuilds"],actual["episode_start_cache_rebuilds"]),(16632,600))
        self.assertEqual((actual["dev_moving_episodes"],actual["dev_hover_episodes"],actual["dev_episodes"]),(72,84,156))
        self.assertEqual((actual["dev_lower_inference"],actual["dev_upper_inference"]),(280800,13032))
        self.assertEqual(actual["physics_steps"]+actual["dev_physics_steps"],6804000)
        self.assertEqual(actual["checkpoint_saves"],608)


if __name__=="__main__": unittest.main()
