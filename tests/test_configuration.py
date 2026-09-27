import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hrl_v2.paths import default_config
from hrl_v2.run import read_config, train
from hrl_v2.schedule import expected_counts
from hrl_v2.timing import ClockConfig


class ConfigurationTests(unittest.TestCase):
    def config(self):
        return json.loads(default_config().read_text(encoding="utf-8"))

    def test_rollout_counts_derived_from_one_value(self):
        c=self.config()
        self.assertEqual(ClockConfig.from_config(c).counts(),dict(physical_steps=9000,lower_steps=1800,upper_steps=180))
        c["task"]["rollout_seconds"]=60
        self.assertEqual(ClockConfig.from_config(c).counts(),dict(physical_steps=600,lower_steps=120,upper_steps=12))
        self.assertEqual(expected_counts(c)["physics_steps"],600*600)

    def test_bad_clocks_rejected(self):
        for key,value in (("rollout_seconds",61),("rollout_seconds",0),
                          ("physical_dt",float("nan")),("lower_interval",.35),
                          ("upper_interval",4.7),("rollout_seconds",True)):
            c=self.config(); c["task"][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                ClockConfig.from_config(c)

    def test_60_second_config_works_without_source_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            c=self.config(); c["task"]["rollout_seconds"]=60
            path=Path(directory)/"short.json"; path.write_text(json.dumps(c),encoding="utf-8")
            self.assertEqual(read_config(path,directory),c)
            result=subprocess.run([sys.executable,"-I","-B","-m","hrl_v2","dry-run","--config",str(path)],
                                  capture_output=True,text=True,cwd=directory)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(result.stdout)
            self.assertEqual(data["rollout"],dict(physical_steps=600,lower_steps=120,upper_steps=12))
            self.assertFalse(data["training_started"])

    def test_training_fails_before_outputs_or_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"no_run"
            with self.assertRaisesRegex(RuntimeError,"Research training is disabled"):
                train(Path(directory),default_config(),4101,output)
            self.assertFalse(output.exists())

    def test_objective_units_cannot_silently_disagree_with_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            c=self.config(); c["objective"]["energy_coefficient"]=1.
            path=Path(directory)/"wrong.json"; path.write_text(json.dumps(c),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"physical objective"):
                read_config(path,directory)

    def test_resume_fails_before_loading_any_checkpoint(self):
        result=subprocess.run([sys.executable,"-I","-B","-m","hrl_v2","resume","absent.pt"],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Research training is disabled",result.stderr)
        self.assertNotIn("FileNotFoundError",result.stderr)


if __name__=="__main__": unittest.main()
