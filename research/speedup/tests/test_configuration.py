import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hrl_v2.paths import default_config
from hrl_v2.run import read_config, train, technical_status
from hrl_v2.schedule import expected_counts
from hrl_v2.timing import ClockConfig


def isolated_cli():
    # -I ignores PYTHONPATH and editable installs may point at another worktree.
    source = Path(__file__).resolve().parents[1]/"src"
    bootstrap = ("import sys,runpy;sys.path.insert(0,"+repr(str(source))+");"
                 "import hrl_v2,hrl_uav;from pathlib import Path;"
                 "assert all(Path("+repr(str(source))+ ") in Path(m.__file__).resolve().parents "
                 "for m in (hrl_v2,hrl_uav));runpy.run_module('hrl_v2',run_name='__main__')")
    return [sys.executable,"-I","-B","-c",bootstrap]


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
            result=subprocess.run(isolated_cli()+["dry-run","--config",str(path)],
                                  capture_output=True,text=True,cwd=directory)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(result.stdout)
            self.assertEqual(data["rollout"],dict(physical_steps=600,lower_steps=120,upper_steps=12))
            self.assertFalse(data["training_started"])
            self.assertFalse(data["training_authorized"])
            self.assertTrue(data["research_protocol_confirmed"])

    def test_training_fails_before_outputs_or_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"no_run"
            config=self.config(); config["research"]["training_enabled"]=False
            path=Path(directory)/"unconfirmed.json"; path.write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"Research training is disabled"):
                train(Path(directory),path,4101,output)
            self.assertFalse(output.exists())

    def test_confirmed_configuration_still_requires_matching_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"no_run"
            with self.assertRaisesRegex(ValueError,"same --workspace"):
                train(Path(directory),default_config(),4101,output)
            self.assertFalse(output.exists())

    def test_readiness_is_not_authorization_and_identity_is_checked(self):
        from hrl_v2.checkpoint import identity,runtime_record
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/"validation/validation.json"
            path.parent.mkdir()
            record=dict(status="passed",identity=identity(root,default_config()),runtime=runtime_record())
            path.write_text(json.dumps(record),encoding="utf-8")
            status=technical_status(root,default_config(),self.config())
            self.assertTrue(status["training_ready"])
            self.assertFalse(status["training_authorized"])
            self.assertFalse(status["training_started"])
            record["identity"]["config_sha256"]="stale"
            path.write_text(json.dumps(record),encoding="utf-8")
            self.assertFalse(technical_status(root,default_config(),self.config())["validation_ready"])

    def test_objective_units_cannot_silently_disagree_with_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            c=self.config(); c["objective"]["energy_coefficient"]=1.
            path=Path(directory)/"wrong.json"; path.write_text(json.dumps(c),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"physical objective"):
                read_config(path,directory)

    def test_resume_fails_before_loading_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config=self.config(); config["research"]["training_enabled"]=False
            path=Path(directory)/"unconfirmed.json"; path.write_text(json.dumps(config),encoding="utf-8")
            result=subprocess.run(isolated_cli()+["resume","absent.pt","--config",str(path)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Research training is disabled",result.stderr)
        self.assertNotIn("FileNotFoundError",result.stderr)


if __name__=="__main__": unittest.main()
