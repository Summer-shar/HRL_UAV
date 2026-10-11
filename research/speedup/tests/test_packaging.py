import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hrl_v2.paths import source_root,default_config
from hrl_v2.checkpoint import identity,runtime_signature


def isolated_cli():
    # -I ignores PYTHONPATH and editable installs may point at another worktree.
    source = Path(__file__).resolve().parents[1]/"src"
    bootstrap = ("import sys,runpy;sys.path.insert(0,"+repr(str(source))+");"
                 "import hrl_v2,hrl_uav;from pathlib import Path;"
                 "assert all(Path("+repr(str(source))+ ") in Path(m.__file__).resolve().parents "
                 "for m in (hrl_v2,hrl_uav));runpy.run_module('hrl_v2',run_name='__main__')")
    return [sys.executable,"-I","-B","-c",bootstrap]


class PackagingTests(unittest.TestCase):
    def test_candidate_discounts_preserve_other_algorithm_numeric_choices(self):
        current=json.loads(default_config().read_text(encoding="utf-8"))
        baseline=json.loads((source_root()/"hrl_v2/_reference/hrl_v2/configs/v2_1_0.json").read_text(encoding="utf-8"))
        baseline["runtime"].pop("preserve_local_python")
        for key in ("runtime","network","action","sac","information"):
            self.assertEqual(current[key],baseline[key],key)
        for key in ("holding_coefficient","energy_coefficient","reward_scale"):
            self.assertEqual(current["objective"][key],baseline["objective"][key])
        for key,value in baseline["ppo"].items():
            if key not in ("macro_samples_per_batch","advantage_normalization","gamma","gae_lambda"):
                self.assertEqual(current["ppo"][key],value,key)
        self.assertEqual(current["objective"]["gamma_micro"],.99)
        self.assertAlmostEqual(current["objective"]["gamma_macro"],.99**10)
        self.assertEqual(current["ppo"]["gamma"],current["objective"]["gamma_macro"])
        self.assertEqual(current["ppo"]["gae_lambda"],.95)
        self.assertTrue(current["research"]["training_enabled"])

    def test_identity_does_not_depend_on_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(identity(Path(directory)/"one",default_config()),identity(Path(directory)/"two",default_config()))

    def test_runtime_path_is_not_numerical_contract(self):
        a=dict(python="3.11",torch="2.7",numpy="1.26",cuda="12.8",device="GPU",deterministic=True,executable="/one/python",platform="Windows")
        b=dict(a,executable="/two/python",platform="Linux")
        self.assertEqual(runtime_signature(a),runtime_signature(b))
        self.assertNotEqual(runtime_signature(a),runtime_signature(dict(b,torch="changed")))

    def test_cli_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run(isolated_cli()+["--workspace",str(Path(directory)/"work"),"dry-run"],cwd=directory,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(result.stdout)
            self.assertFalse(data["training_started"])
            self.assertEqual(data["expected_counts"]["critic_fits"],70020)
            self.assertFalse((Path(directory)/"work").exists())

    def test_embedded_reference_is_self_contained(self):
        from hrl_v2.benchmark_optimization import verify_baseline
        self.assertGreater(len(verify_baseline()),20)

if __name__=="__main__": unittest.main()
