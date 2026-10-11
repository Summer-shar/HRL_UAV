"""Count real physical steps while running existing bounded validation suites."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--suite",choices=("package","entry","cuda"),required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from hrl_uav.industrial_env import IndustrialEnv
    original=IndustrialEnv.micro_step
    count=[0]
    def counted(*a,**k):
        result=original(*a,**k); count[0]+=1; return result
    IndustrialEnv.micro_step=counted
    started=time.perf_counter(); passed=False
    try:
        if args.suite=="package":
            sys.argv=["validate","--output",str(args.output),"--skip-benchmark"]
            runpy.run_module("hrl_v2.validate",run_name="__main__")
            passed=True
        else:
            if args.suite=="cuda":
                import os
                os.environ["V2_EQ_DEVICE"]="cuda:0"
                from hrl_v2.tests.test_optimizer_reuse import OptimizerReuseTests
                suite=unittest.defaultTestLoader.loadTestsFromTestCase(OptimizerReuseTests)
            else:
                suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
            with (args.output/"tests.txt").open("w",encoding="utf-8") as stream:
                result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            passed=result.wasSuccessful()
            record=dict(passed=passed,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                        skipped=[dict(test=test.id(),reason=reason) for test,reason in result.skipped])
            if args.suite=="cuda": record["measurements"]=OptimizerReuseTests.measurements
            (args.output/"results.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
            print(json.dumps({k:v for k,v in record.items() if k!="measurements"}),flush=True)
    finally:
        IndustrialEnv.micro_step=original
        record=dict(suite=args.suite,physical_steps=count[0],seconds=time.perf_counter()-started,passed=passed,
                    research_training_episodes=0)
        (args.output/"execution.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        print(json.dumps(record),flush=True)
    if not passed: raise SystemExit(1)


if __name__=="__main__": main()
