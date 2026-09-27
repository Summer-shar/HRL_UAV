"""Public CLI; help and no-argument invocation never start training."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def invoke(module,args):
    old=sys.argv
    try:
        sys.argv=[module,*args]
        runpy.run_module(module,run_name="__main__")
    finally: sys.argv=old


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version",action="version",version="hrl-uav-v2 2.1.0")
    parser.add_argument("--workspace",type=Path,default=Path.cwd()/"outputs",help="Writable validation/run directory")
    parser.add_argument("command",nargs="?",default="check",choices=("check","dry-run","validate","validate-cuda","train","resume","evaluate","fixture","benchmark","budget"))
    parser.add_argument("arguments",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    rest=args.arguments
    workspace=args.workspace.resolve()
    if args.command=="check":
        if rest: parser.error("unrecognized arguments: "+" ".join(rest))
        import torch
        from .paths import default_config
        from .runtime import configure
        from .run import read_config,technical_status
        configure(test_cpu=True)
        config=read_config(default_config(),workspace)
        print(json.dumps(dict(version="2.1.0",python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),workspace=str(workspace),
            **technical_status(workspace,default_config(),config),updates=0),indent=2))
    elif args.command in ("dry-run","train","resume"):
        extra=[]
        if args.command=="train": extra=["--fresh"]
        if args.command=="resume":
            if not rest: parser.error("resume requires a checkpoint path from latest.json")
            extra=["--resume"]
        invoke("hrl_v2.run",["--project-root",str(workspace),*extra,*rest])
    elif args.command=="evaluate": invoke("hrl_v2.evaluate",["--project-root",str(workspace),*rest])
    elif args.command=="validate": invoke("hrl_v2.validate",["--output",str(workspace/"validation"),*rest])
    elif args.command=="validate-cuda":
        import os,unittest
        from .checkpoint import atomic_json
        os.environ["V2_EQ_DEVICE"]="cuda:0"
        from .tests.test_optimizer_reuse import OptimizerReuseTests
        output=workspace/"validation"; output.mkdir(parents=True,exist_ok=True)
        with (output/"cuda_tests.txt").open("w",encoding="utf-8") as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OptimizerReuseTests))
        record=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),measurements=OptimizerReuseTests.measurements)
        atomic_json(output/"cuda_tests.json",record)
        print(json.dumps({key:value for key,value in record.items() if key!="measurements"}))
        if not result.wasSuccessful(): raise SystemExit(1)
    else: invoke("hrl_v2."+{"fixture":"fixture","benchmark":"benchmark_optimization","budget":"budget"}[args.command],rest)
