"""Detached supervisor: resume own stream anchors, then aggregate once both finish."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import write_json
from hrl_uav.wireless_replicates import load_json


def main():
    output = ROOT / "results/sac_only_wireless6_recovery_supervisor"
    output.mkdir(exist_ok=False)
    state = dict(status="STARTING", children={}, start_time=time.time(), training_initializations=[102, 103])
    children, handles = {}, []
    try:
        for seed in (102, 103):
            folder = ROOT / f"results/sac_only_wireless6_long_seed{seed}"
            source = load_json(folder / "manifest.json")["latest_resume_anchor"]
            command = [sys.executable, str(ROOT / "scripts/run_sac_only_wireless6_replicates.py"),
                       "--seed", str(seed), "--resume-anchor", source["path"]]
            stdout = (output / f"seed{seed}.stdout.txt").open("w", encoding="utf-8", buffering=1)
            stderr = (output / f"seed{seed}.stderr.txt").open("w", encoding="utf-8", buffering=1)
            handles.extend([stdout, stderr])
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            children[seed] = child
            state["children"][str(seed)] = dict(pid=child.pid, command=command,
                source_anchor=source, exit_code=None)
        state["status"] = "RUNNING"
        write_json(output / "status.json", state)
        while any(child.poll() is None for child in children.values()):
            for seed, child in children.items():
                state["children"][str(seed)]["exit_code"] = child.poll()
            state["last_check_time"] = time.time()
            write_json(output / "status.json", state)
            time.sleep(30)
        for seed, child in children.items():
            state["children"][str(seed)]["exit_code"] = child.returncode
        if any(child.returncode != 0 for child in children.values()):
            state["status"] = "CHILD_ERROR"
        else:
            state["status"] = "AGGREGATING"
            write_json(output / "status.json", state)
            with (output / "aggregate.txt").open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, str(ROOT / "scripts/run_sac_only_wireless6_replicates.py"),
                    "--aggregate"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(status="COMPLETE" if result.returncode == 0 else "AGGREGATE_ERROR",
                         aggregate_exit_code=result.returncode)
        state["end_time"] = time.time()
        write_json(output / "status.json", state)
    except BaseException as error:
        state.update(status="SUPERVISOR_ERROR", error=repr(error), end_time=time.time())
        write_json(output / "status.json", state)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
