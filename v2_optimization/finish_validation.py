"""Run the final V2 suite and record CPU/CUDA integration differences."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["V2_EQ_DEVICE"] = "cpu"

from hrl_v2 import validate
from hrl_v2.tests.test_optimizer_reuse import OptimizerReuseTests


def record(device, passed, tests, elapsed):
    result = dict(
        baseline="137d2f1", device=device, passed=passed, tests_run=tests,
        elapsed_seconds=elapsed,
        scope="Temporary real V2 networks; 17 SAC samples with microbatch 16; 3 PPO samples with microbatch 2; no research training",
        tolerances=dict(parameters=dict(atol=3e-5, rtol=2e-5),
                        clipped_gradients=dict(atol=2e-6, rtol=2e-3),
                        adam_state=dict(atol=2e-7, rtol=2e-3),
                        graph_outputs=dict(atol=2e-5, rtol=2e-5),
                        graph_gradients=dict(atol=2e-5, rtol=2e-4)),
        note="Float32 summation and Q decomposition are not bitwise identical. First-step Adam with eps=1e-8 can amplify differences in near-zero gradients; exact parameter positions and gradients are recorded.",
        measurements=OptimizerReuseTests.measurements,
    )
    suffix="cuda" if device.startswith("cuda") else "cpu"
    path=ROOT / f"hrl_v2/validation/optimization_equivalence_{suffix}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    sys.argv=["hrl_v2.validate", "--skip-benchmark"]
    start=time.perf_counter()
    validate.main()
    record("cpu", True, 4, time.perf_counter()-start)
    OptimizerReuseTests.measurements={}
    os.environ["V2_EQ_DEVICE"]="cuda:0"
    start=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(OptimizerReuseTests)
    with (ROOT / "hrl_v2/validation/equivalence_cuda_tests.txt").open("w", encoding="utf-8") as log:
        result=unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    record("cuda:0", result.wasSuccessful(), result.testsRun, time.perf_counter()-start)
    print(json.dumps(dict(cuda_equivalence_passed=result.wasSuccessful(), tests=result.testsRun,
                          failures=len(result.failures), errors=len(result.errors))), flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)
    from cold_inference import run
    run(ROOT)
    # run.persist hashes the checkpoint again for latest.json. Its digest
    # uses the same full-file reader; freshly written files are usually
    # in the OS cache, as this bounded reference is after its first read.
    from hrl_v2.checkpoint import digest
    fixture=ROOT / "v2_optimization/independent_fixture.pkl"
    digest(fixture)
    samples=[]
    for _ in range(5):
        start=time.perf_counter()
        fingerprint=digest(fixture)
        samples.append(time.perf_counter()-start)
    io=dict(file_bytes=fixture.stat().st_size, seconds=statistics.median(samples),
            samples_seconds=samples, sha256=fingerprint,
            scope="Production SHA256 read_bytes digest on a bounded fixture after one warmup; used for the additional checkpoint identity reads, not serialization")
    (ROOT / "hrl_v2/validation/throughput_extra_io.json").write_text(json.dumps(io, indent=2)+"\n", encoding="utf-8")
