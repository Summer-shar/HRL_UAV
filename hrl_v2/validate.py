"""Run bounded implementation tests and record real device measurements."""
import argparse
import json
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from .checkpoint import atomic_json,identity,runtime_record


def benchmark(episode):
    from .runtime import configure,Learner
    from .replay import unpack,Replay
    from .randomness import Streams
    from .policy import Policy
    from .tests.test_learning import fingerprint
    device=configure(); learner=Learner(4101,device); history=unpack(episode.history)
    indices=sorted(range(1800),key=lambda i:len(episode.report(episode.observations[i]).fragments))
    states={"median_fragment_count":episode.report(episode.observations[indices[900]]),
            "max_fragment_count":episode.report(episode.observations[indices[-1]])}
    policy=Policy(learner.sac.actor,learner.ppo.actor,None,Streams(4101))
    results={}
    for label,report in states.items():
        for _ in range(3): policy.lower(report,history)
        timings=[]
        for _ in range(30):
            torch.cuda.synchronize(); start=time.perf_counter()
            policy.lower(report,history); torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        results[label]=dict(physical_tick=report.tick,parents=len(report.parents),fragments=len(report.fragments),
            history_frames=report.history_cursor,report_bytes=len(__import__("pickle").dumps(report,protocol=5)),
            samples=30,p50_seconds=float(np.percentile(timings,50)),p95_seconds=float(np.percentile(timings,95)),
            p99_seconds=float(np.percentile(timings,99)))
    replay=Replay(); replay.set_upper_version(episode.upper_version); replay.add(episode)
    streams=Streams(4101); upper_before=fingerprint(learner.ppo.actor.state_dict())
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    samples=replay.sample(streams.numpy)
    critic=learner.sac._critic_fit(samples,.005,streams)
    torch.cuda.synchronize(); critic_seconds=time.perf_counter()-start
    start=time.perf_counter()
    actor=learner.sac._actor_fit(replay.sample(streams.numpy),.005,streams)
    torch.cuda.synchronize(); actor_seconds=time.perf_counter()-start
    if fingerprint(learner.ppo.actor.state_dict())!=upper_before: raise AssertionError("SAC update changed PPO")
    import ctypes
    from ctypes import wintypes
    class ProcessMemory(ctypes.Structure):
        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
    memory=ProcessMemory(); memory.cb=ctypes.sizeof(memory)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemory),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(memory),memory.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    results["training_batch"]=dict(batch_size=128,microbatch_size=16,critic_fit_seconds=critic_seconds,
        actor_update_seconds=actor_seconds,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),critic=critic,actor=actor,
        process_working_set_bytes=memory.WorkingSetSize,process_peak_working_set_bytes=memory.PeakWorkingSetSize,
        note="One measured critic fit and actor update on scripted-trajectory replay; no research training")
    results["episode_storage_bytes"]=episode.storage_bytes
    results["network_parameter_counts"]={name:sum(p.numel() for p in module.parameters())
        for name,module in (("lower_actor",learner.sac.actor),("q1",learner.sac.q1),("q2",learner.sac.q2),
                            ("upper_actor",learner.ppo.actor),("value",learner.ppo.value))}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("hrl_v2/validation"))
    parser.add_argument("--skip-benchmark",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).absolute().parents[1]
    output=args.output if args.output.is_absolute() else root/args.output
    from .tests.test_physics import PhysicsTests
    discovered=unittest.defaultTestLoader.discover(str(root/"hrl_v2/tests"),top_level_dir=str(root))
    def cases(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite): yield from cases(item)
            else: yield item
    # Keep the full physical replay fixture available to the checkpoint test,
    # while automatically including new integration/optimization test modules.
    order={name:i for i,name in enumerate(("MathTests","PhysicsTests","LearningTests","PPOTests"))}
    suites=unittest.TestSuite(sorted(cases(discovered),
        key=lambda test:(order.get(type(test).__name__,4),test.id())))
    output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with (output/"tests.txt").open("w",encoding="utf-8") as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suites)
    record=dict(status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),test_seconds=time.perf_counter()-start,
        runtime=runtime_record(),identity=identity(root,root/"hrl_v2/configs/v2_1_0.json"),
        research_training_episodes=0,performance_claim=False)
    atomic_json(output/"validation.json",record)
    print(json.dumps(record|{"identity":"recorded in validation.json"},ensure_ascii=False),flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)
    if not args.skip_benchmark:
        measurement=benchmark(PhysicsTests.episode)
        atomic_json(output/"benchmark.json",measurement)
        print(json.dumps(measurement,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
