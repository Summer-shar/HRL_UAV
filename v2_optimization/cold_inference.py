"""Continuous optimized Policy windows with no cross-repeat graph cache hits."""
import hashlib
import json
import pickle
from pathlib import Path
import statistics
import time

from hrl_v2.benchmark_optimization import sync
from hrl_v2.policy import Policy
from hrl_v2.prepared import fixed_graph
from hrl_v2.randomness import Streams
from hrl_v2.replay import unpack
from hrl_v2.runtime import configure, Learner


def run(root):
    fixture=root/"v2_optimization/independent_fixture.pkl"
    episode=pickle.loads(fixture.read_bytes())["episode"]
    history=unpack(episode.history)
    reports=[episode.report(i) for i in episode.observations]
    high=[episode.report(row["report"]) for row in episode.high]
    order=sorted(range(1800),key=lambda i:len(reports[i].fragments))
    learner=Learner(4101,configure())
    results={}
    for label,index in (("ordinary",order[900]),("congested",order[-1])):
        for purpose,upper,value in (("lower",False,False),("upper",True,False),("upper_with_value",True,True)):
            data=high if upper else reports
            first=max(0,min(index//10-8,164)) if upper else max(0,min(index-8,1784))
            times=[]
            for _ in range(3):
                # Real online reports advance; old repeats must not make the
                # graph LRU look like a throughput gain. Keep within-window
                # sharing between the independent upper actor/value inputs.
                fixed_graph.cache_clear()
                policy=Policy(learner.sac.actor,learner.ppo.actor,learner.ppo.value if value else None,Streams(4101))
                if first: (policy.upper if upper else policy.lower)(data[first-1],history)
                sync()
                start=time.perf_counter()
                for position in range(first,first+16):
                    (policy.upper if upper else policy.lower)(data[position],history)
                sync()
                times.append((time.perf_counter()-start)/16)
            key=f"{label}_{purpose}_inference"
            results[key]=dict(seconds=statistics.median(times),samples_seconds=times,
                frames_per_sample=16,fixed_graph_cache_cleared_before_each_window=True,
                incremental_history_primed=True)
            print(json.dumps(dict(cold_graph_inference=key,seconds=results[key]["seconds"])),flush=True)
    record=dict(status="complete",fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        method="Replace optimized online-window timing only: clear fixed CPU graph LRU before each repeat, retaining within-window sharing. Baseline single-state Encoder.forward rebuilds the graph on every call.",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),measurements=results)
    (root/"hrl_v2/validation/throughput_cold_inference.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return record
