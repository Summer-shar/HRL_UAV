"""Generate five bounded scripted trajectories, with zero learning updates."""
import argparse
import json
import pickle
from pathlib import Path
import time
from .benchmark_optimization import Scripted
from .collector import collect
from .physics_adapter import PhysicsAdapter
from .paths import default_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("data/fixtures/independent_fixture.pkl"))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    scene=json.loads(default_config().with_name("physical_scene.json").read_text(encoding="utf-8"))
    episodes=[]; durations=[]
    for index in range(5):
        seeds=dict(arrival_seed=1+10*index,layout_seed=2+10*index,load_seed=3+10*index)
        adapter=PhysicsAdapter(scene); adapter.reset(**seeds)
        start=time.perf_counter()
        episodes.append(collect(adapter,Scripted(),upper_version="benchmark-script",lower_version=0,identity=seeds))
        durations.append(time.perf_counter()-start)
    payload=dict(episode=episodes[0],episodes=episodes,seconds=durations[0],collection_seconds=durations,learning_updates=0,physics_steps=45000)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as output: pickle.dump(payload,output,protocol=5)
    print(json.dumps(dict(path=str(args.output),episodes=5,physical_steps=45000,learning_updates=0)))

if __name__=="__main__": main()
