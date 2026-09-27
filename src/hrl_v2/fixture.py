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
from .timing import ClockConfig

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("data/fixtures/independent_fixture.pkl"))
    parser.add_argument("--config",type=Path,default=default_config())
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    scene=json.loads(default_config().with_name("physical_scene.json").read_text(encoding="utf-8"))
    from .run import read_config
    config=read_config(args.config,None)
    timing=ClockConfig.from_config(config)
    episodes=[]; durations=[]
    for index in range(5):
        seeds=dict(arrival_seed=1+10*index,layout_seed=2+10*index,load_seed=3+10*index)
        adapter=PhysicsAdapter(scene,timing=timing); adapter.reset(**seeds)
        start=time.perf_counter()
        episodes.append(collect(adapter,Scripted(),upper_version="benchmark-script",lower_version=0,identity=seeds,
                                reward_scale=config["objective"]["reward_scale"]))
        durations.append(time.perf_counter()-start)
    steps=len(episodes)*timing.physical_steps
    payload=dict(episode=episodes[0],episodes=episodes,seconds=durations[0],collection_seconds=durations,learning_updates=0,physics_steps=steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as output: pickle.dump(payload,output,protocol=5)
    print(json.dumps(dict(path=str(args.output),episodes=len(episodes),physical_steps=steps,learning_updates=0)))

if __name__=="__main__": main()
