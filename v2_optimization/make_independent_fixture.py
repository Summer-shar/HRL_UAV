"""Five bounded scripted trajectories, no learned policy or optimizer steps."""
import json
import pickle
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hrl_v2.benchmark_optimization import Scripted
from hrl_v2.collector import collect
from hrl_v2.physics_adapter import PhysicsAdapter


def main():
    target=ROOT/"v2_optimization/independent_fixture.pkl"
    if target.exists(): raise FileExistsError("Do not replace an existing measured fixture")
    original=pickle.loads((ROOT/"v2_optimization/bounded_fixture.pkl").read_bytes())
    episodes=[original["episode"]]
    seconds=[original["seconds"]]
    scene=json.loads((ROOT/"hrl_v2/configs/physical_scene.json").read_text())
    for index in range(1,5):
        seeds=dict(arrival_seed=1+index*10,layout_seed=2+index*10,load_seed=3+index*10)
        adapter=PhysicsAdapter(scene)
        adapter.reset(**seeds)
        start=time.perf_counter()
        episode=collect(adapter,Scripted(),upper_version="benchmark-script",lower_version=0,
                        identity=dict(bounded_fixture=True,**seeds))
        seconds.append(time.perf_counter()-start)
        episodes.append(episode)
        print(json.dumps(dict(fixture_episode=index+1,seconds=seconds[-1],physics_steps=9000,
                              learning_updates=0,storage_bytes=episode.storage_bytes)),flush=True)
    payload=dict(original,episodes=episodes,episode=episodes[0],collection_seconds=seconds,
                 total_fixture_physics_steps=45000,
                 description="Five distinct seeded scripted finite trajectories; zero learning updates")
    target.write_bytes(pickle.dumps(payload,protocol=5))
    print(json.dumps(dict(path=str(target),episodes=5,bytes=target.stat().st_size)),flush=True)


if __name__=="__main__": main()
