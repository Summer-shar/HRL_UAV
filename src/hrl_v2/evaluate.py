"""Frozen stochastic/secondary deterministic panels with paired external seeds."""
import argparse
import json
from pathlib import Path
from .paths import source_root, default_config
import torch
from .checkpoint import atomic_json,json_digest,identity,load,digest
from .randomness import derive,Streams
from .physics_adapter import PhysicsAdapter
from .policy import Policy
from .collector import collect_metrics
from .metrics import PooledMetrics


def manifest(split):
    if split not in ("dev","test"): raise ValueError("Unknown evaluation split")
    return dict(schema="hrl_v2_evaluation_manifest_1_0",split=split,
        scenarios=[dict(scenario=i,**{p+"_seed":derive(0,episode=i,purpose=p,split=split)
                                    for p in ("layout","load","arrival")})
                   for i in range(1,7 if split=="dev" else 21)],
        policy_streams=[1,2] if split=="dev" else [1,2,3,4])


def panel(learner,scene,master,panel_manifest,output,checkpoint_id,*,modes=("moving","hover"),stochastic=True):
    split=panel_manifest["split"]
    if panel_manifest!=manifest(split): raise ValueError("Evaluation manifest differs from sealed seed contract")
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    policy_label="stochastic" if stochastic else "deterministic"
    key=dict(checkpoint=checkpoint_id,manifest=json_digest(panel_manifest),policy=policy_label)
    results={mode:PooledMetrics() for mode in modes}
    # All training random states are restored, including unused global states.
    devices=list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        for scenario in panel_manifest["scenarios"]:
            for stream in panel_manifest["policy_streams"] if stochastic else [0]:
                for mode in modes:
                    item=dict(key,scenario=scenario["scenario"],stream=stream,mode=mode,master=master)
                    path=output/f"{policy_label}_{mode}_{scenario['scenario']:02d}_{stream}.json"
                    if path.exists():
                        saved=json.loads(path.read_text(encoding="utf-8"))
                        if saved["identity"]!=item: raise ValueError("Existing evaluation belongs to another contract")
                        results[mode].add(saved["metrics"]); continue
                    streams=Streams(master,episode=scenario["scenario"],split=split,policy_stream=stream)
                    adapter=PhysicsAdapter(scene)
                    adapter.reset(**{k:v for k,v in scenario.items() if k.endswith("_seed")})
                    policy=Policy(learner.sac.actor,learner.ppo.actor,None,streams,mode=mode,stochastic=stochastic)
                    metrics=collect_metrics(adapter,policy,identity=item)
                    metrics.update(split=split,policy=policy_label,mode=mode,training_seed=master,
                                   learning_updates=0,replay_writes=0)
                    atomic_json(path,dict(identity=item,metrics=metrics))
                    results[mode].add(metrics)
    summary=dict(identity=key,by_mode={k:v.result() for k,v in results.items()},learning_updates=0,replay_writes=0)
    atomic_json(output/f"summary_{policy_label}.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd()/"outputs")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=default_config())
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("paired","moving","hover","centroid"),default="paired")
    parser.add_argument("--policy",choices=("stochastic","deterministic"),default="stochastic")
    args=parser.parse_args()
    root=args.project_root.resolve()
    rooted=lambda p:p if p.is_absolute() else root/p
    from .runtime import configure,Learner
    from .run import read_config
    config=read_config(rooted(args.config),root)
    binding=identity(root,rooted(args.config))
    saved=load(rooted(args.checkpoint),binding)
    m=json.loads(rooted(args.manifest).read_text(encoding="utf-8"))
    if m["split"]=="test" and (saved["completed_episode"]!=600 or saved["pending_evaluation"] is not None):
        raise ValueError("Final test requires the completed 600-episode endpoint")
    learner=Learner(saved["seed"],configure()); learner.load_state_dict(saved["learner"])
    modes=("moving","hover") if args.mode=="paired" else (args.mode,)
    scene=json.loads((source_root()/"hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
    result=panel(learner,scene,saved["seed"],m,rooted(args.output),digest(rooted(args.checkpoint)),
                 modes=modes,stochastic=args.policy=="stochastic")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
