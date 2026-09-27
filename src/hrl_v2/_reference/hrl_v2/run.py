"""Explicit fresh/resume CLI. Default action is a zero-step dry run."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import uuid
import torch
from . import SCHEMA
from .checkpoint import atomic_json,digest,identity,json_digest,load,save,runtime_record
from .schedule import phase,expected_counts
from .randomness import derive,Streams
from .replay import Replay

CONFIG_SHA="bd72f535b125acee3e0019cb60b33831995db0b90593eb621392f65f8307d45a"
SCENE_SHA="daea8771a2cb06754724c2af373704c279c080c7216d55ad990c86f0918b6136"


def read_config(path,root):
    if digest(path)!=CONFIG_SHA: raise ValueError("V2 1.0 config changed: create an explicitly versioned protocol")
    if digest(Path(root)/"hrl_v2/configs/physical_scene.json")!=SCENE_SHA:
        raise ValueError("Physical scene differs from the documented source")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append(path,value):
    with Path(path).open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")


def train(root,config_path,seed,output,resume=None):
    from .runtime import configure,Learner,parameter_hash
    from .physics_adapter import PhysicsAdapter
    from .collector import collect
    from .policy import Policy
    from .evaluate import manifest,panel
    root,output=Path(root),Path(output)
    config=read_config(config_path,root); binding=identity(root,config_path)
    if Path(sys.executable).resolve()!=Path(config["runtime"]["preserve_local_python"]).resolve():
        raise ValueError("Use the registered local Python interpreter; no runtime substitution")
    validation_path=root/"hrl_v2/validation/validation.json"
    if not validation_path.exists():
        raise ValueError("Run python -m hrl_v2.validate before research training")
    validation=json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status")!="passed" or validation.get("identity")!=binding:
        raise ValueError("Implementation validation is missing, failed or belongs to another code/config version")
    saved=load(resume,binding) if resume is not None else None
    if saved is not None:
        seed=saved["seed"]
        if saved["output_relative"]!=str(output.relative_to(root)).replace("\\","/"):
            raise ValueError("Resume must use the checkpoint's original run directory")
        latest_path=output/"latest.json"
        if latest_path.exists():
            latest=json.loads(latest_path.read_text(encoding="utf-8"))
            if latest["sha256"]!=digest(resume):
                raise ValueError("Resume from latest.json; older snapshots are evaluation-only and cannot overwrite later work")
    elif output.exists(): raise FileExistsError("Fresh runs never overwrite an existing directory")
    device=configure(); learner=Learner(seed,device)
    if saved is not None and saved.get("runtime")!=runtime_record():
        raise ValueError("Resume runtime differs from the saved Python/PyTorch/CUDA/device contract")
    scene=json.loads((root/"hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True,exist_ok=True)
    replay=Replay(); ppo_batch=[]; completed=0; pending=None; dev_results={}; last_streams=None
    attempt_id=uuid.uuid4().hex
    if saved:
        learner.load_state_dict(saved["learner"])
        replay=saved["replay"]; ppo_batch=saved["ppo_batch"]
        completed=saved["completed_episode"]; pending=saved["pending_evaluation"]
        dev_results=saved["development_results"]; last_streams=saved["streams"]
        torch.set_rng_state(saved["global_cpu_rng"])
        torch.cuda.set_rng_state_all(saved["global_cuda_rng"])
    else:
        training_manifest=[]
        for number in range(1,601):
            p=phase(number)
            training_manifest.append(dict(global_episode=number,stage=p.stage,cycle=p.cycle,block_episode=p.block_episode,
                **{k+"_seed":derive(seed,p.stage,p.cycle,p.block_episode,purpose=k) for k in ("arrival","layout","load")}))
        atomic_json(output/"training_manifest.json",training_manifest)
        atomic_json(output/"development_manifest.json",manifest("dev"))
        atomic_json(output/"test_manifest.json",manifest("test"))
        atomic_json(output/"contract.json",dict(identity=binding,config=config,seed=seed,
            runtime=runtime_record(),expected_counts=expected_counts(),
            initial_parameters={name:parameter_hash(module) for name,module in
                (("lower_actor",learner.sac.actor),("q1",learner.sac.q1),("q2",learner.sac.q2),
                 ("upper_actor",learner.ppo.actor),("value",learner.ppo.value))},
            physical_source="HANDOVER_20260925.md:77; locally verified byte copy; supplied physical_source_binding.json absent"))
    def persist(label):
        state=dict(schema=SCHEMA,safe_episode_boundary=True,completed_episode=completed,
            pending_evaluation=pending,identity=binding,seed=seed,
            output_relative=str(output.relative_to(root)).replace("\\","/"),
            learner=learner.state_dict(),replay=replay,ppo_batch=ppo_batch,streams=last_streams,
            global_cpu_rng=torch.get_rng_state(),global_cuda_rng=torch.cuda.get_rng_state_all(),
            development_results=dev_results,scene=scene,config=config,
            runtime=runtime_record(),
            discarded_work_policy="restart from last completed episode; incomplete episode never admitted")
        path=output/"checkpoints"/f"episode-{completed:04d}-{label}.pt"
        if path.exists():
            # Resume of an earlier boundary must not overwrite evidence from a
            # later attempt. A new immutable checkpoint gets a unique suffix.
            path=path.with_name(path.stem+"-"+uuid.uuid4().hex[:8]+".pt")
        save(path,state)
        atomic_json(output/"latest.json",dict(path=str(path.relative_to(root)).replace("\\","/"),sha256=digest(path)))
        # Keep two rolling episode boundaries and all registered development
        # nodes. Delete only this run's own superseded complete snapshots after
        # the new snapshot and latest pointer have both committed atomically.
        rolling=sorted((output/"checkpoints").glob("episode-*-complete*.pt"),key=lambda p:p.stat().st_mtime)
        for old in rolling[:-2]:
            if old.parent.resolve()!=(output/"checkpoints").resolve(): raise RuntimeError("Checkpoint retention path escaped run")
            old.with_suffix(".json").unlink()
            old.unlink()
        return path
    def evaluate_pending(checkpoint_path):
        nonlocal pending
        result=panel(learner,scene,seed,manifest("dev"),output/"evaluations"/f"episode-{pending:04d}",
                     checkpoint_id=digest(checkpoint_path),
                     modes=("hover",) if pending==300 else ("moving","hover"))
        dev_results[str(pending)]=result
        pending=None
        return persist("evaluated")
    if saved is None: persist("initial")
    append(output/"attempts.jsonl",dict(attempt_id=attempt_id,resumed=bool(saved),
        restored_completed_episode=completed,checkpoint_sha256=digest(resume) if resume else None,
        prior_uncommitted_work_excluded=bool(saved),unrecorded_discarded_physical_steps=None,
        note="Uncommitted records from previous attempts are retained by attempt_id, excluded from completed-episode totals; unknown unlogged work is not fabricated."))
    if pending is not None: evaluate_pending(resume)
    for number in range(completed+1,601):
        if identity(root,config_path)!=binding: raise RuntimeError("Source changed during training")
        p=phase(number)
        streams=Streams(seed,p.stage,p.cycle,p.block_episode)
        adapter=PhysicsAdapter(scene)
        external={k+"_seed":derive(seed,p.stage,p.cycle,p.block_episode,purpose=k)
                  for k in ("arrival","layout","load")}
        adapter.reset(**external)
        version="hover-0" if p.stage==1 else f"ppo-{learner.ppo.version}-stochastic"
        if p.layer=="lower": replay.set_upper_version(version)
        policy=Policy(learner.sac.actor,learner.ppo.actor,learner.ppo.value,streams,
                      mode="hover" if p.stage==1 else "moving")
        started=time.perf_counter()
        def event():
            row=learner.sac.event(replay,p.alpha,p.actor_enabled,streams)
            append(output/"learning.jsonl",dict(attempt_id=attempt_id,global_episode=number,
                physical_tick=adapter.env.step_index,event=learner.sac.counts["triggers"],**row))
        episode=collect(adapter,policy,upper_version=version,lower_version=learner.sac.version,
                        identity=dict(global_episode=number,external=external),on_event=event if p.layer=="lower" else None)
        if p.layer=="lower": replay.add(episode)
        else:
            ppo_batch.append(episode)
            if len(ppo_batch)==5:
                updates=learner.ppo.update(ppo_batch,streams); ppo_batch.clear()
                append(output/"learning.jsonl",dict(attempt_id=attempt_id,global_episode=number,ppo=updates))
        completed=number; last_streams=streams.state_dict()
        metrics=episode.metrics
        atomic_json(output/"episodes"/f"episode-{number:04d}.json",dict(attempt_id=attempt_id,phase=p.__dict__,metrics=metrics,
            wall_seconds=time.perf_counter()-started,storage_bytes=episode.storage_bytes,
            sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts)))
        pending=number if number in config["schedule"]["development_global_episodes"] else None
        checkpoint_path=persist("pending" if pending is not None else "complete")
        if pending is not None: checkpoint_path=evaluate_pending(checkpoint_path)
        atomic_json(output/"progress.json",dict(status="complete" if number==600 else "running",
            completed_episode=number,sac=learner.sac.counts,ppo=learner.ppo.counts,latest=str(checkpoint_path.relative_to(root))))
    actual=dict(episodes=completed,physics_steps=completed*9000,sac_events=learner.sac.counts["events"],
                critic_fits=learner.sac.counts["critic_fits"],sac_actor_updates=learner.sac.counts["actor_updates"],
                ppo_batches=learner.ppo.counts["batches"])
    if actual!=expected_counts(): raise RuntimeError(f"Budget mismatch: {actual}")
    eligible=[(int(k),v["by_mode"]["moving"]["cost_per_s"]) for k,v in dev_results.items() if k!="300"]
    best=min(eligible,key=lambda x:(x[1],x[0]))[0]
    result=dict(status="training_complete_performance_not_yet_accepted",actual_counts=actual,
                final_episode=600,development_best_episode=best,development_results=dev_results)
    atomic_json(output/"result.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).absolute().parents[1])
    parser.add_argument("--config",type=Path,default=Path("hrl_v2/configs/v2_1_0.json"))
    parser.add_argument("--seed",type=int,choices=range(4101,4106))
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--fresh",action="store_true"); group.add_argument("--resume",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(); root=args.project_root.absolute()
    rooted=lambda p:p if p.is_absolute() else root/p
    config_path=rooted(args.config); read_config(config_path,root)
    binding=identity(root,config_path)
    if not args.fresh and args.resume is None:
        print(json.dumps(dict(status="dry_run",training_started=False,physical_steps=0,updates=0,
            config_sha256=digest(config_path),expected_counts=expected_counts()),indent=2)); return
    if args.fresh and args.seed is None: parser.error("--fresh requires --seed")
    if args.resume is not None:
        if args.seed is not None: parser.error("--resume reads the saved training seed")
        saved=load(rooted(args.resume),binding)
        output=root/saved["output_relative"]
    else:
        output=rooted(args.output) if args.output else root/"v2_runs"/json_digest(binding)[:12]/"ppo_sac"/f"seed_{args.seed}"
    try:
        result=train(root,config_path,args.seed,output,rooted(args.resume) if args.resume else None)
    except Exception as error:
        if output.exists(): atomic_json(output/"technical_failure.json",dict(error=type(error).__name__,message=str(error)))
        raise
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
