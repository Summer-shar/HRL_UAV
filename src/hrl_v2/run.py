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
from .paths import source_root, default_config
from .checkpoint import runtime_signature
from .timing import ClockConfig
from .baseline_protocol import reference_modes_complete

SCENE_SHA="e978ea4f9f0a87d860dab55fffcbdd303e1e2e792947640969d7292636622187"


def read_config(path,root):
    if digest(source_root()/"hrl_v2/configs/physical_scene.json")!=SCENE_SHA:
        raise ValueError("Physical scene differs from the reviewed physical parameters")
    config=json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get('encoder')=='lite' and config.get('observation_version')!='lite_110_729_v2_2_1':
        raise ValueError('This Lite implementation requires the v2_2_1 observation version; archived configs are not upgraded implicitly')
    if config.get("schema")!=SCHEMA:
        raise ValueError("Expected continuing-service configuration")
    clocks=ClockConfig.from_config(config)
    scene=json.loads((source_root()/"hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
    if clocks!=ClockConfig.from_scene(scene,rollout_seconds=clocks.rollout_seconds):
        raise ValueError("Control and physical clocks must match the physical scene")
    objective=config["objective"]
    physical=scene["base_scene"]["objective"]
    import math
    if (objective["priority_weights"]!=physical["priority_weights"] or
        not math.isclose(objective["holding_coefficient"],physical["holding_weight"]/physical["holding_reference_task_s"],rel_tol=0.,abs_tol=1e-12) or
        not math.isclose(objective["energy_coefficient"],(1-physical["holding_weight"])/physical["energy_reference_j"],rel_tol=0.,abs_tol=1e-12) or
        objective.get('energy_mode','full')!=physical['energy_mode'] or
        objective.get('propulsion_weight',1.)!=physical.get('propulsion_weight',1.)):
        raise ValueError("Configured costs must match the reviewed physical objective and units")
    if (objective.get("normal_rollout_terminated") is not False or
            objective.get("normal_rollout_truncated") is not True):
        raise ValueError("Artificial rollout boundaries must truncate, never terminate")
    if not isinstance(objective["reward_scale"],(float,int)) or not 0<objective["reward_scale"]<float("inf"):
        raise ValueError("reward_scale must be finite and positive")
    validate_discounts(config,clocks)
    return config


def validate_discounts(config,clocks):
    """Both layers discount the same physical time: gamma_macro = gamma_micro ** lower_per_upper."""
    import math
    objective,ppo=config["objective"],config["ppo"]
    micro,macro=objective["gamma_micro"],objective["gamma_macro"]
    if not math.isclose(ppo["gamma"],macro,rel_tol=0.,abs_tol=1e-12):
        raise ValueError("ppo.gamma must equal objective.gamma_macro")
    if not math.isclose(macro,micro**clocks.lower_per_upper,rel_tol=0.,abs_tol=1e-9):
        raise ValueError("gamma_macro must equal gamma_micro ** lower_per_upper (same physical horizon)")
    if not 0<=ppo["gae_lambda"]<=1:
        raise ValueError("gae_lambda must lie in [0, 1]")


def require_research_protocol(config):
    # Continuing service needs a strict discount: with gamma=1 the bootstrapped
    # truncation target is an unbounded sum of costs.
    research=config.get("research",{})
    objective=config.get("objective",{})
    missing=[k for k in ("training_enabled","discounts_confirmed","schedule_confirmed")
             if research.get(k) is not True]
    if missing:
        raise RuntimeError("Research training is disabled: config research flags not confirmed: "+", ".join(missing))
    if not (0<objective.get("gamma_micro",1.)<1 and 0<objective.get("gamma_macro",1.)<1):
        raise RuntimeError("Continuing-service training requires 0 < gamma < 1 for both layers")


def technical_status(root,config_path,config):
    """Readiness is a technical report, never authorization to launch a run."""
    try:
        require_research_protocol(config)
        confirmed=True
    except RuntimeError:
        confirmed=False
    path=Path(root)/"validation/validation.json"
    validation=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ready=(validation.get("status")=="passed" and
           validation.get("identity")==identity(root,config_path) and
           runtime_signature(validation.get("runtime",{}))==runtime_signature(runtime_record()))
    return dict(validation_ready=ready,research_protocol_confirmed=confirmed,
                training_ready=bool(ready and confirmed),training_authorized=False,
                training_started=False)


def append(path,value):
    with Path(path).open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")


def collection_policy(learner,streams,phase):
    from .policy import Policy
    return Policy(learner.sac.actor,learner.ppo.actor,
                  learner.ppo.value if phase.layer=="upper" else None,streams,
                  mode="hover" if phase.stage==1 else "moving")


def train(root,config_path,seed,output,resume=None,baseline_reference=None):
    require_research_protocol(read_config(config_path,root))
    from .runtime import configure,Learner,parameter_hash
    from .physics_adapter import PhysicsAdapter
    from .collector import collect
    from .evaluate import manifest,panel
    root,output=Path(root),Path(output)
    config=read_config(config_path,root); binding=identity(root,config_path)
    clocks=ClockConfig.from_config(config)
    validation_path=root/"validation/validation.json"
    if not validation_path.exists():
        raise ValueError("Run hrl-v2 validate with the same --workspace before research training")
    validation=json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status")!="passed" or validation.get("identity")!=binding:
        raise ValueError("Implementation validation is missing, failed or belongs to another code/config version")
    saved=load(resume,binding) if resume is not None else None
    reference=None
    if config.get('observation_version')=='lite_110_729_v2_2_1':
        if saved is not None:
            reference=saved['baseline_reference']
        else:
            if baseline_reference is None: raise ValueError('Freeze the paired baseline reference before this fresh Lite run')
            reference=json.loads(Path(baseline_reference).read_text(encoding='utf-8'))
        from .evaluate import gate_manifest
        if reference['identity']!=binding or reference['master']!=seed:
            raise ValueError('Baseline reference source/config/seed mismatch')
        count=len(reference['manifest']['scenarios'])
        if reference['manifest']!=gate_manifest(count) or reference['required_wins']!={6:4,20:12}[count]:
            raise ValueError('Baseline gate contract mismatch')
        if not reference_modes_complete(config,reference['per_scenario']):
            raise ValueError('Incomplete baseline reference')
    evaluation_manifest=reference['manifest'] if reference else manifest('dev')
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
    device=configure()
    if runtime_signature(validation.get("runtime",{}))!=runtime_signature(runtime_record()):
        raise ValueError("Run local validation in the current Python/PyTorch/CUDA environment")
    learner=Learner(seed,device,config=config)
    if saved is not None and runtime_signature(saved.get("runtime",{}))!=runtime_signature(runtime_record()):
        raise ValueError("Resume runtime differs from the saved Python/PyTorch/CUDA/device contract")
    scene=json.loads((source_root()/"hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
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
        atomic_json(output/"development_manifest.json",evaluation_manifest)
        if reference: atomic_json(output/'baseline_reference.json',reference)
        atomic_json(output/"test_manifest.json",manifest("test"))
        atomic_json(output/"contract.json",dict(identity=binding,config=config,seed=seed,
            runtime=runtime_record(),expected_counts=expected_counts(config),
            initial_parameters={name:parameter_hash(module) for name,module in
                (("lower_actor",learner.sac.actor),("q1",learner.sac.q1),("q2",learner.sac.q2),
                 ("upper_actor",learner.ppo.actor),("value",learner.ppo.value))},
            physical_source="Bundled physical source derived from dbf2b97; see docs/PROVENANCE.md"))
    def persist(label):
        state=dict(schema=SCHEMA,safe_episode_boundary=True,completed_episode=completed,
            pending_evaluation=pending,identity=binding,seed=seed,
            output_relative=str(output.relative_to(root)).replace("\\","/"),
            learner=learner.state_dict(),replay=replay,ppo_batch=ppo_batch,streams=last_streams,
            global_cpu_rng=torch.get_rng_state(),global_cuda_rng=torch.cuda.get_rng_state_all(),
            development_results=dev_results,scene=scene,config=config,baseline_reference=reference,
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
        evaluation_number=pending
        result=panel(learner,scene,seed,evaluation_manifest,output/"evaluations"/f"episode-{pending:04d}",
                     checkpoint_id=digest(checkpoint_path),
                     modes=("hover",) if pending==300 else ("moving","hover"),
                     timing=clocks,reward_scale=config["objective"]["reward_scale"])
        dev_results[str(pending)]=result
        if reference:
            from .gates import g1,g2
            gate=(g1(result['per_scenario']['hover'],reference['per_scenario'],reference['required_wins'])
                  if pending==300 else g2(result['per_scenario']['moving'],result['per_scenario']['hover']))
            result['gate']=gate
            atomic_json(output/'gates'/f'episode-{evaluation_number:04d}.json',gate)
            if not gate.get('passed',True) or gate.get('weak_upper_signal',False):
                atomic_json(output/'diagnosis_required.json',dict(episode=pending,gate=gate,
                    stage_delay_s={k:v['stage_delay_s'] for k,v in result['by_mode'].items()},
                    candidate_training_authorized=False))
        pending=None
        return persist("evaluated")
    if saved is None: persist("initial")
    append(output/"attempts.jsonl",dict(attempt_id=attempt_id,resumed=bool(saved),
        restored_completed_episode=completed,checkpoint_sha256=digest(resume) if resume else None,
        prior_uncommitted_work_excluded=bool(saved),unrecorded_discarded_physical_steps=None,
        note="Uncommitted records from previous attempts are retained by attempt_id, excluded from completed-episode totals; unknown unlogged work is not fabricated."))
    if pending is not None: evaluate_pending(resume)
    if saved is not None and reference and completed:
        # A completed checkpoint may survive a crash before its log append.
        metric_path=output/'train_metrics.jsonl'
        recorded=[]
        if metric_path.exists():
            for line in metric_path.read_text(encoding='utf-8').splitlines():
                try: recorded.append(json.loads(line)['global_episode'])
                except (ValueError,KeyError): continue
        if completed not in recorded:
            last=json.loads((output/'episodes'/f'episode-{completed:04d}.json').read_text())
            recovered=last['train_metric']
            recovered['recovered_after_interruption']=True
            append(metric_path,recovered)
    def paused_g1():
        failed=reference and not dev_results.get('300',{}).get('gate',{}).get('passed',True)
        if failed:
            record=dict(status='paused_G1',completed_episode=completed,gate=dev_results['300']['gate'],
                        resumable_checkpoint=json.loads((output/'latest.json').read_text()),
                        note='PPO stage blocked. Resume preserves this gate; no automatic threshold override.')
            atomic_json(output/'progress.json',record)
            return record
    paused=paused_g1()
    if paused: return paused
    for number in range(completed+1,601):
        if identity(root,config_path)!=binding: raise RuntimeError("Source changed during training")
        p=phase(number)
        print(f'[{number}/600] {p.layer} episode started; stage={p.stage}',flush=True)
        streams=Streams(seed,p.stage,p.cycle,p.block_episode)
        adapter=PhysicsAdapter(scene,timing=clocks)
        external={k+"_seed":derive(seed,p.stage,p.cycle,p.block_episode,purpose=k)
                  for k in ("arrival","layout","load")}
        adapter.reset(**external)
        version="hover-0" if p.stage==1 else f"ppo-{learner.ppo.version}-stochastic"
        if p.layer=="lower": replay.set_upper_version(version)
        policy=collection_policy(learner,streams,p)
        started=time.perf_counter()
        counters_before=dict(learner.sac.counts)
        residuals=[]
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        def event():
            row=learner.sac.event(replay,p.alpha,p.actor_enabled,streams)
            residuals.extend(v for fit in row.get('critics',[]) for v in fit['residual_mse'])
            append(output/"learning.jsonl",dict(attempt_id=attempt_id,global_episode=number,
                physical_tick=adapter.env.step_index,event=learner.sac.counts["triggers"],alpha=p.alpha,**row))
        episode=collect(adapter,policy,upper_version=version,lower_version=learner.sac.version,
                        identity=dict(global_episode=number,external=external),on_event=event if p.layer=="lower" else None,
                        reward_scale=config["objective"]["reward_scale"],gamma_lower=config["objective"]["gamma_micro"])
        if p.layer=="lower": replay.add(episode)
        else:
            ppo_batch.append(episode)
            if len(ppo_batch)==5:
                updates=learner.ppo.update(ppo_batch,streams); ppo_batch.clear()
                append(output/"learning.jsonl",dict(attempt_id=attempt_id,global_episode=number,ppo=updates))
        completed=number; last_streams=streams.state_dict()
        metrics=episode.metrics
        work_seconds=time.perf_counter()-started
        from .telemetry import compact,resources
        row=dict(attempt_id=attempt_id,global_episode=number,layer=p.layer,stage=p.stage,
            reward_physical=float(sum(episode.rewards)),
            reward_learning_macro_sum=float(sum(h['reward_learning'] for h in episode.high)),
            work_seconds=work_seconds,checkpoint_seconds=None,evaluation_seconds=None,
            wall_seconds=None,alpha=p.alpha,
            critic_residual_mse=sum(residuals)/len(residuals) if residuals else None,
            actor_relative_entropy=policy.entropy_sum/policy.entropy_count if policy.entropy_count else None,
            entropy_samples=policy.entropy_count,entropy_scope='executed lower samples plus real bootstrap successor',
            sac_delta={k:v-counters_before[k] for k,v in learner.sac.counts.items()},
            resources=resources(),**compact(metrics))
        atomic_json(output/"episodes"/f"episode-{number:04d}.json",dict(attempt_id=attempt_id,phase=p.__dict__,metrics=metrics,
            wall_seconds=work_seconds,storage_bytes=episode.storage_bytes,
            sac=dict(learner.sac.counts),ppo=dict(learner.ppo.counts),train_metric=row))
        pending=number if number in config["schedule"]["development_global_episodes"] else None
        saved_at=time.perf_counter()
        checkpoint_path=persist("pending" if pending is not None else "complete")
        checkpoint_seconds=time.perf_counter()-saved_at
        evaluated_at=time.perf_counter()
        if pending is not None: checkpoint_path=evaluate_pending(checkpoint_path)
        evaluation_seconds=time.perf_counter()-evaluated_at
        row.update(checkpoint_seconds=checkpoint_seconds,evaluation_seconds=evaluation_seconds,
                   wall_seconds=time.perf_counter()-started,resources=resources())
        append(output/'train_metrics.jsonl',row)
        print(f'[{number}/600] {p.layer} reward={row["reward_physical"]:.6f} '
              f'truncated_delay={row["truncated_delay_s"]:.6f}s '
              f'work={work_seconds:.2f}s SAC_events={row["sac_delta"]["events"]} '
              f'critic_fits={row["sac_delta"]["critic_fits"]}',flush=True)
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
        paused=paused_g1()
        if paused: return paused
        atomic_json(output/"progress.json",dict(status="complete" if number==600 else "running",
            completed_episode=number,sac=learner.sac.counts,ppo=learner.ppo.counts,latest=str(checkpoint_path.relative_to(root))))
    actual=dict(episodes=completed,physics_steps=completed*clocks.physical_steps,sac_events=learner.sac.counts["events"],
                critic_fits=learner.sac.counts["critic_fits"],sac_actor_updates=learner.sac.counts["actor_updates"],
                ppo_batches=learner.ppo.counts["batches"])
    if actual!=expected_counts(config): raise RuntimeError(f"Budget mismatch: {actual}")
    eligible=[(int(k),v["by_mode"]["moving"]["cost_per_s"]) for k,v in dev_results.items() if k!="300"]
    best=min(eligible,key=lambda x:(x[1],x[0]))[0]
    result=dict(status="training_complete_performance_not_yet_accepted",actual_counts=actual,
                final_episode=600,development_best_episode=best,development_results=dev_results)
    atomic_json(output/"result.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd()/"outputs")
    parser.add_argument("--config",type=Path,default=default_config())
    parser.add_argument("--seed",type=int,choices=range(4101,4106))
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--fresh",action="store_true"); group.add_argument("--resume",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument('--baseline-reference',type=Path,help='Frozen paired baseline summary for the Lite gated run')
    args=parser.parse_args(); root=args.project_root.resolve()
    rooted=lambda p:p if p.is_absolute() else root/p
    config_path=rooted(args.config); config=read_config(config_path,root)
    binding=identity(root,config_path)
    if not args.fresh and args.resume is None:
        from .runtime import configure
        configure(test_cpu=True)
        print(json.dumps(dict(status="dry_run",physical_steps=0,updates=0,
            **technical_status(root,config_path,config),rollout=ClockConfig.from_config(config).counts(),
            rollout_seconds=config["task"]["rollout_seconds"],
            config_sha256=digest(config_path),expected_counts=expected_counts(config)),indent=2)); return
    require_research_protocol(config)
    if args.fresh and args.seed is None: parser.error("--fresh requires --seed")
    if args.resume is not None:
        if args.seed is not None: parser.error("--resume reads the saved training seed")
        saved=load(rooted(args.resume),binding)
        output=root/saved["output_relative"]
    else:
        output=rooted(args.output) if args.output else root/"v2_runs"/json_digest(binding)[:12]/"ppo_sac"/f"seed_{args.seed}"
    try:
        result=train(root,config_path,args.seed,output,rooted(args.resume) if args.resume else None,
                     rooted(args.baseline_reference) if args.baseline_reference else None)
    except Exception as error:
        if output.exists(): atomic_json(output/"technical_failure.json",dict(error=type(error).__name__,message=str(error)))
        raise
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
