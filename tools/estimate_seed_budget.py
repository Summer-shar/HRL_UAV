"""Count the current run exactly and price it using this round's measurements.

This tool never constructs a model, advances physics, or launches training.
Example: python tools/estimate_seed_budget.py --profile outputs/profile_current
    --fixture-manifest outputs/perf_fixture/manifest.json --output outputs/seed_budget.json
Use --counts-only before measurements are available. Missing prices remain
unknown; a partial subtotal is never presented as a complete seed budget.
"""
import argparse
import ast
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT=Path(__file__).resolve().parents[1]
SCHEMA="current_seed_budget_1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_constants(source):
    """Read execution literals, rejecting drift between code and config."""
    trees={name:ast.parse((source/"hrl_v2"/(name+".py")).read_text(encoding="utf-8"))
           for name in ("run","collector","sac","ppo","replay")}
    def one(values,label):
        values=set(values)
        if len(values)!=1: raise ValueError("Cannot identify current execution constant: "+label)
        return values.pop()
    def literal(node):
        return node.value if isinstance(node,ast.Constant) else None
    def ranges(tree,arity):
        return [n.args for n in ast.walk(tree) if isinstance(n,ast.Call) and
                isinstance(n.func,ast.Name) and n.func.id=="range" and len(n.args)==arity]
    periods=[literal(n.right) for n in ast.walk(trees["collector"]) if isinstance(n,ast.BinOp)
             and isinstance(n.op,ast.Mod) and isinstance(n.left,ast.Name) and n.left.id=="completed_lower"]
    actor_period=[literal(n.right) for n in ast.walk(trees["sac"]) if isinstance(n,ast.BinOp)
                  and isinstance(n.op,ast.Mod)]
    cap=[literal(n.comparators[0]) for n in ast.walk(trees["replay"]) if isinstance(n,ast.Compare)
         and len(n.ops)==1 and isinstance(n.ops[0],ast.Gt) and isinstance(n.left,ast.Call)
         and isinstance(n.left.func,ast.Name) and n.left.func.id=="len"]
    batch=[literal(n.comparators[0]) for n in ast.walk(trees["ppo"]) if isinstance(n,ast.Compare)
           and len(n.ops)==1 and isinstance(n.ops[0],ast.NotEq) and isinstance(n.left,ast.Call)
           and isinstance(n.left.func,ast.Name) and n.left.func.id=="len"
           and isinstance(n.left.args[0],ast.Name) and n.left.args[0].id=="episodes"]
    total=[literal(n.iter.args[-1])-1 for n in ast.walk(trees["run"]) if isinstance(n,ast.For)
           and isinstance(n.target,ast.Name) and n.target.id=="number" and isinstance(n.iter,ast.Call)
           and isinstance(n.iter.func,ast.Name) and n.iter.func.id=="range"
           and literal(n.iter.args[-1]) is not None]
    return dict(total_windows=one(total,"run windows"),event_period=one(periods,"event period"),
        critic_fits_per_event=one([literal(a[0]) for a in ranges(trees["sac"],1)],"critic fits"),
        actor_every_events=one(actor_period,"actor period"),replay_capacity=one(cap,"replay capacity"),
        ppo_batch_windows=one(batch,"PPO windows"),
        ppo_epochs=one([literal(a[0]) for a in ranges(trees["ppo"],1)],"PPO epochs"),
        ppo_minibatch=one([literal(a[2]) for a in ranges(trees["ppo"],3)
                          if literal(a[2]) is not None],"PPO minibatch"))


def enumerate_protocol(config,source):
    from hrl_v2.schedule import expected_counts,phase
    from hrl_v2.timing import ClockConfig
    clocks=ClockConfig.from_config(config)
    actual=source_constants(source)
    configured=dict(total_windows=config["schedule"]["total_episodes_per_seed"],
        event_period=config["sac"]["learn_every_routing_frames"],
        critic_fits_per_event=config["sac"]["critic_fits_per_event"],
        actor_every_events=config["sac"]["actor_every_events"],
        replay_capacity=config["replay"]["max_completed_episodes"],
        ppo_batch_windows=config["ppo"]["batch_episodes"],ppo_epochs=config["ppo"]["epochs"],
        ppo_minibatch=config["ppo"]["minibatch_size"])
    if actual!=configured: raise ValueError("Execution literals differ from configured budget: "+repr((actual,configured)))
    count=Counter(checkpoint_saves=1)
    ledger=[]
    checkpoints=[dict(window=0,kind="initial",replay=0,pending_ppo=0)]
    occupancy={key:Counter() for key in ("critic_fits","actor_updates","events")}
    prefixes={key:Counter() for key in ("lower_cached","lower_rebuild","lower_cold")}
    replay_version=None
    replay_slots=ppo_slots=ppo_version=0
    development=set(config["schedule"]["development_global_episodes"])
    for number in range(1,actual["total_windows"]+1):
        current=phase(number)
        before=count.copy()
        count["episodes"]+=1
        count["physics_steps"]+=clocks.physical_steps
        count["lower_inference"]+=clocks.lower_steps
        count["episode_start_cache_rebuilds"]+=1
        if current.stage!=1:
            count["upper_inference"]+=clocks.upper_steps+1
            count["cutoff_extra_upper_inference"]+=1
            count["upper_cold_inference"]+=1
        if current.layer=="upper":
            count["value_inference"]+=clocks.upper_steps+1
            count["cutoff_extra_value_inference"]+=1
            count["value_cold_inference"]+=1
        rebuild_at=set()
        if current.layer=="lower":
            version="hover-0" if current.stage==1 else f"ppo-{ppo_version}-stochastic"
            if version!=replay_version:
                replay_slots=0
                replay_version=version
                count["replay_flushes"]+=1
            for frame in range(actual["event_period"],clocks.lower_steps+1,actual["event_period"]):
                count["sac_triggers"]+=1
                count["learning_log_rows"]+=1
                if not replay_slots:
                    count["skipped_events"]+=1
                    continue
                count["sac_events"]+=1
                count["critic_fits"]+=actual["critic_fits_per_event"]
                count["replay_samples"]+=actual["critic_fits_per_event"]
                count["target_polyak_updates"]+=1
                occupancy["events"][replay_slots]+=1
                occupancy["critic_fits"][replay_slots]+=actual["critic_fits_per_event"]
                if current.actor_enabled and count["sac_events"]%actual["actor_every_events"]==0:
                    count["sac_actor_updates"]+=1
                    count["replay_samples"]+=1
                    occupancy["actor_updates"][replay_slots]+=1
                    if frame<clocks.lower_steps:
                        count["interior_actor_cache_rebuilds"]+=1
                        rebuild_at.add(frame+1)
                    else:
                        count["final_actor_updates_without_online_rebuild"]+=1
            replay_slots=min(actual["replay_capacity"],replay_slots+1)
        else:
            ppo_slots+=1
            if ppo_slots==actual["ppo_batch_windows"]:
                ppo_slots=0
                ppo_version+=1
                count["ppo_batches"]+=1
                count["learning_log_rows"]+=1
        for cursor in range(1,clocks.lower_steps+1):
            mode="lower_cold" if cursor==1 else "lower_rebuild" if cursor in rebuild_at else "lower_cached"
            prefixes[mode][cursor]+=1
        copies=2 if number in development else 1
        count["checkpoint_saves"]+=copies
        count["checkpoint_replay_episode_records"]+=copies*replay_slots
        count["checkpoint_pending_ppo_episode_records"]+=copies*ppo_slots
        checkpoints.append(dict(window=number,kind="pending" if number in development else "complete",
                                replay=replay_slots,pending_ppo=ppo_slots))
        if number in development:
            panel=config["evaluation"]["development_scenarios"]*config["evaluation"]["development_policy_streams"]
            count["development_panels"]+=1
            count["dev_hover_episodes"]+=panel
            count["dev_moving_episodes"]+=panel if number!=300 else 0
            count["development_checkpoint_hash_episode_records"]+=replay_slots+ppo_slots
            checkpoints.append(dict(window=number,kind="evaluated",replay=replay_slots,pending_ppo=ppo_slots))
        ledger.append(dict(window=number,stage=current.stage,cycle=current.cycle,layer=current.layer,
            block_episode=current.block_episode,actor_enabled=current.actor_enabled,upper_version=ppo_version,
            replay_slots_after=replay_slots,pending_ppo_after=ppo_slots,
            events=count["sac_events"]-before["sac_events"],skipped=count["skipped_events"]-before["skipped_events"],
            actor_updates=count["sac_actor_updates"]-before["sac_actor_updates"],
            internal_cache_rebuilds=len(rebuild_at),checkpoint_copies=copies))
    expected=expected_counts(config)
    if {key:count[key] for key in expected}!=expected:
        raise AssertionError("Independent event enumeration disagrees with schedule.expected_counts")
    rows=actual["ppo_batch_windows"]*clocks.upper_steps
    per_epoch=math.ceil(rows/actual["ppo_minibatch"])
    count["ppo_rows_per_batch"]=rows
    count["ppo_value_optimizer_steps"]=count["ppo_batches"]*actual["ppo_epochs"]*per_epoch
    count["ppo_actor_optimizer_steps_attempted_max"]=count["ppo_value_optimizer_steps"]
    count["ppo_actor_optimizer_steps_attempted_min"]=count["ppo_batches"]*per_epoch
    count["ppo_actor_optimizer_steps_accepted_min"]=0
    count["ppo_actor_optimizer_steps_accepted_max"]=count["ppo_value_optimizer_steps"]
    count["ppo_kl_sweeps_max"]=count["ppo_batches"]*actual["ppo_epochs"]
    count["ppo_kl_sweeps_min"]=count["ppo_batches"]
    count["lower_cached_inference"]=count["lower_inference"]-count["episode_start_cache_rebuilds"]-count["interior_actor_cache_rebuilds"]
    count["upper_inference_without_value"]=count["upper_inference"]-count["value_inference"]
    count["upper_cold_without_value"]=count["upper_cold_inference"]-count["value_cold_inference"]
    count["upper_cached_without_value"]=count["upper_inference_without_value"]-count["upper_cold_without_value"]
    count["upper_cached_with_value"]=count["value_inference"]-count["value_cold_inference"]
    count["dev_episodes"]=count["dev_hover_episodes"]+count["dev_moving_episodes"]
    count["dev_physics_steps"]=count["dev_episodes"]*clocks.physical_steps
    count["dev_lower_inference"]=count["dev_episodes"]*clocks.lower_steps
    count["dev_upper_inference"]=count["dev_moving_episodes"]*(clocks.upper_steps+1)
    test_per_mode=config["evaluation"]["test_scenarios"]*config["evaluation"]["test_policy_streams"]
    count["optional_test_episodes"]=2*test_per_mode
    count["optional_test_physics_steps"]=count["optional_test_episodes"]*clocks.physical_steps
    count["optional_test_lower_inference"]=count["optional_test_episodes"]*clocks.lower_steps
    count["optional_test_upper_inference"]=test_per_mode*(clocks.upper_steps+1)
    count["additional_checkpoint_hash_reads"]=count["checkpoint_saves"]+count["development_panels"]
    count["source_identity_calls_including_fresh_cli"]=count["episodes"]+2
    count["scene_integrity_hash_reads"]=3
    count["initial_parameter_hashes"]=5
    count["training_atomic_json_outside_save"]=2*count["episodes"]+count["checkpoint_saves"]+5
    count["development_atomic_json"]=count["dev_episodes"]+count["development_panels"]
    count["optional_test_atomic_json"]=count["optional_test_episodes"]+1
    count["attempt_log_rows"]=1
    # Per fresh window: reset report, precommit reports, prepared lower reports.
    count["reports_generated_training"]=count["episodes"]*(clocks.lower_steps+clocks.upper_steps+3)
    count["stored_report_records_training"]=count["episodes"]*(2*clocks.lower_steps+2*clocks.upper_steps)
    # Replay.sample draws four episode indices with replacement. These are
    # exact probabilities, not claimed realized counts for a future RNG path.
    draws=config["sac"]["episodes_per_batch"]
    distinct={}
    for operation,by_size in occupancy.items():
        expected_distinct=Counter()
        for size,calls in by_size.items():
            probabilities={0:1.}
            for _ in range(draws):
                following=Counter()
                for unique,probability in probabilities.items():
                    following[unique]+=probability*unique/size
                    following[unique+1]+=probability*(size-unique)/size
                probabilities=following
            for unique,probability in probabilities.items():
                if probability: expected_distinct[unique]+=calls*probability
        distinct[operation]=dict(sorted(expected_distinct.items()))
    return dict(counts=dict(count),clocks=asdict(clocks),execution_constants=actual,
        schedule_expected_counts=expected,per_window=ledger,checkpoints=checkpoints,
        replay_occupancy_by_operation={key:dict(sorted(value.items())) for key,value in occupancy.items()},
        expected_unique_replay_objects_per_operation=distinct,
        online_lower_prefix_counts={key:dict(sorted(value.items())) for key,value in prefixes.items()})


def coefficient(times,keys):
    rows=[(key,times[key]) for key in keys if key in times]
    if not rows: return None
    if any(not math.isfinite(row["median_s"]) or row["median_s"]<0 for _,row in rows):
        raise ValueError("Invalid timing coefficient")
    def samples(row):
        return [item["seconds"] for item in row.get("samples",[])] or [row["median_s"]]
    return dict(keys=[key for key,_ in rows],low_s=min(min(samples(row)) for _,row in rows),
        high_s=max(max(samples(row)) for _,row in rows),
        range_kind="Observed sample minimum/maximum across these fixtures; not a confidence or tail bound",
        median_range_s=[min(row["median_s"] for _,row in rows),max(row["median_s"] for _,row in rows)])


def priced_component(name,count,price,method,unit="calls"):
    return dict(name=name,count=count,unit=unit,status="unknown" if price is None else "conditional_estimate",
        seconds=None if price is None else [count*price["low_s"],count*price["high_s"]],
        coefficient=price,method=method)


def profile_inputs(directory,config):
    directory=Path(directory)
    metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    times=json.loads((directory/"times.json").read_text(encoding="utf-8"))
    if metadata.get("schema")!="fixed_performance_workloads_1":
        raise ValueError("Only this round's fixed-workload profile is accepted")
    if metadata["config"]!=config: raise ValueError("Profile and current budget configurations differ")
    for key,filename in (("checkpoint.save0","measured_checkpoint0.pt"),
                         ("checkpoint.save2","measured_checkpoint2.pt")):
        path=directory/filename
        if key in times and path.exists():
            times[key]["bytes"]=path.stat().st_size
            times[key]["bytes_provenance"]=dict(path=str(path.resolve()),method="stat after timed save")
    return metadata,times


def full_window_collection_seconds(row,clocks):
    # ``physical_seconds`` is measured kernel wall time in this artifact;
    # ``seconds`` is the duration of the simulated window.
    duration=row["seconds"]
    if not math.isclose(duration,row["physical_steps"]*clocks["physical_dt"]):
        raise ValueError("Collection-overhead duration and physical tick count differ")
    return row["wall_s"]*clocks["rollout_seconds"]/duration


def evaluation_prices(clocks,times,*,windows,moving,panel_summaries,physical_price):
    """Use inclusive full replay once; retain a separate hover sensitivity."""
    lower_steps=round(clocks["rollout_seconds"]/clocks["lower_interval"])
    upper_steps=round(clocks["rollout_seconds"]/clocks["upper_interval"])
    hover=windows-moving
    replay=coefficient(times,["evaluation.replay900s"])
    if replay is not None:
        work=times["evaluation.replay900s"].get("workload",{})
        if (work.get("lower_calls")!=lower_steps or work.get("upper_calls")!=upper_steps+1):
            raise ValueError("Complete evaluation replay does not match the configured control window")
    lower_cold=coefficient(times,["inference.lower.h1.rebuild"])
    lower_cached=coefficient(times,[key for key in times if re.fullmatch(r"inference\.lower\.h\d+\.cached",key)])
    json_price=coefficient(times,["io.metrics_json"])
    physical=priced_component("physics_reports_metrics",windows,physical_price,
        "Separately measured streamed physical/report/metric construction proxy; contains no neural policy inference or JSON write. "
        "Short-window 60-to-900-second growth remains unmeasured.")
    rows=[physical]
    if replay is not None:
        rows.append(priced_component("moving_complete_report_replay",moving,replay,
            "Full real source-0 report sequence: all lower actions, upper actions including cutoff, metric pooling and one atomic JSON. "
            "These inclusive operations are not charged again as standalone inference or per-window JSON."))
        lower_windows=hover
        json_writes=hover+panel_summaries
    else:
        # An incomplete profile may still be inspected; name the fallback so
        # it cannot masquerade as a measured full-trajectory workload.
        lower_windows=windows
        json_writes=windows+panel_summaries
        for name,count,keys in (("moving_upper_cold_fallback",moving,["inference.upper.h0.rebuild"]),
                                ("moving_upper_cached_fallback",moving*upper_steps,
                                 [key for key in times if re.fullmatch(r"inference\.upper\.h\d+\.cached",key)])):
            rows.append(priced_component(name,count,coefficient(times,keys),
                "Full replay was not measured; fixed-section inference remains a conditional fallback."))
    label="hover" if replay is not None else "all_modes_fallback"
    rows.append(priced_component(label+"_lower_cold_inference",lower_windows,lower_cold,
        "No full pure-lower hover replay was measured; retain one cold lower call per hover window."))
    rows.append(priced_component(label+"_lower_cached_inference",lower_windows*(lower_steps-1),lower_cached,
        "Hover retains the fixed source-1 section model, with no parameter changes inside an evaluation window."))
    rows.append(priced_component("uncovered_window_and_panel_json",json_writes,json_price,
        "Hover window JSON and panel summaries only when complete moving replay is used; moving-window JSON is already included. "
        "Measured replay writes a pooled summary payload, so production per-window payload-size differences remain unpriced."))
    scenario_rows=[physical,
        priced_component("all_windows_priced_as_moving_complete_replay",windows,replay,
            "Additional budget scenario: charge hover the measured moving replay including extra upper work. "
            "This is not a measured hover trajectory or a strict upper bound across different queue workloads."),
        priced_component("panel_summary_json_only",panel_summaries,json_price,
            "Each substituted complete replay already includes one atomic JSON; only panel summaries remain.")]
    missing=[r["name"] for r in scenario_rows if r["seconds"] is None]
    scenario=dict(name="hover_priced_as_moving_complete_replay",components=scenario_rows,
        conditional_seconds=None if missing else [sum(r["seconds"][i] for r in scenario_rows) for i in (0,1)],
        unknown_components=missing,strict_upper_bound=False,measured_hover_replay=False,
        method="Conservative workload-budget scenario with extra upper inference for hover; reported separately from the primary estimate.")
    return rows,scenario


def inference_window_crosscheck(clocks,times):
    """Compare scopes, never add nested times or rescale training blindly."""
    lower_steps=round(clocks["rollout_seconds"]/clocks["lower_interval"])
    upper_steps=round(clocks["rollout_seconds"]/clocks["upper_interval"])
    specs=((1,["inference.lower.h1.rebuild"]),
           (lower_steps-1,[k for k in times if re.fullmatch(r"inference\.lower\.h\d+\.cached",k)]),
           (1,["inference.upper.h0.rebuild"]),
           (upper_steps,[k for k in times if re.fullmatch(r"inference\.upper\.h\d+\.cached",k)]),
           (1,["io.metrics_json"]))
    terms=[(count,coefficient(times,keys)) for count,keys in specs]
    replay=coefficient(times,["evaluation.replay900s"])
    model=None if any(price is None for _,price in terms) else [
        sum(count*price[field] for count,price in terms) for field in ("low_s","high_s")]
    actual=None if replay is None else [replay["low_s"],replay["high_s"]]
    return dict(fixed_section_window_model_seconds=model,complete_report_replay_seconds=actual,
        complete_report_replay_median_s=times.get("evaluation.replay900s",{}).get("median_s"),
        replay_minus_component_model_seconds=None if model is None or actual is None else [actual[0]-model[1],actual[1]-model[0]],
        replay_to_component_model_ratio=None if model is None or actual is None or model[0]<=0 else [actual[0]/model[1],actual[1]/model[0]],
        applied_as_training_multiplier=False,added_to_training_seconds=False,
        interpretation="Source-0 complete sequential replay versus source-1 fixed report sections plus JSON proxy. "
        "The difference includes trajectory/graph variation, sequencing and pooling/payload effects, not an isolated optimization gain. "
        "Training retains exact update counts and component pricing; this cross-check quantifies unmodeled trajectory sensitivity.")


def prices(protocol,times,manifest=None,collection_overhead=None):
    """Keep every component visible, including absent measurements."""
    times=dict(times)
    c=protocol["counts"]
    groups={"training":[],"development":[],"optional_primary_test":[]}
    def add(group,name,count,keys,method):
        groups[group].append(priced_component(name,count,coefficient(times,keys),method))
    def matching(pattern):
        return [key for key in times if re.fullmatch(pattern,key)]
    for kind,count,operation in (("critic",c["critic_fits"],"critic_fits"),
                                 ("actor",c["sac_actor_updates"],"actor_updates")):
        by_unique={n:coefficient(times,matching(kind+rf"\.e{n}\.h1800\.(normal|backlog)")) for n in (1,2,4)}
        if by_unique[2] and by_unique[4]:
            by_unique[3]=dict(keys=by_unique[2]["keys"]+by_unique[4]["keys"],
                low_s=(by_unique[2]["low_s"]+by_unique[4]["low_s"])/2,
                high_s=(by_unique[2]["high_s"]+by_unique[4]["high_s"])/2,
                interpolation="Distinct=3 is the midpoint of measured distinct=2/4, not a separate measurement")
        weights=protocol["expected_unique_replay_objects_per_operation"][operation]
        price=None
        if all(by_unique.get(int(n)) is not None for n in weights):
            price=dict(keys=sorted({key for n in weights for key in by_unique[int(n)]["keys"]}),
                low_s=sum(weight*by_unique[int(n)]["low_s"] for n,weight in weights.items())/count,
                high_s=sum(weight*by_unique[int(n)]["high_s"] for n,weight in weights.items())/count,
                expected_unique_weights=weights,by_unique_coefficients=by_unique)
        groups["training"].append(priced_component("sac_"+kind,count,price,
            "128-sample h1800 fixture envelope over normal/backlog reports, weighted by exact probabilities of "
            "1..4 distinct objects from four replay draws at each enumerated occupancy. Distinct=3 uses linear "
            "interpolation of e2/e4; expected object counts are not a realized RNG path or a trained-policy confidence bound."))
    add("training","replay_sampling",c["replay_samples"],matching(r"replay\.sample128.*"),
        "One replay.sample per critic fit and actor fit; sampling is excluded from measured optimizer calls.")
    add("training","target_polyak",c["target_polyak_updates"],["target.polyak"],
        "Once after every nonempty SAC event, not once per critic fit.")
    add("training","ppo_batches_fixture_KL",c["ppo_batches"],["ppo.rows900"],
        "Measured complete 900-row batch times 40; all six value epochs are included. "
        "The sample minimum/maximum applies only to this fixture's fixed KL rollback behavior, "
        "not theoretical wall-time bounds over arbitrary early-stop outcomes. Actor attempted steps are "
        "240..1440 per seed; accepted steps can be zero after rollback. Separately priced actor/value/KL "
        "measurements are unavailable, so no early-stop timing bounds are inferred from these counts.")
    groups["training"][-1].update(interval_scope="empirical_samples_for_fixed_fixture_KL_behavior",
        covers_arbitrary_early_stop_outcomes=False,theoretical_wall_time_bounds_seconds=None)
    collections=[]
    streamed=[]
    if collection_overhead:
        for row in collection_overhead["cases"]:
            seconds=full_window_collection_seconds(row,protocol["clocks"])
            key="collection_overhead."+row["mode"]
            times[key]=dict(median_s=seconds,p95_s=seconds,workload=row,
                           single_short_window_extrapolation=True)
            (streamed if row["mode"]=="streamed" else collections).append(key)
    if manifest:
        for index,row in enumerate(manifest.get("trajectories",[])):
            seconds=row.get("collection_seconds")
            if seconds is not None:
                key=f"fixture.collection.{index}"
                times[key]=dict(median_s=seconds,p95_s=seconds,workload=row)
                collections.append(key)
    collection_method=("Scripted collection including physics, reports, compression and metrics; 60-second stored/streamed "
        "measurements are scaled by 900/60 when supplied. Short-window single measurements do not establish "
        "900-second history/queue/metrics growth or repeat variability; these remain unknown.")
    add("training","physics_reports_compression_metrics",c["episodes"],collections,collection_method)
    add("training","lower_cold_inference",c["episode_start_cache_rebuilds"],["inference.lower.h1.rebuild"],
        "One fresh-history action at the start of every independently reset window.")
    add("training","lower_cached_inference",c["lower_cached_inference"],matching(r"inference\.lower\.h\d+\.cached"),
        "Cached next-frame inference; envelope over the measured prefixes, with exact prefix counts retained.")
    add("training","lower_post_update_rebuild",c["interior_actor_cache_rebuilds"],
        matching(r"inference\.lower\.h(120|600|1800)\.rebuild"),
        "After an interior actor update the next call recomputes its observed prefix. "
        "Final-frame updates are excluded because run.train discards that world.")
    upper_keys=matching(r"inference\.upper\.h\d+\.cached")
    value_keys=matching(r"inference\.upper_value\.h\d+\.cached")
    add("training","upper_without_value_cold",c["upper_cold_without_value"],["inference.upper.h0.rebuild"],
        "One empty-prefix upper call at the start of each moving lower-training window.")
    add("training","upper_without_value_cached",c["upper_cached_without_value"],upper_keys,
        "Moving lower-training windows call upper without value, including the committed cutoff action; macro history delta is ten.")
    add("training","upper_with_value_cold",c["value_cold_inference"],["inference.upper_value.h0.rebuild"],
        "One empty-prefix upper/value call at the start of every upper-training window.")
    add("training","upper_with_value_cached",c["upper_cached_with_value"],value_keys,
        "Upper-training windows call upper and value together, including the precommit cutoff successor value; macro history delta is ten.")
    evaluation_scenarios={}
    for group,windows,upper,panels in (("development",c["dev_episodes"],c["dev_upper_inference"],c["development_panels"]),
                                      ("optional_primary_test",c["optional_test_episodes"],c["optional_test_upper_inference"],1)):
        moving=upper//(int(protocol["clocks"]["rollout_seconds"]/protocol["clocks"]["upper_interval"])+1)
        groups[group],evaluation_scenarios[group]=evaluation_prices(protocol["clocks"],times,
            windows=windows,moving=moving,panel_summaries=panels,physical_price=coefficient(times,streamed))
    add("training","learning_and_attempt_logs",c["learning_log_rows"]+c["attempt_log_rows"],["io.learning_jsonl"],
        "Includes skipped SAC trigger rows, PPO batch rows, and the one fresh-run attempt record.")
    add("training","atomic_json_outside_checkpoint_save",c["training_atomic_json_outside_save"],["io.metrics_json"],
        "Episode metrics, progress, latest pointers, four initial manifests/contract and final result. "
        "An episode-metrics-sized write is a proxy for differently sized JSON records.")
    add("training","source_identity_hashing",c["source_identity_calls_including_fresh_cli"],["source.identity"],
        "One CLI identity, one train binding, and one identity check before each of 600 windows.")
    add("training","initial_parameter_hashing",c["initial_parameter_hashes"],["source.parameter_hash"],
        "Five parameter hashes in the fresh-run contract; per-network workload variation is unmeasured unless supplied.")
    # save includes its first digest and sidecar; persist/latest and dev use
    # additional full-file reads, priced separately below.
    zero=coefficient(times,["checkpoint.save0"])
    two=coefficient(times,["checkpoint.save2"])
    checkpoint_details=None
    save_price=None
    hash_price=None
    if zero and two:
        records=c["checkpoint_replay_episode_records"]+c["checkpoint_pending_ppo_episode_records"]
        per_low=max(0.,(two["low_s"]-zero["high_s"])/2)
        per_high=max(0.,(two["high_s"]-zero["low_s"])/2)
        sums=[c["checkpoint_saves"]*zero["low_s"]+records*per_low,
              c["checkpoint_saves"]*zero["high_s"]+records*per_high]
        save_price=dict(keys=zero["keys"]+two["keys"],low_s=sums[0]/c["checkpoint_saves"],high_s=sums[1]/c["checkpoint_saves"])
        def byte_size(key):
            row=times[key]
            return row.get("bytes",row.get("workload",{}).get("bytes"))
        b0,b2=byte_size("checkpoint.save0"),byte_size("checkpoint.save2")
        checkpoint_details=dict(occupancy_measured=[0,2],episode_record_seconds=[per_low,per_high],
            full55_seconds=[zero["low_s"]+55*per_low,zero["high_s"]+55*per_high],
            cumulative_save_seconds=sums,measured_save0_bytes=b0,measured_save2_bytes=b2)
        digest=coefficient(times,["checkpoint.digest2"])
        hashed_bytes=byte_size("checkpoint.digest2") if digest else None
        if b0 is not None and b2 is not None:
            per_bytes=max(0.,(b2-b0)/2)
            written=c["checkpoint_saves"]*b0+records*per_bytes
            extra=written+c["development_panels"]*b0+c["development_checkpoint_hash_episode_records"]*per_bytes
            checkpoint_details.update(full55_bytes=b0+55*per_bytes,cumulative_written_bytes=written,
                                      additional_hash_read_bytes=extra)
            if digest and hashed_bytes:
                hash_price=dict(keys=digest["keys"],low_s=digest["low_s"]*extra/hashed_bytes,
                                high_s=digest["high_s"]*extra/hashed_bytes)
    groups["training"].append(priced_component("checkpoint_save",c["checkpoint_saves"],save_price,
        "Affine 0/2 completed-episode calibration, extrapolated separately to actual replay and pending-PPO occupancy; "
        "55-slot size/time is extrapolated, never measured. Adam state maturity and large-buffer nonlinear effects remain unknown."))
    groups["training"].append(priced_component("additional_checkpoint_digests",1,hash_price,
        "Byte-scaled measured sequential SHA256 proxy: latest.json digests for every save plus seven pending-development "
        "checkpoint identities. Production uses read_bytes rather than chunked reads, so full-file allocation overhead is not measured.",unit="all reads"))
    summaries={}
    for group,items in groups.items():
        known=[item for item in items if item["seconds"] is not None]
        missing=[item["name"] for item in items if item["seconds"] is None]
        subtotal=[sum(item["seconds"][i] for item in known) for i in (0,1)]
        summaries[group]=dict(known_component_subtotal_seconds=subtotal,unknown_components=missing,
            all_listed_components_priced=not missing,complete_seed_wall_time_measured=False,
            interval_interpretation="Conditional sum of empirical fixture ranges; not theoretical or confidence bounds",
            total_wall_time_bounds_seconds=None)
    for label,parts in (("training_plus_development",("training","development")),
                        ("training_development_primary_test",tuple(groups))):
        summaries[label]=dict(known_component_subtotal_seconds=[sum(summaries[p]["known_component_subtotal_seconds"][i] for p in parts) for i in (0,1)],
            unknown_components=[p+"."+name for p in parts for name in summaries[p]["unknown_components"]],
            all_listed_components_priced=all(summaries[p]["all_listed_components_priced"] for p in parts),
            complete_seed_wall_time_measured=False,
            interval_interpretation="Conditional sum of empirical fixture ranges; does not bound arbitrary PPO early-stop or workload drift",
            total_wall_time_bounds_seconds=None)
    residuals={
        "training":["Unmeasured initialization, per-window reset and initial parameter hashes",
                    "Three scene integrity hashes, checkpoint pruning, GC and read_bytes allocation overhead",
                    "PPO early-stop timing beyond the measured fixture behavior",
                    "60-to-900-second physics/report growth and 0/2-to-55-record checkpoint nonlinearity"],
        "development":["Per-window reset, RNG-scope and hover pooling overhead outside the component timings",
                       "No complete pure-lower hover replay measurement; production JSON payload differences",
                       "60-to-900-second physics/report growth and closed-loop policy workload variation"],
        "optional_primary_test":["Fresh CLI learner construction, final-checkpoint load/GPU restore and per-window reset",
                                 "No complete pure-lower hover replay measurement; production JSON payload differences",
                                 "60-to-900-second physics/report growth and closed-loop policy workload variation"]}
    for group in groups: summaries[group]["unpriced_residuals"]=residuals[group]
    for label,parts in (("training_plus_development",("training","development")),
                        ("training_development_primary_test",tuple(groups))):
        summaries[label]["unpriced_residuals"]=[p+": "+item for p in parts for item in residuals[p]]
    training=summaries["training"]["known_component_subtotal_seconds"]
    dev_scenario=evaluation_scenarios["development"]["conditional_seconds"]
    test_scenario=evaluation_scenarios["optional_primary_test"]["conditional_seconds"]
    evaluation_scenarios["training_plus_development_known_subtotal_seconds"]=None if dev_scenario is None else [training[i]+dev_scenario[i] for i in (0,1)]
    evaluation_scenarios["training_development_primary_test_known_subtotal_seconds"]=None if dev_scenario is None or test_scenario is None else [training[i]+dev_scenario[i]+test_scenario[i] for i in (0,1)]
    return dict(components=groups,subtotals=summaries,checkpoint_extrapolation=checkpoint_details,
        separate_conservative_evaluation_scenario=evaluation_scenarios,
        training_inference_crosscheck=inference_window_crosscheck(protocol["clocks"],times))


def markdown(result):
    c=result["protocol"]["counts"]
    lines=["# Current single-seed conditional budget","", "No research training or physical steps are executed by this tool.","",
           "| Exact current count | Value |","|---|---:|"]
    for key in ("episodes","physics_steps","sac_triggers","skipped_events","sac_events","critic_fits",
                "sac_actor_updates","ppo_batches","ppo_rows_per_batch","lower_inference","upper_inference",
                "value_inference","interior_actor_cache_rebuilds","final_actor_updates_without_online_rebuild",
                "dev_episodes","dev_physics_steps","dev_upper_inference","checkpoint_saves"):
        lines.append(f"| {key} | {c[key]:,} |")
    estimate=result.get("estimate")
    if estimate:
        lines.extend(["","| Scope / component | Conditional hours |","|---|---:|"])
        for group,items in estimate["components"].items():
            for item in items:
                value="unknown" if item["seconds"] is None else " – ".join(f"{s/3600:.3f}" for s in item["seconds"])
                lines.append(f"| {group} / {item['name']} | {value} |")
            summary=estimate["subtotals"][group]
            value=" – ".join(f"{s/3600:.2f}" for s in summary["known_component_subtotal_seconds"])
            lines.append(f"| **{group}: known subtotal** | **{value}** |")
        lines.extend(["","Unknown components are not zero. These are measured-workload sensitivity estimates, not confidence bounds or a completed seed runtime."])
        scenarios=estimate["separate_conservative_evaluation_scenario"]
        lines.extend(["","| Scope | Primary conditional hours | Separate hover-as-moving scenario hours |",
                      "|---|---:|---:|"])
        for group in ("development","optional_primary_test","training_plus_development","training_development_primary_test"):
            primary=estimate["subtotals"][group]["known_component_subtotal_seconds"]
            alternative=(scenarios[group]["conditional_seconds"] if group in ("development","optional_primary_test")
                         else scenarios[group+"_known_subtotal_seconds"])
            show=lambda values:"unknown" if values is None else " – ".join(f"{s/3600:.2f}" for s in values)
            lines.append(f"| {group} | {show(primary)} | {show(alternative)} |")
        lines.extend(["", "The separate scenario prices hover using the moving replay, including additional upper work. "
            "It is neither a measured pure-hover runtime nor a strict upper bound. Moving replay already contains "
            "all its inference, metric pooling and one JSON write; these are not added again."])
        cross=estimate["training_inference_crosscheck"]
        if cross["replay_to_component_model_ratio"] is not None:
            model=" – ".join(f"{s:.2f}" for s in cross["fixed_section_window_model_seconds"])
            measured=" – ".join(f"{s:.2f}" for s in cross["complete_report_replay_seconds"])
            ratio=" – ".join(f"{s:.2f}" for s in cross["replay_to_component_model_ratio"])
            lines.extend(["", f"Inference cross-check: complete source-0 report replay {measured} s versus fixed source-1 "
                f"section model {model} s per moving window ({ratio} times). The difference includes trajectory, "
                "sequencing and pooling/JSON effects. It is not added to training or applied as a training multiplier."])
        lines.extend(["", "PPO's empirical timing range holds its measured fixture KL behavior fixed. "
            "It does not bound arbitrary early stopping: actor steps attempted are "
            f"{c['ppo_actor_optimizer_steps_attempted_min']:,}–{c['ppo_actor_optimizer_steps_attempted_max']:,}, "
            f"accepted steps can be {c['ppo_actor_optimizer_steps_accepted_min']:,}; "
            f"value steps remain {c['ppo_value_optimizer_steps']:,}. "
            "Startup/reset/GC, unmeasured hashing and workload-growth residuals remain unpriced; total wall-time bounds are unknown."])
    lines.extend(["","Limitations:",""]+["- "+item for item in result["limitations"]])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"src")
    parser.add_argument("--config",type=Path,default=ROOT/"src/hrl_v2/configs/v2_1_0.json")
    parser.add_argument("--profile",type=Path)
    parser.add_argument("--fixture-manifest",type=Path)
    parser.add_argument("--collection-overhead",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--counts-only",action="store_true")
    args=parser.parse_args()
    if not args.counts_only and args.profile is None: parser.error("Use --profile or --counts-only")
    sys.path.insert(0,str(args.source_dir.resolve()))
    config=json.loads(args.config.read_text(encoding="utf-8"))
    protocol=enumerate_protocol(config,args.source_dir)
    names=("run","schedule","collector","policy","replay","sac","ppo","evaluate","checkpoint","timing")
    result=dict(schema=SCHEMA,physical_steps_executed=0,training_started=False,
        config_sha256=sha(args.config),source_sha256={name:sha(args.source_dir/"hrl_v2"/(name+".py")) for name in names},
        protocol=protocol,limitations=[
            "Counts describe a fresh uninterrupted 600-window run under the current code/config; resumed or failed work adds unknown discarded work and I/O.",
            "Warmup is two lower windows per block. Empty replay skips each lower block's first window; upper-version changes flush replay. Event parity is global across nonempty events.",
            "Each moving window includes one extra committed upper action at cutoff; upper-training windows also infer its value. That successor is created before the final SAC event and the world is then discarded by run.train.",
            "The two measured scripted physical trajectories are not an empirical distribution of trained policies. Queue growth, graph size, history lengths, replay uniqueness and hardware variability can move runtime outside every shown range.",
            "Profile operation times include their documented boundaries only. Moving evaluation uses full evaluation.replay900s in place of its standalone lower/upper inference and per-window JSON; physical/report construction remains separate. Standalone report/graph/GRU diagnostics are not added again.",
            "No complete pure-lower hover replay was measured. The primary estimate retains hover component prices; a separate scenario charges it the full moving replay with extra upper work, without claiming a strict upper bound. The whole-replay versus fixed-section discrepancy is reported but not used to rescale training.",
            "Checkpoint occupancy is enumerated exactly; 0/2 to 55 episode-record time/bytes are affine extrapolations. Pending PPO records use replay-record size as a proxy.",
            "PPO fixture KL behavior does not determine future rollback frequency. Actor optimizer steps attempted range from 240 to 1440; accepted steps can be zero after rollback. Value steps remain 1440. Empirical fixture timing ranges are not theoretical bounds over arbitrary early-stop outcomes.",
            "Unpriced residuals include scene/reset/startup cost outside measured collection, three scene integrity digests, initial module construction, report-history retention/GC, filesystem contention, checkpoint pruning, and future workload drift. No arbitrary reserve converts these unknowns into measured time.",
            "Optional final-checkpoint paired stochastic testing is shown separately; secondary checkpoints, deterministic panels, baselines and five-seed multiplication are outside the one-seed training-plus-development scope."])
    if args.profile:
        metadata,times=profile_inputs(args.profile,config)
        manifest=json.loads(args.fixture_manifest.read_text(encoding="utf-8")) if args.fixture_manifest else None
        if manifest and manifest["artifact_sha256"]!=metadata["artifact_sha256"]:
            raise ValueError("Profile and physical collection manifest differ")
        result["profile"]=dict(path=str(args.profile.resolve()),metadata_sha256=sha(args.profile/"metadata.json"),
            times_sha256=sha(args.profile/"times.json"),metadata=metadata,
            fixture_manifest_sha256=sha(args.fixture_manifest) if args.fixture_manifest else None)
        overhead=json.loads(args.collection_overhead.read_text(encoding="utf-8")) if args.collection_overhead else None
        if overhead:
            result["collection_overhead_source"]=dict(path=str(args.collection_overhead.resolve()),
                sha256=sha(args.collection_overhead),raw=overhead)
        result["estimate"]=prices(protocol,times,manifest,overhead)
        result["raw_measured_coefficients"]=times
        ppo=times.get("ppo.rows900")
        if ppo:
            before=metadata.get("initial_counts",{}).get("ppo",{})
            after=ppo.get("counts_after",{}).get("ppo",{})
            result["ppo_fixture_update_counts"]={key:value-before.get(key,0) for key,value in after.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output.resolve()),counts=protocol["counts"],
        subtotals=result.get("estimate",{}).get("subtotals")),indent=2))


if __name__=="__main__": main()
