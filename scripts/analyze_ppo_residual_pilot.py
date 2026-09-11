"""Offline analysis of the fixed residual pilot; no environment or policy execution."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import append_csv, write_json
from hrl_uav.geometry import cooperation_aligned_basis
from hrl_uav.ppo_frozen import load_json
from hrl_uav.priority_grid import file_hash
from hrl_uav.target_tracking_check import numeric_rows
from scripts.run_progressive_centroid_recheck import FIELDS


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or min(x[mask].std(), y[mask].std()) < 1e-12:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def summarize(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def analyze(run):
    run = Path(run).resolve()
    manifest, report = load_json(run/"manifest.json"), load_json(run/"report.json")
    if (report["status"] != "RESIDUAL_PPO120_DEVELOPMENT_COMPLETE" or report["ppo_updates"] != 12
            or report["macro_transitions_consumed"] != 3600 or report["forced_updates"]
            or report["reserved_scenes_executed"]):
        raise ValueError("Incomplete or invalid formal pilot")
    for name, digest in manifest["source_hashes"].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError(f"Run source changed before analysis: {name}")
    for name, digest in manifest["reused_file_hashes"].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError(f"Reused reference changed: {name}")
    if file_hash(run/"final.pt") != manifest["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint changed")
    output = run/"analysis"
    output.mkdir(exist_ok=False)
    refs = load_json(run/"reference_records.json")
    final = report["final_development"]
    combined = refs+final
    summaries, pairs, observations, observation_means = [], [], [], []
    for split, seeds in (("validation", (33600,33601)), ("development_recheck", (33610,33611,33612)),
                         ("all_development", (33600,33601,33610,33611,33612))):
        for policy in ("A", "B", "PPO120", "Residual_PPO"):
            selected = [r for r in combined if r["policy"] == policy and int(r["seed"]) in seeds]
            if len(selected) != len(seeds):
                raise ValueError("Missing/duplicate comparison scene")
            summaries.append(dict(policy=policy, split=split, scene_count=len(selected), **summarize(selected, FIELDS)))
    for r in final:
        for label in ("A", "B", "PPO120"):
            ref = next(x for x in refs if x["policy"] == label and x["seed"] == r["seed"])
            pairs.append(dict(pair="Residual-"+label, seed=r["seed"],
                              **{"delta_"+k: r[k]-ref[k] for k in FIELDS}))
    extra = ("energy_propulsion_j", "mean_uav_centroid_distance_m", "fleet_distance_m", "residual_magnitude_mean",
             "residual_magnitude_max", "projection_uav_fraction", "projection_macro_fraction")
    for r in report["observations"]:
        observations.append({k:r[k] for k in ("window", "seed", *FIELDS, *extra)})
    for w in (0,30,60,90,120):
        selected = [r for r in observations if r["window"] == w]
        if len(selected) != 2:
            raise ValueError("Missing observation point")
        observation_means.append(dict(window=w, scenes=2, **summarize(selected, (*FIELDS, *extra))))
    append_csv(output/"summary.csv", summaries)
    append_csv(output/"per_seed.csv", [{k:r[k] for k in ("policy", "seed", "source_path", *FIELDS)} for r in combined])
    append_csv(output/"paired_deltas.csv", pairs)
    append_csv(output/"observations.csv", observations)
    append_csv(output/"observation_means.csv", observation_means)

    batches = numeric_rows(run/"ppo_samples/updates.csv")
    if len(batches) != 12 or any(b["transitions"] != 300 for b in batches):
        raise ValueError("Expected twelve complete saved PPO batches")
    batch_analysis, sample_rows, roles, input_hashes = [], [], [], {}
    slices = manifest["observation_spec"]["macro"]["slices"]
    for b in batches:
        update = int(b["update"])
        path = run/f"ppo_samples/update{update:03d}.npz"
        input_hashes[str(path.relative_to(ROOT))] = file_hash(path)
        with np.load(path) as z:
            data = {k:z[k].copy() for k in z.files}
        if len(data["state"]) != 300 or not all(np.isfinite(v).all() for v in data.values()):
            raise ValueError("Incomplete/nonfinite sample batch")
        np.testing.assert_array_equal(data["update_id"], update)
        np.testing.assert_array_equal(data["global_macro_index"], np.arange((update-1)*300, update*300))
        if not set(data["scene_seed"]).issubset(range(72000,72040)):
            raise ValueError("Unexpected training scenes")
        state = data["state"]

        def feature(name, shape, scale=None):
            lo, hi = slices[name]
            value = state[:, lo:hi].astype(float).reshape((-1, *shape))
            return np.sign(value)*np.expm1(np.abs(value))*scale if scale is not None else value

        q = feature("compute_queue", (2,3), 10).sum(axis=1)
        inbound = feature("inbound_transfer_queue", (2,3), 10).sum(axis=1)
        upload = feature("upload_queue", (2,3), 10).sum(axis=1)
        quality = feature("u2u_spectral_efficiency", (3,3), 1).sum(axis=2)/2
        phase = feature("phase_age", (1,)).ravel()*10
        means = data["residual_mean"].reshape(-1,3,2)
        delta = data["sampled_delta"].reshape(-1,3,2)
        mean_norm = np.linalg.norm(means.reshape(-1,6), axis=1)
        sample_norm = np.linalg.norm(delta.reshape(-1,6), axis=1)
        adv = data["advantage_normalized"]
        gap = data["current_centroid"]-data["current_uav"]
        gap_norm = np.linalg.norm(gap, axis=2)
        worlds = []
        for i in range(300):
            basis, _ = cooperation_aligned_basis(data["current_centroid"][i],data["current_uav"][i],data["current_load"][i])
            worlds.append(60*np.einsum("nij,nj->ni", basis, means[i]))
        worlds = np.asarray(worlds)
        projection = np.divide((worlds*gap).sum(axis=2), gap_norm, out=np.zeros_like(gap_norm), where=gap_norm>1e-12)
        order = np.argsort(sample_norm, kind="stable")
        row = dict(b, mean_residual_norm=float(mean_norm.mean()), sampled_residual_norm=float(sample_norm.mean()),
            mean_residual_norm_p90=float(np.percentile(mean_norm,90)),
            corr_sample_norm_adv=corr(sample_norm,adv), corr_mean_norm_adv=corr(mean_norm,adv),
            near_B_quartile_adv=float(adv[order[:75]].mean()), far_B_quartile_adv=float(adv[order[-75:]].mean()),
            corr_mean_norm_compute_queue=corr(mean_norm,q.sum(axis=1)),
            corr_mean_norm_upload_queue=corr(mean_norm,upload.sum(axis=1)),
            corr_mean_norm_transfer_queue=corr(mean_norm,inbound.sum(axis=1)),
            corr_mean_norm_centroid_gap=corr(mean_norm,gap_norm.mean(axis=1)),
            corr_mean_norm_reference_u2u_quality=corr(mean_norm,quality.mean(axis=1)),
            corr_mean_x_load=corr(means[:,:,0].ravel(),data["current_load"].ravel()),
            corr_mean_x_source_compute=corr(means[:,:,0].ravel(),q.ravel()),
            corr_mean_x_reference_quality=corr(means[:,:,0].ravel(),quality.ravel()),
            corr_mean_toward_centroid_gap=corr(projection.ravel(),gap_norm.ravel()),
            mean_unprojected_toward_centroid_m=float(projection.mean()),
            kl_early_stop=int(b["actual_actor_epochs"] < 10))
        batch_analysis.append(row)
        for i in range(300):
            sample_rows.append(dict(update=update, global_macro_index=int(data["global_macro_index"][i]),
                scene_seed=int(data["scene_seed"][i]), macro_index=int(data["macro_index"][i]),
                mean_norm=float(mean_norm[i]), sampled_norm=float(sample_norm[i]),
                compute_queue_mbit=float(q[i].sum()), upload_queue_mbit=float(upload[i].sum()),
                inbound_transfer_mbit=float(inbound[i].sum()), centroid_gap_m=float(gap_norm[i].mean()),
                reference_u2u_quality=float(quality[i].mean()), phase_age=float(phase[i]),
                normalized_advantage=float(adv[i]), reward=float(data["reward"][i]),
                raw_advantage=float(data["advantage"][i]),
                mean_toward_centroid_unprojected_m=float(projection[i].mean()),
                projection_count=int(data["projected"][i].sum())))
            rank = np.argsort(-data["current_load"][i],kind="stable")
            for role, source in zip(("heavy","medium","light"),rank):
                roles.append(dict(update=update,role=role,source=int(source),local_x=float(means[i,source,0]),
                    local_y=float(means[i,source,1]),mean_norm=float(np.linalg.norm(means[i,source])),
                    sampled_norm=float(np.linalg.norm(delta[i,source])),
                    load_mbps=float(data["current_load"][i,source]),queue_mbit=float(q[i,source]),
                    mean_toward_centroid_unprojected_m=float(projection[i,source])))
    append_csv(output/"batch_analysis.csv",batch_analysis)
    append_csv(output/"sample_mechanism.csv",sample_rows)
    role_summary=[]
    for update in range(1,13):
        for role in ("heavy","medium","light"):
            rows=[r for r in roles if r["update"]==update and r["role"]==role]
            keys=("local_x","local_y","mean_norm","sampled_norm","load_mbps","queue_mbit","mean_toward_centroid_unprojected_m")
            role_summary.append(dict(update=update,role=role,samples=len(rows),**summarize(rows,keys)))
    append_csv(output/"role_action_summary.csv",role_summary)
    high_norm_states=[]
    for update in range(1,13):
        rows=[r for r in sample_rows if r["update"]==update]
        sorted_rows=sorted(rows,key=lambda r:r["mean_norm"])
        for group,selected in (("bottom_quartile",sorted_rows[:75]),("top_quartile",sorted_rows[-75:])):
            keys=("mean_norm","compute_queue_mbit","upload_queue_mbit","inbound_transfer_mbit","centroid_gap_m","reference_u2u_quality","phase_age","normalized_advantage")
            high_norm_states.append(dict(update=update,group=group,samples=75,**summarize(selected,keys)))
    append_csv(output/"residual_state_quartiles.csv",high_norm_states)
    route_rows=[]
    for r in final:
        route_rows.extend(numeric_rows(Path(r["source_path"])/"route_summary.csv"))
    append_csv(output/"final_route_summary.csv",route_rows)
    diagnostic=dict(actor_epochs=[int(b["actual_actor_epochs"]) for b in batch_analysis],
        kl_early_stops=sum(b["kl_early_stop"] for b in batch_analysis),
        actor_gradient_steps=report["ppo_actor_gradient_steps"],critic_gradient_steps=report["ppo_critic_gradient_steps"],
        projected_uav_count=int(sum(b["projection_uav_count"] for b in batch_analysis)),
        projected_macro_count=int(sum(b["projection_macro_count"] for b in batch_analysis)),
        projection_uav_denominator=10800,projection_macro_denominator=3600,
        relationships={k:dict(positive_batches=sum(b[k] is not None and b[k]>0 for b in batch_analysis),
                             negative_batches=sum(b[k] is not None and b[k]<0 for b in batch_analysis))
                       for k in batch_analysis[0] if k.startswith("corr_")},
        limitations=["Single training initialization; within-batch/state correlations are not causal effects.",
            "Positive GAE is an estimated advantage, not proof that a residual beats B at the same state.",
            "U2U observation is fixed reference quality, NOT allocated transmission rate.",
            "Unprojected mean direction is not the realized movement after clipping/safety limits.",
            "Initial-batch mean is zero; tied residual-magnitude groups have no directional interpretation.",
            "No same-state counterfactual residual-vs-zero rollouts were authorized."])
    write_json(output/"diagnostics.json",diagnostic)
    for name in ("manifest.json","report.json","reference_records.json","ppo_samples/updates.csv"):
        input_hashes[str((run/name).relative_to(ROOT))]=file_hash(run/name)
    write_json(output/"manifest.json",dict(environment_steps=0,learning_updates=0,policy_inference=False,
        source_commit=manifest["actual_head"],input_sha256=input_hashes,reserved_scenes_executed=[]))
    table=["# Residual PPO120 development pilot", "", "## Five used development scenes", "",
        "|Policy|J|Holding|Energy|Weighted ms|High ms|Low ms|Power W|Terminal queue Mbit|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["split"]=="all_development":
            table.append(f"|{r['policy']}|{r['J']:.9f}|{r['J_holding']:.9f}|{r['J_energy']:.9f}|{r['task_delay_weighted_ms']:.4f}|{r['task_delay_high_ms']:.4f}|{r['task_delay_low_ms']:.4f}|{r['power_w']:.4f}|{r['queue_total_mbit']:.4f}|")
    table.extend(["", "## Residual minus B", "", "|Scene|J|Holding|Energy|Weighted ms|Power W|", "|---|---:|---:|---:|---:|---:|"])
    for r in pairs:
        if r["pair"]=="Residual-B":
            table.append(f"|{r['seed']}|{r['delta_J']:+.9f}|{r['delta_J_holding']:+.9f}|{r['delta_J_energy']:+.9f}|{r['delta_task_delay_weighted_ms']:+.4f}|{r['delta_power_w']:+.4f}|")
    table.extend(["", "All completion rates, unfinished counts, mean/terminal stage queues and slopes are retained in per_seed.csv and paired_deltas.csv.",
        "Observation means and per-scene values: observation_means.csv and observations.csv.", "", "## Training facts", "",
        f"12 unforced batches, 300 transitions each; {diagnostic['actor_gradient_steps']} actor steps and {diagnostic['critic_gradient_steps']} critic steps.",
        f"Actor epochs: {diagnostic['actor_epochs']}; KL early stops: {diagnostic['kl_early_stops']}.",
        f"Projected macro actions: {diagnostic['projected_macro_count']}/3600; projected UAV rows: {diagnostic['projected_uav_count']}/10800.",
        "SAC frozen, 64800 replay transitions collected, no replay sampling, no reserved scenes.", "",
        "## Mechanism scope", "", *diagnostic["limitations"], "",
        "Batch correlations, residual-state quartiles and role-conditioned means are descriptive only.",
        "Full sample data, values, returns and advantages were saved before each native update cleared its rollout.",
        "Stopped at window120; no extra parameter changes, continuation or alternate learner experiment."])
    (output/"result.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    print("Offline analysis complete:",output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    analyze(parser.parse_args().run)
