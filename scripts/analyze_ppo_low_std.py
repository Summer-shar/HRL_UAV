"""Post-run analysis of the fixed low-std pilot. No environment execution."""

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import append_csv, write_json
from hrl_uav.ppo_diagnostics import action_geometry
from hrl_uav.ppo_frozen import load_json, SOURCE_DIGESTS
from hrl_uav.priority_grid import file_hash
from hrl_uav.resumption import state_equal
from hrl_uav.target_tracking_check import numeric_rows
from hrl_uav.agents.ppo_agent import ActorNet
from scripts.analyze_ppo120_macro_mechanism import evaluate_geometry
from scripts.run_progressive_centroid_recheck import FIELDS, normalize, compare_processes


OLD = ROOT / "results/ppo_frozen_sac_pilot120_seed101"
B_RUN = ROOT / "results/progressive_centroid_dev_recheck"
SEEDS = (33600, 33601, 33610, 33611, 33612)


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or min(x[valid].std(), y[valid].std()) < 1e-12:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def geometry_summary(rows, window):
    b = np.array([r["B_norm"] for r in rows])
    d = np.array([r["distance_to_B"] for r in rows])
    angles = [r["angle_deg"] for r in rows if r["angle_deg"] is not None]
    return dict(window=window, macro_samples=len(rows), distance_to_B_mean=float(d.mean()),
                B_norm_mean=float(b.mean()), distance_ratio_to_zero=float(d.mean()/b.mean()),
                closer_than_zero_fraction=float(np.mean(d < b-1e-10)),
                angle_deg_mean=float(np.mean(angles)) if angles else None)


def analyze(run):
    run = Path(run).resolve()
    output = run / "analysis"
    if output.exists():
        raise FileExistsError(output)
    manifest, report = load_json(run/"manifest.json"), load_json(run/"report.json")
    old_manifest, old_report = load_json(OLD/"manifest.json"), load_json(OLD/"report.json")
    bm = load_json(B_RUN/"manifest.json")
    if report["ppo_windows"] != 120 or report["ppo_updates"] != 12 or report["reserved_scenes_executed"]:
        raise AssertionError("Not the complete, authorized pilot")
    original_cfg = copy.deepcopy(manifest["config"])
    original_cfg["training"]["ppo_initial_std"] = 1/3
    assert original_cfg == old_manifest["config"]
    for key in ("environment", "observation", "cpu_ghz", "phase_loads", "phase_dwell", "reward",
                "waypoint_radius_m", "max_displacement_m"):
        assert original_cfg[key] == bm["config"][key]
    assert manifest["source_checkpoint_sha256"] == old_manifest["source_checkpoint_sha256"] == bm["source_digests"]["checkpoint"] == SOURCE_DIGESTS["checkpoint"]
    assert file_hash(ROOT/"hrl_uav/target_tracking_check.py") == bm["B_controller_sha256"]
    assert file_hash(OLD/"manifest.json") == manifest["reference_pilot"]["manifest_sha256"]
    assert file_hash(OLD/"report.json") == manifest["reference_pilot"]["report_sha256"]
    for name, digest in bm["reused_file_hashes"].items():
        assert file_hash(ROOT/name) == digest, name
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(ROOT/name) == digest, name
    output.mkdir()
    inputs = {}

    def remember(path):
        inputs[str(Path(path).relative_to(ROOT))] = file_hash(path)

    for path in (run/"manifest.json", run/"report.json", OLD/"manifest.json", OLD/"report.json",
                 B_RUN/"manifest.json", B_RUN/"per_seed.csv"):
        remember(path)
    historical = numeric_rows(B_RUN/"per_seed.csv")
    cfg, hover = manifest["config"], bm["hover_baseline_power_w"]
    observations, current_rows, paired, crn = [], [], [], []
    history = report["validation_history"]
    assert [h["window"] for h in history] == [0, 30, 60, 90, 120]
    source = torch.load(manifest["source_checkpoint"], map_location="cpu", weights_only=False)
    remember(manifest["source_checkpoint"])
    for h in history:
        w = h["window"]
        checkpoint = source if w == 0 else torch.load(run/f"checkpoints/window{w}.pt", map_location="cpu", weights_only=False)
        if w:
            remember(run/f"checkpoints/window{w}.pt")
        for key in ("sac_actor", "sac_critic", "sac_target", "log_alpha"):
            assert state_equal(checkpoint[key], source[key]), key
        assert state_equal(checkpoint["optimizers"][:3], source["optimizers"][:3])
        delta = max(float((checkpoint["macro_actor"][k]-v).abs().max()) for k, v in source["macro_actor"].items() if k != "log_std")
        for raw in h["ppo"]:
            r = normalize(raw, "low_std_PPO120", cfg, hover)
            observations.append(dict(window=w, seed=r["seed"], ppo_updates=checkpoint["updates"]["ppo"],
                                     ppo_gradient_steps=checkpoint["updates"]["ppo_gradients"],
                                     macro_parameter_max_delta=delta, **{k:r[k] for k in FIELDS}))
            if w == 120:
                current_rows.append(r)
    current_rows.extend(normalize(r,"low_std_PPO120",cfg,hover) for r in report["development_final"]["ppo"])
    for r in current_rows:
        seed = r["seed"]
        label = "validation" if seed in SEEDS[:2] else "development_final"
        prefix = run/f"{label}_ppo_w120_{seed}"
        r.update(reused=False, source_path=str(run/"report.json"),
                 crn_path=str(prefix)+"_crn.csv", trajectory_path=str(prefix)+"_trajectory.npz")
        for method in ("A", "B", "PPO120"):
            reference = next(x for x in historical if x["policy"] == method and x["seed"] == seed)
            if method == "B":
                directory = Path(reference["source_path"])
                reference.update(crn_path=str(directory/"crn.csv"), trajectory_path=str(directory/"trajectory.npz"))
            else:
                old_label = "centroid" if method == "A" else "ppo"
                p = OLD/f"{label}_{old_label}_w120_{seed}"
                reference.update(crn_path=str(p)+"_crn.csv", trajectory_path=str(p)+"_trajectory.npz")
            crn.append(compare_processes(reference, r))
            for row in (reference, r):
                remember(row["crn_path"])
                remember(row["trajectory_path"])
            paired.append(dict(pair="low_std-"+method, seed=seed, **{"delta_"+k:r[k]-reference[k] for k in FIELDS}))
        baseline_raw = next(x for x in (history[-1] if seed in SEEDS[:2] else report["development_final"])["centroid"] if x["seed"] == seed)
        baseline = normalize(baseline_raw,"A",cfg,hover)
        a = next(x for x in historical if x["policy"] == "A" and x["seed"] == seed)
        for key in FIELDS:
            np.testing.assert_allclose(baseline[key], a[key], atol=1e-9, rtol=0)
    combined = historical + current_rows
    summaries = []
    for split, seeds in (("validation", SEEDS[:2]), ("development_recheck", SEEDS[2:]), ("all_development", SEEDS)):
        for policy in ("A", "B", "PPO120", "low_std_PPO120"):
            rows = [r for r in combined if r["policy"]==policy and r["seed"] in seeds]
            assert len(rows) == len(seeds)
            summaries.append(dict(policy=policy, split=split, scenes=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in FIELDS}))
    append_csv(output/"per_seed.csv", [{k:r[k] for k in ("policy","seed","reused","source_path",*FIELDS)} for r in combined])
    append_csv(output/"observations.csv", observations)
    append_csv(output/"summary.csv", summaries)
    append_csv(output/"paired_deltas.csv", paired)
    write_json(output/"pairing_checks.json", crn)

    # Same saved geometry readback as the original offline audit; first macro is omitted.
    geometries, g_summary = [], []
    for w in (0,30,60,90,120):
        rows = []
        for seed in SEEDS[:2]:
            p = run/f"validation_ppo_w{w}_{seed}"
            remember(str(p)+"_trajectory.npz")
            remember(str(p)+"_queues.csv")
            with np.load(str(p)+"_trajectory.npz") as t:
                local, _ = evaluate_geometry(t, numeric_rows(str(p)+"_queues.csv"), cfg, w, seed)
            rows.extend(local)
        geometries.extend(rows)
        g_summary.append(geometry_summary(rows,w))
    append_csv(output/"evaluation_actions.csv", geometries)
    append_csv(output/"evaluation_action_summary.csv", g_summary)

    updates = numeric_rows(run/"ppo_samples/updates.csv")
    assert len(updates) == 12 and sum(int(u["transitions"]) for u in updates) == 3600
    sample_batches, samples = [], []
    for u in updates:
        path = run/f"ppo_samples/update{int(u['update']):03d}.npz"
        remember(path)
        with np.load(path) as z:
            distance, adv = z["distance_to_B"], z["advantage_normalized"]
            order = np.argsort(distance)
            near, far = order[:75], order[-75:]
            cosine = np.cos(np.radians(z["angle_deg"]))
            rms = float(np.sqrt(np.mean(z["B_action"]**2)))
            sample_batches.append(dict(u, closeness_adv_corr=correlation(-distance,adv),
                                       cosine_adv_corr=correlation(cosine,adv),
                                       projection_adv_corr=correlation(z["projection_length"],adv),
                                       action_norm_adv_corr=correlation(z["action_norm"],adv),
                                       near_quartile_adv=float(adv[near].mean()), far_quartile_adv=float(adv[far].mean()),
                                       near_minus_far_adv=float(adv[near].mean()-adv[far].mean()),
                                       B_component_rms=rms, std_over_B_rms=float(u["exploration_std"])/rms))
            samples.append({k:z[k].copy() for k in ("state", "B_action")})
    append_csv(output/"batch_analysis.csv", sample_batches)
    remember(run/"ppo_samples/updates.csv")

    # Frozen offline inference on identical new-run sample states, not a new rollout.
    probe = []
    for label, directory, metadata in (("PPO120",OLD,old_manifest), ("low_std_PPO120",run,manifest)):
        path = Path(metadata["final_checkpoint"])
        assert file_hash(path) == metadata["final_checkpoint_sha256"]
        remember(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        net = ActorNet(saved["macro_state_dim"], 6)
        net.load_state_dict(saved["macro_actor"])
        net.eval()
        for index, sample in enumerate(samples,1):
            with torch.no_grad():
                mean, _ = net(torch.as_tensor(sample["state"], dtype=torch.float32))
            g = action_geometry(mean.numpy(),sample["B_action"])
            probe.append(dict(policy=label, update_states=index, samples=len(mean),
                              distance_mean=float(g["distance_to_B"].mean()), B_norm_mean=float(g["B_norm"].mean()),
                              distance_ratio_to_zero=float(g["distance_to_B"].mean()/g["B_norm"].mean()),
                              closer_than_zero_fraction=float(np.mean(g["distance_to_B"] < g["B_norm"]))))
        del saved, net
    append_csv(output/"common_state_probe.csv",probe)
    diagnostics = dict(early_KL_stops=sum(bool(u["actual_actor_epochs"] < cfg["training"]["ppo_epochs"]) for u in updates),
                       actual_actor_epochs=[int(u["actual_actor_epochs"]) for u in updates],
                       ppo_actor_gradient_steps=report["ppo_gradients"],
                       old_actor_gradient_steps=None,
                       caveats=["Old per-sample advantages unavailable; correlations are not causal effects.",
                                "Raw distance is confounded by action magnitude, geometry and queue state.",
                                "Common-state probe uses new stochastic training states, not unseen scenes.",
                                "Evaluation geometry is measured on each policy's induced states; initial macro omitted."])
    old_final = torch.load(OLD/"final.pt", map_location="cpu", weights_only=False)
    diagnostics["old_actor_gradient_steps"] = old_final["updates"]["ppo_gradients"]
    write_json(output/"diagnostics.json",diagnostics)
    write_json(output/"manifest.json",dict(input_sha256=inputs, training=False, environment_steps=0,
                                           policy_inference="offline final macro policies on saved states only",
                                           reserved_scenes_executed=[], reused_historical_rows=15))
    table = ["# Low-exploration PPO120 development pilot", "", "## Five development scenes", "",
             "|Policy|J|Weighted ms|High ms|Low ms|Power W|Terminal queue Mbit|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["split"] == "all_development":
            table.append(f"|{r['policy']}|{r['J']:.9f}|{r['task_delay_weighted_ms']:.4f}|{r['task_delay_high_ms']:.4f}|{r['task_delay_low_ms']:.4f}|{r['power_w']:.4f}|{r['queue_total_mbit']:.4f}|")
    table.extend(["", "Full completion, unfinished-task, queue-component and paired metrics: per_seed.csv and paired_deltas.csv.",
                  "Validation (33600/33601) and development recheck (33610/33611/33612) are separated in summary.csv.",
                  "", "## Learning diagnostics", "",
                  f"12 PPO updates; actor steps {diagnostics['ppo_actor_gradient_steps']} (original {diagnostics['old_actor_gradient_steps']}); KL early stops {diagnostics['early_KL_stops']}.",
                  f"Actual actor epochs: {diagnostics['actual_actor_epochs']}",
                  "", "Batch-specific distance/angle/projection correlations with actual normalized GAE are in batch_analysis.csv.",
                  "Readback of deterministic evaluation actions: evaluation_action_summary.csv. Identical saved-state inference: common_state_probe.csv.",
                  "No original per-sample GAE was saved, so a matched old/new advantage comparison is unavailable.",
                  "Closeness can reflect action magnitude rather than directional learning. These are associations, not causal proof.",
                  "", "## Scope", "", "Only initial std changed from 1/3 to 1/6; original update implementation retained.",
                  "SAC actor/critic/target/alpha/optimizers frozen. No SAC updates. No reserved scenes.",
                  "Final window120 only. This is one paired development pilot, not convergence, significance or generalization evidence.",
                  "Stopped after 120 windows. No parameter search, no further training."])
    (output/"result.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    print("Offline analysis complete:",output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    analyze(parser.parse_args().run)
