"""Offline descriptive summaries of the fixed Stage3A matrix; no simulation."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .experiment import append_csv, write_json
from .mission_opportunity import C_ATOL, R_ATOL, SCENES, TIMES, actions
from scripts.analyze_local_counterfactual import distribution, ranking


def summarize(rows, states):
    if len(rows) != 208 or len(states) != 16:
        raise ValueError("Require all208 branches and16 states; no selective summary")
    per_state, fixed, associations = [], [], []
    for feature in states:
        seed, start = feature["seed"], feature["start_s"]
        selected = [r for r in rows if r["seed"] == seed and r["start_s"] == start]
        if len(selected) != 13 or {r["action"] for r in selected} != {a["name"] for a in actions()}:
            raise ValueError("Incomplete or duplicated action set")
        zero = next(r for r in selected if r["action"] == "zero")
        effective = [r for r in selected if r["action"] != "zero" and not r["ineffective"]]
        if not effective:
            raise ValueError("No effective action: report missing oracle rather than invent a direction")
        best = min(effective, key=lambda r: (r["C60"], r["action"]))
        rank = ranking(selected, C_ATOL, R_ATOL)
        c, ret = np.array([r["C60"] for r in selected]), np.array([r["ppo_return60"] for r in selected])
        corr = float(spearmanr(c, -ret).statistic) if np.ptp(c) > C_ATOL and np.ptp(ret) > R_ATOL else None
        row = dict(seed=seed, start_s=start, zero_C60=zero["C60"], best_action=best["action"],
                   effective_nonzero=len(effective), improving_effective=sum(r["delta_C60"] < -C_ATOL for r in effective),
                   ineffective_nonzero=12-len(effective),
                   zero_is_best=all(r["delta_C60"] >= -C_ATOL for r in effective),
                   best_delta_C60=best["delta_C60"], best_delta_holding=best["delta_C60_holding"],
                   best_delta_energy=best["delta_C60_energy"], best_relative_percent=best["relative_C60_percent"],
                   best_delta_ppo_return=best["delta_ppo_return60"],
                   best_also_return_improves=best["delta_ppo_return60"]>R_ATOL,
                   best_sets_overlap=rank["best_sets_overlap"], same_exact_argbest=rank["physical_training_exact_argbest_equal"],
                   spearman_physical_vs_negative_return=corr,
                   concordant_pairs=rank["pair_concordant"], discordant_pairs=rank["pair_discordant"],
                   tied_pairs=rank["pair_ties"], conflict_actions=len(rank["conflict_indices"]),
                   conflict_action_names=",".join(selected[i]["action"] for i in rank["conflict_indices"]),
                   physical_best_actions=",".join(selected[i]["action"] for i in rank["physical_indices"]),
                   return_best_actions=",".join(selected[i]["action"] for i in rank["training_indices"]))
        per_state.append(row)
        uav = (best["dimension"]-1)//2
        tx = np.asarray(feature["transfer_queue"])
        association = dict(seed=seed, start_s=start, best_action=best["action"], uav=uav,
                           axis="xy"[(best["dimension"]-1)%2], sign=best["sign"], role=feature["roles"][uav],
                           current_load_mbps=feature["current_load"][uav],
                           compute_queue_mbit=feature["compute_queue"][uav],
                           outgoing_transfer_mbit=float(tx[uav].sum()), inbound_transfer_mbit=float(tx[:, uav].sum()),
                           upload_queue_mbit=feature["upload_queue"][uav],
                           u2g_mean_rate_mbps=feature["u2g_mean_rate_mbps"][uav],
                           u2u_outgoing_reference_mean=float(np.sum(feature["u2u_reference_quality"][uav])/2),
                           centroid_offset_x_m=feature["centroid_offset"][uav][0],
                           centroid_offset_y_m=feature["centroid_offset"][uav][1],
                           centroid_velocity_x_mps=feature["centroid_velocity"][uav][0],
                           centroid_velocity_y_mps=feature["centroid_velocity"][uav][1],
                           best_delta_C60=best["delta_C60"], relative_percent=best["relative_C60_percent"])
        associations.append(association)
    for action in actions()[1:]:
        chosen = [r for r in rows if r["action"] == action["name"]]
        if len(chosen) != 16:
            raise ValueError("Fixed direction does not cover all16 states")
        fixed.append(dict(action=action["name"], mean_delta_C60=float(np.mean([r["delta_C60"] for r in chosen])),
                          mean_delta_holding=float(np.mean([r["delta_C60_holding"] for r in chosen])),
                          mean_delta_energy=float(np.mean([r["delta_C60_energy"] for r in chosen])),
                          mean_relative_percent=float(np.mean([r["relative_C60_percent"] for r in chosen])),
                          improving_states=sum(r["delta_C60"] < -C_ATOL and not r["ineffective"] for r in chosen),
                          ineffective_states=sum(r["ineffective"] for r in chosen)))
    fixed.sort(key=lambda r: (r["mean_delta_C60"], r["action"]))
    bests = [r["best_delta_C60"] for r in per_state]
    improving = [r for r in per_state if r["best_delta_C60"] < -C_ATOL]
    nonzero = [r for r in rows if r["action"] != "zero"]
    grouped = []
    for field in ("role", "uav", "best_action"):
        for value in sorted({r[field] for r in associations}, key=str):
            group = [r for r in associations if r[field] == value]
            grouped.append(dict(group_by=field, group=value, states=len(group),
                                mean_delta_C60=float(np.mean([r["best_delta_C60"] for r in group])),
                                mean_compute_queue_mbit=float(np.mean([r["compute_queue_mbit"] for r in group])),
                                mean_inbound_transfer_mbit=float(np.mean([r["inbound_transfer_mbit"] for r in group])),
                                mean_upload_queue_mbit=float(np.mean([r["upload_queue_mbit"] for r in group])),
                                mean_current_load_mbps=float(np.mean([r["current_load_mbps"] for r in group])),
                                mean_u2g_rate_mbps=float(np.mean([r["u2g_mean_rate_mbps"] for r in group]))))
    comparable = sum(r["concordant_pairs"]+r["discordant_pairs"] for r in per_state)
    summary = dict(states=16, branches=208, improving_states=len(improving),
                   zero_best_states=sum(r["zero_is_best"] for r in per_state),
                   best_nonzero_delta_distribution=distribution(bests),
                   relative_percent_distribution=distribution([r["best_relative_percent"] for r in per_state]),
                   statewise_mean_delta_C60=float(np.mean(bests)),
                   statewise_mean_delta_J=float(np.mean(bests)/600),
                   statewise_mean_holding_delta=float(np.mean([r["best_delta_holding"] for r in per_state])),
                   statewise_mean_energy_delta=float(np.mean([r["best_delta_energy"] for r in per_state])),
                   zero_fallback_oracle_mean_delta_C60=float(np.mean(np.minimum(bests, 0))),
                   best_fixed_direction=fixed[0],
                   oracle_extra_gain_C60=fixed[0]["mean_delta_C60"]-float(np.mean(bests)),
                   improving_best_holding_down=sum(r["best_delta_holding"] < -C_ATOL for r in improving),
                   improving_best_energy_down=sum(r["best_delta_energy"] < -C_ATOL for r in improving),
                   improving_best_both_down=sum(r["best_delta_holding"] < -C_ATOL and r["best_delta_energy"] < -C_ATOL for r in improving),
                   improving_best_holding_up_energy_win=sum(r["best_delta_holding"] > C_ATOL and r["best_delta_energy"] < -C_ATOL for r in improving),
                   best_action_overlap_states=sum(r["best_sets_overlap"] for r in per_state),
                   same_exact_argbest_states=sum(r["same_exact_argbest"] for r in per_state),
                   improving_best_also_return_improves=sum(r["best_also_return_improves"] for r in improving),
                   mean_rank_spearman=float(np.mean([r["spearman_physical_vs_negative_return"] for r in per_state if r["spearman_physical_vs_negative_return"] is not None])),
                   comparable_action_pairs=comparable,
                   pairwise_ordering_agreement=sum(r["concordant_pairs"] for r in per_state)/max(comparable, 1),
                   conflict_states=sum(r["conflict_actions"] > 0 for r in per_state),
                   conflict_nonzero_actions=sum(r["conflict_actions"] for r in per_state),
                   ineffective_nonzero=sum(r["ineffective"] for r in nonzero),
                   ineffective_denominator=192, projection_numerical_tolerance_m=1e-8,
                   future_replicates=1, independent_training_runs=0,
                   projected_nonzero_branches=sum(r["component_clip_uavs"]+r["unit_disk_projection_uavs"]+r["map_projection_uavs"]+r["safety_projection_uavs"] > 0 for r in nonzero))
    return summary, per_state, fixed, associations, grouped


def analyze(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    if manifest["status"] != "COMPLETE_208_BRANCHES_STOPPED":
        raise ValueError("Analyze only the completed fixed matrix")
    states = json.loads((output/"states.json").read_text())
    if {(s["seed"], s["start_s"]) for s in states} != {(s, t) for s in SCENES for t in TIMES}:
        raise ValueError("Unexpected state selection")
    rows = [json.loads((output/"states"/f'{s["seed"]}_t{s["start_s"]}'/a["name"]/"summary.json").read_text())
            for s in states for a in actions()]
    summary, per_state, fixed, associations, grouped = summarize(rows, states)
    write_json(output/"analysis.json", summary)
    for name, data in (("per_state", per_state), ("fixed_directions", fixed),
                       ("state_associations", associations), ("association_groups", grouped)):
        path = output/f"{name}.csv"
        if path.exists():
            raise FileExistsError("Analysis already exists; do not append duplicate results")
        append_csv(path, data)
    f = summary["best_fixed_direction"]
    lines = ["# Mission V1 Stage 3A: Predictive local opportunity", "",
             "## Scope and verification", "",
             f'- Run HEAD: `{manifest["actual_head"]}`; physics: `{manifest["physics_base_commit"]}`.',
             "- Two reused development scenes33600/33601; eight predeclared states each; one future realization per state.",
             "- Completed208 branches /1248 macro intervals. Predictive prefix replays0..860s only construct real states and verify zero continuations.",
             "- Source checkpoint: SAC101 window2000, deterministic frozen inference. No PPO/TD3 instance, training, replay, extra scene or parameter search.",
             "- All16 zero branches reproduce the full uninterrupted Predictive terminal state. All208 initial snapshots/RNG and exogenous streams verified.",
             "- Prefix trajectories, task births, user motion, mission groups, load schedules and frame ledgers match the archived Predictive pilot.",
             "- See manifest.json, checks.json, per-branch crn.csv and source_evidence/ for provenance.", "",
             "## A-B. Local opportunity and magnitude", "",
             f'- Effective improving nonzero action exists in {summary["improving_states"]}/16 states. Zero is best within tolerance in {summary["zero_best_states"]}/16.',
             f'- Statewise best effective nonzero: mean Delta C60={summary["statewise_mean_delta_C60"]:.9f}; mean Delta J60={summary["statewise_mean_delta_J"]:.9f}.',
             f'- Mean state-relative change={summary["relative_percent_distribution"]["mean"]:.6f}%; median={summary["relative_percent_distribution"]["median"]:.6f}%.',
             f'- Absolute distribution: `{json.dumps(summary["best_nonzero_delta_distribution"], sort_keys=True)}`.',
             f'- Ineffective perturbations: {summary["ineffective_nonzero"]}/192; tolerance1e-8m over all600 trajectory samples, excluded from statewise nonzero oracle.', "",
             "|Scene|Branch time/s|Best effective direction|Delta C60|Holding|Energy|Relative/%|Effective improvements|",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in per_state:
        lines.append(f'|{r["seed"]}|{r["start_s"]}|{r["best_action"]}|{r["best_delta_C60"]:.7f}|{r["best_delta_holding"]:.7f}|{r["best_delta_energy"]:.7f}|{r["best_relative_percent"]:.5f}|{r["improving_effective"]}|')
    lines += ["", "## C. Holding and energy", "",
              f'- Mean selected holding delta={summary["statewise_mean_holding_delta"]:.9f}; energy delta={summary["statewise_mean_energy_delta"]:.9f}; these add to Delta C60.',
              f'- Among improving selected directions: holding decreases in {summary["improving_best_holding_down"]}; energy decreases in {summary["improving_best_energy_down"]}; both in {summary["improving_best_both_down"]}.',
              f'- Holding increases but energy compensates in {summary["improving_best_holding_up_energy_win"]} selected states.',
              "- Full propulsion energy and signed increment are separately recorded. Full average power includes hover power; objective energy uses the original signed increment.", "",
              "## D. Retrospective statewise selection versus one fixed direction", "",
              f'- Best fixed direction on all16 states: {f["action"]}; mean Delta C60={f["mean_delta_C60"]:.9f}; mean relative change={f["mean_relative_percent"]:.6f}%.',
              f'- Additional statewise gain in mean C60={summary["oracle_extra_gain_C60"]:.9f}. This is a descriptive in-sample oracle gap, not statistical significance or a realizable policy.',
              "- fixed_directions.csv retains all12 directions, including ineffective and negative cases; state_associations.csv lists role, queues, load, U2G rate, reference U2U quality and centroid motion for all selected states.",
              "- e1/e2=UAV0 world x/y, e3/e4=UAV1, e5/e6=UAV2. Heavy/medium/light roles are recomputed from current load, not hard-coded UAV IDs.", "",
              "## E. Native PPO-return ordering", "",
              f'- Physical minimum / return maximum best-set overlap: {summary["best_action_overlap_states"]}/16; exact argbest agreement: {summary["same_exact_argbest_states"]}/16.',
              f'- Mean Spearman rank correlation(C60, -return)={summary["mean_rank_spearman"]:.6f}; pairwise agreement excluding tolerance ties={summary["pairwise_ordering_agreement"]:.6f}.',
              f'- Physical improvement but worse return: {summary["conflict_nonzero_actions"]} actions in {summary["conflict_states"]} states; see per_state.csv for exact conflicts.',
              f'- Selected physical improving direction also improves return in {summary["improving_best_also_return_improves"]}/{summary["improving_states"]} states.',
              "- This is a truncated60s return without value bootstrap or entropy. Frame reward and20-frame macro aggregation match experiment.rollout; macro discount is gamma^20 as in learning.py. All600-step cost sums and direct discounted sums close numerically.", "",
              "## F-G. Decision boundary and missing evidence", "",
              "- Any nonzero gain is conditional on these16 real states and their single future paths. Searching12 actions retrospectively introduces oracle selection optimism.",
              "- A statewise advantage over the best fixed direction is a reason to consider a separately authorized multi-future check, not proof that current observations predict its sign or that PPO can learn it.",
              "- Neither weak gains nor absent gains bound all feasible policies: only1m pre-projection impulses, one intervention followed by Predictive for50s, no continuation reoptimization and no horizon tail were tested.",
              "- No final-test generalization, training stability, significance, RL necessity or superiority claim follows. Stage3A stops here.",
              "- Per-branch queues are120 frame-end samples; completed delay includes parents born before the branch and completed during it. Completion denominators include initially unfinished parents. Unfinished does not mean deadline violation.",
              "- U2G/U2U service, sender time and actual work-weighted effective rates are logged, not inferred from the9D reference quality. Per-task critical-path stage delays are not available and are not reconstructed.", ""]
    (output/"result.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
