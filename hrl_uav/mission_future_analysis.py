"""Offline Stage3B summaries; frozen actions are never selected again."""

import json
from pathlib import Path

import numpy as np

from .experiment import append_csv, write_json
from .mission_future_check import LABELS, FUTURES, C_ATOL, R_ATOL
from scripts.analyze_local_counterfactual import ranking


def stats(values):
    a = np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()))


def summarize(rows,plan):
    if len(rows) != 320 or len(plan) != 16:
        raise ValueError("Require all320 branches and16 frozen selections")
    paired, per_state = [], []
    for item in plan:
        scene,start = item["scene"],item["start_s"]
        state_pairs = []
        for k in FUTURES:
            group = [r for r in rows if r["seed"]==scene and r["start_s"]==start and r["future"]==k]
            if len(group)!=4 or {r["action"] for r in group}!=set(LABELS):
                raise ValueError("Missing/duplicated frozen candidate")
            by = {r["action"]:r for r in group}
            for candidate in item["candidates"]:
                if by[candidate["label"]]["direction"] != candidate["name"]:
                    raise ValueError("Validation action differs from predeclared selection")
            rank = ranking(group,C_ATOL,R_ATOL)
            pair = dict(seed=scene,start_s=start,future=k,selected_direction=item["selected_direction"],
                        zero_C60=by["zero"]["C60"],best_sets_overlap=rank["best_sets_overlap"],
                        pair_concordant=rank["pair_concordant"],pair_discordant=rank["pair_discordant"],
                        pair_ties=rank["pair_ties"],ranking_conflicts=len(rank["conflict_indices"]))
            for label in LABELS[1:]:
                r = by[label]
                for name in ("C60","C60_holding","C60_energy","ppo_return60","power_w",
                             "task_delay_weighted_ms","task_delay_high_ms","task_delay_low_ms"):
                    pair[f"{label}_delta_{name}"] = r[name]-by["zero"][name]
                pair[f"{label}_relative_percent"] = 100*pair[f"{label}_delta_C60"]/by["zero"]["C60"]
                pair[f"{label}_improves"] = pair[f"{label}_delta_C60"] < -C_ATOL
                pair[f"{label}_return_improves"] = pair[f"{label}_delta_ppo_return60"] > R_ATOL
                pair[f"{label}_ineffective"] = r["ineffective"]
            for label in ("fixed_best","opposite"):
                pair[f"selected_minus_{label}_C60"] = by["stage3a_best"]["C60"]-by[label]["C60"]
                pair[f"selected_minus_{label}_holding"] = by["stage3a_best"]["C60_holding"]-by[label]["C60_holding"]
                pair[f"selected_minus_{label}_energy"] = by["stage3a_best"]["C60_energy"]-by[label]["C60_energy"]
            paired.append(pair)
            state_pairs.append(pair)
        state = dict(seed=scene,start_s=start,selected_direction=item["selected_direction"],
                     selection_delta_C60=item["stage3a_delta_C60"],selection_delta_holding=item["stage3a_delta_holding"],
                     selection_delta_energy=item["stage3a_delta_energy"],selection_relative_percent=item["stage3a_relative_percent"])
        for label in LABELS[1:]:
            for key,value in stats([r[f"{label}_delta_C60"] for r in state_pairs]).items():
                state[f"{label}_delta_C60_{key}"] = value
            state[f"{label}_improving_futures"] = sum(r[f"{label}_improves"] for r in state_pairs)
            for field in ("C60_holding","C60_energy","ppo_return60","power_w","task_delay_weighted_ms",
                          "task_delay_high_ms","task_delay_low_ms"):
                state[f"{label}_delta_{field}_mean"] = float(np.mean([r[f"{label}_delta_{field}"] for r in state_pairs]))
            state[f"{label}_relative_percent_mean"] = float(np.mean([r[f"{label}_relative_percent"] for r in state_pairs]))
        for label in ("fixed_best","opposite"):
            state[f"selected_minus_{label}_C60_mean"] = float(np.mean([r[f"selected_minus_{label}_C60"] for r in state_pairs]))
        state["fresh_minus_selection_C60"] = state["stage3a_best_delta_C60_mean"]-state["selection_delta_C60"]
        state["best_opposite_directional"] = state["stage3a_best_delta_C60_mean"] < -C_ATOL and state["opposite_delta_C60_mean"] > C_ATOL
        state["best_opposite_both_improve"] = state["stage3a_best_delta_C60_mean"] < -C_ATOL and state["opposite_delta_C60_mean"] < -C_ATOL
        per_state.append(state)
    aggregate = []
    for label in LABELS[1:]:
        summary = dict(action=label,state_count=16,futures_each=5,
                       improving_state_future_pairs=sum(p[f"{label}_improves"] for p in paired),
                       improving_pair_fraction=sum(p[f"{label}_improves"] for p in paired)/80,
                       new_mean_improving_states=sum(s[f"{label}_delta_C60_mean"] < -C_ATOL for s in per_state),
                       four_or_five_improving_future_states=sum(s[f"{label}_improving_futures"] >= 4 for s in per_state),
                       ineffective_pairs=sum(p[f"{label}_ineffective"] for p in paired))
        summary.update({f"state_mean_delta_C60_{k}":v for k,v in stats([s[f"{label}_delta_C60_mean"] for s in per_state]).items()})
        summary["mean_delta_J60"] = summary["state_mean_delta_C60_mean"]/600
        for field in ("C60_holding","C60_energy","ppo_return60","power_w","task_delay_weighted_ms","task_delay_high_ms","task_delay_low_ms"):
            summary[f"mean_delta_{field}"] = float(np.mean([s[f"{label}_delta_{field}_mean"] for s in per_state]))
        summary["mean_relative_percent"] = float(np.mean([s[f"{label}_relative_percent_mean"] for s in per_state]))
        aggregate.append(summary)
    selected = aggregate[0]
    original = float(np.mean([s["selection_delta_C60"] for s in per_state]))
    comparable = sum(p["pair_concordant"]+p["pair_discordant"] for p in paired)
    overall = dict(states=16,state_future_pairs=80,branches=320,training_initializations=0,
                   conditioned_on_original_load_schedule=True,
                   actions=aggregate,selection_original_mean_delta_C60=original,
                   selection_original_mean_relative_percent=float(np.mean([s["selection_relative_percent"] for s in per_state])),
                   fresh_minus_selection_delta_C60=selected["state_mean_delta_C60_mean"]-original,
                   signed_gain_retained_fraction=selected["state_mean_delta_C60_mean"]/original,
                   best_set_overlap_pairs=sum(p["best_sets_overlap"] for p in paired),
                   pairwise_ordering_agreement=sum(p["pair_concordant"] for p in paired)/max(comparable,1),
                   comparable_rank_pairs=comparable,tied_rank_pairs=sum(p["pair_ties"] for p in paired),
                   conflict_action_instances=sum(p["ranking_conflicts"] for p in paired),
                   conflict_state_future_pairs=sum(p["ranking_conflicts"]>0 for p in paired),
                   selected_physical_improvements_also_return_improve=sum(p["stage3a_best_improves"] and p["stage3a_best_return_improves"] for p in paired),
                   selected_vs_opposite_directional_states=sum(s["best_opposite_directional"] for s in per_state),
                   selected_and_opposite_both_improve_states=sum(s["best_opposite_both_improve"] for s in per_state))
    for label in ("fixed_best","opposite"):
        values = [s[f"selected_minus_{label}_C60_mean"] for s in per_state]
        overall[f"selected_minus_{label}"] = dict(**stats(values),
            favorable_states=sum(v < -C_ATOL for v in values),adverse_states=sum(v > C_ATOL for v in values),
            tied_states=sum(abs(v)<=C_ATOL for v in values),
            favorable_pairs=sum(p[f"selected_minus_{label}_C60"] < -C_ATOL for p in paired),
            adverse_pairs=sum(p[f"selected_minus_{label}_C60"] > C_ATOL for p in paired),
            tied_pairs=sum(abs(p[f"selected_minus_{label}_C60"])<=C_ATOL for p in paired))
    return overall,per_state,paired,aggregate


def analyze(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    if manifest["status"] != "COMPLETE_320_BRANCHES_STOPPED":
        raise ValueError("Only analyze completed320-branch validation")
    plan = json.loads((output/"frozen_candidates.json").read_text())
    rows = [json.loads((output/"states"/f'{p["scene"]}_t{p["start_s"]}'/f"future{k}"/label/"summary.json").read_text())
            for p in plan for k in FUTURES for label in LABELS]
    overall,states,pairs,aggregate = summarize(rows,plan)
    for name,data in (("per_state",states),("paired_deltas",pairs),("summary",aggregate)):
        path = output/f"{name}.csv"
        if path.exists():
            raise FileExistsError("Do not append duplicate analysis results")
        append_csv(path,data)
    write_json(output/"analysis.json",overall)
    best = aggregate[0]
    lines = ["# Mission Stage3B: frozen-candidate fresh-future validation", "",
             "## Scope", "",
             f'- Stage3A archive: `{manifest["stage3a_archive_commit"]}`; run HEAD: `{manifest["actual_head"]}`.',
             "-16 original real snapshots; five new futures each; four frozen labels;320x60s branches completed. No900s run or new snapshot state.",
             "- Fresh task arrivals and user-local innovations only. The full original pre-generated load-switch schedule and mission reference routes remain conditioned/fixed.",
             "- This is NOT robustness to new load-switch dwell schedules. Original physical/history state is unchanged except replacing the future environment RNG stream.",
             "- Original Stage3A future is selection data only; none of its outcomes enters fresh-validation means. No candidate was reselected.",
             "- No training/replay/PPO/TD3; SAC101 deterministic frozen. No other scenes, rewards, physics, scales or extra futures.", "",
             "## Fresh validation", "",
             "|Frozen action|Improving pairs/80|Mean-improving states/16|Mean Delta C60|Holding|Energy|Mean relative/%|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f'|{r["action"]}|{r["improving_state_future_pairs"]}|{r["new_mean_improving_states"]}|{r["state_mean_delta_C60_mean"]:.8f}|{r["mean_delta_C60_holding"]:.8f}|{r["mean_delta_C60_energy"]:.8f}|{r["mean_relative_percent"]:.6f}|')
    lines += ["",f'- Selected direction has4/5 or5/5 improving futures in {best["four_or_five_improving_future_states"]}/16 states.',
              f'- Selected direction mean Delta J60={best["mean_delta_J60"]:.9f}. This is not a900s policy score.',
              f'- Original selection mean Delta C60={overall["selection_original_mean_delta_C60"]:.9f}; fresh minus original={overall["fresh_minus_selection_delta_C60"]:.9f}.',
              f'- Signed gain retained ratio={overall["signed_gain_retained_fraction"]:.6f}; negative would mean a sign reversal, not retained benefit.',
              "- First average five futures per state, then16 states. Reported std is across16 state means, not independent training initialization variability.", "",
              "|Scene|t/s|Selected direction|Original Delta C60|Fresh selected mean|Fresh selected std|Improving/5|Holding mean|Energy mean|Fixed mean|Opposite mean|",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in states:
        lines.append(f'|{s["seed"]}|{s["start_s"]}|{s["selected_direction"]}|{s["selection_delta_C60"]:.7f}|{s["stage3a_best_delta_C60_mean"]:.7f}|{s["stage3a_best_delta_C60_std"]:.7f}|{s["stage3a_best_improving_futures"]}|{s["stage3a_best_delta_C60_holding_mean"]:.7f}|{s["stage3a_best_delta_C60_energy_mean"]:.7f}|{s["fixed_best_delta_C60_mean"]:.7f}|{s["opposite_delta_C60_mean"]:.7f}|')
    lines += ["", "## Direction comparisons", ""]
    for label in ("fixed_best","opposite"):
        r = overall[f"selected_minus_{label}"]
        lines += [f'- Selected minus {label}: mean Delta C60={r["mean"]:.9f}; state means favorable/adverse/tied={r["favorable_states"]}/{r["adverse_states"]}/{r["tied_states"]}; paired futures={r["favorable_pairs"]}/{r["adverse_pairs"]}/{r["tied_pairs"]}.']
    lines += [f'- Selected improves and opposite worsens in {overall["selected_vs_opposite_directional_states"]}/16 state means; both improve in {overall["selected_and_opposite_both_improve_states"]}/16.',
              "- Identical actual directions under different labels are intentionally retained, independently simulated and checked bitwise. They must not be counted as competing independent policies.", "",
              "## PPO return versus physical objective", "",
              f'- Best-set overlap among four frozen labels: {overall["best_set_overlap_pairs"]}/80.',
              f'- Pairwise ordering agreement={overall["pairwise_ordering_agreement"]:.6f}, among {overall["comparable_rank_pairs"]} comparable pairs; {overall["tied_rank_pairs"]} tolerance ties excluded.',
              f'- Physical-improving but return-worse: {overall["conflict_action_instances"]} action instances in {overall["conflict_state_future_pairs"]} state/future pairs.',
              f'- Selected physical improvement also improves return in {overall["selected_physical_improvements_also_return_improve"]}/{best["improving_state_future_pairs"]} improving pairs.',
              "- Exact native frame/macro reward chronology and energy ledgers reused from Stage3A.60s truncation has no value tail or entropy bonus. Ranking is diagnostic, not new action selection.", "",
              "## Evidence boundaries", "",
              "- These results describe persistence of frozen selections under five fresh innovation streams conditional on the same full states and load schedules. They are not a learned state-to-action predictor or unseen-state validation.",
              "- First-macro1m pre-projection impulses followed by50s Predictive do not bound larger, sustained or longer-horizon feasible controls.",
              "- Small means, sign variation and in-sample state selection require caution; no significance, PPO learnability or global optimality claim is made.",
              "- All primary/auxiliary records remain available per branch: original physical cost, holding/energy, completion/unfinished, queues, actual radio work/time, routes and projections.",
              "- Stopped after320 branches. No automatic training, additional future, model modification or push follows.", ""]
    (output/"result.md").write_text("\n".join(lines),encoding="utf-8")
    return overall
