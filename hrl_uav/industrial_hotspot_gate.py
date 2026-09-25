"""Pure, versioned admission decisions. No simulation or neural imports."""
import copy
import hashlib
import json
import math


class TechnicalEvaluationError(ValueError):
    """An invalid evaluation is never an ordinary quality failure."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _number(value, name, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TechnicalEvaluationError("Nonfinite or invalid " + name)
    if nonnegative and value < 0:
        raise TechnicalEvaluationError("Negative " + name)
    return float(value)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise TechnicalEvaluationError("Invalid count " + name)
    return value


def validate_metric_row(row):
    if row.get("status") != "complete" or row.get("technical_valid") is not True:
        raise TechnicalEvaluationError("An incomplete or technically invalid trajectory cannot be scored")
    if row.get("frozen_verified") is not True or row.get("metrics_frozen_before_priming") is not True:
        raise TechnicalEvaluationError("Missing frozen-policy or pre-priming metric verification")
    if not isinstance(row.get("scenario_id"), (str, int)) or isinstance(row.get("scenario_id"), bool):
        raise TechnicalEvaluationError("Missing scenario identifier")
    for key in ("scenario_sha", "realization_sha", "event_sha256"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise TechnicalEvaluationError("Missing paired identity " + key)
    m = row["metrics"]
    if _number(m["seconds"], "seconds") <= 0:
        raise TechnicalEvaluationError("Empty evaluation duration")
    _number(m["cost_sum"], "physical cost")
    _number(m["holding_task_s"], "holding")
    for value in m["energy_j"].values():
        _number(value, "energy")
    if len(m["priority"]) != 2:
        raise TechnicalEvaluationError("Expected two priorities")
    for priority in m["priority"]:
        arrived, completed = (_count(priority[name], name) for name in ("arrived", "completed"))
        if completed > arrived:
            raise TechnicalEvaluationError("Completed parents exceed arrivals")
        _number(priority["delay_sum_s"], "delay sum")
    unfinished = m["priority"][1]["arrived"] - m["priority"][1]["completed"]
    if row.get("low_unfinished") != unfinished:
        raise TechnicalEvaluationError("Low-priority cutoff count does not equal arrived minus completed")
    return row


def paired_rows(left, right):
    def indexed(rows):
        result = {}
        for row in rows:
            validate_metric_row(row)
            key = str(row["scenario_id"])
            if key in result:
                raise TechnicalEvaluationError("Duplicate evaluation scenario")
            result[key] = row
        return result
    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise TechnicalEvaluationError("Evaluation scenario sets differ")
    for key in a:
        for identity in ("scenario_sha", "realization_sha", "event_sha256", "profile_permutation"):
            if a[key].get(identity) != b[key].get(identity):
                raise TechnicalEvaluationError("Paired scene/event identity differs: " + identity)
        if a[key]["metrics"]["seconds"] != b[key]["metrics"]["seconds"]:
            raise TechnicalEvaluationError("Paired durations differ")
        for p in range(2):
            if a[key]["metrics"]["priority"][p]["arrived"] != b[key]["metrics"]["priority"][p]["arrived"]:
                raise TechnicalEvaluationError("Paired parent arrivals differ")
    return [(a[key], b[key]) for key in sorted(a)]


def pooled_metrics(rows, priority_weights=(10., 1.)):
    if not rows:
        raise TechnicalEvaluationError("Cannot pool empty evaluations")
    for row in rows:
        validate_metric_row(row)
    priorities = []
    for p in range(2):
        arrived = sum(row["metrics"]["priority"][p]["arrived"] for row in rows)
        completed = sum(row["metrics"]["priority"][p]["completed"] for row in rows)
        delay = sum(row["metrics"]["priority"][p]["delay_sum_s"] for row in rows)
        priorities.append(dict(arrived=arrived, completed=completed, unfinished=arrived-completed,
            completion_rate=completed/arrived if arrived else None,
            delay_sum_s=delay, mean_delay_ms=1000*delay/completed if completed else None))
    seconds = sum(row["metrics"]["seconds"] for row in rows)
    cost = sum(row["metrics"]["cost_sum"] for row in rows)
    denominator = sum(w*p["completed"] for w, p in zip(priority_weights, priorities))
    energy_keys = set(rows[0]["metrics"]["energy_j"])
    if any(set(row["metrics"]["energy_j"]) != energy_keys for row in rows):
        raise TechnicalEvaluationError("Energy schemas differ")
    return dict(seconds=seconds, cost_sum=cost, mean_cost_per_s=cost/seconds,
        physical_return=-cost, holding_task_s=sum(r["metrics"]["holding_task_s"] for r in rows),
        energy_j={key: sum(r["metrics"]["energy_j"][key] for r in rows) for key in sorted(energy_keys)},
        priority=priorities, weighted_delay_ms=(1000*sum(w*p["delay_sum_s"] for w, p in
            zip(priority_weights, priorities))/denominator if denominator else None))


def evaluate_admission(sac_rows, local_rows, cvx_rows, rules):
    pairs = paired_rows(sac_rows, local_rows)
    paired_rows(sac_rows, cvx_rows)
    if len(pairs) != 6:
        raise TechnicalEvaluationError("Admission requires the six sealed development scenarios")
    s, l, c = (pooled_metrics(rows) for rows in (sac_rows, local_rows, cvx_rows))
    if s["priority"][0]["completion_rate"] is None or l["priority"][0]["completion_rate"] is None:
        raise TechnicalEvaluationError("No high-priority arrivals to evaluate admission")
    small_loss = int(sum(bool(a["metrics"]["cost_sum"] <= rules["per_scene_cost_ratio_to_local_max"]*
        b["metrics"]["cost_sum"]) for a, b in pairs))
    low_s, low_l = s["priority"][1]["unfinished"], l["priority"][1]["unfinished"]
    low_ok = bool(low_s <= rules["pooled_low_unfinished_ratio_to_local_max"]*low_l)
    checks = dict(total_cost=bool(s["cost_sum"] <= rules["cost_ratio_to_local_max"]*l["cost_sum"]),
        scene_coverage=bool(small_loss >= rules["per_scene_required_count"]),
        high_completion=bool(s["priority"][0]["completion_rate"] >= l["priority"][0]["completion_rate"]-
            rules["pooled_high_completion_deficit_max_percentage_points"]/100),
        low_unfinished=low_ok)
    return dict(schema="industrial_hotspot_admission_measurement_v1", rule_version=rules["version"],
        quality_passed=all(checks.values()), checks=checks, qualifying_scenes=small_loss,
        pooled=dict(sac=s, local=l, cvx=c), cvx_reference_only=True)


def initial_gate_state(check_episodes, max_episodes):
    nodes = list(check_episodes)
    if (not nodes or any(type(n) is not int or n <= 0 for n in nodes) or
            nodes != sorted(set(nodes)) or nodes[-1] != max_episodes):
        raise ValueError("Admission nodes must be increasing and end at the maximum episode")
    return dict(schema="industrial_hotspot_admission_state_v1", status="continue",
        check_episodes=nodes, max_episodes=max_episodes, consecutive=0, last_node=None,
        admitted=False, admitted_candidate_sha=None, history=[], context=None,
        consecutive_passes=0, last_processed_episode=None, next_check_episode=nodes[0])


def advance_gate(state, episode, candidate_sha, result):
    """Commit a complete candidate once; caller atomically checkpoints returned state."""
    current = copy.deepcopy(state)
    if current.get("schema") != "industrial_hotspot_admission_state_v1":
        raise ValueError("Unknown admission-state schema")
    if result.get("status") != "complete" or result.get("candidate_sha") != candidate_sha:
        raise TechnicalEvaluationError("Candidate evaluation is incomplete or belongs to another checkpoint")
    identity = result.get("identity", {})
    required = {"candidate_sha", "scene_contract_sha", "dev_manifest_sha", "admission_rule_version", "evaluator_source_sha"}
    if set(identity) != required or identity["candidate_sha"] != candidate_sha or any(not isinstance(v, str) or not v for v in identity.values()):
        raise TechnicalEvaluationError("Incomplete candidate evaluation identity")
    evaluation_sha = result.get("evaluation_file_sha")
    if not isinstance(evaluation_sha, str) or len(evaluation_sha) != 64:
        raise TechnicalEvaluationError("A committed evaluation file digest is required")
    measured = result.get("gate_result", {})
    if (type(result.get("quality_passed")) is not bool or measured.get("quality_passed") != result["quality_passed"]
            or measured.get("rule_version") != identity["admission_rule_version"]):
        raise TechnicalEvaluationError("Admission result and measured checks disagree")
    checks = measured.get("checks", {})
    if (set(checks) != {"total_cost", "scene_coverage", "high_completion", "low_unfinished"}
            or any(type(value) is not bool for value in checks.values())
            or all(checks.values()) != result["quality_passed"]):
        raise TechnicalEvaluationError("Quality flag does not match the recorded admission checks")
    commit = dict(episode=episode, candidate_sha=candidate_sha,
        evaluation_file_sha=evaluation_sha, identity=identity, quality_passed=result["quality_passed"])
    for old in current["history"]:
        if old["episode"] == episode:
            if old != commit:
                raise TechnicalEvaluationError("A committed node cannot be replaced")
            return current
        if old["candidate_sha"] == candidate_sha:
            raise TechnicalEvaluationError("One candidate cannot count as two independent checkpoints")
    if current["admitted"] or current["status"] == "needs_review":
        raise TechnicalEvaluationError("Admission is already final")
    index = len(current["history"])
    if index >= len(current["check_episodes"]) or episode != current["check_episodes"][index]:
        raise TechnicalEvaluationError("Missing or out-of-order admission node")
    context = {key: value for key, value in identity.items() if key != "candidate_sha"}
    if current["context"] is not None and current["context"] != context:
        raise TechnicalEvaluationError("Scene, panel, evaluator or admission rule changed")
    current["context"] = context
    current["history"].append(commit)
    current["last_node"] = episode
    current["consecutive"] = current["consecutive"]+1 if result["quality_passed"] else 0
    current["last_processed_episode"] = episode
    current["consecutive_passes"] = current["consecutive"]
    current["next_check_episode"] = (current["check_episodes"][index+1]
        if index+1 < len(current["check_episodes"]) else None)
    if current["consecutive"] >= 2:
        current.update(status="admitted", admitted=True, admitted_candidate_sha=candidate_sha)
        current["next_check_episode"] = None
    elif episode == current["max_episodes"]:
        current["status"] = "needs_review"
    return current


advance = advance_gate
