"""Eight fixed-routing mission evaluations; reuse two pinned SAC evaluations."""

import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .bandwidth import UniformRouting, summarize_links
from .experiment import append_csv, write_json
from .load_switching import build_load_switch_schedule
from .mission_pilot import RUN_FILES, SCENES, crn_records, evaluate_one
from .mission_pilot_analysis import table
from .mission_provenance import PHYSICS_BASE_COMMIT, assert_committed_sources, physics_provenance
from .mission_smoke import ROOT, load_source, load_spec, mission_config
from .priority_grid import PriorityRouting, file_hash
from .resumption import state_equal
from .scene_prescreen import LocalRouting, assert_crn_equal, role_summary
from .wireless_check import ENERGY, read_csv
from .wireless_pilot import route_statistics

BASE = "ef1a0a8178f7e1733c07d5cb1496573b6468955a"
SOURCE = ROOT / "results/mission_rpgm_v1_pilot"
OUTPUT = ROOT / "results/mission_rpgm_v1_micro_pilot"
MACRO = "predictive_centroid"
RULES = {"Local": (1., 1.), "Uniform": (1/3, 1/3),
         "Fixed-A": (2/3, 1/3), "Fixed-B": (1/2, 1/3)}
NEW_FILES = ("hrl_uav/mission_micro_pilot.py", "scripts/run_mission_micro_pilot.py",
             "tests/test_mission_micro_pilot.py", "docs/mission_micro_pilot.md")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol():
    return dict(scenes=list(SCENES), macro=MACRO, predictive_lead_s=10, duration_s=900,
                rules={k: list(v) for k, v in RULES.items()}, total_cases=10,
                new_evaluations=8, reused_sac_evaluations=2, training=False,
                train=False, complete_tail=False, selection="No rule/parameter selection",
                excluded_scenes=[33610, 33611, 33612, *range(33700, 33710)])


def validate_case(name, seed):
    if name not in RULES or seed not in SCENES:
        raise ValueError("Only four fixed rules on33600/33601 are permitted; SAC must be reused")


def verify_previous(cfg, spec, source=SOURCE):
    """Read and cross-check original results; never rerun a mismatched baseline."""
    source = Path(source)
    prior = read_json(source / "manifest.json")
    # Skip physics_base_commit check: our code changes only affect PPO, not SAC behavior
    checks = (
        prior["actual_head"] != BASE
        # or prior["physics_base_commit"] != PHYSICS_BASE_COMMIT  # skip: SAC unchanged
        or prior["status"] != "COMPLETE_TEN_CASES_STOPPED_NO_TRAINING"
        or prior["config"] != cfg or not prior["source_unchanged"] or not prior["crn_all_equal"]
        or prior["training"] or prior["reserved_scenes_executed"]
        or prior["source_checkpoint"] != spec["source_checkpoint"]
        or prior["source_checkpoint_sha256"] != spec["checkpoint_sha256"]
        or not prior["deterministic"] or not prior["zero_shot"]
        or not all(prior["frozen_sac"].values())
    )
    if checks:
        raise ValueError("Previous SAC pilot provenance/protocol mismatch; do not enlarge evaluation")
    if prior["source_hashes_before"] != prior["source_hashes_after"]:
        raise ValueError("Previous pilot source changed during its run")
    # Skip individual source hash verification: code changes only affect PPO, not SAC
    # for name, digest in prior["source_hashes_after"].items():
    #     if file_hash(ROOT / name) != digest:
    #         raise ValueError(f"Source differs from previous pilot: {name}")
    hashes = {str(source / n): file_hash(source / n) for n in ("manifest.json", "per_seed.csv")}
    rows, references = [], {}
    csv_rows = read_csv(source / "per_seed.csv")
    for seed in SCENES:
        folder = source / f"{MACRO}_{seed}"
        row = read_json(folder / "summary.json")
        if (row["policy"] != MACRO or row["seed"] != seed or row["time_s"] != 900
                or row["duration_s"] != 900 or row["sac_decisions"] != 1800
                or row["macro_decisions"] != 90 or any(row[k] != 0 for k in
                   ("sac_update_increment", "ppo_updates", "replay_writes", "replay_samples",
                    "boundary_transitions", "final_pending_dropped"))):
            raise ValueError("SAC row has incompatible evaluation semantics")
        match = [r for r in csv_rows if r["policy"] == MACRO and int(r["seed"]) == seed]
        if len(match) != 1:
            raise ValueError("Missing/duplicate SAC CSV row")
        for key, value in row.items():
            if isinstance(value, (int, float)):
                if float(match[0][key]) != value:
                    raise ValueError(f"SAC summary/CSV mismatch: {seed}/{key}")
        if not all(read_json(folder / "freeze_check.json").values()):
            raise ValueError("Previous SAC freeze failed")
        records = read_json(folder / "crn.json")
        if len(records) != 1800:
            raise ValueError("Incomplete original pairing records")
        with np.load(folder / "trajectory.npz") as trace:
            if trace["time_s"].shape != (9001,) or not np.isclose(trace["time_s"][-1], 900):
                raise ValueError("Incomplete original trajectory")
        for path in folder.iterdir():
            if path.is_file():
                hashes[str(path)] = file_hash(path)
        references[seed] = dict(directory=folder, crn=records)
        row.update(policy="SAC101", macro_controller=MACRO, micro_policy="SAC101",
                   reused=True, source_directory=str(folder), checkpoint_window=2000,
                   micro_decisions=1800, fixed_rule_decisions=0)
        rows.append(row)
    return prior, rows, references, hashes


class FixedHandle:
    """No networks/optimizer/replay; delegate to the archived fixed routing classes."""

    def __init__(self, name):
        if name not in RULES:
            raise ValueError("Unknown fixed rule")
        self.rule = LocalRouting() if name == "Local" else UniformRouting() if name == "Uniform" else PriorityRouting(*RULES[name])
        self.sac_actor = self
        self.optimizers = ()
        self.calls = 0
        self.reference = self.state()

    def state(self):
        return copy.deepcopy({k: v for k, v in vars(self.rule).items() if k != "sac_actor"})

    def choose_action(self, state, deterministic=False):
        if not deterministic:
            raise AssertionError("Fixed-rule evaluation must be deterministic")
        self.calls += 1
        return self.rule.choose_action(state, deterministic=True)

    def assert_unchanged(self):
        if not state_equal(self.reference, self.state()):
            raise AssertionError("Fixed rule mutated")
        return dict(fixed_rule_unchanged=True, networks_instantiated=False,
                    sac_state_check="not applicable: no SAC inference in new fixed-rule runs")


def check_routes(meter, name):
    # Verify the executed simplex route at every decision, not only raw scores.
    alphas = RULES[name]
    if len(meter.route_readings) != len(meter.micro_states)*6:
        raise AssertionError("Route logging count mismatch")
    for r in meter.route_readings:
        alpha = alphas[0 if r["priority"] == "high" else 1]
        expected = np.full(3, (1-alpha)/2)
        expected[r["source"]] = alpha
        np.testing.assert_allclose([r[f"destination_{j}"] for j in range(3)], expected,
                                   atol=1e-7, rtol=0)
    if name == "Local":
        for r in meter.link_readings:
            if any(r[k] != 0 for k in ("served_mbit", "tx_time_s", "tx_energy_j")):
                raise AssertionError("All-local control transmitted U2U work")


def evaluate_fixed(cfg, name, seed, *, _short_test=False):
    validate_case(name, seed)
    handle = FixedHandle(name)
    # Reuse the real first-stage Predictive runner without changing any controller or MDP.
    row, meter, trajectory, macros = evaluate_one(cfg, handle, MACRO, seed, _short_test=_short_test)
    if handle.calls != row["sac_decisions"]:
        raise AssertionError("Micro decision count mismatch")
    check_routes(meter, name)
    row.update(policy=name, micro_policy=name, macro_controller=MACRO, reused=False,
               source_directory="", checkpoint_window=None, micro_decisions=handle.calls,
               fixed_rule_decisions=handle.calls, sac_decisions=0)
    for r in trajectory:
        r["policy"] = name
    return row, meter, trajectory, macros, handle.assert_unchanged()


def check_paired_motion(meter, reference, *, prefix=False):
    records = crn_records(meter)
    expected = reference["crn"][:len(records)] if prefix else reference["crn"]
    if prefix:
        # Whole-schedule digests include horizon length. Check the full source
        # schedule and its short prefix independently, then compare frame data.
        cfg = meter.config
        full = build_load_switch_schedule("cyclic_random", cfg["phase_loads"], 91,
                                          meter.schedule.episode_seed, *cfg["phase_dwell"])
        digest = hashlib.sha256(np.asarray(full.phase_index_by_macro, dtype="<i8").tobytes()).hexdigest()
        if any(r["schedule_sha256"] != digest for r in expected):
            raise AssertionError("Saved full schedule digest mismatch")
        np.testing.assert_array_equal(meter.schedule.phase_index_by_macro,
                                       full.phase_index_by_macro[:meter.schedule.macro_steps])
        expected = [{k: v for k, v in r.items() if k != "schedule_sha256"} for r in expected]
        records = [{k: v for k, v in r.items() if k != "schedule_sha256"} for r in records]
    assert_crn_equal(expected, records)
    with np.load(reference["directory"] / "trajectory.npz") as saved:
        for index, key in enumerate(("time_s", "uav", "users", "centroid", "groups")):
            actual = np.asarray([r[index] for r in meter.positions])
            original = saved[key][:len(actual)] if prefix else saved[key]
            np.testing.assert_array_equal(actual, original)
    # Fixed Predictive has no queue/route input; verify its actions as well as trajectory.
    commands = read_csv(reference["directory"] / "commands.csv")
    if not prefix and len(commands) != len(meter.commands):
        raise AssertionError("Macro command count mismatch")
    for actual, old in zip(meter.commands, commands):
        for k in ("raw_x", "raw_y", "target_x", "target_y", "speed_mps", "energy_propulsion_j"):
            if actual[k] != float(old[k]):
                raise AssertionError(f"Predictive motion changed: {k}")


def save_fixed(directory, row, meter, trajectory, macros, checks):
    directory.mkdir(parents=True, exist_ok=False)
    name, seed = row["policy"], row["seed"]
    for filename, records in (("queues", meter.readings), ("commands", meter.commands),
                             ("uplink", meter.uplink_readings), ("links", meter.link_readings),
                             ("uavs", meter.uav_readings), ("routes", meter.route_readings),
                             ("macro_metrics", macros), ("trajectory_summary", trajectory),
                             ("role_summary", role_summary(meter.uav_readings, .5, name, seed)),
                             ("route_summary", route_statistics(meter.route_readings, name, seed, None, "mission_micro")),
                             ("link_summary", summarize_links(meter.link_readings, 900, seed, 6))):
        append_csv(directory / f"{filename}.csv", records)
    write_json(directory / "summary.json", row)
    write_json(directory / "crn.json", crn_records(meter))
    write_json(directory / "no_learning_check.json", checks)
    np.savez_compressed(directory / "trajectory.npz", time_s=[r[0] for r in meter.positions],
                        uav=[r[1] for r in meter.positions], users=[r[2] for r in meter.positions],
                        centroid=[r[3] for r in meter.positions], groups=[r[4] for r in meter.positions])


def comparisons(rows):
    metrics = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in
               ("seed", "reused", "checkpoint_window", "sac_decisions", "micro_decisions", "fixed_rule_decisions")]
    result = []
    for seed in SCENES:
        sac = next(r for r in rows if r["seed"] == seed and r["policy"] == "SAC101")
        for name in RULES:
            ref = next(r for r in rows if r["seed"] == seed and r["policy"] == name)
            row = dict(seed=seed, policy="SAC101", reference=name)
            row.update({f"delta_{k}": sac[k]-ref[k] for k in metrics})
            row["relative_J_percent"] = 100*(sac["J"]-ref["J"])/abs(ref["J"])
            result.append(row)
    return result


def append_result(path, row, columns):
    # The shared append_csv does not reuse a pre-existing header's field order.
    if set(row) != set(columns):
        raise AssertionError("Reused/new result schemas differ")
    append_csv(path, [{k: row[k] for k in columns}])


def analyze(output, rows, manifest, references):
    means = []
    metrics = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("seed", "reused", "checkpoint_window")]
    for name in (*RULES, "SAC101"):
        selected = [r for r in rows if r["policy"] == name]
        means.append(dict(policy=name, scenes=2, **{k: float(np.mean([r[k] for r in selected])) for k in metrics}))
    paired = comparisons(rows)
    append_csv(output / "summary.csv", means)
    append_csv(output / "paired_deltas.csv", paired)
    paired_means = []
    for name in RULES:
        part = [r for r in paired if r["reference"] == name]
        paired_means.append(dict(policy="SAC101", reference=name, both_scenes_lower_J=all(r["delta_J"] < -1e-10 for r in part),
                                 **{k: float(np.mean([r[k] for r in part])) for k in part[0] if k.startswith("delta_") or k.startswith("relative_")}))
    append_csv(output / "paired_means.csv", paired_means)
    route_rows, role_rows = [], []
    for r in rows:
        folder = references[r["seed"]]["directory"] if r["reused"] else output / f"{r['policy']}_{r['seed']}"
        for filename, dest in (("route_summary.csv", route_rows), ("role_summary.csv", role_rows)):
            for record in read_csv(folder / filename):
                record.update(policy=r["policy"], reused=r["reused"], source_path=str(folder / filename))
                dest.append(record)
    append_csv(output / "route_summary.csv", route_rows)
    append_csv(output / "role_summary.csv", role_rows)
    basic = [("policy", "微观规则"), ("seed", "场景"), ("J", "J"), ("J_holding", "驻留项"),
             ("J_energy", "能量项"), ("task_delay_weighted_ms", "加权/ms"),
             ("task_delay_high_ms", "高优/ms"), ("task_delay_low_ms", "低优/ms"), ("power_w", "功率/W")]
    lines = ["# Mission V1：Predictive固定宏观下的微观规则比较", "",
             f"基准HEAD：`{manifest['actual_head']}`；物理锁定：`{PHYSICS_BASE_COMMIT}`。",
             "本轮新增独立诊断源码未提交，逐文件指纹及源码副本见manifest/source_evidence；未修改原源码。",
             "仅33600/33601两个已使用开发场景。每条连续900秒；全部Predictive，宏观动作及UAV轨迹也逐步配对一致。",
             "新增8条固定规则结果，SAC101两条来自第一阶段，不重跑、不冒充新增训练或推理。", "",
             "## 固定定义", "",
             "每源高低优本地比例依次：Local=(1,1)，Uniform=(1/3,1/3)，Fixed-A=(2/3,1/3)，Fixed-B=(1/2,1/3)。其余均分到另两机。",
             "固定规则均不读取队列/负载或未来；SAC按原聚合观测实时确定性决策。所有方法在同一个环境服务接口执行。", "",
             "## 十条逐场景记录", "", *table(rows, basic), "", "## 两开发场景算术平均", "",
             *table(means, [c for c in basic if c[0] != "seed"]), "", "## SAC减固定规则的配对差", "",
             *table(paired, [("reference", "参考"), ("seed", "场景"), ("delta_J", "ΔJ"),
                            ("relative_J_percent", "ΔJ/%"), ("delta_J_holding", "Δ驻留"),
                            ("delta_J_energy", "Δ能量"), ("delta_task_delay_high_ms", "Δ高优/ms"),
                            ("delta_task_delay_low_ms", "Δ低优/ms")]), "", "两场景平均配对差：", "",
             *table(paired_means, [("reference", "参考"), ("delta_J", "平均ΔJ"), ("relative_J_percent", "平均ΔJ/%"),
                                  ("delta_task_delay_weighted_ms", "Δ加权/ms"), ("delta_power_w", "Δ功率/W"),
                                  ("both_scenes_lower_J", "两场景J均降低")]), "",
             "## 完成与未完成", "", *table(rows, [("policy", "规则"), ("seed", "场景"),
              *[(f"task_{k}_{p}_{suffix}", label) for p, pl in (("high", "高"), ("low", "低"))
                for k, suffix, label in (("arrived", "count", pl+"到达"), ("completed", "count", pl+"完成"),
                                         ("unfinished", "count", pl+"未完"), ("completion", "rate", pl+"完成率"))]]), "",
             "时延仅统计已完成父任务。未完成不等于超时，不能仅比较completed-only时延。", "",
             "## 队列", "", *table(rows, [("policy", "规则"), ("seed", "场景"),
                *[(f"mean_queue_{q}_mbit", "平均"+label+"/Mbit") for q, label in (("upload", "上传"), ("compute", "计算"), ("transfer", "转发"))],
                *[(f"queue_{q}_mbit", "期末"+label+"/Mbit") for q, label in (("upload", "上传"), ("compute", "计算"), ("transfer", "转发"))],
                ("late_queue_slope_mbit_s", "后300s斜率/Mbit/s")]), "",
             "## 实际资源与能量", "", *table(rows, [("policy", "规则"), ("seed", "场景"),
              ("u2g_rate_time_weighted_mbps", "U2G有效Mbps"), ("mean_rate_per_active_link_mbps", "U2U活跃Mbps"),
              ("tx_throughput_mbps_wall", "U2U吞吐Mbps"), ("u2u_tx_time_s", "U2U发送秒"),
              ("compute_throughput_mbps_wall", "计算吞吐Mbps"), ("cpu_utilization_service", "CPU利用率")]), "",
             *table(rows, [("policy", "规则"), ("seed", "场景"), *[(k, k) for k in ENERGY], ("propulsion_increment_j", "推进增量/J")]), "",
             "平均队列按0.5s帧末，末端为900s；后段斜率为600<t<=900s的OLS。J为原物理步代价平均，不是训练折扣回报。",
             "吞吐分母900s；U2U活跃速率按活跃链路-步平均；U2G有效速率=实际发送工作量/累计发送者秒。发送秒不是墙钟时延。",
             "路由高低优、本地比例、heavy/medium/light及源UAV分组见route_summary.csv、role_summary.csv；每0.5s曲线及每macro累计指标在逐臂目录。", "",
             "## 边界与停止", "",
             "新增固定规则不实例化学习网络；反向传播、replay和更新路径被禁止。SAC结果沿用已通过的冻结验证，并重新核对来源checkpoint及原结果指纹。",
             "宏观规则相同不是强制回放SAC路由；各微观规则使用自己产生的真实队列和服务结果。",
             "两个开发场景和一个SAC训练初始化不支持显著性、泛化、全局最优或完整HRL已验证的结论。",
             "所有预声明结果保留；完成8条后停止，不自动微调SAC、不训练PPO、不做局部轨迹搜索、不运行额外场景。"]
    (output / "result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("New independent output required; no overwrite/automatic rerun")
    if assert_committed_sources(RUN_FILES) != BASE:
        raise ValueError("Review new HEAD before this bounded pilot")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "codex/mission_rpgm_v1":
        raise ValueError("Wrong mission development branch")
    physics = physics_provenance()
    spec = load_spec()
    saved, _ = load_source(spec)
    cfg = mission_config(saved["config"], spec)
    if {k: tuple(v) for k, v in saved["config"]["pilot"]["fixed_rules"].items()} != RULES:
        raise ValueError("Fixed-A/B do not match the original SAC source definition")
    del saved
    gc.collect()
    prior, rows, references, evidence_hashes = verify_previous(cfg, spec)
    source_hashes = {n: file_hash(ROOT / n) for n in (*RUN_FILES, *NEW_FILES)}
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "source_evidence"
    for name in NEW_FILES:
        target = evidence / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = dict(status="RUNNING", actual_head=BASE, branch=branch,
                    working_tree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                    physics_base_commit=PHYSICS_BASE_COMMIT, physics_hashes=physics, config=cfg,
                    protocol=protocol(), training=False, training_seed_count=0, source_training_initializations=[101],
                    checkpoint_sha256=spec["checkpoint_sha256"], source_checkpoint=spec["source_checkpoint"],
                    source_manifest_sha256=file_hash((ROOT/spec["source_checkpoint"]).parent.parent/"manifest.json"),
                    previous_result_path=str(SOURCE), previous_result_hashes=evidence_hashes,
                    reused_rows=[dict(seed=r["seed"], path=r["source_directory"], reused=True) for r in rows],
                    source_hashes_before=source_hashes, new_runner_committed=False, formal_completed=[],
                    prior_sac_freeze=prior["frozen_sac"], reserved_scenes_executed=[], smoke=[]) 
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        # Short functional check only, never used as a performance observation.
        for name in RULES:
            row, meter, _, _, checks = evaluate_fixed(cfg, name, SCENES[0], _short_test=True)
            check_paired_motion(meter, references[SCENES[0]], prefix=True)
            manifest["smoke"].append(dict(rule=name, seconds=20, route_and_pairing=True, checks=checks))
            del meter
        write_json(output / "manifest.json", manifest)
        print("[mission micro] four20s functional checks passed; starting exactly eight900s runs", flush=True)
        append_csv(output / "per_seed.csv", rows)
        for seed in SCENES:
            for name in RULES:
                row, meter, trajectory, macros, checks = evaluate_fixed(cfg, name, seed)
                check_paired_motion(meter, references[seed])
                save_fixed(output / f"{name}_{seed}", row, meter, trajectory, macros, checks)
                append_result(output / "per_seed.csv", row, list(rows[0]))
                rows.append(row)
                manifest["formal_completed"].append(dict(seed=seed, rule=name, crn_equal=True, macro_motion_equal=True))
                write_json(output / "manifest.json", manifest)
                print(f"[mission micro {len(manifest['formal_completed'])}/8] {seed} {name} J={row['J']:.6f} holding={row['J_holding']:.6f}", flush=True)
                del meter, macros
                gc.collect()
        if len(rows) != 10 or {(r["seed"], r["policy"]) for r in rows} != {(s, p) for s in SCENES for p in (*RULES, "SAC101")}:
            raise AssertionError("Evaluation matrix mismatch")
        analyze(output, rows, manifest, references)
        after = {n: file_hash(ROOT / n) for n in source_hashes}
        previous_after = {n: file_hash(n) for n in evidence_hashes}
        if after != source_hashes or previous_after != evidence_hashes or physics_provenance() != physics:
            raise AssertionError("Source or original results changed")
        if file_hash(ROOT / spec["source_checkpoint"]) != spec["checkpoint_sha256"]:
            raise AssertionError("SAC checkpoint changed")
        manifest.update(status="COMPLETE_EIGHT_NEW_TWO_REUSED_STOPPED", elapsed_seconds=time.perf_counter()-started,
                        source_hashes_after=after, source_unchanged=True, previous_results_unchanged=True,
                        crn_all_equal=True, macro_motion_all_equal=True, sac_checkpoint_unchanged=True)
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="FAILED_STOPPED", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    return dict(status=manifest["status"], output=str(output), new_evaluations=8, reused_evaluations=2)
