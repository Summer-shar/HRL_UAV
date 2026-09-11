"""Three fixed B-controller development rechecks; no training or new policies."""

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav import target_tracking_check as tracking
from hrl_uav import trajectory_check as base
from hrl_uav.experiment import append_csv, write_json
from hrl_uav.scene_prescreen import assert_crn_equal, source_hashes
from hrl_uav.wireless_check import read_csv


PRIOR = ROOT / "results/target_tracking_sac101"
OUTPUT = ROOT / "results/progressive_centroid_dev_recheck"
EXPLORATION = (33600, 33601)
RECHECK = (33610, 33611, 33612)
B_SOURCE_SHA256 = "415b42fcd5b4852604da9b3220a9445b33096352b24819147d22cc9868f72180"
CRN_KEYS = ("time_s", "task_arrived_count", "arrivals_sha256", "load_sha256",
            "schedule_sha256", "users_microsteps_sha256")
FIELDS = (
    "J", "J_holding", "J_energy", "power_w", "task_delay_weighted_ms",
    "task_delay_high_ms", "task_delay_low_ms", "task_completion_rate",
    "task_arrived_high_count", "task_completed_high_count", "task_unfinished_high_count",
    "task_completion_high_rate", "task_arrived_low_count", "task_completed_low_count",
    "task_unfinished_low_count", "task_completion_low_rate", "unfinished",
    "mean_queue_total_mbit", "mean_queue_upload_mbit", "mean_queue_compute_mbit",
    "mean_queue_transfer_mbit", "queue_total_mbit", "queue_upload_mbit",
    "queue_compute_mbit", "queue_transfer_mbit", "late_queue_slope_mbit_s",
)


def normalize(row, label, cfg, hover_wall_power):
    """Recover only algebraically identifiable fields, with explicit provenance."""
    r = copy.deepcopy(row)
    r["policy"] = label
    r["seed"] = int(r["seed"])
    r["split"] = "exploration" if r["seed"] in EXPLORATION else "development_recheck"
    dt = .1  # Frozen physical step in the checked runtime source.
    energy = cfg["reward"]["energy_weight"] * (r["power_w"] - hover_wall_power) * dt / cfg["reward"]["energy_norm_j"]
    if "J_energy" in r:
        np.testing.assert_allclose(r["J_energy"], energy, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(r["J"], r["J_holding"] + r["J_energy"], atol=1e-10, rtol=1e-12)
        r["J_components_source"] = "recorded integrated components; power identity crosschecked"
    else:
        r.update(J_energy=energy, J_holding=r["J"] - energy,
                 J_components_source="algebraic recovery from recorded full power and frozen hover baseline; holding=J-energy")
    for priority in ("high", "low"):
        completed = r[f"task_completed_{priority}_count"]
        if f"task_unfinished_{priority}_count" not in r:
            r[f"task_unfinished_{priority}_count"] = r[f"task_alive_{priority}_count"]
        alive = r[f"task_unfinished_{priority}_count"]
        arrived = r.get(f"task_arrived_{priority}_count", completed + alive)
        np.testing.assert_allclose(arrived, completed + alive, atol=0, rtol=0)
        r[f"task_arrived_{priority}_count"] = arrived
        r[f"task_completion_{priority}_rate"] = completed / arrived
    arrived = r["task_arrived_high_count"] + r["task_arrived_low_count"]
    if "task_arrived_count" in r:
        assert arrived == r["task_arrived_count"]
    assert r["unfinished"] == r["task_unfinished_high_count"] + r["task_unfinished_low_count"]
    r["mean_queue_total_mbit"] = sum(r[f"mean_queue_{stage}_mbit"] for stage in ("upload", "compute", "transfer"))
    np.testing.assert_allclose(r["queue_total_mbit"], sum(r[f"queue_{stage}_mbit"] for stage in ("upload", "compute", "transfer")), atol=1e-10)
    if not all(np.isfinite(r[k]) for k in FIELDS):
        raise ValueError("Missing or nonfinite common metric")
    return r


def compare_processes(a, b):
    ca, cb = read_csv(a["crn_path"]), read_csv(b["crn_path"])
    if len(ca) != 1800 or len(cb) != 1800:
        raise ValueError("Expected all 1800 resource frames of one continuous 900s run")
    assert_crn_equal([{k: r[k] for k in CRN_KEYS} for r in ca], [{k: r[k] for k in CRN_KEYS} for r in cb])
    with np.load(a["trajectory_path"]) as x, np.load(b["trajectory_path"]) as y:
        for key in ("users", "centroid"):
            np.testing.assert_array_equal(x[key], y[key])
        assert x["uav"].shape == y["uav"].shape == (90, 3, 2)
    return dict(pair=b["policy"] + "-" + a["policy"], seed=b["seed"], frames=1800,
                births=True, load=True, physical_step_users=True, recorded_centroids=True,
                centroid_all_step_basis="same causal arrival/user inputs and unchanged deterministic centroid recurrence; old PPO record lacks per-step centroid digest")


def collect_reuse(cfg, source_saved):
    prior = base.load_json(PRIOR / "manifest.json")
    post = base.load_json(PRIOR / "postprocess_manifest.json")
    if prior["status"] != "COMPLETE" or prior["new_900s_runs"] != 4 or prior["config"] != cfg:
        raise ValueError("Previous B result provenance mismatch")
    if prior["source_digests"] != base.SOURCE_DIGESTS or not all(prior["frozen"].values()):
        raise ValueError("Previous source or frozen-state mismatch")
    assert prior["source_hashes_before"] == prior["source_hashes_after"]
    for name, h in prior["source_hashes_after"].items():
        if base.file_hash(ROOT / name) != post["postrun_diagnostic_source_hashes"].get(name, h):
            raise ValueError(f"Previous implementation changed: {name}")
    if base.file_hash(ROOT / "hrl_uav/target_tracking_check.py") != B_SOURCE_SHA256:
        raise ValueError("B controller is not the already validated implementation")
    hashes = dict(prior["reuse_file_hashes"])
    for name, h in hashes.items():
        if base.file_hash(ROOT / name) != h:
            raise ValueError(f"Previous reused file changed: {name}")

    def remember(path):
        hashes[str(Path(path).relative_to(ROOT))] = base.file_hash(path)

    for name in ("manifest.json", "postprocess_manifest.json", "protocol.json", "per_seed.csv", "smoke/report.json"):
        remember(PRIOR / name)
    old = tracking.numeric_rows(PRIOR / "per_seed.csv")
    a_old = [r for r in old if r["policy"] == "A"]
    hover_values = [r["energy_hover_j"] / (900 * .9) for r in a_old]
    np.testing.assert_allclose(hover_values, hover_values[0], atol=1e-10, rtol=1e-12)
    hover = float(hover_values[0])
    # Independent scalar evaluation of the unchanged zero-speed power formula.
    np.testing.assert_allclose(hover, 3 * (78 + 22 * 265 ** .25 + 28), atol=1e-10)

    hist_manifest = base.load_json(base.HISTORY / "manifest.json")
    hist = base.load_json(base.HISTORY / "report.json")
    for name in ("manifest.json", "report.json"):
        remember(base.HISTORY / name)
    if hist["ppo_windows"] != 120 or hist["source_sac_window"] != 2000 or hist["source_seed"] != 101:
        raise ValueError("Historical PPO is not the fixed window120 endpoint")
    if hist["reserved_scenes_executed"] or not all(hist["sac_frozen_checks"].values()):
        raise ValueError("Historical PPO audit inconsistent")
    endpoint = Path(hist_manifest["final_checkpoint"])
    if base.file_hash(endpoint) != hist_manifest["final_checkpoint_sha256"]:
        raise ValueError("Historical PPO endpoint hash mismatch")
    remember(endpoint)
    saved = torch.load(endpoint, map_location="cpu", weights_only=False)
    for k in ("sac_actor", "sac_critic", "sac_target", "log_alpha", "observations"):
        if not base.state_equal(source_saved[k], saved[k]):
            raise ValueError(f"Historical PPO SAC component differs: {k}")
    if not base.state_equal(source_saved["optimizers"][:3], saved["optimizers"][:3]):
        raise ValueError("Historical SAC optimizers differ")
    if saved["config"] != hist_manifest["config"]:
        raise ValueError("PPO checkpoint configuration differs from manifest")
    normalized = copy.deepcopy(saved["config"])
    normalized["training"]["joint_windows"] = 0
    if normalized != cfg:
        raise ValueError("PPO/source configuration differs beyond historical macro budget")
    del saved

    validation = hist["validation_history"][-1]
    development = hist["development_final"]
    rows, checks = [], []
    for group, seeds in ((validation, EXPLORATION), (development, RECHECK)):
        if group["window"] != 120 or group["evaluation_macros"] != 90 or not group["learning_state_unchanged"] or not group["rng_unchanged"]:
            raise ValueError("Historical endpoint evaluation scope mismatch")
        for old_label, label in (("centroid", "A"), ("ppo", "PPO120")):
            if tuple(r["seed"] for r in group[old_label]) != seeds:
                raise ValueError("Historical seed list mismatch")
            for raw in group[old_label]:
                if raw["time_s"] != 900:
                    raise ValueError("Historical evaluation is not 900s")
                prefix = base.HISTORY / f"{raw['policy']}_{raw['seed']}"
                paths = {kind: str(prefix) + suffix for kind, suffix in (
                    ("crn_path", "_crn.csv"), ("trajectory_path", "_trajectory.npz"),
                    ("queues_path", "_queues.csv"), ("uavs_path", "_uavs.csv"))}
                for path in paths.values():
                    remember(Path(path))
                queue_rows = tracking.numeric_rows(paths["queues_path"])
                np.testing.assert_array_equal([r["time_s"] for r in queue_rows], np.arange(10, 901, 10))
                for k in ("J", "power_w", "queue_total_mbit", "unfinished", "task_delay_weighted_ms"):
                    np.testing.assert_allclose(raw[k], queue_rows[-1][k], atol=1e-10, rtol=1e-12)
                r = normalize(raw, label, cfg, hover)
                r.update(reused=True, source_path=str(base.HISTORY / "report.json"), **paths)
                if label == "A" and r["seed"] in EXPLORATION:
                    previous = next(x for x in a_old if int(x["seed"]) == r["seed"])
                    for k in FIELDS:
                        np.testing.assert_allclose(r[k], previous[k], atol=1e-9, rtol=1e-12, err_msg=k)
                    with np.load(paths["trajectory_path"]) as x, np.load(Path(previous["source_path"]) / "trajectory.npz") as y:
                        for k in ("users", "centroid", "uav", "executed_route"):
                            np.testing.assert_array_equal(x[k], y[k])
                rows.append(r)
    for seed in EXPLORATION:
        previous = next(r for r in old if r["policy"] == "B" and int(r["seed"]) == seed)
        d = Path(previous["source_path"])
        for f in d.iterdir():
            if f.is_file():
                remember(f)
        audit = base.load_json(d / "audit.json")
        if audit["duration_s"] != 900 or audit["reset_count"] != 1 or not all(audit["learning_unchanged"].values()):
            raise ValueError("Previous B evaluation audit mismatch")
        r = normalize(previous, "B", cfg, hover)
        r.update(reused=True, crn_path=str(d / "crn.csv"), trajectory_path=str(d / "trajectory.npz"))
        rows.append(r)
    for seed in EXPLORATION + RECHECK:
        a = next(r for r in rows if r["policy"] == "A" and r["seed"] == seed)
        p = next(r for r in rows if r["policy"] == "PPO120" and r["seed"] == seed)
        checks.append(compare_processes(a, p))
        if seed in EXPLORATION:
            checks.append(compare_processes(a, next(r for r in rows if r["policy"] == "B" and r["seed"] == seed)))
    return rows, hashes, checks, hover


def evaluate_b(cfg, sac, seed, output):
    if type(seed) is not int or seed not in RECHECK:
        raise ValueError("Only B on 33610/33611/33612 is authorized")
    # Only the diagnostic admission list changes in this process. Controller code is untouched.
    with patch.object(tracking, "SEEDS", RECHECK):
        return tracking.evaluate(cfg, sac, "zero", .5, seed, 90, output, "B")


def aggregate(rows, seeds, split):
    result = []
    for label in ("A", "B", "PPO120"):
        selected = [r for r in rows if r["policy"] == label and r["seed"] in seeds]
        if sorted(r["seed"] for r in selected) != sorted(seeds):
            raise ValueError("Missing or duplicate comparison record")
        result.append(dict(policy=label, split=split, scene_count=len(seeds),
                           **{k: float(np.mean([r[k] for r in selected])) for k in FIELDS}))
    return result


def make_report(output, summary, pairs):
    lines = ["# 渐进质心0.5：固定控制器开发复核", "",
             "只新增B在33610/33611/33612各900秒；原探索33600/33601单列。A和PPO120全部复用已核验的同SAC记录。",
             "B的目标、系数、动作及原start_macro执行不变；只在独立入口临时设置本次许可场景列表。没有改变PPO零动作含义。",
             "全部为已使用开发场景，不是未见泛化测试。各场景等权平均，未按已完成任务数重新加权。"]

    def table(title, rows, keys, labels):
        lines.extend(["", "## " + title, "", "| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"])
        for r in rows:
            lines.append("| " + " | ".join(f"{r[k]:.6f}" if isinstance(r[k], (float, np.floating)) else str(r[k]) for k in keys) + " |")

    table("B-A逐场景差", pairs, ["split", "seed", "delta_J", "delta_J_holding", "delta_J_energy", "delta_power_w", "delta_task_delay_weighted_ms", "delta_task_delay_high_ms", "delta_task_delay_low_ms"],
          ["组别", "场景", "ΔJ", "ΔJ驻留", "ΔJ能量", "Δ功率W", "Δ加权ms", "Δ高优ms", "Δ低优ms"])
    for split in ("exploration", "development_recheck", "all_development"):
        table(split + "同口径均值", [r for r in summary if r["split"] == split],
              ["policy", "scene_count", "J", "J_holding", "J_energy", "power_w", "task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms"],
              ["控制器", "场景数", "J", "J驻留", "J能量", "功率W", "加权ms", "高优ms", "低优ms"])
    table("完成与未完成（比例0至1，计数为场景均值）", summary,
          ["split", "policy", "task_completion_high_rate", "task_completion_low_rate", "task_unfinished_high_count", "task_unfinished_low_count"],
          ["组别", "控制器", "高优完成率", "低优完成率", "高优未完成", "低优未完成"])
    table("积压（Mbit，平均为0.5秒帧末采样，期末为900秒）", summary,
          ["split", "policy", "mean_queue_upload_mbit", "mean_queue_compute_mbit", "mean_queue_transfer_mbit", "queue_upload_mbit", "queue_compute_mbit", "queue_transfer_mbit"],
          ["组别", "控制器", "平均上传", "平均计算", "平均转发", "期末上传", "期末计算", "期末转发"])
    lines += ["", "## 口径与来源", "",
              "A/PPO120来自同一个固定window120终点的既有成对评价。有效配置除历史PPO训练预算外与SAC源配置一致；本次重新比对PPO终点中的SAC网络、温度参数和三套优化器与SAC101源文件逐项相同。未运行PPO推理或训练。",
              "A/PPO120没有直接记录J分项。按原模型全悬停基线功率P_hover还原：J_E=0.1*(power_w-P_hover)*0.1/14，J_H=J-J_E。原J本身完全复用；分解是代数还原，不是新奖励。P_hover由旧悬停账和原零速公式双重核对；在原两场景A、B的直接累计分项上交叉验证。B三次新评价保存直接积分分项。",
              "旧高低优到达数以已完成数+仍存活未完成数补齐，校验总到达数守恒；完成率以各优先级到达数为分母。不是把未完成任务称为超时。所有时间平均积压沿用原统计器，期末积压单列。",
              "外生核验比较1800帧累计逐任务出生、负载调度及物理步用户运动哈希，并比较已记录的90个时刻用户和业务质心坐标。旧PPO记录不含逐物理步质心哈希；逐步质心的一致性另由相同到达、用户输入及未改的因果质心递推公式保证，不冒称旧记录含此哈希。",
              "没有参数扫描、额外控制器或额外900秒评价；没有33700-33709、学习更新、replay写入/采样。原文件哈希前后保留。新tracking.csv记录实际目标、可行投影、位移、速度和部署能量。",
              "完整逐场景指标见per_seed.csv；B-A差值见paired_deltas.csv及paired_summary.csv；探索与复核分别见summary.csv。"]
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite or silently rerun an existing recheck")
    if base.git("rev-parse", "HEAD") != base.REVIEWED_HEAD or base.git("diff", base.REVIEWED_HEAD, "--", "hrl_uav", "scripts", "configs"):
        raise ValueError("Review runtime differences before evaluating")
    cfg, saved, _ = base.source_bundle()
    rows, hashes, checks, hover = collect_reuse(cfg, saved)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    protocol = dict(controller="B", target="original current centroid", coefficient=.5,
                    seeds=RECHECK, macros=90, duration_s=900, new_evaluations=3,
                    deterministic=True, train=False, complete_tail=False, replay=False,
                    exploration_separate=EXPLORATION, forbidden_scenes=list(range(33700, 33710)))
    write_json(OUTPUT / "protocol.json", protocol)
    manifest = dict(status="PREFLIGHT_PASSED", actual_head=base.git("rev-parse", "HEAD"),
                    branch=base.git("branch", "--show-current"), worktree_before=base.git("status", "--short"),
                    config=cfg, source_checkpoint=str(base.DEFAULT_SOURCE), source_digests=base.SOURCE_DIGESTS,
                    B_controller_sha256=B_SOURCE_SHA256, source_hashes_before=before,
                    reused_file_hashes=hashes, reused_records=len(rows), new_900s_runs=0,
                    hover_baseline_power_w=hover, training=False, replay_writes=0, replay_reads=0,
                    ppo_updates=0, sac_update_increment=0, reserved_scenes_run=[])
    write_json(OUTPUT / "manifest.json", manifest)
    write_json(OUTPUT / "reuse_checks.json", checks)
    write_json(OUTPUT / "reused_records.json", rows)
    print("Preflight passed: 12 records reused; only three B evaluations will run", flush=True)
    sac = base.ReadOnlySAC(saved)
    del saved
    projections = []
    for seed in RECHECK:
        if source_hashes() != before:
            raise ValueError("Source changed during recheck")
        directory = OUTPUT / "evaluations" / "B" / str(seed)
        raw, meter = evaluate_b(cfg, sac, seed, directory)
        r = normalize(raw, "B", cfg, hover)
        r.update(reused=False, source_path=str(directory), crn_path=str(directory / "crn.csv"),
                 trajectory_path=str(directory / "trajectory.npz"))
        a = next(x for x in rows if x["policy"] == "A" and x["seed"] == seed)
        checks.append(compare_processes(a, r))
        rows.append(r)
        projections.append(dict(seed=seed, deployments=len(meter.tracking),
                                feasible_projection_count=sum(t["projected"] for t in meter.tracking),
                                max_displacement_m=max(t["distance_m"] for t in meter.tracking),
                                max_speed_mps=max(t["speed_mps"] for t in meter.tracking)))
        append_csv(OUTPUT / "new_per_seed.csv", [{k: r[k] for k in ("policy", "seed", "split", *FIELDS)}])
        manifest["new_900s_runs"] += 1
        write_json(OUTPUT / "manifest.json", manifest)
        print(f"Completed B {seed}: J={r['J']:.9f}, B-A={r['J']-a['J']:+.9f}", flush=True)
    summaries = []
    for seeds, split in ((EXPLORATION, "exploration"), (RECHECK, "development_recheck"),
                         (EXPLORATION + RECHECK, "all_development")):
        summaries.extend(aggregate(rows, seeds, split))
    pairs = []
    for seed in EXPLORATION + RECHECK:
        a = next(r for r in rows if r["policy"] == "A" and r["seed"] == seed)
        b = next(r for r in rows if r["policy"] == "B" and r["seed"] == seed)
        pairs.append(dict(pair="B-A", seed=seed, split=b["split"], **{"delta_"+k: b[k]-a[k] for k in FIELDS}))
    pair_means = []
    for split in ("exploration", "development_recheck", "all_development"):
        ps = [r for r in pairs if split == "all_development" or r["split"] == split]
        pair_means.append(dict(pair="B-A", split=split, scene_count=len(ps), **{"delta_"+k: float(np.mean([r["delta_"+k] for r in ps])) for k in FIELDS}))
    append_csv(OUTPUT / "per_seed.csv", [{k: r[k] for k in ("policy", "seed", "split", "reused", "source_path", "J_components_source", *FIELDS)} for r in sorted(rows, key=lambda r: (r["policy"], r["seed"]))])
    append_csv(OUTPUT / "summary.csv", summaries)
    append_csv(OUTPUT / "paired_deltas.csv", pairs)
    append_csv(OUTPUT / "paired_summary.csv", pair_means)
    append_csv(OUTPUT / "projection_summary.csv", projections)
    write_json(OUTPUT / "pairing_checks.json", checks)
    after = source_hashes()
    assert after == before
    assert all(base.file_hash(ROOT / p) == h for p, h in hashes.items())
    manifest.update(status="COMPLETE", source_hashes_after=after, frozen=sac.assert_unchanged(),
                    reused_files_unchanged=True, protocol_sha256=base.file_hash(OUTPUT / "protocol.json"))
    write_json(OUTPUT / "manifest.json", manifest)
    make_report(OUTPUT, summaries, pairs)
    print("COMPLETE: three evaluations; stopped", flush=True)


if __name__ == "__main__":
    run()
