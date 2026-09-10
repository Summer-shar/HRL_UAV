"""Predeclared fixed priority routes on the frozen candidate wireless model."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from .bandwidth import UniformRouting, summarize_meter
from .experiment import append_csv, evaluate, isolated_rng, write_json
from .scene_prescreen import LocalRouting, ROOT, SEEDS, assert_crn_equal, source_hashes
from .wireless_check import AuditMeter, ENERGY, MEASURES, PSD, PRIORITY_MEASURES, fixed_rollout, read_csv


LEVELS = (("1of3", 1/3), ("1of2", 1/2), ("2of3", 2/3), ("1", 1.0))
SOURCE = ROOT / "results/wireless_model_fixed_rule_check"
BASE_HEAD = "d7a4e8d0ef0cfa7da83e7af09844f65070b443e8"
SUFFIXES = ("queues.csv", "service.csv", "crn.csv", "trajectory.npz", "uavs.csv", "links.csv")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_csv(path):
    rows = read_csv(path)
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


class PriorityRouting(UniformRouting):
    def __init__(self, alpha_high, alpha_low):
        super().__init__()
        if alpha_high not in dict(LEVELS).values() or alpha_low not in dict(LEVELS).values():
            raise ValueError("Only the four predeclared fractions are allowed")
        self.alphas = (alpha_high, alpha_low)
        self.expected = np.stack([np.full((3, 3), (1-a)/2) for a in self.alphas])
        for priority, alpha in enumerate(self.alphas):
            np.fill_diagonal(self.expected[priority], alpha)

    def choose_action(self, state, deterministic=False):
        if self.alphas == (1/3, 1/3):
            return UniformRouting.choose_action(self, state, deterministic)
        if self.alphas == (1, 1):
            return LocalRouting.choose_action(self, state, deterministic)
        # A simplex point is a valid bounded score and is unchanged by projection.
        return self.expected.ravel().copy()


def conditions(base):
    arms = []
    for bandwidth in (3, 6):
        for h_name, high in LEVELS:
            for l_name, low in LEVELS:
                if bandwidth == 3 and high == low == 1:
                    continue
                config = copy.deepcopy(base)
                config["environment"]["u2u_bandwidth_mhz"] = bandwidth
                source_policy = None
                if high == low == 1:
                    source_policy = "w1_local_b6"
                elif high == low == 1/3:
                    source_policy = f"w1_uniform_b{bandwidth}"
                arms.append(dict(policy=f"h{h_name}_l{l_name}_b{bandwidth}",
                                 alpha_high=high, alpha_low=low,
                                 high_fraction=h_name, low_fraction=l_name,
                                 bandwidth_mhz=bandwidth, config=config,
                                 source_policy=source_policy))
    return arms


def verify_source(source=SOURCE):
    source = Path(source).resolve()
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    current = source_hashes()
    changed = [name for name, digest in manifest["source_hashes"].items() if current.get(name) != digest]
    if changed:
        raise ValueError(f"Source no longer matches previous run: {changed}")
    if (manifest["base_head"] != BASE_HEAD or manifest["status"] != "FIXED_RULE_CHECK_COMPLETE"
            or manifest["seeds"] != SEEDS or manifest["duration_s"] != 900
            or manifest["train"] or manifest["training"] or manifest["complete_tail"]
            or manifest["macro_enabled"] or not manifest["source_unchanged"] or not manifest["crn_equal"]):
        raise ValueError("Previous evaluation protocol does not match")
    base = manifest["arm_configs"]["w1_uniform_b6"]
    env = base["environment"]
    if (env["wireless_model"] != "psd_per_uav_budget" or not env["compute_during_deployment"]
            or not env["pause_service_during_flight"] or any(env[k] != v for k, v in PSD.items())
            or base["training"]["window_macros"] != 30):
        raise ValueError("Candidate configuration mismatch")
    lookup = {(r["policy"], int(r["seed"])): r for r in numeric_csv(source/"per_seed.csv")}
    report = json.loads((source/"report.json").read_text(encoding="utf-8"))
    recorded = {(r["policy"], r["seed"]): r for r in report["rows"]}
    origins = {}
    for arm in conditions(base):
        label = arm["source_policy"]
        if label is None:
            continue
        if arm["config"] != manifest["arm_configs"][label]:
            raise ValueError(f"Reuse configuration mismatch: {label}")
        for seed in SEEDS:
            row = lookup[label, seed]
            if row["time_s"] != 900 or row["wireless_model"] != "psd_per_uav_budget":
                raise ValueError("Incomplete reused row")
            for key in MEASURES:
                if row[key] != recorded[label, seed][key]:
                    raise ValueError(f"CSV/report mismatch: {label}/{seed}/{key}")
            files = {suffix: str(source/f"{label}_{seed}_{suffix}") for suffix in SUFFIXES}
            if len(read_csv(files["service.csv"])) != 1800 or len(read_csv(files["queues.csv"])) != 90:
                raise ValueError("Reused time series is incomplete")
            origins[f"{arm['policy']}:{seed}"] = dict(
                source_policy=label, source_per_seed=str(source/"per_seed.csv"),
                files=files, hashes={suffix: file_hash(path) for suffix, path in files.items()})
    # Hash all files in the specifically authorized prior result directory, not other archives.
    protected = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
    return manifest, lookup, origins, protected


def check_crn(reference, other):
    # CSV values are typed here so a newly collected frame can be compared exactly.
    assert_crn_equal(reference, other)


def check_trajectory(path, reference, expected):
    with np.load(path) as current, np.load(reference) as baseline:
        for key in ("uav", "users", "centroid", "macro_action"):
            np.testing.assert_array_equal(current[key], baseline[key])
        np.testing.assert_allclose(current["executed_route"],
                                   np.broadcast_to(expected, current["executed_route"].shape),
                                   rtol=1e-12, atol=1e-12)


def window_metrics(service, macro_rows, policy, seed):
    """Recover 300 s summaries from existing cumulative counters, without resets."""
    result, previous = [], None
    duration = float(service[-1]["time_s"])
    ends = list(np.arange(300, duration+1e-6, 300)) or [duration]
    for end in ends:
        start = 0.0 if previous is None else float(previous["time_s"])
        current = next(r for r in macro_rows if abs(r["time_s"]-end) < 1e-7)
        sampled = [r for r in service if start < r["time_s"] <= end]
        if len(sampled) != round((end-start)/0.5):
            raise ValueError("Window is missing 0.5 s samples")
        out = dict(policy=policy, seed=seed, start_s=start, end_s=float(end),
                   J=(current["J"]*end - (previous["J"]*start if previous else 0))/(end-start),
                   queue_total_mbit=current["queue_total_mbit"],
                   **summarize_meter(sampled, end-start))
        out.update({k: sum(r[k] for r in sampled) for k in ENERGY})
        out["power_w"] = sum(out[k] for k in ENERGY)/(end-start)
        np.testing.assert_allclose(out["power_w"],
            (current["power_w"]*end - (previous["power_w"]*start if previous else 0))/(end-start),
            rtol=1e-12, atol=1e-7)
        delay_sums, completed_counts = {}, {}
        for priority in ("high", "low"):
            count_key = f"task_completed_{priority}_count"
            delay_key = f"task_delay_{priority}_ms"
            completed = int(current[count_key] - (previous[count_key] if previous else 0))
            # These are exact sufficient aggregates up to CSV floating-point rounding.
            delay_sum = current[count_key]*current[delay_key]
            delay_sum -= previous[count_key]*previous[delay_key] if previous else 0
            completed_counts[priority], delay_sums[priority] = completed, delay_sum
            out[count_key] = completed
            out[delay_key] = delay_sum/completed if completed else None
            arrived_key = f"task_arrived_{priority}_count"
            before = next((r[arrived_key] for r in service if r["time_s"] == start), 0)
            out[arrived_key] = int(sampled[-1][arrived_key]-before)
            out[f"task_unfinished_{priority}_count"] = int(sampled[-1][f"task_unfinished_{priority}_count"])
            out[f"cumulative_completion_{priority}_rate"] = sampled[-1][f"task_completion_{priority}_rate"]
        n = 10*completed_counts["high"] + completed_counts["low"]
        out["task_delay_weighted_ms"] = (10*delay_sums["high"]+delay_sums["low"])/n if n else None
        result.append(out)
        previous = current
    return result


def run_arm(arm, seed, macros, output, reference_crn, reference_trajectory):
    meters = []

    def factory(config):
        meter = AuditMeter(config)
        meter.check_local = arm["alpha_high"] == arm["alpha_low"] == 1
        meters.append(meter)
        return meter

    agent = PriorityRouting(arm["alpha_high"], arm["alpha_low"])
    with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", factory), \
         patch("hrl_uav.experiment.rollout", fixed_rollout), \
         patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("No learning networks")):
        row = evaluate(arm["config"], agent, [seed], macros, output, arm["policy"], macro_enabled=False)[0]
    meter, env = meters[0], meters[0].env
    if (meter.reset_count != 1 or row["time_s"] != macros*10 or len(meter.readings) != macros*20
            or env.wireless_model != "psd_per_uav_budget" or not env.compute_during_deployment
            or not env.pause_service_during_flight or env.B_u2u_total/1e6 != arm["bandwidth_mhz"]):
        raise AssertionError("Effective configuration/time/reset mismatch")
    check_crn(reference_crn, meter.crn_frames)
    paths = {suffix: str(output/f"{arm['policy']}_{seed}_{suffix}") for suffix in SUFFIXES}
    check_trajectory(paths["trajectory.npz"], reference_trajectory, agent.expected)
    row.update(summarize_meter(meter.readings, row["time_s"]))
    row.update({k: sum(r[k] for r in meter.readings) for k in ENERGY})
    row.update({k: meter.readings[-1][k] for k in PRIORITY_MEASURES})
    np.testing.assert_allclose(sum(row[k] for k in ENERGY), row["power_w"]*row["time_s"], rtol=1e-12, atol=1e-6)
    for suffix, data in (("service.csv", meter.readings), ("crn.csv", meter.crn_frames),
                         ("uavs.csv", meter.uav_readings), ("links.csv", meter.link_readings)):
        append_csv(paths[suffix], data)
    windows = window_metrics(meter.readings, numeric_csv(paths["queues.csv"]), arm["policy"], seed)
    error = max(max(abs(r["balance_error"]) for r in meter.link_readings),
                max(abs(r["compute_balance_error"]) for r in meter.uav_readings))
    return row, windows, paths, error


def local_alias(arm, bandwidth):
    return arm["bandwidth_mhz"] == bandwidth or arm["alpha_high"] == arm["alpha_low"] == 1


def summaries(rows, arms):
    means = []
    for bandwidth in (3, 6):
        group = []
        for arm in arms:
            if not local_alias(arm, bandwidth):
                continue
            selected = [r for r in rows if r["policy"] == arm["policy"]]
            group.append(dict(policy=arm["policy"], comparison_bandwidth_mhz=bandwidth,
                              evaluated_bandwidth_mhz=arm["bandwidth_mhz"],
                              local_bandwidth_alias=bandwidth != arm["bandwidth_mhz"],
                              high_fraction=arm["high_fraction"], low_fraction=arm["low_fraction"],
                              alpha_high=arm["alpha_high"], alpha_low=arm["alpha_low"], scene_count=len(selected),
                              **{key: float(np.mean([r[key] for r in selected])) for key in MEASURES}))
        for rank, row in enumerate(sorted(group, key=lambda r: r["J"]), 1):
            means.append(dict(rank=rank, **row))
    return means


def paired_deltas(rows, arms):
    lookup = {(r["policy"], r["seed"]): r for r in rows}
    comparisons = set()
    for bandwidth in (3, 6):
        uniform = f"h1of3_l1of3_b{bandwidth}"
        for arm in arms:
            if local_alias(arm, bandwidth):
                comparisons.add((arm["policy"], uniform, bandwidth, "versus_uniform"))
        for h_name, _ in LEVELS:
            for l_name, _ in LEVELS:
                if h_name != l_name:
                    a = f"h{h_name}_l{l_name}_b{bandwidth}"
                    for equal in (h_name, l_name):
                        b = f"h{equal}_l{equal}_b{6 if equal == '1' else bandwidth}"
                        comparisons.add((a, b, bandwidth, "versus_equal_priority_fraction"))
    for h_name, _ in LEVELS:
        for l_name, _ in LEVELS:
            if h_name == l_name == "1":
                continue
            comparisons.add((f"h{h_name}_l{l_name}_b6", f"h{h_name}_l{l_name}_b3", 6, "bandwidth_6_minus_3"))
    return [dict(treatment=a, control=b, comparison_bandwidth_mhz=bandwidth, comparison=kind,
                 seed=seed, **{key: lookup[a, seed][key]-lookup[b, seed][key] for key in MEASURES})
            for a, b, bandwidth, kind in sorted(comparisons) for seed in SEEDS]


def write_report(output, means, rows, windows, arms):
    lines = ["# 高低优固定路由开发集诊断", "",
             "候选无线模式；部署期CPU开启、通信暂停；动态质心。31个不同条件、93条场景记录，其中9条复用、84条本轮运行。"
             "33500/33501/33502是复用开发场景；每条连续900秒，不训练、不加载权重。",
             "所有比例预先固定，alpha表示本地份额，其余两目的各分(1-alpha)/2。以下排序仅按三个场景平均物理代价J，"
             "不是训练折扣回报或soft策略目标的严格等价最优解。Local只运行6MHz一次，3MHz表引用同一记录。", "",
             "## 1. 相对Uniform的变化与代价", ""]
    by_band = {b: [r for r in means if r["comparison_bandwidth_mhz"] == b] for b in (3, 6)}
    for bandwidth, group in by_band.items():
        baseline = next(r for r in group if r["alpha_high"] == r["alpha_low"] == 1/3)
        better = [r for r in group if r["J"] < baseline["J"]]
        lines += [f"### {bandwidth} MHz", "",
                  f"16条规则中{len(better)}条平均J低于Uniform；改善项和代价并列如下（场景指标算术平均）。",
                  "|J排名|高优本地|低优本地|J|相对Uniform的J差|加权ms|高优ms|低优ms|高优完成率%|低优完成率%|高/低未完成均数|期末Mbit|功率W|",
                  "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for r in group:
            h, l = r["high_fraction"].replace("of", "/"), r["low_fraction"].replace("of", "/")
            lines.append(f"|{r['rank']}|{h}|{l}|{r['J']:.6f}|{r['J']-baseline['J']:+.6f}|"
                f"{r['task_delay_weighted_ms']:.3f}|{r['task_delay_high_ms']:.3f}|{r['task_delay_low_ms']:.3f}|"
                f"{100*r['task_completion_high_rate']:.4f}|{100*r['task_completion_low_rate']:.4f}|"
                f"{r['task_unfinished_high_count']:.2f}/{r['task_unfinished_low_count']:.2f}|"
                f"{r['queue_total_mbit']:.3f}|{r['power_w']:.3f}|")
    lines += ["", "## 2. 不同优先级比例的额外收益", ""]
    for b, group in by_band.items():
        equal = min((r for r in group if r["alpha_high"] == r["alpha_low"]), key=lambda r: r["J"])
        different = min((r for r in group if r["alpha_high"] != r["alpha_low"]), key=lambda r: r["J"])
        lines.append(f"- {b}MHz：所测同份额规则中J较低者为{equal['high_fraction']}/{equal['low_fraction']}"
                     f"（J={equal['J']:.6f}）；异份额规则中J较低者为{different['high_fraction']}/{different['low_fraction']}"
                     f"（J={different['J']:.6f}），后者减前者为{different['J']-equal['J']:+.6f}。"
                     "这是开发集候选比较，不是事后新增通过门槛。")
    lines += ["", "## 3. 同一规则的跨带宽表现", "",
              "|高优本地|低优本地|3MHz J排名|6MHz J排名|3MHz J|6MHz J|",
              "|---|---|---:|---:|---:|---:|"]
    for h, _ in LEVELS:
        for l, _ in LEVELS:
            selected = [next(r for r in by_band[b] if r["high_fraction"] == h and r["low_fraction"] == l) for b in (3, 6)]
            a, b = selected
            lines.append(f"|{h.replace('of','/')}|{l.replace('of','/')}|{a['rank']}|{b['rank']}|{a['J']:.6f}|{b['J']:.6f}|")
    lines += ["", "## 4. 逐场景和300秒窗口", "",
              "下表给每个带宽平均J排名前两条候选和Uniform的全部场景、全部窗口；完整31条件的逐场景并列表见"
              "`per_scene_comparison.md`与`per_seed.csv`，全部窗口见`window_metrics.csv`。33502另完整列于`scene33502.md`。",
              "窗口时延按该窗完成任务的完整出生至完成时间计，可能包含更早窗口出生的任务；不表示该窗出生任务的完成时延。"
              "完成率列为截至窗口末累计完成/累计到达，避免把当窗完成数/当窗到达数误称为队列系统的完成率。",
              "|带宽|规则|场景|窗口s|窗口J|窗内完成高/低|高/低时延ms|累计完成率高/低%|期末未完成高/低|平均计算/转发Mbit|期末总Mbit|",
              "|---:|---|---:|---|---:|---|---|---|---|---|---:|"]
    for b, group in by_band.items():
        names = list(dict.fromkeys([r["policy"] for r in group[:2]]+[f"h1of3_l1of3_b{b}"]))
        for name in names:
            for w in (r for r in windows if r["policy"] == name):
                lines.append(f"|{b}|{name}|{w['seed']}|{w['start_s']:g}-{w['end_s']:g}|{w['J']:.6f}|"
                    f"{w['task_completed_high_count']}/{w['task_completed_low_count']}|"
                    f"{w['task_delay_high_ms']:.2f}/{w['task_delay_low_ms']:.2f}|"
                    f"{100*w['cumulative_completion_high_rate']:.4f}/{100*w['cumulative_completion_low_rate']:.4f}|"
                    f"{w['task_unfinished_high_count']}/{w['task_unfinished_low_count']}|"
                    f"{w['mean_queue_compute_mbit']:.3f}/{w['mean_queue_transfer_mbit']:.3f}|{w['queue_total_mbit']:.3f}|")
    lines += ["", "## 5. 未解决问题与口径", "",
              "- 只称所测规则中的开发集较好候选；不是最优策略，不作显著性或长期稳定主张。"
              "两个带宽的排序变化不能证明必须使用SAC，本轮没有运行PPO/SAC。",
              "- 有限窗口的未完成任务不一律称为超时；时延只覆盖已完成父任务，应并读高低优计数和积压。"
              "未额外设置QoS硬约束或事后筛选门槛。",
              "- 固定规则共享所有源的比例，没有测试状态反馈或源间不同份额；不从本次有限网格推导任何方法均不可行。",
              "- J沿用原每物理步代价的平均值；窗口J由累计J乘物理步数后作差（固定时间步下等价于乘时长作差）。"
              "能量分项为900秒或指定窗口实计总量，完整功率为能量/墙钟时长。",
              "- 窗口完成时延由原累计完成数乘累计均值还原延时和、跨窗口作差；仅有原CSV浮点舍入误差，"
              "不假造逐任务时间戳。新旧复用数据用同一计算方式。",
              "- 平均积压是0.5秒帧末采样，包含部署；期末积压是该窗口末时刻。CPU利用率以可计算物理步为分母；"
              "活跃链路数按通信服务物理步平均，链路速率按活跃链路物理步平均，吞吐除完整墙钟时长，不是固定容量。",
              "- 9条旧记录在per_seed标记reused，来源路径、配置与文件SHA-256见manifest.record_provenance。"
              "旧文件未复制改写；时间过程文件可沿该来源直接读取。",
              "- 本轮结束后停止，不细化网格、不调奖励、不改模型、不训练、不修改论文或历史成绩。", ""]
    (output/"result.md").write_text("\n".join(lines), encoding="utf-8")
    for name, selected in (("per_scene_comparison.md", rows), ("scene33502.md", [r for r in rows if r["seed"] == 33502])):
        detail = ["# 逐场景完整指标", "", "时延为已完成任务均值；规则按比例标签列示，不按该场景时延挑选。",
                  "|规则|场景|复用|J|高/低到达数|高/低完成数|高/低未完成数|高/低完成率%|加权/高/低ms|平均上传/计算/转发Mbit|期末Mbit|功率W|",
                  "|---|---:|---|---:|---|---|---|---|---|---|---:|---:|"]
        for r in selected:
            detail.append(f"|{r['policy']}|{r['seed']}|{r['reused']}|{r['J']:.6f}|"
                + "|".join(f"{int(r[f'task_{k}_high_count'])}/{int(r[f'task_{k}_low_count'])}" for k in ("arrived","completed","unfinished"))
                + f"|{100*r['task_completion_high_rate']:.4f}/{100*r['task_completion_low_rate']:.4f}|"
                f"{r['task_delay_weighted_ms']:.3f}/{r['task_delay_high_ms']:.3f}/{r['task_delay_low_ms']:.3f}|"
                f"{r['mean_queue_upload_mbit']:.3f}/{r['mean_queue_compute_mbit']:.3f}/{r['mean_queue_transfer_mbit']:.3f}|"
                f"{r['queue_total_mbit']:.3f}|{r['power_w']:.3f}|")
        if name == "scene33502.md":
            detail += ["", "## 全部300秒窗口", "",
                       "|规则|窗口s|J|高/低完成时延ms|累计高/低完成率%|末尾高/低未完成数|平均计算/转发Mbit|期末Mbit|",
                       "|---|---|---:|---|---|---|---|---:|"]
            for w in (r for r in windows if r["seed"] == 33502):
                detail.append(f"|{w['policy']}|{w['start_s']:g}-{w['end_s']:g}|{w['J']:.6f}|"
                    f"{w['task_delay_high_ms']:.3f}/{w['task_delay_low_ms']:.3f}|"
                    f"{100*w['cumulative_completion_high_rate']:.4f}/{100*w['cumulative_completion_low_rate']:.4f}|"
                    f"{w['task_unfinished_high_count']}/{w['task_unfinished_low_count']}|"
                    f"{w['mean_queue_compute_mbit']:.3f}/{w['mean_queue_transfer_mbit']:.3f}|{w['queue_total_mbit']:.3f}|")
        (output/name).write_text("\n".join(detail)+"\n", encoding="utf-8")


def run_grid(output, smoke=False, smoke_report=None, source=SOURCE):
    prior, lookup, origins, protected = verify_source(source)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "codex/u2u-model":
        raise ValueError("Unexpected branch")
    arms = conditions(prior["arm_configs"]["w1_uniform_b6"])
    sources = source_hashes()
    if not smoke:
        proof = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
        if (proof["status"] != "SMOKE_COMPLETE" or proof["source_hashes"] != sources
                or proof["conditions"] != arms or proof["base_head"] != head or not proof["crn_equal"]
                or proof["protected_source_files"] != protected):
            raise ValueError("A matching successful smoke is required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", base_head=head, branch=branch, prior_base_head=prior["base_head"],
        source_manifest=str(Path(source)/"manifest.json"), source_manifest_sha256=file_hash(Path(source)/"manifest.json"),
        source_hashes=sources, conditions=arms, protected_source_files=protected,
        worktree_status=subprocess.check_output(["git","status","--porcelain"], cwd=ROOT, text=True),
        tracked_diff=subprocess.check_output(["git","diff","HEAD","--","hrl_uav","scripts","tests","configs"], cwd=ROOT, text=True),
        training=False, training_seed_count=0, checkpoint=None, train=False, complete_tail=False, macro_enabled=False,
        seeds=[33500] if smoke else SEEDS, duration_s=20 if smoke else 900,
        scene_scope="Reused development scenes, not unseen final tests", alpha_levels=dict(LEVELS),
        condition_count=31, expected_records=31 if smoke else 93,
        expected_reused=0 if smoke else 9, expected_new=31 if smoke else 84,
        local_alias="3MHz (1,1) references the same 6MHz Local records; no duplicate simulation",
        smoke_report=str(smoke_report) if smoke_report else None, record_provenance={})
    write_json(output/"manifest.json", manifest)
    rows, windows = [], []
    started, evaluated, reused, balance_error = time.perf_counter(), 0, 0, 0.0
    reference_dir = Path(source).with_name("wireless_model_fixed_rule_check_smoke") if smoke else Path(source)
    references = {seed: numeric_csv(reference_dir/f"w1_uniform_b6_{seed}_crn.csv") for seed in manifest["seeds"]}
    try:
        for arm in arms:
            for seed in manifest["seeds"]:
                key = f"{arm['policy']}:{seed}"
                is_reused = not smoke and arm["source_policy"] is not None
                if is_reused:
                    record = lookup[arm["source_policy"], seed]
                    provenance = copy.deepcopy(origins[key])
                    paths = provenance["files"]
                    service = numeric_csv(paths["service.csv"])
                    check_crn(references[seed], numeric_csv(paths["crn.csv"]))
                    check_trajectory(paths["trajectory.npz"], reference_dir/f"w1_uniform_b6_{seed}_trajectory.npz",
                                     PriorityRouting(arm["alpha_high"], arm["alpha_low"]).expected)
                    win = window_metrics(service, numeric_csv(paths["queues.csv"]), arm["policy"], seed)
                    reused += 1
                else:
                    print(f"[start {evaluated+1}/{manifest['expected_new']}] {key}", flush=True)
                    record, win, paths, error = run_arm(arm, seed, 2 if smoke else 90, output,
                        references[seed], reference_dir/f"w1_uniform_b6_{seed}_trajectory.npz")
                    balance_error = max(balance_error, error)
                    evaluated += 1
                    provenance = dict(source_policy=arm["policy"], source_per_seed=str(output/"per_seed.csv"),
                                      files=paths, hashes={k: file_hash(p) for k,p in paths.items()})
                    if smoke and arm["source_policy"]:
                        historical = next(r for r in numeric_csv(reference_dir/"per_seed.csv")
                                          if r["policy"] == arm["source_policy"])
                        for k in MEASURES:
                            np.testing.assert_allclose(record[k], historical[k], rtol=1e-12, atol=1e-8, err_msg=k)
                row = dict(policy=arm["policy"], seed=seed, high_fraction=arm["high_fraction"], low_fraction=arm["low_fraction"],
                    alpha_high=arm["alpha_high"], alpha_low=arm["alpha_low"], bandwidth_mhz=arm["bandwidth_mhz"],
                    reused=is_reused, source_policy=provenance["source_policy"], source_per_seed=provenance["source_per_seed"],
                    **{k: record[k] for k in MEASURES})
                # Window aggregates reconstruct full-run physical J and completion delays.
                np.testing.assert_allclose(np.mean([w["J"] for w in win]), row["J"], rtol=1e-12, atol=1e-9)
                for p in ("high", "low"):
                    n = sum(w[f"task_completed_{p}_count"] for w in win)
                    total = sum((w[f"task_delay_{p}_ms"] or 0)*w[f"task_completed_{p}_count"] for w in win)
                    np.testing.assert_allclose(total/max(n,1), row[f"task_delay_{p}_ms"], rtol=1e-12, atol=1e-8)
                rows.append(row)
                windows.extend(win)
                provenance.update(reused=is_reused, physical_source_hashes=prior["source_hashes"] if is_reused else sources)
                manifest["record_provenance"][key] = provenance
                append_csv(output/"per_seed.csv", [row])
                append_csv(output/"window_metrics.csv", win)
                manifest.update(completed_records=len(rows), evaluated_records=evaluated, reused_records=reused,
                                elapsed_seconds=time.perf_counter()-started)
                write_json(output/"manifest.json", manifest)
                print(f"[{'reuse' if is_reused else 'done'} {len(rows)}/{manifest['expected_records']}] {key} "
                      f"J={row['J']:.6f} delay={row['task_delay_weighted_ms']:.3f}ms "
                      f"endQ={row['queue_total_mbit']:.3f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if (evaluated != manifest["expected_new"] or reused != manifest["expected_reused"]
                or len(rows) != manifest["expected_records"]):
            raise AssertionError("Evaluation/reuse budget mismatch")
        if sources != source_hashes() or any(file_hash(p) != h for p,h in protected.items()):
            raise AssertionError("Source or protected previous results changed")
        means = summaries(rows, arms)
        append_csv(output/"summary.csv", means)
        if not smoke:
            pairs = paired_deltas(rows, arms)
            append_csv(output/"paired_deltas.csv", pairs)
            write_report(output, means, rows, windows, arms)
        manifest.update(status="SMOKE_COMPLETE" if smoke else "PRIORITY_GRID_COMPLETE",
            crn_equal=True, parent_conservation=True, queue_balance_max_error=balance_error,
            source_unchanged=True, source_hashes_after=source_hashes(), protected_results_unchanged=True,
            elapsed_seconds=time.perf_counter()-started)
        write_json(output/"manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        evaluated_records=evaluated, reused_records=reused)
        write_json(output/"manifest.json", manifest)
        raise
