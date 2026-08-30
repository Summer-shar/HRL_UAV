"""Evaluate the v64 causal-fair HRL comparison and produce paper figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v45_fair_screen as v45  # noqa: E402
import evaluate_v48_priority_gated as v48  # noqa: E402
from compare_sca import solve_sca_micro_action  # noqa: E402
from compare_xu2024_sca import (  # noqa: E402
    configure_xu2024_wrapper,
    solve_xu2024_micro_action,
)
from train_centralized_macro import sca_micro_policy  # noqa: E402


RESULTS_VERSION = "tcom_v64_causal_fair_hrl"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v64_causal_fair_hrl_seed1"
METHOD_ORDER = (
    "hrl_residual",
    "hrl_direct",
    "xu_cvx",
    "qa_psca_causal",
    "no_coop",
    "random",
)
METHOD_LABELS = {
    "hrl_residual": "PPO+残差SAC（本文）",
    "hrl_direct": "PPO+纯SAC",
    "xu_cvx": "Xu-CVX（适配）",
    "qa_psca_causal": "因果QA-PSCA",
    "no_coop": "不协作",
    "random": "随机路由",
}
COLORS = {
    "hrl_residual": "#D1495B",
    "hrl_direct": "#2878B5",
    "xu_cvx": "#6A4C93",
    "qa_psca_causal": "#2A9D8F",
    "no_coop": "#7F7F7F",
    "random": "#E9C46A",
}


def checkpoint(method: str) -> Path:
    return (
        ROOT
        / "results"
        / RESULTS_VERSION
        / "formal"
        / "seed1"
        / method
        / "checkpoints"
        / "final.pt"
    )


def zero_macro_policy(env, _local_obs, _deterministic=False):
    return np.zeros((env.N, 2), dtype=np.float32)


def cloned_args(args):
    return argparse.Namespace(**vars(args))


def static_policy(name: str, base_args, seed: int) -> dict:
    args = cloned_args(base_args)
    args.sca_load_source = "current_arrivals"
    args.sca_update_interval = 1
    args.sca_max_iter = 3
    args.sca_tol = 1e-3
    args.sca_solver = "CLARABEL"
    args.sca_delay_weight = float(args.delay_objective_weight)
    args.sca_energy_weight = float(args.energy_objective_weight)
    args.sca_proximal_weight = 1e-3

    if name == "qa_psca_causal":
        micro_policy = sca_micro_policy(solve_sca_micro_action, args)
    elif name == "xu_cvx":
        args.xu_sca_max_iter = 10
        args.xu_sca_tol = 0.01
        args.xu_sca_proximal_rho = 1e-3
        args.xu_sca_proximal_cpu = 1e-3
        args.xu_sca_warm_start = 0
        args.xu_sca_task_aggregation = "source_priority"
        args.xu_sca_enforce_frame_deadline = 0
        args.xu_sca_use_priority_delay_weights = 1
        args.xu_sca_queue_aware = 0

        def micro_policy(env, deterministic=False):
            configure_xu2024_wrapper(env, args)
            return solve_xu2024_micro_action(env, deterministic)
    elif name == "no_coop":
        def micro_policy(env, _deterministic=False):
            env.force_local_routing = True
            return np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
    elif name == "random":
        rng = np.random.default_rng(int(seed) + 64000)

        def micro_policy(env, _deterministic=False):
            route = rng.dirichlet(np.ones(env.N), size=2 * env.N)
            return route.reshape(-1).astype(np.float32)
    else:
        raise ValueError(name)

    return {
        "name": name,
        "path": Path("<fixed-policy>"),
        "args": args,
        "macro_policy": zero_macro_policy,
        "micro_policy": micro_policy,
        "micro_solver": name,
        "macro_solver": "centroid",
    }


def evaluate(policy: dict, seed: int, trajectory: bool = False):
    row, high, low, trace = v48.evaluate_detailed(
        policy,
        int(seed),
        trajectory=trajectory,
    )
    row["method"] = policy["name"]
    return row, high, low, trace


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], methods: list[str], seeds: list[int]) -> dict:
    numeric = [key for key in rows[0] if key not in {"seed", "method"}]
    indexed = {
        (str(row["method"]), int(row["seed"])): row
        for row in rows
    }
    summaries = {
        method: {
            metric: v45.mean_ci95(
                [float(indexed[(method, seed)][metric]) for seed in seeds]
            )
            for metric in numeric
        }
        for method in methods
    }
    paired = {}
    if "hrl_residual" in methods:
        for method in methods:
            if method == "hrl_residual":
                continue
            paired[f"hrl_residual_minus_{method}"] = {
                metric: v45.mean_ci95([
                    float(indexed[("hrl_residual", seed)][metric])
                    - float(indexed[(method, seed)][metric])
                    for seed in seeds
                ])
                for metric in numeric
            }
    return {
        "scope": (
            "One independent training seed for each learned method and paired "
            "CRN environment seeds. QA-PSCA uses current realized arrivals, "
            "not model arrival rates. Xu-CVX is the source-priority adapter."
        ),
        "training_seed": 1,
        "evaluation_seeds": seeds,
        "method_summaries": summaries,
        "paired_contrasts": paired,
        "delay_ranking": sorted(
            methods,
            key=lambda item: summaries[item]["task_delay_weighted_ms"]["mean"],
        ),
    }


def configure_plotting() -> None:
    v45.configure_plotting()
    plt.rcParams["font.size"] = 10.5


def plot_main_results(report: dict, methods: list[str], output: Path) -> None:
    configure_plotting()
    panels = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 时延"),
        ("queue_total_mbits", "平均队列积压 / Mbit", "(b) 队列"),
        ("average_power_w", "平均功率 / W", "(c) 功率"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    x = np.arange(len(methods))
    labels = [METHOD_LABELS[item] for item in methods]
    for axis, (metric, ylabel, title) in zip(axes, panels):
        stats = [report["method_summaries"][m][metric] for m in methods]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        axis.bar(
            x,
            means,
            yerr=errors,
            capsize=3,
            color=[COLORS[item] for item in methods],
            width=0.7,
        )
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("统一因果信息口径下的方法比较（20个共同环境种子）")
    fig.tight_layout()
    fig.savefig(output / "01_主结果_时延队列功率.png", dpi=300, facecolor="white")
    plt.close(fig)


def _read_eval_curve(path: Path, offset: int):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([offset + int(float(row["episode"])) for row in rows]),
        np.asarray([float(row["task_delay_weighted_ms"]) for row in rows]),
    )


def plot_convergence(output: Path) -> None:
    configure_plotting()
    root = ROOT / "results" / RESULTS_VERSION / "formal" / "seed1"
    sources = {
        "hrl_residual": (
            root / "micro_pretrain_joint_ppo_sac" / "eval_central_macro.csv",
            root / "joint_ppo_sac" / "eval_central_macro.csv",
        ),
        "hrl_direct": (
            root / "micro_pretrain_joint_ppo_direct_sac" / "eval_central_macro.csv",
            root / "joint_ppo_direct_sac" / "eval_central_macro.csv",
        ),
    }
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    for method, (pretrain, joint) in sources.items():
        x1, y1 = _read_eval_curve(pretrain, 0)
        x2, y2 = _read_eval_curve(joint, 200)
        axis.plot(
            np.concatenate([x1, x2]),
            np.concatenate([y1, y2]),
            marker="o",
            markersize=3.2,
            linewidth=2.0,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.axvline(200, color="#333333", linestyle="--", linewidth=1.2)
    axis.text(204, 0.98, "上下层联训开始", transform=axis.get_xaxis_transform(), va="top")
    axis.set_xlabel("累计训练轮次")
    axis.set_ylabel("固定策略评估端到端时延 / ms")
    axis.set_title("微观预训练与上下层联合训练收敛曲线")
    axis.grid(linestyle="--", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "02_完整收敛曲线.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectories(traces: dict, output: Path) -> None:
    configure_plotting()
    methods = [item for item in ("hrl_residual", "hrl_direct", "xu_cvx", "qa_psca_causal") if item in traces]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0), sharex=True, sharey=True)
    colors = ("#2878B5", "#E67E22", "#2A9D8F")
    for axis, method in zip(axes.flat, methods):
        trace = traces[method]
        allocations = np.asarray(trace["user_allocations"], dtype=np.int64)
        for user in range(trace["users"].shape[1]):
            zone = int(allocations[user])
            axis.plot(
                trace["users"][:, user, 0],
                trace["users"][:, user, 1],
                color=colors[zone],
                linewidth=0.5,
                alpha=0.14,
            )
        for index, color in enumerate(colors):
            axis.plot(
                trace["centroids"][:, index, 0],
                trace["centroids"][:, index, 1],
                color=color,
                linestyle="--",
                linewidth=1.1,
            )
            axis.plot(
                trace["uavs"][:, index, 0],
                trace["uavs"][:, index, 1],
                color=color,
                linewidth=2.4,
                label=f"UAV {index + 1}",
            )
            axis.scatter(*trace["uavs"][0, index], marker="s", s=34, facecolor="white", edgecolor=color)
            axis.scatter(*trace["uavs"][-1, index], marker="*", s=85, color=color, edgecolor="#222222")
        axis.set_title(METHOD_LABELS[method])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(linestyle="--", alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("X坐标 / m")
    for axis in axes[:, 0]:
        axis.set_ylabel("Y坐标 / m")
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper center")
    fig.suptitle("同一环境种子下的UAV、用户与业务质心轨迹")
    fig.tight_layout()
    fig.savefig(output / "03_轨迹对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_report_md(report: dict, output: Path) -> None:
    ranking = report["delay_ranking"]
    lines = [
        "# v64 统一因果口径结果",
        "",
        "本轮把 QA-PSCA 的输入从真实期望到达率改为当帧已观测到达量；",
        "Xu-CVX 在无到达帧不再生成使用真实到达率的伪任务。学习方法各自从头训练。",
        "",
        "| 排名 | 方法 | 时延/ms | 队列/Mbit | 功率/W | 协作比例 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, method in enumerate(ranking, 1):
        summary = report["method_summaries"][method]
        lines.append(
            f"| {rank} | {METHOD_LABELS[method]} | "
            f"{summary['task_delay_weighted_ms']['mean']:.3f} | "
            f"{summary['queue_total_mbits']['mean']:.3f} | "
            f"{summary['average_power_w']['mean']:.3f} | "
            f"{summary['coop_ratio']['mean']:.4f} |"
        )
    lines += [
        "",
        "## 证据边界",
        "",
        "- 学习方法目前只有一个训练种子；20个种子是共同环境评估种子。",
        "- Xu-CVX保留本仓库固定区域关联，并将任务聚合为源UAV×优先级，因此是适配版而非逐项复刻原论文仿真。",
        "- QA-PSCA仍是仓库自研的队列感知优化器，本轮只修正了其动态负载输入公平性。",
        "- 完整原始数据、配对差和图均在本目录，结论不依赖单个示例轨迹。",
    ]
    (output / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--trajectory-seed", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=checkpoint("joint_ppo_sac"),
    )
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = v45.parse_seed_spec(args.eval_seeds)

    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    base = v45.load_method_policy("base", args.base_checkpoint, args.device)
    learned = {}
    if not args.static_only:
        learned = {
            "hrl_residual": v45.load_method_policy(
                "hrl_residual", checkpoint("joint_ppo_sac"), args.device
            ),
            "hrl_direct": v45.load_method_policy(
                "hrl_direct", checkpoint("joint_ppo_direct_sac"), args.device
            ),
        }
    methods = list(learned) + ["xu_cvx", "qa_psca_causal", "no_coop", "random"]
    rows = []
    routing_high = {}
    routing_low = {}
    for seed in seeds:
        for method in methods:
            policy = learned.get(method) or static_policy(method, base["args"], seed)
            row, high, low, _ = evaluate(policy, seed)
            rows.append(row)
            routing_high.setdefault(method, []).append(high)
            routing_low.setdefault(method, []).append(low)
            print(
                f"seed={seed} method={method} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms "
                f"queue={row['queue_total_mbits']:.2f}Mbit",
                flush=True,
            )

    report = summarize(rows, methods, seeds)
    report["information_contract"] = {
        "qa_psca_load_source": "current_arrivals",
        "xu_cvx_task_source": "current_arrival_tasks_only",
        "hidden_expected_arrival_fallback": False,
    }
    report["mean_routing_high"] = {
        method: np.mean(values, axis=0).tolist()
        for method, values in routing_high.items()
    }
    report["mean_routing_low"] = {
        method: np.mean(values, axis=0).tolist()
        for method, values in routing_low.items()
    }
    write_csv(output / "per_seed_raw.csv", rows)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_main_results(report, methods, output)

    traces = {}
    for method in methods:
        if method not in {"hrl_residual", "hrl_direct", "xu_cvx", "qa_psca_causal"}:
            continue
        policy = learned.get(method) or static_policy(
            method, base["args"], args.trajectory_seed
        )
        _, _, _, trace = evaluate(policy, args.trajectory_seed, trajectory=True)
        traces[method] = trace
        np.savez_compressed(
            output / f"trajectory_{method}_seed{args.trajectory_seed}.npz",
            **trace,
        )
    plot_trajectories(traces, output)
    if not args.static_only:
        plot_convergence(output)
    write_report_md(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
