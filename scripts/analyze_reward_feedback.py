import argparse
import csv
import json
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def configure_chinese_font():
    installed = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "STSong"):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training log: {path}")
    return rows


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def moving_average(series, window):
    series = np.asarray(series, dtype=np.float64)
    width = min(max(int(window), 1), series.size)
    result = np.convolve(
        series,
        np.ones(width, dtype=np.float64) / width,
        mode="valid",
    )
    return np.concatenate(
        [np.full(width - 1, np.nan, dtype=np.float64), result]
    )


def plot_smoothed(axis, x, y, window, label, color, raw=False):
    if raw:
        axis.plot(x, y, color=color, alpha=0.18, linewidth=0.7)
    axis.plot(
        x,
        moving_average(y, window),
        color=color,
        linewidth=1.8,
        label=label,
    )


def paired_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def linear_tail_slope(series, fraction=0.25):
    series = np.asarray(series, dtype=np.float64)
    width = max(int(np.ceil(series.size * float(fraction))), 3)
    tail = series[-width:]
    x = np.arange(width, dtype=np.float64)
    return float(np.polyfit(x, tail, deg=1)[0])


def window_mean(series, width, head):
    series = np.asarray(series, dtype=np.float64)
    actual = min(max(int(width), 1), series.size)
    selected = series[:actual] if head else series[-actual:]
    return float(np.mean(selected))


def parse_per_uav(rows, key):
    parsed = [
        np.asarray(json.loads(row[key]), dtype=np.float64)
        for row in rows
    ]
    return np.stack(parsed, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["smoke", "diagnostic", "pilot", "formal"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--results-version",
        default="tcom_communication_convergence_v1",
    )
    args = parser.parse_args()

    root = os.path.join(
        PROJECT_ROOT,
        "results",
        args.results_version,
        args.stage,
        f"seed{args.seed}",
    )
    micro_rows = read_rows(
        os.path.join(
            root,
            "micro_pretrain",
            "train_centralized.csv",
        )
    )
    joint_rows = read_rows(
        os.path.join(
            root,
            "joint_td3_sac",
            "train_td3.csv",
        )
    )
    output_dir = os.path.join(root, "figures")
    os.makedirs(output_dir, exist_ok=True)

    micro_episode = values(micro_rows, "episode")
    joint_episode = values(joint_rows, "episode")
    joint_reward = values(joint_rows, "macro_actor_reward") / np.maximum(
        values(joint_rows, "service_resource_decisions") / 18.0,
        1.0,
    )

    correlation_inputs = {
        "加权排队压力": "micro_cost_delay_mean",
        "总队列": "micro_q_total_mean",
        "平均U2G距离": "u2g_m",
        "实际上行时延代价": "micro_upload_delay_cost_mean",
        "预测U2U时间": "projected_u2u_time_s_mean",
        "通信代价": "comm_cost",
        "平均移动距离": "move_m",
        "约束能耗比": "macro_energy_ratio_mean",
    }
    correlations = {
        label: paired_correlation(joint_reward, values(joint_rows, key))
        for label, key in correlation_inputs.items()
    }

    configure_chinese_font()
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 12.0))

    micro_components = (
        ("delay_objective_train", "时延目标项", "#2878B5"),
        (
            "micro_lagrangian_energy_cost_mean",
            "能耗拉格朗日项",
            "#D95F02",
        ),
        ("micro_drift_cost_mean", "队列漂移项", "#2CA02C"),
        ("micro_overflow_cost_mean", "溢出项", "#9467BD"),
        ("J_train", "总成本J", "#222222"),
    )
    for key, label, color in micro_components:
        plot_smoothed(
            axes[0, 0],
            micro_episode,
            values(micro_rows, key),
            args.window,
            label,
            color,
            raw=(key == "J_train"),
        )
    axes[0, 0].set_title("(a) SAC预训练奖励成本分解")
    axes[0, 0].set_xlabel("预训练轮次")
    axes[0, 0].set_ylabel("每积分步平均成本")

    joint_components = (
        ("delay_objective_train", "时延目标项", "#2878B5"),
        (
            "micro_lagrangian_energy_cost_mean",
            "能耗拉格朗日项",
            "#D95F02",
        ),
        ("micro_drift_cost_mean", "队列漂移项", "#2CA02C"),
        ("micro_overflow_cost_mean", "溢出项", "#9467BD"),
        ("J_train", "总成本J", "#222222"),
    )
    for key, label, color in joint_components:
        plot_smoothed(
            axes[0, 1],
            joint_episode,
            values(joint_rows, key),
            args.window,
            label,
            color,
            raw=(key == "J_train"),
        )
    axes[0, 1].set_title("(b) 联合训练奖励成本分解")
    axes[0, 1].set_xlabel("联合训练轮次")
    axes[0, 1].set_ylabel("每积分步平均成本")

    macro_components = (
        (
            "macro_global_micro_signal_mean",
            "小时隙反馈",
            "#C82423",
            1.0,
        ),
        ("macro_shaping_signal_mean", "位置塑形", "#2CA02C", 1.0),
        ("macro_movement_penalty_mean", "移动惩罚", "#9467BD", -1.0),
        (
            "macro_propulsion_penalty_mean",
            "推进惩罚",
            "#8C564B",
            -1.0,
        ),
    )
    for key, label, color, sign in macro_components:
        plot_smoothed(
            axes[1, 0],
            joint_episode,
            sign * values(joint_rows, key),
            args.window,
            label,
            color,
        )
    axes[1, 0].set_title("(c) 宏观奖励反馈分解")
    axes[1, 0].set_xlabel("联合训练轮次")
    axes[1, 0].set_ylabel("对宏观奖励的平均贡献")

    link_axis = axes[1, 1]
    plot_smoothed(
        link_axis,
        joint_episode,
        values(joint_rows, "u2g_m"),
        args.window,
        "U2G距离",
        "#2878B5",
        raw=True,
    )
    link_axis.set_ylabel("平均U2G距离（m）", color="#2878B5")
    time_axis = link_axis.twinx()
    plot_smoothed(
        time_axis,
        joint_episode,
        values(joint_rows, "micro_upload_delay_cost_mean"),
        args.window,
        "实际上行时延代价",
        "#D95F02",
    )
    time_axis.set_ylabel("每积分步实际上行时延代价")
    handles_a, labels_a = link_axis.get_legend_handles_labels()
    handles_b, labels_b = time_axis.get_legend_handles_labels()
    link_axis.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
    )
    link_axis.set_title("(d) U2G距离与实际上行时延反馈")
    link_axis.set_xlabel("联合训练轮次")

    power = parse_per_uav(joint_rows, "battery_power_per_uav_w")
    budgets = parse_per_uav(
        joint_rows,
        "battery_power_budget_per_uav_w",
    )
    for uav_index in range(power.shape[1]):
        plot_smoothed(
            axes[2, 0],
            joint_episode,
            power[:, uav_index],
            args.window,
            f"UAV-{uav_index + 1}",
            ("#2878B5", "#D95F02", "#2CA02C")[uav_index],
        )
    axes[2, 0].axhline(
        float(np.mean(budgets)),
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="单机长期预算",
    )
    axes[2, 0].set_title("(e) 单机平均电池功率")
    axes[2, 0].set_xlabel("联合训练轮次")
    axes[2, 0].set_ylabel("功率（W）")

    labels = list(correlations)
    corr_values = np.asarray(
        [correlations[label] for label in labels],
        dtype=np.float64,
    )
    colors = np.where(corr_values >= 0.0, "#C82423", "#2878B5")
    axes[2, 1].barh(labels, corr_values, color=colors, alpha=0.85)
    axes[2, 1].axvline(0.0, color="#333333", linewidth=0.8)
    axes[2, 1].set_xlim(-1.0, 1.0)
    axes[2, 1].set_title("(f) 各状态量与真实宏观回报的相关性")
    axes[2, 1].set_xlabel("Pearson相关系数（仅描述性）")

    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
        if axis is not axes[1, 1] and axis is not axes[2, 1]:
            axis.legend(frameon=False)

    figure.suptitle("奖励分量及物理量反馈诊断")
    figure.tight_layout()
    figure_path = os.path.join(output_dir, "reward_feedback.png")
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        os.path.join(output_dir, "reward_feedback.pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)

    correlation_path = os.path.join(
        output_dir,
        "reward_feedback_correlations.csv",
    )
    with open(
        correlation_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["parameter", "source_field", "pearson_r"],
        )
        writer.writeheader()
        for label, key in correlation_inputs.items():
            writer.writerow(
                {
                    "parameter": label,
                    "source_field": key,
                    "pearson_r": correlations[label],
                }
            )

    report_window = min(max(args.window, 5), len(joint_rows))
    summary = {
        "results_version": args.results_version,
        "stage": args.stage,
        "seed": args.seed,
        "micro_pretrain_episodes": len(micro_rows),
        "joint_training_episodes": len(joint_rows),
        "window": report_window,
        "micro_j_first_window": window_mean(
            values(micro_rows, "J_train"),
            report_window,
            True,
        ),
        "micro_j_last_window": window_mean(
            values(micro_rows, "J_train"),
            report_window,
            False,
        ),
        "joint_j_first_window": window_mean(
            values(joint_rows, "J_train"),
            report_window,
            True,
        ),
        "joint_j_last_window": window_mean(
            values(joint_rows, "J_train"),
            report_window,
            False,
        ),
        "joint_reward_first_window": window_mean(
            joint_reward,
            report_window,
            True,
        ),
        "joint_reward_last_window": window_mean(
            joint_reward,
            report_window,
            False,
        ),
        "joint_reward_tail_slope_per_episode": linear_tail_slope(
            joint_reward
        ),
        "joint_j_tail_slope_per_episode": linear_tail_slope(
            values(joint_rows, "J_train")
        ),
        "micro_reward_reconstruction_max_error": float(
            np.max(
                values(
                    micro_rows,
                    "micro_reward_reconstruction_max_error",
                )
            )
        ),
        "joint_micro_reward_reconstruction_max_error": float(
            np.max(
                values(
                    joint_rows,
                    "micro_reward_reconstruction_max_error",
                )
            )
        ),
        "macro_reward_reconstruction_max_error": float(
            np.max(
                values(
                    joint_rows,
                    "macro_reward_reconstruction_max_error",
                )
            )
        ),
        "mean_episode_seconds_micro": float(
            np.mean(values(micro_rows, "episode_seconds"))
        ),
        "mean_episode_seconds_joint": float(
            np.mean(values(joint_rows, "episode_seconds"))
        ),
        "correlations": correlations,
    }
    summary_path = os.path.join(
        output_dir,
        "reward_feedback_summary.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(figure_path)
    print(correlation_path)
    print(summary_path)


if __name__ == "__main__":
    main()
