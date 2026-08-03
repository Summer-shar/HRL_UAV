"""Generate vector-first figures for the revised multi-timescale UAV-MEC paper.

The figures deliberately mirror the current v23 implementation:
fixed logical user ownership, centroid-relative large-timescale waypoints,
deterministically feasible small-timescale routing, active-link U2U
bandwidth sharing, and the deployment-boundary SMDP correction.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "paper_v3"

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.linewidth": 0.8,
    }
)

INK = "#242424"
MUTED = "#686868"
GRID = "#b8b8b8"
BLUE = "#3d6f8e"
ORANGE = "#b46a3c"
GREEN = "#557a56"
LIGHT_BLUE = "#dceaf2"
LIGHT_ORANGE = "#f2e3d7"
LIGHT_GREEN = "#e1eadf"
LIGHT_GRAY = "#eeeeee"


def _save(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc="white", ec=INK, fontsize=9, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )
    return patch


def _arrow(ax, start, end, text=None, color=INK, style="-|>", lw=1.1, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2
        ax.text(
            x,
            y + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    return arrow


def figure_system_architecture():
    rng = np.random.default_rng(23)
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("水平坐标 x/m")
    ax.set_ylabel("水平坐标 y/m")
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.grid(color=GRID, linewidth=0.5, linestyle=":", zorder=0)

    anchors = np.array([[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]])
    colors = [BLUE, ORANGE, GREEN]
    fills = [LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREEN]
    users = []

    for n, (anchor, color, fill) in enumerate(zip(anchors, colors, fills), start=1):
        ax.add_patch(
            Circle(
                anchor,
                150,
                facecolor=fill,
                edgecolor=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.42,
                zorder=1,
            )
        )
        pts = rng.normal(anchor, [105, 105], size=(13, 2))
        if n == 1:
            pts[:2] += np.array([300.0, 120.0])
        pts = np.clip(pts, 15, 985)
        users.append(pts)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
            label=f"固定归属用户集合 $\\mathcal{{K}}_{n}$",
        )

        triangle = Polygon(
            [
                anchor + [-17, -12],
                anchor + [17, -12],
                anchor + [0, 22],
            ],
            closed=True,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.8,
            zorder=5,
        )
        ax.add_patch(triangle)
        uav_label_offsets = [(0, 34), (0, 34), (48, 16)]
        label_dx, label_dy = uav_label_offsets[n - 1]
        ax.text(
            anchor[0] + label_dx,
            anchor[1] + label_dy,
            f"UAV {n}",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.5,
            zorder=6,
        )

        for user in pts[[0, 4, 8]]:
            ax.plot(
                [user[0], anchor[0]],
                [user[1], anchor[1]],
                color=color,
                linewidth=0.7,
                linestyle=":",
                zorder=2,
            )

    for i, j, rad in [(0, 1, 0.08), (1, 2, 0.08), (2, 0, 0.08)]:
        _arrow(
            ax,
            anchors[i],
            anchors[j],
            color=INK,
            lw=1.25,
            rad=rad,
        )

    ax.text(
        535,
        800,
        "活跃有向U2U链路共享总带宽\n$B_{\\mathrm{UU}}=3$ MHz",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax.text(
        985,
        985,
        "$h_{\\mathrm{U}}=50$ m\n$\\|\\mathbf{q}_i-\\mathbf{q}_j\\|\\geq 5$ m",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3"),
    )
    ax.text(
        20,
        20,
        "固定逻辑关联；用户可跨越虚线区域，仍上传至原归属UAV",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88),
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    _save(fig, "fig1-system-architecture")


def figure_multiscale_timeline():
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    ax.set_xlim(-0.2, 10.35)
    ax.set_ylim(-0.2, 4.1)
    ax.axis("off")

    ax.text(-0.05, 3.55, "大时间尺度", ha="right", va="center", fontsize=10)
    ax.add_patch(Rectangle((0, 3.2), 10, 0.7, facecolor="white", edgecolor=INK))
    ax.text(5, 3.55, "第 $t$ 个大时间尺度周期 $\\Delta_{\\mathrm{L}}=10$ s", ha="center", va="center")

    ax.text(-0.05, 2.55, "运行阶段", ha="right", va="center", fontsize=10)
    ax.add_patch(Rectangle((0, 2.2), 1, 0.7, facecolor=LIGHT_ORANGE, edgecolor=INK))
    ax.add_patch(Rectangle((1, 2.2), 9, 0.7, facecolor=LIGHT_BLUE, edgecolor=INK))
    ax.text(0.5, 2.55, "部署\n1 s", ha="center", va="center", fontsize=9)
    ax.text(5.5, 2.55, "服务阶段 9 s", ha="center", va="center", fontsize=10)

    ax.text(-0.05, 1.55, "0.5 s资源帧", ha="right", va="center", fontsize=10)
    for idx in range(20):
        x = idx * 0.5
        fc = LIGHT_ORANGE if idx < 2 else ("white" if idx % 2 else LIGHT_BLUE)
        ax.add_patch(Rectangle((x, 1.25), 0.5, 0.6, facecolor=fc, edgecolor=INK, linewidth=0.65))
        if idx < 2:
            label = f"部署{idx+1}"
        elif idx in (2, 3, 18, 19):
            label = f"$r={idx-2}$"
        else:
            label = "$\\cdots$" if idx in (8, 13) else ""
        ax.text(x + 0.25, 1.55, label, ha="center", va="center", fontsize=8)

    ax.annotate(
        "2个部署帧：到达和移动继续，U2G/U2U/CPU暂停",
        xy=(0.5, 1.22),
        xytext=(0.5, 0.62),
        ha="center",
        va="top",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
        color=MUTED,
    )
    ax.annotate(
        "18个服务帧，每帧 $\\Delta_{\\mathrm{S}}=0.5$ s",
        xy=(5.5, 1.22),
        xytext=(5.5, 0.62),
        ha="center",
        va="top",
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
        color=MUTED,
    )

    ax.text(-0.05, 0.0, "物理演化步", ha="right", va="center", fontsize=10)
    for idx in range(5):
        x = 1.0 + idx * 0.1
        ax.add_patch(Rectangle((x, -0.13), 0.1, 0.3, facecolor="white", edgecolor=INK, linewidth=0.6))
    ax.text(1.72, 0.02, "$5\\times\\delta=5\\times0.1$ s", ha="left", va="center", fontsize=9)

    ax.annotate("", xy=(10, 4.02), xytext=(0, 4.02), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(5, 4.04, "连续运行由相邻大时间尺度周期串联构成", ha="center", va="bottom", fontsize=9)
    _save(fig, "fig2-multiscale-timeline")


def figure_task_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.02, 0.39, 0.12, 0.22, "父任务产生\n高/低优先级", fc=LIGHT_GRAY)
    _box(ax, 0.18, 0.39, 0.13, 0.22, "用户上传FIFO\n按任务先入先出", fc=LIGHT_BLUE)
    _box(ax, 0.35, 0.39, 0.13, 0.22, "源UAV标记上传完成\n下一物理步拆分", fc="white")
    _box(ax, 0.52, 0.39, 0.13, 0.22, "比例路由拆分\n$\\rho_{ij,p}$", fc=LIGHT_ORANGE)
    _box(ax, 0.70, 0.67, 0.13, 0.20, "本地计算队列\n$Q_{i,p}^{\\mathrm{loc}}$", fc=LIGHT_GREEN)
    _box(ax, 0.70, 0.15, 0.13, 0.20, "U2U发送队列\n$Q_{ij,p}^{\\mathrm{tx}}$", fc=LIGHT_BLUE)
    _box(ax, 0.86, 0.15, 0.12, 0.20, "目标计算队列\n$Q_{j,p}^{\\mathrm{loc}}$", fc=LIGHT_GREEN)
    _box(ax, 0.86, 0.67, 0.12, 0.20, "父任务汇合\n最慢分支完成", fc=LIGHT_GRAY)

    _arrow(ax, (0.14, 0.50), (0.18, 0.50))
    _arrow(ax, (0.31, 0.50), (0.35, 0.50))
    _arrow(ax, (0.48, 0.50), (0.52, 0.50))
    _arrow(ax, (0.65, 0.51), (0.70, 0.77), "$j=i$")
    _arrow(ax, (0.65, 0.49), (0.70, 0.25), "$j\\ne i$")
    _arrow(ax, (0.83, 0.25), (0.86, 0.25))
    _arrow(ax, (0.83, 0.77), (0.86, 0.77))
    _arrow(ax, (0.92, 0.35), (0.92, 0.67), "完成的远端分支", rad=0.0)

    ax.text(0.16, 0.545, "泊松到达", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(0.33, 0.545, "$R_k^{\\mathrm{UG}}$", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(0.845, 0.295, "$R_{ij}^{\\mathrm{UU}}$", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    ax.text(
        0.765,
        0.04,
        "活跃有向链路共享3 MHz；链路容量先服务高优先级",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        0.985,
        0.945,
        "父任务完成：\n$T_\\omega^{\\mathrm{fin}}=\\max_{b\\in\\mathcal{B}_\\omega}T_{\\omega,b}^{\\mathrm{fin}}$",
        ha="right",
        va="top",
        fontsize=8.5,
    )
    ax.text(
        0.50,
        0.93,
        "父任务在上传、U2U等待和计算阶段始终计入加权任务驻留代价",
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    _save(fig, "fig3-task-pipeline")


def figure_hierarchical_framework():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.72, 0.14, 0.16, "大时间尺度状态\n$\\mathbf{s}_t^{\\mathrm{L}}\\in\\mathbb{R}^{81}$", fc=LIGHT_GRAY)
    _box(ax, 0.22, 0.72, 0.13, 0.16, "TD3 Actor\n确定性策略", fc=LIGHT_BLUE)
    _box(ax, 0.40, 0.72, 0.14, 0.16, "6维动作\n单位圆盘映射", fc="white")
    _box(ax, 0.59, 0.72, 0.16, 0.16, "质心相对航点\n$R_{\\mathrm{w}}=200$ m", fc=LIGHT_ORANGE)
    _box(ax, 0.80, 0.72, 0.17, 0.16, "每周期最多20 m\n最小间距5 m", fc=LIGHT_GREEN)

    for start, end in [
        ((0.17, 0.80), (0.22, 0.80)),
        ((0.35, 0.80), (0.40, 0.80)),
        ((0.54, 0.80), (0.59, 0.80)),
        ((0.75, 0.80), (0.80, 0.80)),
    ]:
        _arrow(ax, start, end)

    _box(ax, 0.03, 0.23, 0.14, 0.16, "小时间尺度状态\n$\\mathbf{s}_{t,r}^{\\mathrm{S}}\\in\\mathbb{R}^{48}$", fc=LIGHT_GRAY)
    _box(ax, 0.22, 0.23, 0.13, 0.16, "SAC Actor\nTanh-Gaussian", fc=LIGHT_BLUE)
    _box(ax, 0.40, 0.23, 0.14, 0.16, "18维动作\n两组$3\\times3$矩阵", fc="white")
    _box(ax, 0.59, 0.23, 0.16, 0.16, "逐行确定性\n可行比例映射", fc=LIGHT_ORANGE)
    _box(ax, 0.80, 0.23, 0.17, 0.16, "可稀疏协作路由\n加权队列CPU分配", fc=LIGHT_GREEN)

    for start, end in [
        ((0.17, 0.31), (0.22, 0.31)),
        ((0.35, 0.31), (0.40, 0.31)),
        ((0.54, 0.31), (0.59, 0.31)),
        ((0.75, 0.31), (0.80, 0.31)),
    ]:
        _arrow(ax, start, end)

    _box(
        ax,
        0.35,
        0.47,
        0.30,
        0.14,
        "UAV-MEC环境\n移动用户、上传FIFO、活跃链路带宽共享、队列与能耗",
        fc="white",
        lw=1.2,
    )
    _arrow(ax, (0.885, 0.72), (0.65, 0.56), "位置改变U2G/U2U和推进能耗", color=BLUE)
    _arrow(ax, (0.885, 0.39), (0.65, 0.52), "路由改变流量与队列", color=ORANGE)
    _arrow(ax, (0.35, 0.54), (0.10, 0.72), "队列/流量/负载摘要", color=BLUE, rad=0.15)
    _arrow(ax, (0.35, 0.50), (0.10, 0.39), "队列/链路/引导矩阵", color=ORANGE, rad=-0.15)

    ax.text(
        0.50,
        0.10,
        "大时间尺度回放奖励：$-0.1[\\bar c_{\\mathrm{P}}+20(g^{\\mathrm{L}}-1)^+]$；"
        "小时间尺度：加权任务驻留代价及有界溢出项",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(facecolor=LIGHT_GRAY, edgecolor=GRID, boxstyle="round,pad=0.35"),
    )
    ax.text(
        0.50,
        0.02,
        "当前系统乘子固定为0；能量Critic仅作辅助估计，不参与Actor更新",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    _save(fig, "fig4-hierarchical-framework")


def figure_action_mapping():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axhline(0, color=GRID, linewidth=0.6)
    ax.axvline(0, color=GRID, linewidth=0.6)
    ax.add_patch(Circle((0, 0), 1.0, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.3))
    raw = np.array([1.18, 0.92])
    projected = raw / np.linalg.norm(raw)
    _arrow(ax, (0, 0), raw, "$\\widetilde{\\mathbf{a}}_n^{\\mathrm{L}}$", color=MUTED, style="->", lw=1.1)
    _arrow(ax, (0, 0), projected, "$\\mathbf{a}_n^{\\mathrm{L}}$", color=BLUE, style="-|>", lw=1.7)
    ax.plot(raw[0], raw[1], marker="x", color=MUTED, markersize=7)
    ax.plot(projected[0], projected[1], marker="o", color=BLUE, markersize=5)
    ax.text(0, -1.22, "逐无人机单位圆盘映射", ha="center", va="center", fontsize=10)
    ax.text(
        0,
        1.23,
        "$\\mathbf{a}_n^{\\mathrm{L}}="
        "\\widetilde{\\mathbf{a}}_n^{\\mathrm{L}}/"
        "\\max(1,\\|\\widetilde{\\mathbf{a}}_n^{\\mathrm{L}}\\|_2)$",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("$a_x$")
    ax.set_ylabel("$a_y$")

    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.08, 0.98)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    triangle = Polygon(vertices, closed=True, facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=1.4)
    ax.add_patch(triangle)
    ax.text(-0.02, -0.04, "$(1,0,0)$", ha="left", va="top", fontsize=9)
    ax.text(1.02, -0.04, "$(0,1,0)$", ha="right", va="top", fontsize=9)
    ax.text(0.5, 0.91, "$(0,0,1)$", ha="center", va="bottom", fontsize=9)
    points = np.array(
        [
            [0.20, 0.00],
            [0.82, 0.00],
            [0.50, np.sqrt(3) / 2],
            [0.38, 0.38],
        ]
    )
    markers = ["s", "^", "o", "D"]
    for point, marker in zip(points, markers):
        ax.plot(
            point[0],
            point[1],
            marker=marker,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=ORANGE,
        )
    ax.annotate(
        "严格零分量",
        xy=(0.20, 0.0),
        xytext=(0.15, 0.26),
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    ax.text(
        0.50,
        0.58,
        "$\\boldsymbol{\\rho}_{i,p}=\\Pi_{\\Delta_N}^{\\mathrm{Euc}}(\\mathbf{z}_{i,p})$",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(0.5, -0.07, "欧氏概率单纯形映射（$N=3$）", ha="center", va="top", fontsize=10)
    ax.axis("off")
    _save(fig, "fig5-action-mapping")


def figure_smdp_boundary():
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    items = [
        (0.3, 1.85, 2.0, 1.0, "周期 $t$ 最后服务帧\n状态 $\\mathbf{s}_{t,17}^{\\mathrm{S}}$\n奖励 $R_{\\mathrm{S}}^{\\mathrm{last}}[t]$", LIGHT_BLUE),
        (3.0, 1.85, 1.7, 1.0, "下周期部署帧1\n无服务\n驻留奖励 $R_{\\mathrm{hold}}^{\\mathrm{D},1}$", LIGHT_ORANGE),
        (5.3, 1.85, 1.7, 1.0, "下周期部署帧2\n无服务\n驻留奖励 $R_{\\mathrm{hold}}^{\\mathrm{D},2}$", LIGHT_ORANGE),
        (7.7, 1.85, 2.0, 1.0, "周期 $t+1$ 首个服务态\n下一状态 $\\mathbf{s}_{t+1,0}^{\\mathrm{S}}$", LIGHT_GREEN),
    ]
    for x, y, w, h, text, fc in items:
        _box(ax, x, y, w, h, text, fc=fc, fontsize=9)

    _arrow(ax, (2.3, 2.35), (3.0, 2.35), "$\\gamma_{\\mathrm{S}}$")
    _arrow(ax, (4.7, 2.35), (5.3, 2.35), "$\\gamma_{\\mathrm{S}}^2$")
    _arrow(ax, (7.0, 2.35), (7.7, 2.35), "$\\gamma_{\\mathrm{S}}^3$")

    ax.text(
        5.0,
        3.50,
        "$\\widetilde R_{\\mathrm{S}}^{\\mathrm{bd}}="
        "R_{\\mathrm{S}}^{\\mathrm{last}}+"
        "\\gamma_{\\mathrm{S}}R_{\\mathrm{hold}}^{\\mathrm{D},1}+"
        "\\gamma_{\\mathrm{S}}^2R_{\\mathrm{hold}}^{\\mathrm{D},2},"
        "\\quad \\Gamma_{\\mathrm{S}}^{\\mathrm{bd}}=\\gamma_{\\mathrm{S}}^3$",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.text(
        5.0,
        0.90,
        "部署期任务到达和用户移动继续，但U2G、U2U和CPU服务暂停",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )
    ax.annotate(
        "",
        xy=(7.0, 0.48),
        xytext=(0.3, 0.48),
        arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0),
    )
    ax.text(3.65, 0.51, "持续时间 $0.5+0.5+0.5=1.5$ s", ha="center", va="bottom", fontsize=9)
    _save(fig, "fig6-smdp-boundary")


def main():
    figure_system_architecture()
    figure_multiscale_timeline()
    figure_task_pipeline()
    figure_hierarchical_framework()
    figure_action_mapping()
    figure_smdp_boundary()
    print(f"Generated paper figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
