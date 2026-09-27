"""Plot saved diagnostics only; this module never runs physical simulation."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    panel = json.loads((root / "panel.json").read_text())
    followup = json.loads((root / "priority_followup.json").read_text())
    mechanisms = json.loads((root / "mechanisms.json").read_text())
    groups = [(layout, pattern) for layout in (92, 102)
              for pattern in ("staggered_busy", "synchronous_busy")]
    labels = ["A: staggered", "A: synchronous", "B: staggered", "B: synchronous"]
    routes = ("local", "equal", "high_local_low_equal")
    names = ("Local", "Equal split", "High local / low equal")
    colors = ("#355C7D", "#149488", "#D68135")

    def row(group, motion, route):
        return next(r for r in panel + followup if (r["layout_seed"], r["pattern"]) == group
                    and r["motion"] == motion and r["route"] == route)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), layout="constrained")
    x = np.arange(4)
    width = .24
    ax = axes[0, 0]
    for i, (route, name, color) in enumerate(zip(routes, names, colors)):
        values = [row(g, "hover", route)["cost_per_s"] for g in groups]
        bars = ax.bar(x + (i-1)*width, values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, rotation=90)
    ax.set(title="A  Routing changes total physical cost", ylabel="Cost / second (lower is better)", ylim=(0, 22))
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8, loc="upper center", ncol=3)

    ax = axes[0, 1]
    for shift, motion, name, color in ((-.17, "toward", "Toward centroid", colors[1]),
                                      (.17, "away", "Away, matched speed", colors[0])):
        values = [row(g, "hover", "equal")["cost_per_s"] - row(g, motion, "equal")["cost_per_s"] for g in groups]
        bars = ax.bar(x+shift, values, .3, color=color, label=name)
        ax.bar_label(bars, fmt="%+.3f", fontsize=8, padding=3)
    propulsion = [row(g, "hover", "equal")["cost_components_per_s"]["propulsion"] -
                  row(g, "toward", "equal")["cost_components_per_s"]["propulsion"] for g in groups]
    ax.plot(x, propulsion, "s--", color="#777777", label="Propulsion contribution", markersize=4)
    ax.axhline(0, color="#777777", lw=.8)
    ax.set(title="B  Motion benefit is small and conditional", ylabel="Cost / second saved vs hover (equal split)", ylim=(-.14, .23))
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 0]
    conditions = ("idle", "source_busy", "destination_busy")
    for i, (route, label, color) in enumerate(zip(("local", "equal", "remote1"),
                                               ("Local", "Equal split", "Single remote"), colors)):
        values = [next(r["target_finish_s"] for r in mechanisms if r["condition"] == condition and r["action"] == route)
                  for condition in conditions]
        bars = ax.bar(np.arange(3)+(i-1)*width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    ax.set(title="C  A congested fragment delays its whole parent", ylabel="Target ready-to-complete time (seconds)", ylim=(0, 2.95))
    ax.set_xticks(np.arange(3), ("Idle", "Source CPU busy", "Destination CPU busy"))
    ax.legend(fontsize=8, loc="upper center", ncol=3)

    ax = axes[1, 1]
    for i, (route, label, color) in enumerate(zip(routes, names, colors)):
        values = [row(g, "hover", route)["priority"][0]["mean_delay_ms"] for g in groups]
        bars = ax.bar(x+(i-1)*width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set(title="D  Faster high priority alone does not minimize cost", ylabel="High-priority completed mean delay (ms)", ylim=(0, 505))
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Fixed-workload UAV-MEC diagnostics | unchanged physics, no RL training", fontsize=15)
    fig.savefig(root / "control_diagnostics.png", dpi=170)
    fig.savefig(root / "control_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
