import sys
sys.path.insert(0, "D:/Scientific/HRL_UAV")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Setup
OUTPUT_DIR = Path("D:/Scientific/HRL_UAV/paper_current_20260907/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'SAC': '#2868a0',
    'CVX': '#555555',
    'Uniform': '#4a90d9',
    'Random': '#999999',
    'Local': '#777777',
    'Predictive': '#2d8a4e',
    'PPO': '#4a90d9',
    'Joint': '#7ab8f5',
}

# ============================================================
# Figure 1: Micro-layer Routing Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['SAC\n(Ours)', 'CVX-R', 'Uniform', 'Random', 'Local']
J_values = [0.5532, 0.5882, 0.5725, 0.6669, 3.7585]
colors = [COLORS['SAC'], COLORS['CVX'], COLORS['Uniform'], COLORS['Random'], COLORS['Local']]

# Highlight SAC (ours)
colors[0] = '#e41a1c'  # Red for ours

bars = ax.bar(methods, J_values, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, J_values):
    height = bar.get_height()
    ax.annotate(f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Physical Cost J (lower is better)', fontsize=11)
ax.set_title('Fig. 6: Micro-layer Routing Comparison (Macro: PPO120)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 4.5)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#e41a1c', label='Proposed (SAC)'),
    mpatches.Patch(facecolor=COLORS['CVX'], label='CVX-R'),
    mpatches.Patch(facecolor=COLORS['Uniform'], label='Uniform'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig6_micro_routing.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig6_micro_routing.png")

# ============================================================
# Figure 2: Macro-layer Trajectory Control
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['Predictive\n(Oracle)', 'Joint HRL\n(Ours)', 'PPO120']
J_values = [0.4946, 0.5020, 0.5532]
colors = [COLORS['Predictive'], '#e41a1c', COLORS['PPO']]

bars = ax.bar(methods, J_values, color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, J_values):
    height = bar.get_height()
    ax.annotate(f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Physical Cost J (lower is better)', fontsize=11)
ax.set_title('Fig. 7: Macro-layer Trajectory Control (Micro: SAC)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.7)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['Predictive'], label='Oracle (Predictive)'),
    mpatches.Patch(facecolor='#e41a1c', label='Proposed (Joint HRL)'),
    mpatches.Patch(facecolor=COLORS['PPO'], label='PPO120'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig7_macro_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig7_macro_trajectory.png")

# ============================================================
# Figure 3: Load Sensitivity
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

loads = ['0.8x', '1.0x\n(base)', '1.1x']
sac_J = [0.4050, 0.5532, 0.6575]
uniform_J = [0.4201, 0.5725, 0.6739]
cvx_J = [None, 0.5882, None]

x = np.arange(len(loads))
width = 0.25

bars1 = ax.bar(x - width, sac_J, width, label='SAC (Ours)', color='#e41a1c', edgecolor='black')
bars2 = ax.bar(x, uniform_J, width, label='Uniform', color=COLORS['Uniform'], edgecolor='black')

# CVX-R as scattered points
for i, val in enumerate(cvx_J):
    if val is not None:
        ax.scatter(i + width, val, s=100, c=COLORS['CVX'], marker='s', zorder=5, label='CVX-R' if i == 0 else '')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Traffic Load Multiplier', fontsize=11)
ax.set_ylabel('Physical Cost J', fontsize=11)
ax.set_title('Fig. 10: Load Sensitivity Analysis', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig10_load_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig10_load_sensitivity.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
