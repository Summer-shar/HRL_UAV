import sys
sys.path.insert(0, "D:/Scientific/HRL_UAV")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path("D:/Scientific/HRL_UAV/paper_current_20260907/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load PPO trajectory
ppo_path = Path("D:/Scientific/HRL_UAV/results/mission_ppo_stage4a_qualification/seed201/monitor120/33600/trajectory.npz")
if ppo_path.exists():
    ppo_data = np.load(ppo_path)
    ppo_uav = ppo_data["uav"]  # (T, 3, 2)
    ppo_centroid = ppo_data["centroid"] if "centroid" in ppo_data else None
    print(f"PPO trajectory loaded: {ppo_uav.shape}")
else:
    print("PPO trajectory not found")
    ppo_uav = None
    ppo_centroid = None

# Colors
RED = '#e41a1c'
BLUE = '#2868a0'
GREEN = '#2d8a4e'
COLORS_UAV = ['#e41a1c', '#377eb8', '#4daf4a']

# ============================================================
# Figure: UAV Trajectory Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. PPO Trajectory
ax = axes[0]
if ppo_uav is not None:
    for i in range(3):
        ax.plot(ppo_uav[:, i, 0], ppo_uav[:, i, 1], 
                color=COLORS_UAV[i], linewidth=2, alpha=0.8,
                label=f'UAV-{i}')
        ax.scatter(ppo_uav[0, i, 0], ppo_uav[0, i, 1], 
                  color=COLORS_UAV[i], marker='o', s=50, zorder=5)
        ax.scatter(ppo_uav[-1, i, 0], ppo_uav[-1, i, 1], 
                  color=COLORS_UAV[i], marker='*', s=150, zorder=5)
    
    # Plot centroid trajectory
    if ppo_centroid is not None:
        for i in range(3):
            ax.plot(ppo_centroid[:, i, 0], ppo_centroid[:, i, 1], 
                   '--', color='gray', alpha=0.5, linewidth=1)

ax.set_xlim(200, 900)
ax.set_ylim(300, 750)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('(a) PPO120 Trajectory', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# 2. Hover (static) reference
ax = axes[1]
if ppo_uav is not None:
    for i in range(3):
        ax.scatter(ppo_uav[0, i, 0], ppo_uav[0, i, 1], 
                  color=COLORS_UAV[i], marker='s', s=100, 
                  label=f'UAV-{i} (static)')
ax.set_xlim(200, 900)
ax.set_ylim(300, 750)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('(b) Hover (No Movement)', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# 3. Movement Analysis
ax = axes[2]
if ppo_uav is not None:
    for i in range(3):
        dist_from_start = np.sqrt(
            (ppo_uav[:, i, 0] - ppo_uav[0, i, 0])**2 +
            (ppo_uav[:, i, 1] - ppo_uav[0, i, 1])**2
        )
        time_s = np.arange(len(dist_from_start)) * 0.5  # 0.5s per step
        ax.plot(time_s, dist_from_start, color=COLORS_UAV[i], 
                linewidth=1.5, alpha=0.8, label=f'UAV-{i}')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance from Start (m)')
ax.set_title('(c) UAV Movement Over Time', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, 900)

plt.suptitle('Fig. 8: UAV Trajectory Visualization (Scene 33600)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig8_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig8_trajectory.png")

# ============================================================
# Figure: Trajectory Statistics
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

if ppo_uav is not None:
    stats = []
    labels = []
    
    for i in range(3):
        # Total distance
        total_dist = np.sum(np.sqrt(
            np.diff(ppo_uav[:, i, 0])**2 + np.diff(ppo_uav[:, i, 1])**2
        ))
        # Max distance from start
        max_dist = np.max(np.sqrt(
            (ppo_uav[:, i, 0] - ppo_uav[0, i, 0])**2 +
            (ppo_uav[:, i, 1] - ppo_uav[0, i, 1])**2
        ))
        stats.append([total_dist, max_dist])
        labels.append(f'UAV-{i}')
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [s[0] for s in stats], width, label='Total Distance', color=BLUE)
    bars2 = ax.bar(x + width/2, [s[1] for s in stats], width, label='Max Distance', color=RED)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}m', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}m', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    ax.set_ylabel('Distance (m)')
    ax.set_title('PPO Trajectory Statistics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_trajectory_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig_trajectory_stats.png")

print(f"\nAll trajectory figures saved to {OUTPUT_DIR}")
