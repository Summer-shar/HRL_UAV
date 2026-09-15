#!/usr/bin/env python3
"""Generate UAV trajectory comparison plots."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_trajectory(path):
    """Load trajectory from npz file."""
    return np.load(path)

def plot_trajectory_comparison(output_path):
    """Plot trajectories from different policies."""

    # Data paths
    data_dir = Path('results/mission_ppo_continue120to600/seed201/monitor600')
    ppo_33600 = data_dir / '33600' / 'trajectory.npz'
    ppo_33601 = data_dir / '33601' / 'trajectory.npz'

    if not ppo_33600.exists():
        print(f"Data not found: {ppo_33600}")
        return

    # Load PPO trajectory
    traj_ppo = load_trajectory(ppo_33600)
    time_ppo = traj_ppo['time_s']
    uav_ppo = traj_ppo['uav']  # (time, uav, xy)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Colors for UAVs
    colors = ['#e41a1c', '#377eb8', '#4daf4a']  # red, blue, green
    uav_labels = ['UAV-0 (heavy)', 'UAV-1 (medium)', 'UAV-2 (light)']

    # ===== Plot 1: PPO Trajectory (full scene) =====
    ax1 = axes[0, 0]
    for uav_id in range(3):
        positions = uav_ppo[:, uav_id, :]  # (time, xy)
        ax1.plot(positions[:, 0], positions[:, 1],
                color=colors[uav_id], alpha=0.8, linewidth=1.5,
                label=uav_labels[uav_id])
        # Mark start and end
        ax1.scatter(positions[0, 0], positions[0, 1],
                   color=colors[uav_id], marker='o', s=50, zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1],
                   color=colors[uav_id], marker='x', s=80, zorder=5)

    # Add centroid trajectory
    centroid = traj_ppo['centroid'][:, 0, :]  # First timestep centroid
    ax1.scatter(centroid[0], centroid[1], color='gray', marker='*', s=200,
               label='Centroid start', zorder=6, alpha=0.5)

    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Y position (m)')
    ax1.set_title('PPO Trajectory (Scene 33600)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ===== Plot 2: PPO vs Hover comparison =====
    ax2 = axes[0, 1]

    # PPO trajectory
    for uav_id in range(3):
        positions = uav_ppo[:, uav_id, :]
        ax2.plot(positions[:, 0], positions[:, 1],
                color=colors[uav_id], alpha=0.8, linewidth=2,
                linestyle='-', label=f'{uav_labels[uav_id]} (PPO)')

    # Hover trajectory (stays at initial position)
    initial_pos = uav_ppo[0]  # (3, 2) - initial positions
    for uav_id in range(3):
        ax2.scatter(initial_pos[uav_id, 0], initial_pos[uav_id, 1],
                   color=colors[uav_id], marker='s', s=30, alpha=0.5,
                   label=f'{uav_labels[uav_id]} (Hover)')

    ax2.set_xlabel('X position (m)')
    ax2.set_ylabel('Y position (m)')
    ax2.set_title('PPO vs Hover (No Movement)')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ===== Plot 3: UAV movement over time =====
    ax3 = axes[1, 0]
    for uav_id in range(3):
        positions = uav_ppo[:, uav_id, :]
        # Calculate distance from initial position
        dist_from_start = np.sqrt(
            (positions[:, 0] - positions[0, 0])**2 +
            (positions[:, 1] - positions[0, 1])**2
        )
        ax3.plot(time_ppo, dist_from_start,
                color=colors[uav_id], alpha=0.8, linewidth=1.5,
                label=uav_labels[uav_id])

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance from start (m)')
    ax3.set_title('UAV Movement Over Time')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Summary statistics =====
    ax4 = axes[1, 1]

    # Calculate statistics
    stats = []
    for uav_id in range(3):
        positions = uav_ppo[:, uav_id, :]
        total_dist = np.sum(np.sqrt(
            np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2
        ))
        max_dist = np.max(np.sqrt(
            (positions[:, 0] - positions[0, 0])**2 +
            (positions[:, 1] - positions[0, 1])**2
        ))
        stats.append({
            'uav': uav_labels[uav_id].split(' ')[0],
            'total_dist': total_dist,
            'max_dist': max_dist
        })

    # Bar chart
    x = np.arange(3)
    width = 0.35
    total_dists = [s['total_dist'] for s in stats]
    max_dists = [s['max_dist'] for s in stats]

    bars1 = ax4.bar(x - width/2, total_dists, width, label='Total Distance', color='steelblue')
    bars2 = ax4.bar(x + width/2, max_dists, width, label='Max Distance', color='coral')

    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Trajectory Statistics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['UAV-0', 'UAV-1', 'UAV-2'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_trajectory_evolution(output_path):
    """Plot trajectory evolution across training."""

    data_dir = Path('results/mission_ppo_continue120to600/seed201')

    # Load trajectories at different training stages
    stages = [
        ('monitor240', '240 windows', 'blue'),
        ('monitor360', '360 windows', 'green'),
        ('monitor480', '480 windows', 'orange'),
        ('monitor600', '600 windows', 'red'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for idx, (monitor, label, color) in enumerate(stages):
        ax = axes[idx // 2, idx % 2]

        traj_path = data_dir / monitor / '33600' / 'trajectory.npz'
        if not traj_path.exists():
            ax.text(0.5, 0.5, f'Data not found: {monitor}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        traj = np.load(traj_path)
        uav = traj['uav']
        time = traj['time_s']

        for uav_id in range(3):
            positions = uav[:, uav_id, :]
            ax.plot(positions[:, 0], positions[:, 1],
                   color=colors[uav_id], alpha=0.8, linewidth=1.5,
                   label=f'UAV-{uav_id}')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{label} - Scene 33600')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    output_dir = Path('results/figures')
    output_dir.mkdir(exist_ok=True)

    print("Generating trajectory comparison plots...")
    plot_trajectory_comparison(output_dir / 'trajectory_comparison.png')
    plot_trajectory_evolution(output_dir / 'trajectory_evolution.png')
    print("Done!")
