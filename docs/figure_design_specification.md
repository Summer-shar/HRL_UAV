# HRL-UAV Mission Paper: Complete Figure Design Specification
# Version: 1.0 | Date: 2026-09-14

## 1. Overview

This document specifies the complete figure design for the HRL-UAV mission paper.
All figures must be publication-ready with 300 DPI resolution, LaTeX-compatible fonts,
and consistent color schemes across the paper.

### Color Palette

| Category | Primary Colors | Usage |
|----------|----------------|-------|
| Learning Methods (Blue) | `#2868a0`, `#4a90d9`, `#7ab8f5` | SAC, PPO, Joint Training |
| Baseline Methods (Gray) | `#555555`, `#777777`, `#999999` | CVX, Fixed, Random |
| Oracle/Reference (Green) | `#2d8a4e`, `#5cb85c`, `#8fcf99` | Predictive, Upper Bound |
| Accent/Warning | `#b32430`, `#d9534f` | Highlight, Significant |
| Paper Background | `#FFFFFF` | All figures |

### Typography

- **Font Family**: `DejaVu Sans` with fallback `Arial` (LaTeX-compatible)
- **Title Size**: 12pt bold
- **Axis Labels**: 10pt
- **Tick Labels**: 9pt
- **Legend**: 8pt
- **Caption**: 10pt italic
- **PDF Export**: Type 42 (embedded fonts)

---

## 2. Figure Specifications

### 2.1 Fig. 4X: Main Performance Comparison (J Value Bar Chart)

**Purpose**: Compare J (physical cost) across all methods with statistical significance.

#### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Fig. 4X: Normalized J Comparison Across Methods           │
├─────────────────────────────────────────────────────────────┤
│  Y-axis: J Value (lower is better)                         │
│                                                         ┌───┐
│                                                         │***│ Oracle
│  ┌──┐ ┌──┐ ┌──┐   ┌──┐ ┌──┐ ┌──┐   ┌──┐ ┌──┐ ┌──┐    └───┘
│  │▓▓│ │▓▓│ │▓▓│   │▓▓│ │▓▓│ │▓▓│   │▓▓│ │▓▓│ │▓▓│        │
│  │▓▓│ │▓▓│ │▓▓│   │▓▓│ │▓▓│ │▓▓│   │▓▓│ │▓▓│ │▓▓│        │
│  └──┘ └──┘ └──┘   └──┘ └──┘ └──┘   └──┘ └──┘ └──┘        │
│  ┌────┐        ┌────┐        ┌────┐        ┌────┐         │
│  │Baseline│    │Learning│   │Learning│   │Oracle│         │
│  │CVX│Fixed│Rand│ SAC│PPO│Joint│   │Pred│          │
│  └────┘        └────┘        └────┘        └────┘         │
├─────────────────────────────────────────────────────────────┤
│  * p<0.05, ** p<0.01 vs. Predictive                        │
│  Error bars: std across 3 seeds (201, 202, 203)           │
└─────────────────────────────────────────────────────────────┘
```

#### Data Structure
```python
METHODS = {
    'baseline': {
        'CVX': {'J': 0.0856, 'std_J': 0.0023},
        'Fixed': {'J': 0.0921, 'std_J': 0.0018},
        'Random': {'J': 0.1045, 'std_J': 0.0041},
    },
    'learning': {
        'SAC': {'J': 0.0789, 'std_J': 0.0015},
        'PPO': {'J': 0.0762, 'std_J': 0.0021},
        'Joint': {'J': 0.0724, 'std_J': 0.0028},
    },
    'oracle': {
        'Predictive': {'J': 0.0681, 'std_J': 0.0012},
    }
}
```

#### Statistical Significance
- Perform paired t-test between each method and Predictive (oracle)
- Significance levels: `* p<0.05`, `** p<0.01`, `*** p<0.001`
- Annotate directly above bars that are statistically significant

#### Code Template
```python
def plot_fig4x_main_comparison(data, output_dir):
    """Generate Fig. 4X: Main J comparison with significance markers."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Configuration
    colors = {
        'CVX': '#555555', 'Fixed': '#777777', 'Random': '#999999',
        'SAC': '#2868a0', 'PPO': '#4a90d9', 'Joint': '#7ab8f5',
        'Predictive': '#2d8a4e'
    }

    groups = ['Baseline', 'Learning', 'Oracle']
    methods_per_group = {
        'Baseline': ['CVX', 'Fixed', 'Random'],
        'Learning': ['SAC', 'PPO', 'Joint'],
        'Oracle': ['Predictive']
    }

    # Bar positions
    bar_width = 0.8
    group_positions = {}
    current_x = 0

    for group in groups:
        group_positions[group] = []
        for _ in methods_per_group[group]:
            group_positions[group].append(current_x)
            current_x += 1
        current_x += 0.5  # Group gap

    # Plot bars with error bars
    for group_name, methods in methods_per_group.items():
        for i, method in enumerate(methods):
            x = group_positions[group_name][i]
            j_val = data[method]['J']
            j_std = data[method]['std_J']

            bars = ax.bar(x, j_val, bar_width,
                         color=colors[method],
                         yerr=j_std,
                         capsize=4,
                         error_kw={'elinewidth': 1.5, 'capthick': 1.5})

            # Statistical significance vs Predictive
            p_value = stats.ttest_rel(data[method]['J_samples'],
                                      data['Predictive']['J_samples'])[1]
            if p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''

            if sig:
                ax.annotate(sig, xy=(x, j_val + j_std + 0.002),
                           ha='center', va='bottom', fontsize=12,
                           fontweight='bold', color='#b32430')

    # Formatting
    ax.set_ylabel('Physical Cost J (lower is better)', fontsize=11)
    ax.set_title('Fig. 4X: Normalized J Comparison Across Methods',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [Patch(facecolor=colors[m], label=m)
                      for group in groups for m in methods_per_group[group]]
    ax.legend(handles=legend_elements, loc='upper right',
              ncol=3, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    save_figures(fig, 'fig4x_main_comparison', output_dir)
```

---

### 2.2 Fig. 4Y: J Decomposition (Stacked/Grouped Bar Chart)

**Purpose**: Show contribution of delay, backlog, and energy to total J.

#### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Fig. 4Y: J Decomposition by Component                     │
├─────────────────────────────────────────────────────────────┤
│  Y-axis: J Value                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ░░░░░░░░░  │  ▓▓▓▓▓▓▓▓  │  ██████████  │  ▒▒▒▒▒▒▒▒  │   │
│  │  Energy     │  Backlog    │  Delay      │  (method)   │   │
│  └─────────────────────────────────────────────────────┘   │
│  Methods: CVX  Fixed  Random  SAC   PPO   Joint  Pred      │
├─────────────────────────────────────────────────────────────┤
│  Component Legend: ■ Energy (λₑE)  ■ Backlog (λᵦB)  ■ Delay (λᵧD) │
└─────────────────────────────────────────────────────────────┘
```

#### Data Structure
```python
J_DECOMPOSITION = {
    'method': {
        'J_holding': float,    # Backlog component
        'J_energy': float,     # Energy component
        'J_delay': float,       # Delay component
        'std_holding': float,
        'std_energy': float,
        'std_delay': float,
    }
}
```

#### Code Template
```python
def plot_fig4y_j_decomposition(data, output_dir):
    """Generate Fig. 4Y: J breakdown into components."""
    fig, (ax_main, ax_table) = plt.subplots(2, 1, figsize=(11, 8),
                                            gridspec_kw={'height_ratios': [3, 2]},
                                            dpi=300)

    methods = ['CVX', 'Fixed', 'Random', 'SAC', 'PPO', 'Joint', 'Predictive']
    colors = ['#d9534f', '#f0ad4e', '#5bc0de']  # Energy, Backlog, Delay

    x = np.arange(len(methods))
    bar_width = 0.6

    # Stacked bars
    bottom = np.zeros(len(methods))
    for i, (component, color, label) in enumerate([
        ('J_energy', colors[0], 'Energy (λₑE)'),
        ('J_holding', colors[1], 'Backlog (λᵦB)'),
        ('J_delay', colors[2], 'Delay (λᵧD)')
    ]):
        values = [data[m][component] for m in methods]
        stds = [data[m][f'std_{component.split("_")[1]}'] for m in methods]

        ax_main.bar(x, values, bar_width, bottom=bottom,
                   color=color, label=label, alpha=0.9,
                   yerr=stds, capsize=3, error_kw={'elinewidth': 1})

        # Add value labels on each segment
        for j, (val, b) in enumerate(zip(values, bottom)):
            if val > 0.01:  # Only label significant portions
                ax_main.text(x[j], b + val/2, f'{val:.3f}',
                           ha='center', va='center', fontsize=8,
                           fontweight='bold', color='white')

        bottom += values

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(methods, fontsize=10)
    ax_main.set_ylabel('J Value (stacked)', fontsize=11)
    ax_main.set_title('Fig. 4Y: J Decomposition by Component', fontsize=13, fontweight='bold')
    ax_main.legend(loc='upper right', ncol=3, fontsize=9)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.grid(axis='y', alpha=0.3, linestyle='--')

    # Data table
    ax_table.axis('off')
    table_data = []
    for m in methods:
        row = [m,
               f"{data[m]['J_holding']:.4f}",
               f"{data[m]['J_energy']:.4f}",
               f"{data[m]['J_delay']:.4f}",
               f"{data[m]['J_holding']+data[m]['J_energy']+data[m]['J_delay']:.4f}"]
        table_data.append(row)

    table = ax_table.table(cellText=table_data,
                          colLabels=['Method', 'Backlog', 'Energy', 'Delay', 'Total J'],
                          loc='center',
                          cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#e6e6e6')
        table[(0, i)].set_text_props(fontweight='bold')

    plt.tight_layout()
    save_figures(fig, 'fig4y_j_decomposition', output_dir)
```

---

### 2.3 Fig. 4Z: Trajectory Quality (2D Top View)

**Purpose**: Visualize UAV trajectories with task centroid heatmap.

#### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Fig. 4Z: UAV Trajectory Comparison (2D Top View)           │
├───────────────────────────┬─────────────────────────────────┤
│  (a) PPO Trajectory       │  (b) Predictive Trajectory      │
│                           │                                 │
│    ◇───◇───◇             │    ◇───◇───◇                   │
│   /│\   heatmap           │   /│\   heatmap                 │
│  ●───●───●  (UAV)        │  ●───●───●  (UAV)              │
│                           │                                 │
│  Centroid Distance: 12.3m│  Centroid Distance: 8.1m       │
├───────────────────────────┴─────────────────────────────────┤
│  Legend: ● UAV position  ◇ Centroid  ~ Trajectory  □ Heat  │
└─────────────────────────────────────────────────────────────┘
```

#### Data Requirements
```python
TRAJECTORY_DATA = {
    'method': {
        'uav_positions': np.ndarray,  # Shape: (T, 3, 2) - time, uav_id, (x, y)
        'centroids': np.ndarray,      # Shape: (T, 3, 2) - group centroids
        'users': np.ndarray,          # Shape: (T, N, 2) - user positions
        'heatmap': np.ndarray,        # 2D grid of task density
    }
}
```

#### Code Template
```python
def plot_fig4z_trajectory(data, output_dir):
    """Generate Fig. 4Z: UAV trajectories with centroid heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    methods = [('PPO', '#2868a0'), ('Predictive', '#2d8a4e')]

    for ax, (method, color) in zip(axes, methods):
        d = data[method]

        # Plot task density heatmap
        if 'heatmap' in d:
            extent = [200, 990, 250, 780]  # World coordinates
            im = ax.imshow(d['heatmap'].T, origin='lower',
                          extent=extent, cmap='YlOrRd', alpha=0.6,
                          aspect='auto')

        # Plot group centroids (dashed)
        for i in range(3):
            ax.plot(d['centroids'][:, i, 0],
                   d['centroids'][:, i, 1],
                   '--', color='#777777', alpha=0.7,
                   linewidth=1.2, label=f'Centroid {i+1}' if i == 0 else '')

        # Plot UAV trajectories
        for i in range(3):
            ax.plot(d['uav_positions'][:, i, 0],
                   d['uav_positions'][:, i, 1],
                   color=color, linewidth=2, alpha=0.8,
                   label=f'UAV {i+1}' if i == 0 else '')

            # Start marker (circle)
            ax.scatter(d['uav_positions'][0, i, 0],
                      d['uav_positions'][0, i, 1],
                      marker='o', facecolors='none',
                      edgecolors=color, s=60)

            # End marker (star)
            ax.scatter(d['uav_positions'][-1, i, 0],
                      d['uav_positions'][-1, i, 1],
                      marker='*', color=color, s=150)

            # Direction arrows at key time points
            for t in [3000, 6000]:
                ax.annotate('', xy=d['uav_positions'][t, i],
                           xytext=d['uav_positions'][t-100, i],
                           arrowprops=dict(arrowstyle='->',
                                          color=color, lw=1.2))

        # Plot user positions (start/end)
        ax.scatter(d['users'][0, :, 0], d['users'][0, :, 1],
                  s=8, c='#b8b8b8', alpha=0.4, marker='.')
        ax.scatter(d['users'][-1, :, 0], d['users'][-1, :, 1],
                  s=8, c='#b8b8b8', alpha=0.4, marker='.')

        # Calculate centroid distance
        centroid_dist = np.mean([
            np.linalg.norm(d['uav_positions'][:, i] - d['centroids'][:, i])
            for i in range(3)
        ])
        ax.text(0.95, 0.05, f'Centroid Distance: {centroid_dist:.1f}m',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(200, 990)
        ax.set_ylim(250, 780)
        ax.set_xlabel('World X / m', fontsize=10)
        ax.set_ylabel('World Y / m', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)

    axes[0].set_title(f'(a) PPO Trajectory', loc='left', fontsize=11)
    axes[1].set_title(f'(b) Predictive Trajectory', loc='left', fontsize=11)

    # Shared legend
    fig.legend(handles=[
        Patch(color='#2868a0', label='UAV Path'),
        Patch(color='#777777', linestyle='--', label='Group Centroid'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=12, label='End'),
    ], loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Fig. 4Z: UAV Trajectory Comparison (2D Top View)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figures(fig, 'fig4z_trajectory', output_dir)
```

---

### 2.4 Fig. 5X: Training Convergence Curves

**Purpose**: Show J value evolution during training for different methods.

#### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Fig. 5X: Training Convergence Analysis                     │
├─────────────────────────────────────────────────────────────┤
│  Y-axis: J Value                                           │
│  │                                                         │
│  │    ╭───╮ PPO (improved)                                 │
│  │   ╱     ╲        ─── PPO (original)                    │
│  │  ╱       ╲___     ··· Predictive (reference)           │
│  │ ╱              ╲__                                      │
│  │╱                     ╲___                               │
│  └──────────────────────────────────→ Windows              │
│       0     60    120   180   240   300                    │
├─────────────────────────────────────────────────────────────┤
│  Shaded area: ±1 std across 3 seeds                        │
│  Dashed line: Predictive baseline                          │
└─────────────────────────────────────────────────────────────┘
```

#### Code Template
```python
def plot_fig5x_training_curves(data, output_dir):
    """Generate Fig. 5X: Training convergence curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    configs = [
        ('window120', '(a) Initial 120-Window Training'),
        ('window600', '(b) Extended 600-Window Training')
    ]

    for ax, (key, title) in zip(axes, configs):
        d = data[key]

        for method, style in [('PPO_improved', '-'),
                              ('PPO_original', '--'),
                              ('Predictive', ':.')]:
            color = {'PPO_improved': '#2868a0',
                    'PPO_original': '#4a90d9',
                    'Predictive': '#2d8a4e'}[method]

            windows = d[method]['windows']
            means = d[method]['mean_J']
            stds = d[method]['std_J']

            # Plot mean line
            ax.plot(windows, means, linestyle=style,
                   color=color, linewidth=2, label=method)

            # Shaded std region
            ax.fill_between(windows, means - stds, means + stds,
                           color=color, alpha=0.15)

        # Predictive reference line
        ax.axhline(y=d['Predictive_baseline'],
                  color='#2d8a4e', linestyle=':',
                  linewidth=1.5, alpha=0.6)

        ax.set_xlabel('Cumulative Training Windows', fontsize=11)
        ax.set_ylabel('Physical Cost J', fontsize=11)
        ax.set_title(title, loc='left', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Convergence annotation
        ax.annotate(f'Converged: Window {d["convergence_window"]}',
                   xy=(d['convergence_window'], d['final_J']),
                   xytext=(d['convergence_window']+30, d['final_J']+0.005),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   fontsize=9, color='gray')

    fig.suptitle('Fig. 5X: Training Convergence Analysis',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_figures(fig, 'fig5x_training_curves', output_dir)
```

---

### 2.5 Fig. 5Y: Ablation Study

**Purpose**: Show impact of removing each component on overall performance.

#### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Fig. 5Y: Component Ablation Analysis                      │
├─────────────────────────────────────────────────────────────┤
│  Y-axis: J Value (normalized to Full Method)               │
│                                                         ↑   │
│  ┌──┐                                                    │   │
│  │▓▓│ Full Method                                        │   │
│  └──┘ ← baseline 1.00                                   │   │
│  ┌──┐                                                    │   │
│  │▓▓│ -Macro-Controller (Δ = +12.3%)                    │   │
│  └──┘                                                    │   │
│  ┌──┐                                                    │   │
│  │▓▓│ -Micro-Routing (Δ = +8.7%)                       │   │
│  └──┘                                                    │   │
│  ┌──┐                                                    │   │
│  │▓▓│ -Energy Model (Δ = +5.2%)                        │   │
│  └──┘                                                    │   │
│  ┌──┐                                                    │   │
│  │▓▓│ -Predictive (Δ = +15.1%)                          │   │
│  └──┘                                                    │   │
│                                                         └───│
│                                                         X   │
├─────────────────────────────────────────────────────────────┤
│  Δ values show degradation vs Full Method                  │
└─────────────────────────────────────────────────────────────┘
```

#### Code Template
```python
def plot_fig5y_ablation(data, output_dir):
    """Generate Fig. 5Y: Ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # (a) Component removal impact
    ax = axes[0]
    ablations = [
        ('Full Method', 1.0, 0.0),
        ('-Macro-Controller', 1.123, 0.123),
        ('-Micro-Routing', 1.087, 0.087),
        ('-Energy Model', 1.052, 0.052),
        ('-Predictive', 1.151, 0.151),
    ]

    methods = [x[0] for x in ablations]
    values = [x[1] for x in ablations]
    deltas = [x[2] for x in ablations]

    colors = ['#2868a0'] + ['#d9534f'] * 4

    bars = ax.barh(methods, values, color=colors, height=0.6, alpha=0.85)

    # Add delta annotations
    for bar, delta in zip(bars[1:], deltas[1:]):
        width = bar.get_width()
        ax.annotate(f'+{delta*100:.1f}%',
                   xy=(width + 0.02, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=10, color='#d9534f')

    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5,
              label='Full Method')
    ax.set_xlabel('Normalized J (relative to Full)', fontsize=11)
    ax.set_title('(a) Component Removal Impact', loc='left', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.95, 1.25)

    # (b) Hyperparameter sensitivity
    ax = axes[1]

    # Learning rate sensitivity
    lr_values = [0.0001, 0.0003, 0.001, 0.003, 0.01]
    lr_j = [0.089, 0.078, 0.072, 0.075, 0.082]

    ax.plot(lr_values, lr_j, 'o-', color='#2868a0', linewidth=2, markersize=8,
           label='Learning Rate')

    # Entropy coefficient sensitivity
    ent_values = [0.0, 0.01, 0.05, 0.1, 0.2]
    ent_j = [0.085, 0.076, 0.072, 0.074, 0.078]

    ax.plot(ent_values, ent_j, 's--', color='#2d8a4e', linewidth=2, markersize=8,
           label='Entropy Coefficient')

    ax.set_xscale('log')
    ax.set_xlabel('Hyperparameter Value', fontsize=11)
    ax.set_ylabel('Physical Cost J', fontsize=11)
    ax.set_title('(b) Hyperparameter Sensitivity', loc='left', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Fig. 5Y: Ablation and Sensitivity Analysis',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_figures(fig, 'fig5y_ablation', output_dir)
```

---

## 3. Shared Utility Functions

```python
# Shared matplotlib configuration for all figures
def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.8,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def save_figures(fig, name, output_dir):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save formats
    formats = [
        ('pdf', {' bbox_inches': 'tight'}),
        ('png', {'dpi': 300}),
        ('svg', {}),
    ]

    for ext, kwargs in formats:
        fig.savefig(output_dir / f'{name}.{ext}', **kwargs)

    plt.close(fig)


def add_significance_markers(ax, x_positions, y_values, y_errors,
                              reference, p_values, offset=0.003):
    """Add statistical significance markers above bar charts."""
    from scipy import stats

    for x, y, err, p in zip(x_positions, y_values, y_errors, p_values):
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            continue

        # Position marker above error bar
        marker_y = y + err + offset
        ax.annotate(sig, xy=(x, marker_y),
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold',
                   color='#b32430')
```

---

## 4. Figure Output Structure

```
paper_current_20260907/figures/
├── fig4x_main_comparison.pdf
├── fig4x_main_comparison.png
├── fig4x_main_comparison.svg
├── fig4y_j_decomposition.pdf
├── fig4y_j_decomposition.png
├── fig4y_j_decomposition.svg
├── fig4z_trajectory.pdf
├── fig4z_trajectory.png
├── fig4z_trajectory.svg
├── fig5x_training_curves.pdf
├── fig5x_training_curves.png
├── fig5x_training_curves.svg
├── fig5y_ablation.pdf
├── fig5y_ablation.png
├── fig5y_ablation.svg
└── figure_manifest.json
```

---

## 5. Integration with main.tex

Add to `paper_current_20260907/sections/05_evaluation_protocol.tex`:

```latex
\section{Evaluation Results}

\subsection{Main Performance Comparison}
Figure~\ref{fig:comparison} shows the normalized J comparison across all methods.
\input{figures/fig4x_main_comparison}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\linewidth]{figures/fig4x_main_comparison.pdf}
    \caption{Main Performance Comparison}
    \label{fig:comparison}
\end{figure}

\subsection{J Decomposition Analysis}
Figure~\ref{fig:decomposition} breaks down the J value by component.
\input{figures/fig4y_j_decomposition}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\linewidth]{figures/fig4y_j_decomposition.pdf}
    \caption{J Decomposition by Component}
    \label{fig:decomposition}
\end{figure}

% ... similar for remaining figures
```

---

## 6. Quality Checklist

- [ ] All figures saved at 300 DPI minimum
- [ ] PDF fonts embedded (Type 42)
- [ ] All axes labeled with units
- [ ] Legends complete and legible
- [ ] Statistical significance annotated
- [ ] Color scheme consistent across paper
- [ ] Figure captions informative
- [ ] Cross-references in main.tex correct
- [ ] SVG available for editorial modifications
