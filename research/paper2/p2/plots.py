"""Figures for the paper (matplotlib, torch-free). Writes PNG and PDF."""
from pathlib import Path

import numpy as np

from . import common as C
from .report import LABELS, load_rows, per_seed, sweep_table

COLORS = {'Proposed': '#d62728', 'TFOO': '#1f77b4', 'TOO': '#2ca02c', 'NC': '#9467bd',
          'HoverSAC': '#ff7f0e', 'CA': '#7f7f7f'}
MARKERS = {'Proposed': 'o', 'TFOO': 's', 'TOO': '^', 'NC': 'v', 'HoverSAC': 'D', 'CA': 'x'}
GROUP_COLORS = ('#4c72b0', '#55a868', '#c44e52')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3, 'savefig.dpi': 200})
    return plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'))
    fig.savefig(path.with_suffix('.pdf'))


def bars(root, seeds, methods, out):
    plt = _plt()
    rows = load_rows(root, seeds, methods, 'test')
    panels = [('truncated_delay_s', 'Average delay (s)', True), ('high_truncated_delay_s', 'High-priority delay (s)', False),
              ('energy_w_compute_comm', 'Compute + comm. power (W)', False), ('cost_per_s', 'Joint weighted cost (/s)', True)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.4))
    for ax, (key, title, log) in zip(axes, panels):
        means, stds = [], []
        for m in methods:
            v = np.asarray(list(per_seed(rows, m, key).values()) or [np.nan])
            means.append(v.mean())
            stds.append(v.std(ddof=1) if len(v) > 1 else 0.)
        bars_ = ax.bar(range(len(methods)), means, yerr=stds, capsize=3, color=[COLORS[m] for m in methods])
        for b, v in zip(bars_, means):
            ax.annotate(f'{v:.3g}', (b.get_x()+b.get_width()/2, v), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)
        if log:
            ax.set_yscale('log')
        elif key.startswith('energy'):
            lo = min(means)
            ax.set_ylim(lo*0.95, max(means)*1.03)   # zoomed axis; values are labelled on the bars
        ax.set_xticks(range(len(methods)), [LABELS[m].split(' (')[0] for m in methods], rotation=20)
        ax.set_title(title, fontsize=9)
    _save(fig, out/'fig_main_bars')
    plt.close(fig)


def trajectory(root, seed, methods, out, scenario=C.TRAJECTORY_SCENARIO):
    plt = _plt()
    shown = [m for m in methods if (C.seed_dir(root, seed)/'test'/m/'traj'/f'{scenario:02d}.npz').exists()]
    if not shown:
        return
    fig, axes = plt.subplots(1, len(shown), figsize=(3.8*len(shown), 3.9), squeeze=False)
    for ax, m in zip(axes[0], shown):
        t = np.load(C.seed_dir(root, seed)/'test'/m/'traj'/f'{scenario:02d}.npz')
        uav, user, src = t['uav_xy'], t['user_xy'], t['user_source']
        for k in range(user.shape[1]):
            ax.plot(user[:, k, 0], user[:, k, 1], color=GROUP_COLORS[src[k]], lw=.5, alpha=.35)
            ax.plot(*user[-1, k], '.', color=GROUP_COLORS[src[k]], ms=3, alpha=.6)
        for i in range(uav.shape[1]):
            ax.plot(uav[:, i, 0], uav[:, i, 1], color=GROUP_COLORS[i], lw=2.2)
            ax.plot(*uav[0, i], 'o', mfc='white', mec=GROUP_COLORS[i], ms=7, mew=2, clip_on=False, zorder=5)
            ax.plot(*uav[-1, i], '^', color=GROUP_COLORS[i], ms=8, mec='k', mew=.6, clip_on=False, zorder=5)
        ax.set_xlim(-30, 1030); ax.set_ylim(-30, 1030); ax.set_aspect('equal')
        ax.set_title(LABELS[m]); ax.set_xlabel('x (m)')
    axes[0][0].set_ylabel('y (m)')
    fig.suptitle(f'UAV trajectories (thick; ○ start, ▲ end) and user movement (thin), test scenario {scenario}, seed {seed}',
                 fontsize=9)
    _save(fig, out/'fig_trajectory')
    plt.close(fig)


def sweeps(root, seeds, methods, out, smoke=False):
    plt = _plt()
    d = C.design(smoke)
    xlabel = dict(cpu_ghz='UAV CPU frequency (GHz)', load='Task arrival rate (× nominal)')
    for param in d['sweeps']:
        table = sweep_table(root, seeds, methods, param, d)
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
        for ax, (key, title) in zip(axes, [('truncated_delay_s', 'Average delay (s)'),
                                           ('energy_w_compute_comm', 'Compute + comm. power (W)'),
                                           ('cost_per_s', 'Joint weighted cost (/s)')]):
            for m in methods:
                xs = [r['value'] for r in table if r['method'] == m and r.get(key) is not None]
                ys = [r[key] for r in table if r['method'] == m and r.get(key) is not None]
                ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=LABELS[m].split(' (')[0])
            values = list(d['sweeps'][param])
            ax.set_xticks(values, [f'{v:g}\n({C.utilization(param, v):.2f})' for v in values])
            ax.set_xlabel(xlabel[param]+'\n(mean demand / total CPU capacity)', fontsize=9); ax.set_title(title)
            if key in ('truncated_delay_s', 'cost_per_s'):
                ax.set_yscale('log')
        axes[0].legend(fontsize=8)
        _save(fig, out/f'fig_sweep_{param}')
        plt.close(fig)


STAGE_COLORS = {'M2': '#d62728', 'TFOO': '#1f77b4', 'TOO': '#2ca02c', 'NC': '#9467bd',
                'M1': '#7f7f7f', 'M3': '#d62728', 'HoverSAC': '#ff7f0e'}


def convergence(root, seeds, out, window=10):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for ax, stages, title in ((axes[0], ('M2', 'TFOO', 'TOO', 'NC'), 'Upper PPO training'),
                              (axes[1], ('M1', 'M3', 'HoverSAC'), 'Lower SAC training')):
        for stage in stages:
            curves = []
            for seed in seeds:
                rows = C.read_jsonl(C.run_dir(root, seed, stage)/'train.jsonl')
                if rows:
                    curves.append([r['truncated_delay_s'] for r in sorted(rows, key=lambda r: r['episode'])])
            if not curves:
                continue
            n = min(map(len, curves))
            y = np.asarray([c[:n] for c in curves]).mean(axis=0)
            x = np.arange(1, n+1)
            if n >= window:
                y = np.convolve(y, np.ones(window)/window, mode='valid')
                x = np.arange(window, n+1)        # trailing window: point at its last episode
            name = dict(M2='Proposed upper (M2)', M1='M1 pre-train (random deployments)', M3='M3 fine-tune',
                        HoverSAC='Hover-SAC fine-tune').get(stage, stage)
            ax.plot(x, y, label=name, color=STAGE_COLORS[stage])
        ax.set_xlabel('training episode (900 s each)')
        ax.set_ylabel('average delay (s)')
        ax.set_title(f'{title}\n(mean over seeds, {window}-episode trailing average)', fontsize=9)
        ax.set_yscale('log'); ax.legend(fontsize=8)
    _save(fig, out/'fig_convergence')
    plt.close(fig)


def cooperation(root, seeds, methods, out):
    plt = _plt()
    rows = load_rows(root, seeds, methods, 'test')

    def imbalance(r, key):
        v = r.get(key)
        return None if not v else max(v)-min(v)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (fn, title) in zip(axes, [(lambda r: r.get('offloaded_fraction'), 'Offloaded share of computation'),
                                      (lambda r: imbalance(r, 'cycles_by_destination'),
                                       'Computing-load imbalance (max − min UAV share)')]):
        vals = []
        for m in methods:
            v = [fn(r) for r in rows if r['method'] == m and fn(r) is not None]
            vals.append(np.mean(v) if v else np.nan)
        ax.bar(range(len(methods)), vals, color=[COLORS[m] for m in methods])
        ax.set_xticks(range(len(methods)), [LABELS[m].split(' (')[0] for m in methods], rotation=20)
        ax.set_title(title, fontsize=9)
    src = [imbalance(r, 'cycles_by_source') for r in rows if imbalance(r, 'cycles_by_source') is not None]
    if src:
        axes[1].axhline(np.mean(src), color='k', ls='--', lw=1, label='arrival imbalance')
        axes[1].legend(fontsize=8)
    _save(fig, out/'fig_cooperation')
    plt.close(fig)


def build(root, seeds, smoke=False, methods=C.METHODS):
    out = Path(root)/'results'/'figures'
    rows = load_rows(root, seeds, methods, 'test')
    present = [m for m in methods if any(r['method'] == m for r in rows)]
    bars(root, seeds, present, out)
    trajectory(root, seeds[0], [m for m in ('Proposed', 'TFOO', 'HoverSAC') if m in present], out)
    sweeps(root, seeds, present, out, smoke)
    convergence(root, seeds, out)
    cooperation(root, seeds, present, out)
    return out
