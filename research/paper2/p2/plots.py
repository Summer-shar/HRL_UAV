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


def trajectory_relative(root, seed, methods, out, scenario=C.TRAJECTORY_SCENARIO, smooth=12):
    """Co-moving frame: each UAV relative to the centroid of its own users.
    Removes the common user drift. Thin line: offset path (60 s moving average);
    solid arrow: mean offset after the first 60 s; dashed arrow: direction to the
    other two user groups (where cooperation partners are). Aligned arrows mean
    the UAV sits on the side of its users facing its partners."""
    plt = _plt()
    shown = [m for m in methods if (C.seed_dir(root, seed)/'test'/m/'traj'/f'{scenario:02d}.npz').exists()]
    if not shown:
        return
    fig, axes = plt.subplots(1, len(shown), figsize=(3.8*len(shown), 4.0), squeeze=False, sharex=True, sharey=True)
    lim = 10.
    for ax, m in zip(axes[0], shown):
        t = np.load(C.seed_dir(root, seed)/'test'/m/'traj'/f'{scenario:02d}.npz')
        time, uav, user, src = t['time_s'], t['uav_xy'], t['user_xy'], t['user_source']
        cent = np.stack([user[:, src == k].mean(axis=1) for k in range(uav.shape[1])], axis=1)
        keep = time >= min(60., time[-1]/2)
        for k in range(uav.shape[1]):
            rel = uav[:, k]-cent[:, k]
            if len(rel) > smooth:
                kernel = np.ones(smooth)/smooth
                rel_s = np.stack([np.convolve(rel[:, i], kernel, mode='valid') for i in range(2)], axis=1)
            else:
                rel_s = rel
            ax.plot(rel_s[:, 0], rel_s[:, 1], color=GROUP_COLORS[k], lw=.9, alpha=.45)
            mean = rel[keep].mean(axis=0)
            others = np.mean([cent[keep][:, j] for j in range(uav.shape[1]) if j != k], axis=0)-cent[keep][:, k]
            direction = others.mean(axis=0)
            direction = direction/max(np.linalg.norm(direction), 1e-9)
            scale = max(np.linalg.norm(mean), 5.)
            ax.annotate('', xy=tuple(direction*scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=GROUP_COLORS[k], lw=1.2, ls='--'))
            ax.annotate('', xy=tuple(mean), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=GROUP_COLORS[k], lw=2.4))
            lim = max(lim, np.abs(rel_s).max()*1.1, scale*1.2)
        ax.axhline(0, color='k', lw=.5); ax.axvline(0, color='k', lw=.5)
        ax.set_aspect('equal'); ax.set_title(LABELS[m], fontsize=10); ax.set_xlabel('x offset from own users (m)')
    for ax in axes[0]:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    axes[0][0].set_ylabel('y offset from own users (m)')
    fig.suptitle(f'UAV offset from the centroid of its own users; solid arrow = mean offset, dashed = direction to '
                 f'partner groups (test scenario {scenario}, seed {seed})', fontsize=8.5)
    _save(fig, out/'fig_trajectory_relative')
    plt.close(fig)


def positioning(root, seeds, methods, out):
    from .positioning import collect, summary
    plt = _plt()
    moving = [m for m in methods if C.METHOD_SPEC[m]['upper'] == 'ppo']
    rows = collect(root, seeds, moving)
    if not rows:
        return None
    summ = summary(rows, moving)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4))
    for m in moving:
        rs = [r for r in rows if r['method'] == m]
        n = min(len(r['time_s']) for r in rs)
        t = rs[0]['time_s'][:n]
        axes[0].plot(t, np.mean([r['spread_ratio'][:n] for r in rs], axis=0), color=COLORS[m], label=LABELS[m].split(' (')[0])
        axes[1].plot(t, np.mean([r['tracking_m'][:n] for r in rs], axis=0), color=COLORS[m])
    axes[0].axhline(1, color='k', ls='--', lw=1)
    axes[0].set_title('UAV spread / user-group spread', fontsize=9); axes[0].set_xlabel('time (s)')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Distance UAV ↔ own users\' centroid (m)', fontsize=9); axes[1].set_xlabel('time (s)')
    keys = [('lean_light_m', 'light-load UAV toward heavy'), ('lean_heavy_m', 'heavy-load UAV toward light')]
    width = .8/len(moving)
    for i, m in enumerate(moving):
        vals = [summ[m].get(k, np.nan) for k, _ in keys]
        axes[2].bar(np.arange(len(keys))+i*width, vals, width, color=COLORS[m], label=LABELS[m].split(' (')[0])
    axes[2].axhline(0, color='k', lw=.8)
    axes[2].set_xticks(np.arange(len(keys))+width*(len(moving)-1)/2, [lab for _, lab in keys], fontsize=8)
    axes[2].set_title('Offset toward partner region (m, >0 = leans toward it)', fontsize=9)
    _save(fig, out/'fig_positioning')
    plt.close(fig)
    C.write_json(out.parent/'positioning.json', C.clean(dict(warmup_s=60, by_method=summ)))
    return summ


def build(root, seeds, smoke=False, methods=C.METHODS):
    out = Path(root)/'results'/'figures'
    rows = load_rows(root, seeds, methods, 'test')
    present = [m for m in methods if any(r['method'] == m for r in rows)]
    bars(root, seeds, present, out)
    trajectory(root, seeds[0], [m for m in ('Proposed', 'TFOO', 'HoverSAC') if m in present], out)
    trajectory_relative(root, seeds[0], [m for m in ('Proposed', 'TFOO', 'TOO') if m in present], out)
    positioning(root, seeds, present, out)
    sweeps(root, seeds, present, out, smoke)
    convergence(root, seeds, out)
    cooperation(root, seeds, present, out)
    return out
