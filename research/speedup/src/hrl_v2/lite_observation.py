"""110/729-dimensional causal summaries, computed at collection.

v2.5.0: the upper vector has 3 x K terminal rows (K = 13 gives the original
729 dimensions); the lower vector is unchanged (source-level aggregates)."""
from dataclasses import dataclass
import numpy as np
from .features import L

TAUS = (5., 60.)
LOWER_DT = .5
LOWER_SEGMENTS = dict(demand=36, cpu=18, transfer=24, links=12, parents=6, uav=12, clocks=2)
LOWER_DIM = sum(LOWER_SEGMENTS.values())
GLOBAL_DIM = 66
TERMINAL_DIM = 17
UPPER_DIM = 39 * TERMINAL_DIM + GLOBAL_DIM      # K = 13 terminals per region


def upper_dim(terminals):
    """Upper observation width for a scene with `terminals` = 3 x K terminals."""
    return terminals * TERMINAL_DIM + GLOBAL_DIM


def terminals_in_upper(width):
    """Inverse of upper_dim; raises unless the width encodes 3 x K terminal rows."""
    n, rest = divmod(int(width) - GLOBAL_DIM, TERMINAL_DIM)
    if rest or n < 3 or n % 3:
        raise ValueError(f'{width} is not a Lite upper observation width')
    return n
SLICES = {}
_offset = 0
for _name, _width in LOWER_SEGMENTS.items():
    SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width


@dataclass(frozen=True)
class LiteObservation:
    lower: np.ndarray
    upper: np.ndarray


class LiteObserver:
    def __init__(self, history=None):
        self.reset(history)

    def reset(self, history=None):
        self.history = history
        self.cursor = 0
        # terminal, priority, workload (bits/cycles), decay time (5/60 s);
        # sized from the first report (3 x K terminals).
        self.ema = None
        self._last_report = None
        self.last = None

    def update(self, report, history=None):
        if history is not None:
            if self.history is not None and history is not self.history:
                raise ValueError('A new arrival history requires an explicit observer reset')
            self.history = history
        history = self.history
        cursor = report.history_cursor
        if history is None or not self.cursor <= cursor <= len(history):
            raise ValueError('Lite observer requires a monotone causal arrival cursor')
        if self._last_report is report:
            return self.last
        if self._last_report is not None and report.tick < self._last_report.tick:
            raise ValueError('A new physical episode requires an observer reset')
        n = len(report.terminals)
        if self.ema is None:
            self.ema = np.zeros((n, 2, 2, 2), dtype=np.float64)
        elif self.ema.shape[0] != n:
            raise ValueError('The number of terminals changed within an episode')
        decay = np.exp(-LOWER_DT / np.asarray(TAUS))[None, None, None, :]
        for frame in history[self.cursor:cursor]:
            rate = np.asarray(frame, dtype=np.float64).reshape(n, 2, 3)[:, :, :2] / LOWER_DT
            if not np.isfinite(rate).all() or (rate < 0).any():
                raise ValueError('Invalid observed arrival increment')
            self.ema *= decay
            self.ema += (1 - decay) * rate[:, :, :, None]
        self.cursor = cursor
        raw = physical_summaries(report)
        lower, upper = vectors(report, raw, self.ema)
        lower.setflags(write=False)
        upper.setflags(write=False)
        self.last = LiteObservation(lower, upper)
        self._last_report = report
        return self.last


def physical_summaries(r):
    """Unnormalized quantities for conservation audits; priority then source."""
    terminals = sorted(r.terminals, key=lambda t: t.key)
    k = len(terminals) // 3
    if len(terminals) != 3 * k or any(t.key != i or t.source != i // k for i, t in enumerate(terminals)):
        raise ValueError('Lite requires K fixed terminals per source, numbered source by source')
    terminal_upload = np.asarray([t.upload for t in terminals], dtype=np.float64)
    ready = np.zeros((2, 3, 3))
    upload = terminal_upload[:, :, 0].reshape(3, k, 2).sum(1).T
    parents = np.zeros((2, 3))
    for p in r.parents:
        parents[p.priority, p.source] += 1
        if p.stage == 'ready':
            ready[p.priority, p.source] += (p.bits, p.cycles, 1)
    fragments = {f.key: f for f in r.fragments}
    cpu = np.zeros((2, 3, 3))
    transfer = np.zeros((2, 6, 2))
    directed = [(i, j) for i in range(3) for j in range(3) if i != j]
    for q in r.queues:
        if q.kind == 'cpu':
            items = [fragments[key] for key in q.items]
            # Existing graph age convention: time since this queue was entered.
            cpu[q.owner] = (sum(f.compute_remaining for f in items), len(items),
                            max((r.time - f.entered for f in items), default=0.))
        elif q.kind == 'transfer':
            transfer[q.owner[0], directed.index(q.owner[1:])] = (
                sum(fragments[key].transfer_remaining for key in q.items), len(q.items))
    return dict(ready=ready, upload=upload, cpu=cpu, transfer=transfer,
                parents=parents, terminal_upload=terminal_upload, directed=directed)


def vectors(r, raw, ema):
    n = len(r.terminals)
    k = n // 3
    source_ema = ema[:, :, 1, :].reshape(3, k, 2, 2).sum(1).transpose(1, 0, 2)
    demand = np.concatenate((raw['ready'], raw['upload'][:, :, None], source_ema), axis=-1)
    links = {(link.source, link.destination): link for link in r.links}
    current = L([links[pair].rate for pair in raw['directed']], 1e6)
    potential = L([links[pair].all_active_rate for pair in raw['directed']], 1e6)
    geometry = np.asarray([[*np.asarray(u.position) / r.map_size,
                            *np.asarray(u.target) / r.map_size] for u in r.uavs])
    global_part = np.concatenate((L(raw['cpu'], [1e9, 1, 10]).ravel(),
        L(raw['transfer'], [1e6, 1]).ravel(), current, potential,
        L(raw['parents']).ravel(), geometry.ravel(), L([r.remaining_macro, r.remaining_adjustment])))
    lower = np.r_[L(demand, [1e6, 1e9, 1, 1e6, 1e9, 1e9]).ravel(), global_part].astype(np.float32)
    terminals = sorted(r.terminals, key=lambda t: t.key)
    positions = np.asarray([t.position for t in terminals])
    relative = (positions - np.asarray([u.position for u in r.uavs])[np.arange(n)//k]) / r.map_size
    terminal = np.concatenate((relative, L([t.rate for t in terminals], 1e6)[:, None],
        L(raw['terminal_upload'][:, :, [0, 2, 3]], [1e6, 1, 10]).reshape(n, 6),
        L(ema, np.asarray([1e6, 1e9])[:, None]).reshape(n, 8)), axis=-1)
    upper_global = np.concatenate((geometry[:, :2].ravel(),
        L(raw['cpu'], [1e9, 1, 10]).ravel(), L(raw['ready'], [1e6, 1e9, 1]).ravel(),
        L(raw['transfer'][:, :, 0], 1e6).ravel(), potential, L(raw['parents']).ravel()))
    upper = np.r_[terminal.ravel(), upper_global].astype(np.float32)
    if lower.shape != (LOWER_DIM,) or upper.shape != (upper_dim(n),) or not np.isfinite(upper).all() or not np.isfinite(lower).all():
        raise ValueError('Invalid Lite observation shape or value')
    return lower, upper
