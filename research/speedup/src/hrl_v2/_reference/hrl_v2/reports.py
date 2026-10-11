"""Immutable causal service reports; identifiers are relational keys only."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Terminal:
    key: int
    source: int
    position: tuple
    rate: float
    upload: tuple  # priority x (remaining bits, full cycles, count, oldest age)


@dataclass(frozen=True)
class Parent:
    key: int
    terminal: int
    source: int
    priority: int
    birth: float
    bits: float
    cycles: float
    upload_remaining: float
    stage: str
    routed_at: Optional[float]
    children: tuple
    completed_children: int


@dataclass(frozen=True)
class Fragment:
    key: int
    parent: int
    source: int
    destination: int
    priority: int
    ratio: float
    bits: float
    cycles: float
    transfer_remaining: float
    compute_remaining: float
    stage: str
    entered: float


@dataclass(frozen=True)
class Queue:
    kind: str
    owner: tuple
    items: tuple


@dataclass(frozen=True)
class UAV:
    key: int
    position: tuple
    velocity: tuple
    target: tuple
    cpu_hz: float
    compute: tuple
    in_transit: tuple
    recent_service: tuple
    ready: tuple


@dataclass(frozen=True)
class Link:
    source: int
    destination: int
    rate: float
    all_active_rate: float
    power: float
    all_active_power: float
    active: bool
    bits: tuple
    cycles: tuple


@dataclass(frozen=True)
class ReportV2:
    tick: int
    remaining_ticks: int
    phase: str
    remaining_macro: float
    remaining_adjustment: float
    history_cursor: int
    map_size: float
    max_speed: float
    terminals: Tuple[Terminal, ...]
    parents: Tuple[Parent, ...]
    fragments: Tuple[Fragment, ...]
    uavs: Tuple[UAV, ...]
    links: Tuple[Link, ...]
    queues: Tuple[Queue, ...]

    def __post_init__(self):
        if len(self.terminals) != 39 or len(self.uavs) != 3 or len(self.queues) != 58:
            raise ValueError("Report must preserve 39 terminals, 3 UAVs and all 58 queues")
        if self.tick + self.remaining_ticks != 9000 or not 0 <= self.tick <= 9000:
            raise ValueError("Invalid finite-horizon report clock")
        if self.phase not in ("uncommitted", "adjustment", "hold"):
            raise ValueError("Unknown macro phase")

    @property
    def time(self):
        return self.tick * 0.1

    @property
    def severe(self):
        return (sum(u.ready[1][1] for u in self.uavs) > 0 and
                max(u.compute[1] / u.cpu_hz for u in self.uavs) >= 5)
