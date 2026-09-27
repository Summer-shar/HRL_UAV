"""Independent physical/control clocks and an artificial collection window."""
from dataclasses import asdict, dataclass
import json
import math

from .paths import default_config


def default_protocol():
    return json.loads(default_config().read_text(encoding="utf-8"))


def _count(duration, interval, label):
    ratio = duration / interval
    value = round(ratio)
    if value < 1 or not math.isclose(ratio, value, rel_tol=0., abs_tol=1e-9):
        raise ValueError(label + " must be an integer multiple of its interval")
    return int(value)


@dataclass(frozen=True)
class ClockConfig:
    physical_dt: float
    lower_interval: float
    upper_interval: float
    rollout_seconds: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        # Require every collection boundary to be a control boundary. This is
        # data segmentation, never a service deadline or terminal condition.
        for name in ("physical_per_lower", "physical_per_upper", "lower_per_upper",
                     "physical_steps", "lower_steps", "upper_steps"):
            getattr(self, name)

    @classmethod
    def from_config(cls, config=None):
        task = (default_protocol() if config is None else config)["task"]
        return cls(**{key: task[key] for key in
                     ("physical_dt", "lower_interval", "upper_interval", "rollout_seconds")})

    @classmethod
    def from_scene(cls, scene, rollout_seconds=None):
        default = cls.from_config()
        base = scene["base_scene"]
        return cls(base["time"]["physical_s"], base["time"]["routing_s"],
                   scene["deployment"]["macro_period_s"],
                   default.rollout_seconds if rollout_seconds is None else rollout_seconds)

    @property
    def physical_per_lower(self):
        return _count(self.lower_interval, self.physical_dt, "lower_interval")

    @property
    def physical_per_upper(self):
        return _count(self.upper_interval, self.physical_dt, "upper_interval")

    @property
    def lower_per_upper(self):
        return _count(self.upper_interval, self.lower_interval, "upper_interval")

    @property
    def physical_steps(self):
        return _count(self.rollout_seconds, self.physical_dt, "rollout_seconds / physical_dt")

    @property
    def lower_steps(self):
        return _count(self.rollout_seconds, self.lower_interval, "rollout_seconds / lower_interval")

    @property
    def upper_steps(self):
        return _count(self.rollout_seconds, self.upper_interval, "rollout_seconds / upper_interval")

    def counts(self):
        return dict(physical_steps=self.physical_steps, lower_steps=self.lower_steps,
                    upper_steps=self.upper_steps)
