"""Needs torch: p2.common.derive must equal hrl_v2.randomness.derive."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from p2 import common as C
from hrl_v2.randomness import derive, PURPOSES

rng = np.random.default_rng(0)
for _ in range(2000):
    purpose = list(PURPOSES)[rng.integers(len(PURPOSES))]
    args = dict(master=int(rng.integers(5000)), stage=int(rng.integers(5)), cycle=int(rng.integers(3)),
                episode=int(rng.integers(500)), purpose=purpose, stream=int(rng.integers(300)),
                split=['train', 'dev', 'test'][rng.integers(3)])
    assert C.derive(**args) == derive(**args), args
print('derive identical')
