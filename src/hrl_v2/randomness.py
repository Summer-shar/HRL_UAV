"""Named, checkpointable CPU random streams; no Python hash seed derivation."""
from contextlib import contextmanager
import numpy as np
import torch

PURPOSES = dict(layout=1, load=2, arrival=3, upper=4, support=5, ratio=6,
                replay=7, actor=8, q1=9, q2=10, upper_init=11, value=12)


def derive(master, stage=0, cycle=0, episode=0, purpose="arrival", stream=0, split="train"):
    external = purpose in ("layout", "load", "arrival") and split != "train"
    fields = [2026092602, dict(train=10, dev=20, test=30)[split], 0 if external else master,
              0 if split != "train" else stage, 0 if split != "train" else cycle,
              episode, PURPOSES[purpose], stream]
    return int(np.random.SeedSequence(fields).generate_state(1, dtype=np.uint64)[0]) % (2**63-1)


class Streams:
    def __init__(self, master, stage=0, cycle=0, episode=0, split="train", policy_stream=0):
        self.identity = dict(master=master, stage=stage, cycle=cycle, episode=episode, split=split)
        self.generators = {}
        for purpose, streams in (("upper", [policy_stream]), ("support", [policy_stream,100,200]),
                                 ("ratio", [policy_stream,100,200]), ("replay", [0,300,400])):
            for stream in set(streams):
                self.generators[purpose,stream] = torch.Generator(device="cpu").manual_seed(
                    derive(**self.identity, purpose=purpose, stream=stream))
        self.policy_stream = policy_stream
        self.numpy = np.random.default_rng(derive(**self.identity, purpose="replay"))

    def generator(self, purpose, stream=0):
        return self.generators[purpose, stream]

    @contextmanager
    def fork(self, purpose, stream=0):
        g = self.generator(purpose, stream)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(g.get_state())
            try:
                yield
            finally:
                g.set_state(torch.get_rng_state())

    def state_dict(self):
        return dict(identity=self.identity, policy_stream=self.policy_stream,
                    generators={k:g.get_state() for k,g in self.generators.items()},
                    numpy=self.numpy.bit_generator.state)

    def load_state_dict(self, state):
        if state["identity"] != self.identity or state["policy_stream"] != self.policy_stream:
            raise ValueError("Random stream identity mismatch")
        for key,value in state["generators"].items():
            self.generators[key].set_state(value.cpu())
        self.numpy.bit_generator.state = state["numpy"]


@contextmanager
def initialization(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield
