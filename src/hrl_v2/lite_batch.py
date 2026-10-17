"""Read collected vectors only. No Report/history/learned-feature cache."""
import numpy as np
import torch


class LiteBatch:
    def __init__(self, values, *, owner=None, indices=None):
        self._owner = owner
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.indices = np.arange(len(self.values)) if indices is None else np.asarray(indices, dtype=np.int64)
        if owner is None:
            self._tensors = {}

    @property
    def owner(self):
        return self if self._owner is None else self._owner

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.select(np.arange(len(self))[key])
        return self.owner.values[self.indices[key]]

    def select(self, indices):
        # Keep all input tensors owned by this optimizer step, never a network.
        obj = object.__new__(type(self))
        obj._owner = self.owner
        obj.values = self.owner.values
        obj.indices = self.indices[np.asarray(indices, dtype=np.int64)]
        return obj

    def tensor(self, device):
        key = str(device)
        if key not in self.owner._tensors:
            self.owner._tensors[key] = torch.as_tensor(self.owner.values, device=device)
        values = self.owner._tensors[key]
        return values[torch.as_tensor(self.indices, device=device)]

    @classmethod
    def replay(cls, samples):
        if not samples or any(getattr(s.episode, 'encoder', 'graph') != 'lite' for s in samples):
            raise ValueError('A Lite batch requires only Lite episodes')
        # One fixed assembly/transfer per fit. Sampling itself is unchanged.
        values = np.stack([s.episode.lower_obs[s.index + offset] for offset in (0, 1) for s in samples])
        both = cls(values)
        return both[:len(samples)], both[len(samples):]

    @classmethod
    def upper(cls, rows):
        if any(getattr(e, 'encoder', 'graph') != 'lite' for e, _, _ in rows):
            raise ValueError('A Lite PPO batch requires only Lite episodes')
        return cls(np.stack([e.upper_obs[h['obs_index']] for e, h, _ in rows]))
