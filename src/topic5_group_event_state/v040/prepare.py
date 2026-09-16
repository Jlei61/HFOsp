"""Device-resident tensors plus the frozen common measurement conditions."""
from __future__ import annotations
import numpy as np
import torch

from ..v0312.prepare import Prepared as BasePrepared, HISTORY_TAU_HOURS
from . import conditions as C
from . import data as D

__all__ = ['Prepared', 'HISTORY_TAU_HOURS']


class Prepared(BasePrepared):
    def __init__(self, payload, split, scaling, device, cond_scaling, dtype=torch.float32):
        super().__init__(payload, split, scaling, device, dtype)
        self.cond_scaling = cond_scaling
        self.last_observed = D.observed_support_end(payload)
        self._cond_cache = {}

    def query_conditions(self, queries, role='descriptive'):
        """C10: one interface, identical for every arm and every frozen consumer."""
        qs = np.asarray(queries, int)
        want = [q for q in qs if (role, int(q)) not in self._cond_cache]
        if want:
            raw = C.query_conditions(self.payload, self.split, np.asarray(want, int), self.cond_scaling,
                                     role, self.last_observed)
            for j, q in enumerate(want):
                self._cond_cache[(role, int(q))] = raw[j]
        stacked = np.stack([self._cond_cache[(role, int(q))] for q in qs]) if len(qs) else np.zeros((0, C.DIM), np.float32)
        return torch.as_tensor(stacked, dtype=self.stats.dtype, device=self.device)
