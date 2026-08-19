"""A compact CMA-ES (Hansen's (mu/mu_w, lambda) with rank-mu and step-size control).

Written in-tree rather than adding a dependency to an unattended multi-hour run.
It consumes only the ORDER of the fitness keys, so the Stage 2 objective can hand
it the lexicographic (n_dir, S_rank) tuple directly -- larger is better.
"""
from __future__ import annotations

import numpy as np


class CMAES:
    def __init__(self, x0, sigma0, seed=0, popsize=None):
        self.dim = int(np.asarray(x0, float).size)
        self.mean = np.asarray(x0, float).copy()
        self.sigma = float(sigma0)
        self.popsize = int(popsize or 4 + int(3 * np.log(self.dim)))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.generation = 0

        mu = self.popsize // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = w / w.sum()
        self.mu = mu
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        n = self.dim
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self._last = None
        self._B = None
        self._d = None

    def _decompose(self):
        """Eigen-decompose the current covariance; ``C`` is unchanged by ask."""
        C = np.triu(self.C) + np.triu(self.C, 1).T
        d, B = np.linalg.eigh(C)
        self._B, self._d = B, np.sqrt(np.maximum(d, 1e-20))

    # ---- sampling -------------------------------------------------------
    def ask(self):
        self._decompose()
        B, d = self._B, self._d
        z = self.rng.standard_normal((self.popsize, self.dim))
        y = z @ (B * d).T
        self._last = [self.mean + self.sigma * yi for yi in y]
        return [x.copy() for x in self._last]

    # ---- update ---------------------------------------------------------
    def tell(self, xs, keys):
        if len(xs) != self.popsize or len(keys) != self.popsize:
            raise ValueError(
                f"tell expects {self.popsize} candidates and keys, got "
                f"{len(xs)} and {len(keys)}")
        xs = [np.asarray(x, float) for x in xs]

        def sort_key(i):
            k = keys[i]
            k = k if isinstance(k, (tuple, list)) else (k,)
            # larger is better -> negate for ascending sort; non-finite sorts last
            return tuple(-v if np.isfinite(v) else np.inf for v in k)

        order = sorted(range(self.popsize), key=sort_key)
        sel = np.array([xs[i] for i in order[:self.mu]])

        old_mean = self.mean.copy()
        self.mean = self.weights @ sel

        n = self.dim
        if self._B is None:
            # A supervisor restart can replay a dispatched generation from its
            # pending file, so this process never called ask. C has not moved
            # since that ask, so decomposing it here reproduces the same basis.
            self._decompose()
        Cinv_sqrt = self._B @ np.diag(1.0 / self._d) @ self._B.T
        y = (self.mean - old_mean) / self.sigma
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinv_sqrt @ y))
        self.generation += 1
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
                / self.chiN) < (1.4 + 2 / (n + 1))
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y)

        ys = (sel - old_mean) / self.sigma
        rank_mu = (ys * self.weights[:, None]).T @ ys
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (not hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps)
                             * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))
        if not np.isfinite(self.C).all() or not np.isfinite(self.mean).all():
            raise FloatingPointError("CMA-ES state went non-finite")

    # ---- checkpointing --------------------------------------------------
    def get_state(self):
        return dict(dim=self.dim, mean=self.mean.tolist(), sigma=self.sigma,
                    popsize=self.popsize, seed=self.seed, generation=self.generation,
                    pc=self.pc.tolist(), ps=self.ps.tolist(), C=self.C.tolist(),
                    rng_state=self.rng.bit_generator.state)

    @classmethod
    def from_state(cls, st):
        es = cls(np.asarray(st["mean"], float), st["sigma"],
                 seed=st["seed"], popsize=st["popsize"])
        es.generation = int(st["generation"])
        es.pc = np.asarray(st["pc"], float)
        es.ps = np.asarray(st["ps"], float)
        es.C = np.asarray(st["C"], float)
        es.rng.bit_generator.state = st["rng_state"]
        return es
