"""Topic 4 MZ slow–fast dynamical transition — pure, import-safe, testable functions.

Design contract (BINDING):
  docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md

Tier = model-side mechanism analysis. Every phenotype is a *detection label*; we test whether a frozen fast
system crosses a repeatable OPERATIONAL-runaway boundary (120 Hz / 100 ms). NOT seizure validation.

This module is SIDE-EFFECT-FREE (no sims, no file writes — those live in
scripts/run_topic4_mz_slow_fast_transition.py). It holds only pure helpers:

  branch_rng_state   — deterministic independent PCG64 future-noise branch state (P_runaway, design §3.1)
  wilson_ci          — Wilson score interval for the P_runaway proportion
  recovery_time      — fast-rate return-to-band time after a subthreshold pulse (design §3.3)
  state_step_schedule / matched_d_times — checkpoint step indices (design §2)
  classify_transition — result-neutral 5-label transition classifier (design §5)

Simulation primitives (MZOnsetProbe, run_loop checkpoint/resume, score_runaway, epsilon_c_from_ladder) are
REUSED from src.topic4_mz_onset_dynamics by the runner — NOT reimplemented here, NO engine edits.
"""
from __future__ import annotations

import hashlib

import numpy as np

SCHEMA_VERSION = "mz-slow-fast-transition-1.0"


# ============================================================ P_runaway replay branches (design §3.1)
def branch_rng_state(seed, cond, state, idx):
    """A PCG64 ``bit_generator.state`` dict for one independent future-noise replay branch.

    Deterministic in ``(seed, cond, state, idx)`` and reproducible ACROSS processes (stable SHA-256 key,
    never the salted builtin ``hash``). Distinct ``idx`` -> distinct stream. Swappable directly into a
    ``LoopState.rng_state`` (run_loop restores ``rng.bit_generator.state``), so a frozen checkpoint can be
    replayed under different future noise while V / currents / z / m stay identical."""
    key = f"{int(seed)}|{cond}|{state}|{int(idx)}".encode()
    digest = hashlib.sha256(key).digest()
    entropy = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]   # 4 x uint32
    ss = np.random.SeedSequence(entropy)
    return np.random.default_rng(np.random.PCG64(ss)).bit_generator.state


def wilson_ci(k, n, z=1.96):
    """Wilson score interval (lo, hi) for a binomial proportion k/n, clipped to [0,1]. n=0 -> (nan, nan)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
