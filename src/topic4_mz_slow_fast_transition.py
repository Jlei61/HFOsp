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

# functions added in Tasks 2–4 (TDD).
