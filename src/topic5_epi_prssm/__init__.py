"""Epi-PRSSM v0.1 — Epilepsy Physiology- and Repertoire-Constrained Recurrent State-Space Model.

Contract: docs/superpowers/specs/2026-08-18-topic5-epi-prssm-v0_1.md
Plan:     docs/superpowers/plans/2026-08-18-topic5-epi-prssm-v0_1.md

Three state objects are kept structurally distinct throughout this package and
must never be merged, aliased or logged under one name:

``s_{p,e,k}``  fast event state    -- which contacts the current event has walked
``z_{p,e}``    slow generative state -- (H, r), persists across events, evolves autonomously
``c_{p,e}``    observer state       -- inference memory accumulated from past observations
"""
from __future__ import annotations

CONTRACT = "topic5_epi_prssm_v0_1"
CONTRACT_VERSION = "0.1"
