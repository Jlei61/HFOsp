"""Real time-axis scaffold for the v0.3.3 assays (plan Task 4).

A ``Scaffold`` is everything a synthetic DGP or an oracle estimator needs from a
patient *except the targets*: the 5-minute anchor grid, coverage / partition
eligibility of every target window, the kept event stream with its carry ids,
the vocabulary participation matrix and the registry ``log mu_H`` per horizon.
Real anchors, coverage, split and event times are never altered (D-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.topic5_group_event_state.v032_eval.partition import EVAL_PHASES
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline

from . import boundaries as B

PHASE_NAMES = EVAL_PHASES
CARRY_MODES = ("segment", "session")


@dataclass
class Scaffold:
    subject: str
    horizons: tuple[float, ...]
    t_anchor: np.ndarray            # (A,)
    anchor_carry: np.ndarray        # (A,) carry unit of each anchor
    anchor_phase: np.ndarray        # (A,) 0..3
    eligible: np.ndarray            # (A, H) whole target window inside one target segment and one phase
    window_lo: np.ndarray           # (A, H) first kept event index in [t, t+h)
    window_hi: np.ndarray           # (A, H) one past the last
    last_event_pos: np.ndarray      # (A,) last kept event strictly before the anchor in its carry unit, -1 if none
    event_times: np.ndarray         # (N,) kept events (seizure / postictal events already excluded)
    event_carry: np.ndarray         # (N,)
    event_phase: np.ndarray         # (N,)
    participation: np.ndarray       # (N, C) bool over vocabulary contacts
    log_mu_h: dict[int, np.ndarray] # registry H_strong log mu per horizon (A,), NaN when not published
    log_r_h: dict[int, float | None]
    segment_bounds: np.ndarray      # (S, 2) target segments
    phase_bounds: np.ndarray        # (4, 2)
    carry: str = "segment"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_anchors(self) -> int:
        return int(self.t_anchor.size)

    @property
    def n_events(self) -> int:
        return int(self.event_times.size)

    @property
    def n_contacts(self) -> int:
        return int(self.participation.shape[1])

    def horizon_index(self, horizon: float) -> int:
        return [float(h) for h in self.horizons].index(float(horizon))

    def phase_mask(self, phase: str) -> np.ndarray:
        return self.anchor_phase == PHASE_NAMES.index(phase)

    def anchor_rows(self, phase: str, horizon: float) -> np.ndarray:
        return np.flatnonzero(self.phase_mask(phase) & self.eligible[:, self.horizon_index(horizon)])

    def event_rows(self, phase: str) -> np.ndarray:
        return np.flatnonzero(self.event_phase == PHASE_NAMES.index(phase))

    def event_size(self) -> np.ndarray:
        return self.participation.sum(axis=1).astype(np.int64)

    def independent_blocks(self, phase: str, horizon: float) -> int:
        """Non-overlapping ``horizon`` windows inside (target segment ∩ phase) -- the honest denominator."""

        lo, hi = self.phase_bounds[PHASE_NAMES.index(phase)]
        total = 0
        for a, b in self.segment_bounds:
            start, stop = max(float(a), float(lo)), min(float(b), float(hi))
            if stop > start:
                total += int(math.floor((stop - start) / float(horizon)))
        return total


def _registry_horizons(registry_path: Path, subject: str, t_anchor: np.ndarray,
                       horizons: tuple[float, ...]) -> tuple[dict[int, np.ndarray], dict[int, float | None], dict]:
    registry = json.loads(Path(registry_path).read_text())
    entry = registry["patients"][subject]
    log_mu: dict[int, np.ndarray] = {}
    log_r: dict[int, float | None] = {}
    meta = {"registry_generated": registry.get("generated"), "registry_commit": registry.get("source_commit"),
            "config_sha256": registry.get("config_sha256"), "horizon_status": {}}
    for h in horizons:
        key = str(int(h))
        spec = entry.get("horizons", {}).get(key)
        if spec is None or spec.get("status", "ok") != "ok":
            log_mu[int(h)] = np.full(t_anchor.size, np.nan)
            log_r[int(h)] = None
            meta["horizon_status"][key] = "not_published"
            continue
        with np.load(spec["arrays"], allow_pickle=False) as data:
            t_reg = np.asarray(data["anchor_time"], dtype=np.float64)
            values = np.asarray(data["log_mu_h"], dtype=np.float64)
        if t_reg.shape != t_anchor.shape or not np.allclose(t_reg, t_anchor, atol=1e-3):
            raise ValueError(f"{subject}: registry anchor grid does not match the timeline grid at {key}s")
        log_mu[int(h)] = values
        log_r[int(h)] = float(spec["nb_log_dispersion"])
        meta["horizon_status"][key] = "ok"
    return log_mu, log_r, meta


def load_real_scaffold(subject: str, cfg: Mapping[str, Any], *, carry: str = "segment") -> Scaffold:
    """Read-only: v0.3.2 evaluation timeline + history-baseline registry -> Scaffold."""

    if carry not in CARRY_MODES:
        raise ValueError(f"carry must be one of {CARRY_MODES}")
    tl = load_eval_timeline(subject, cfg)
    horizons = tuple(float(h) for h in tl.horizons_seconds)
    t_anchor = np.asarray(tl.grid.t_anchor, dtype=np.float64)
    eligible = np.stack([B.target_window_valid(t_anchor, h, tl.segments, tl.partition) for h in horizons], axis=1)
    if carry == "segment":
        event_carry = np.asarray(tl.event_segment, dtype=np.int64)
        anchor_carry = np.asarray(tl.grid.segment_index, dtype=np.int64)
        last = np.asarray(tl.grid.last_event_pos, dtype=np.int64)
    else:
        event_carry = np.asarray(tl.event_session, dtype=np.int64)
        anchor_carry = np.asarray(tl.grid.session_id, dtype=np.int64)
        last = B.carry_last_event(tl.event_times, event_carry, np.ones(tl.n_events, dtype=bool), t_anchor, anchor_carry)
    registry_path = Path(cfg["data_root"]) / "shared/history_baseline_registry.json"
    log_mu, log_r, meta = _registry_horizons(registry_path, subject, t_anchor, horizons)
    return Scaffold(
        subject=subject, horizons=horizons, t_anchor=t_anchor, anchor_carry=anchor_carry,
        anchor_phase=np.asarray(tl.anchor_phase_labels(), dtype=np.int64), eligible=eligible,
        window_lo=np.asarray(tl.grid.window_lo, dtype=np.int64), window_hi=np.asarray(tl.grid.window_hi, dtype=np.int64),
        last_event_pos=last, event_times=np.asarray(tl.event_times, dtype=np.float64), event_carry=event_carry,
        event_phase=np.asarray(tl.event_phase_labels(), dtype=np.int64),
        participation=np.asarray(tl.participation[:, tl.vocab_mask], dtype=bool),
        log_mu_h=log_mu, log_r_h=log_r,
        segment_bounds=np.asarray([[s.start_epoch, s.stop_epoch] for s in tl.segments], dtype=np.float64),
        phase_bounds=np.asarray([list(tl.partition.bounds(p)) for p in PHASE_NAMES], dtype=np.float64),
        carry=carry,
        provenance={"subject": subject, "dataset": tl.dataset, "config_sha256": cfg.get("_config_sha256"),
                    "n_vocab_contacts": int(tl.n_vocab), "n_segments": len(tl.segments), **meta},
    )
