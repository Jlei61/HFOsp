"""Load the cached cohort into ``PatientTensors`` and choose eligible strata."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .contracts import FROZEN, OUTPUT_ROOT
from .model import SESSION_OPEN_DELTA_T, PatientTensors

CACHE = OUTPUT_ROOT / "cache/cohort_v0_1.pt"
NUISANCE_ROOT = OUTPUT_ROOT / "nuisance_features/per_subject"


@lru_cache(maxsize=1)
def _raw() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"cohort cache missing: {CACHE}. Run scripts/topic5_epi_prssm/prepare_cohort.py")
    return torch.load(CACHE, map_location="cpu", weights_only=False)


def cohort_subjects() -> tuple[str, ...]:
    return tuple(sorted(_raw().keys()))


def load_tensors(subjects: Sequence[str] | None = None, *, device: str = "cpu"
                 ) -> list[PatientTensors]:
    raw = _raw()
    names = list(subjects) if subjects is not None else list(cohort_subjects())
    out: list[PatientTensors] = []
    for subject in names:
        d = raw[subject]
        nuisance = None
        nuisance_path = NUISANCE_ROOT / f"{subject}.npz"
        if nuisance_path.exists():
            with np.load(nuisance_path) as z:
                nuisance = torch.as_tensor(z["features"], dtype=torch.float32, device=device)
        delta = d["delta_t_raw"].numpy().astype(np.float64)
        delta[~np.isfinite(delta)] = SESSION_OPEN_DELTA_T
        delta = np.maximum(delta, 0.0)
        f32 = lambda t: t.to(torch.float32).to(device)
        out.append(PatientTensors(
            subject=subject, dataset=d["dataset"],
            participation=d["participation"].to(torch.bool).to(device),
            group_ids=d["group_ids"].to(torch.long).to(device),
            n_groups=d["n_groups"].to(torch.long).to(device),
            marks=f32(d["marks"]),
            delta_t=torch.as_tensor(delta, dtype=torch.float32, device=device),
            log_delta_t=torch.as_tensor(np.log1p(delta), dtype=torch.float32, device=device),
            session_open=f32(d["session_opening"]),
            load=f32(d["load"]),
            split=d["split"].to(torch.long).to(device),
            event_time=d["event_time"].numpy(),
            adjacency=f32(d["adjacency"]),
            node_features=f32(d["node_features"]),
            baseline_order=f32(d["baseline_order"]),
            baseline_participation=f32(d["baseline_participation"]),
            baseline_stop=torch.as_tensor(d["baseline_stop"], dtype=torch.float32, device=device),
            n_contacts=int(d["n_contacts"]), n_events=int(d["n_events"]),
            nuisance=nuisance,
            meta={"geometry_available": bool(d["geometry_available"]),
                  "n_geometry_mapped": int(d["n_geometry_mapped"]),
                  "length_scale_mm": float(d["length_scale_mm"]),
                  "n_sessions": int(d["n_sessions"]),
                  "mean_load": float(d["baseline_mean_load"]),
                  "contact_names": list(d["contact_names"]),
                  "session_index": d["session_index"].numpy(),
                  "source_hashes": d["source_hashes"]},
        ))
    return out


def eligible_subjects() -> tuple[str, ...]:
    """Frozen eligibility: enough events and enough contacts to define a graph."""
    raw = _raw()
    keep = []
    for subject in cohort_subjects():
        d = raw[subject]
        if int(d["n_events"]) < FROZEN["min_events_for_eligibility"]:
            continue
        if int(d["n_contacts"]) < FROZEN["min_contacts_for_eligibility"]:
            continue
        keep.append(subject)
    return tuple(keep)


def breadth_pilot_subjects(n: int = 8) -> tuple[str, ...]:
    """Support-stratified development patients, chosen only by support features.

    Strata are dataset and train-event count -- never by any H1/H2/H3 effect.
    """
    raw = _raw()
    rows = [(s, raw[s]["dataset"], int((raw[s]["split"] == 0).sum()), int(raw[s]["n_contacts"]))
            for s in eligible_subjects()]
    picked: list[str] = []
    for dataset in ("epilepsiae", "yuquan"):
        pool = sorted([r for r in rows if r[1] == dataset], key=lambda r: r[2])
        if not pool:
            continue
        take = n // 2
        step = max(len(pool) // max(take, 1), 1)
        picked.extend([pool[min(i * step, len(pool) - 1)][0] for i in range(take)])
    return tuple(sorted(dict.fromkeys(picked)))
