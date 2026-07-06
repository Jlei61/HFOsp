"""Topic 5 V3c — SOZ join + latency-matrix IO (reuses V3a classifier)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import CACHE, classify_subject_contacts  # noqa: E402
from src.topic5_v3c_coverage import coverage_metrics  # noqa: E402
from src.topic5_v3c_latency import first_crossing_latency  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402

SOZ_JSON = {
    "epilepsiae": _ROOT / "results/epilepsiae_soz_core_channels.json",
    "yuquan": _ROOT / "results/yuquan_soz_core_channels.json",
}

# Per-cohort interictal propagation-axis pool (classify_subject_contacts needs it):
# broad classification fails without the broad pool file (e.g. 442/548/958 -> narrow only).
AXIS_POOL = {
    "broad": _ROOT / "results/interictal_propagation_masked_broad/per_subject",
    "narrow": _ROOT / "results/interictal_propagation_masked/per_subject",
}


def load_soz(dataset: str, subject: str) -> list:
    """Clinical SOZ contact names for one subject; [] if the subject is absent."""
    path = SOZ_JSON[dataset]
    data = json.loads(path.read_text())
    return list(data.get(subject, []))


def _discover_cohort(cohort: str) -> list:
    """Data-derived cohort: subjects that have (a) an ictal recruitment cache,
    (b) a propagation-axis pool file for this cohort, and (c) a clinical SOZ list.

    Replaces the earlier hand-coded list, which under-counted (broad 7 vs 11,
    narrow 5 vs 14) — the list was copied from an incomplete feasibility probe.
    Yuquan subjects are held: their interictal participation files are absent, so
    the axis is derivable but participation-based non-axis separation is degraded.
    """
    if not CACHE.exists() or not AXIS_POOL[cohort].exists():
        return []
    cached = {p.stem for p in CACHE.glob("*.npz")}
    pooled = {p.stem for p in AXIS_POOL[cohort].glob("*.json")}
    out = []
    for stem in sorted(cached & pooled):
        if not stem.startswith("epilepsiae_"):
            continue
        dataset, subj = stem.split("_", 1)
        if load_soz(dataset, subj):
            out.append(stem)
    return out


V3C_SUBJECTS = {"broad": _discover_cohort("broad"), "narrow": _discover_cohort("narrow")}


def axis_soz_join(cls: dict, soz_list: list) -> dict:
    """coverage_metrics(A, S) with S restricted to the all-clean pool; adds soz_in_pool."""
    pool = set(cls["all_clean"])
    soz_in_pool = [n for n in soz_list if n in pool]
    m = coverage_metrics(cls["is_axis"], soz_in_pool)
    m["soz_in_pool"] = soz_in_pool
    return m


def extract_latency_matrix(ds_sid: str, cfg: dict, names: list, *, thresholds: list) -> list:
    """Per eligible seizure, per contact in `names`, first-crossing latency at each
    threshold (window/sustain from cfg['v3c']). Rows ordered 1:1 with `names`.

    P1-4 FAIL-CLOSED: every name MUST exist in the cache channel list. A missing
    contact raises ValueError rather than silently shifting the row->name
    alignment (which would assign one contact's latency to another — a science
    contamination bug). `names` always come from all_clean / soz_in_pool, both
    derived from cache channels, so a miss means an upstream bug, not normal data.
    """
    vc = cfg["v3c"]
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.loads((CACHE / f"{ds_sid}.json").read_text())
    cache_names = [str(x) for x in data["channels"]]
    name_to_row = {n: i for i, n in enumerate(cache_names)}
    missing = [n for n in names if n not in name_to_row]
    if missing:
        raise ValueError(f"{ds_sid}: latency requested for contacts absent from cache: {missing}")
    rows = [name_to_row[n] for n in names]     # 1:1 with names (fail-closed above)
    out = []
    for si in meta.get("eligible_idxs", []):
        zk, rk = f"bb_zt__{si}", f"bb_relt__{si}"
        sz = meta.get("seizure", {}).get(str(si))
        if zk not in data.files or rk not in data.files or sz is None:
            continue
        onset = float(sz["eeg_onset_rel"])
        relt = np.asarray(data[rk], dtype=float)
        Z = np.asarray(data[zk], dtype=float)
        kinds, secs = {}, {}
        for thr in thresholds:
            kk, ss = [], []
            for r in rows:
                kind, sec = first_crossing_latency(
                    Z[r], relt, onset, z_cross=thr,
                    window_sec=vc["window_sec"], sustain_frames=vc["sustain_frames"])
                kk.append(kind); ss.append(sec)
            kinds[thr] = kk; secs[thr] = ss
        out.append({"idx": si, "kinds": kinds, "secs": secs})
    return out


def load_axis_coords(dataset: str, subject: str, names: list) -> dict:
    """{name: ras_mm coord} for `names`; {} if MRI/SQL missing (V3c-3 falls back
    to shaft-only metrics — no silent coord fabrication)."""
    try:
        res = load_subject_coords(dataset, subject, names)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[coords-skip] {dataset}_{subject}: {type(exc).__name__}: {exc}", flush=True)
        return {}
    out = {}
    coords = res.coords_array_in_requested_order      # (n, 3), NaN for missing
    mask = res.mapped_mask_in_requested_order          # (n,) bool, index-aligned to names
    for i, n in enumerate(names):
        if bool(mask[i]) and np.all(np.isfinite(coords[i])):
            out[n] = np.asarray(coords[i], dtype=float)
    return out
