"""Topic 5 V3a mode-transition — shared contact pool + classification IO.

DRY foundation for the ``run_topic5_v3_*`` scripts (feasibility now;
avalanche/dynamics/susceptibility in Tasks 6/8/9). ``classify_subject_contacts``
is the SINGLE SOURCE OF TRUTH for a subject's all-clean contact pool,
interictal HFO participation, and axis/non-axis-strict/ambiguous
classification — downstream tasks must import it rather than re-deriving the
pool.

``channel_is_valid`` is the real per-channel QC gate that builds the
all-clean pool: a channel qualifies only if its concatenated envelope has
enough finite samples and is not flat/degenerate. This replaces an earlier
vacuous filter that compared channel names against ``meta["drops"]`` —
``drops`` is a list of PER-SEIZURE exclusion dicts (``{"idx": ...,
"reason": ...}``), never channel names, so that filter removed zero channels
regardless of input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v2_crit_io import load_context, shaft_of  # noqa: E402
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic5_v3_mode_transition import classify_contacts, geometry_sufficient  # noqa: E402

CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
LAGPAT_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")


def channel_is_valid(env_row) -> bool:
    """Real per-channel QC: >=3 finite samples AND std(finite samples) > 0.

    ``env_row`` is a 1-D envelope array (e.g. a contact's ``bb_zt``
    concatenated across a subject's eligible seizures). Excludes all-NaN
    channels (never populated) and flat/degenerate channels (finite but
    constant — e.g. a railed or disconnected contact) — neither carries
    usable signal for the axis/non-axis geometry.
    """
    row = np.asarray(env_row, dtype=float)
    finite = row[np.isfinite(row)]
    if finite.size < 3:
        return False
    return bool(np.std(finite) > 0)


def _load_participation(subj: str, all_clean: list) -> tuple[dict, str]:
    """Interictal HFO participation per clean contact.

    The 0.0 default for contacts absent from the lagPat pool IS the non-axis
    definition (a contact that never fires an interictal HFO has
    participation 0 < thresh -> non-axis-strict). On lagPat load failure,
    participation is all-0 for every clean contact (classification still
    proceeds via axis_template_names) rather than crashing the subject.
    """
    try:
        ev = load_subject_propagation_events(LAGPAT_ROOT / subj / "all_recs")
        part_raw = {n: float(np.mean(ev["bools"][i])) for i, n in enumerate(ev["channel_names"])}
        return {n: part_raw.get(n, 0.0) for n in all_clean}, ""
    except Exception as exc:  # noqa: BLE001 - external mount, any failure must not crash the cohort
        return {n: 0.0 for n in all_clean}, f"lagpat_load_failed:{type(exc).__name__}:{exc}"


def _axis_template_names(ctx: dict, all_clean_set: set) -> list:
    """Names with finite ``typical_rank`` in either template, intersected with ``all_clean``."""
    names = set()
    for rec in (ctx["ta"], ctx["tb"]):
        for c in rec["channels"]:
            r = c.get("typical_rank", np.nan)
            if np.isfinite(r) and c["name"] in all_clean_set:
                names.add(c["name"])
    return sorted(names)


def classify_subject_contacts(ds_sid: str, cohort: str, cfg: dict) -> dict:
    """Shared pool + classification (single source of truth for Tasks 6/8/9).

    Builds the all-clean contact pool from the ictal field long cache using
    real per-channel QC (``channel_is_valid`` on each contact's concatenated
    ``bb_zt`` envelope across ``meta["eligible_idxs"]`` — NOT the vacuous
    ``meta["drops"]`` channel-name filter; ``drops`` is per-seizure, not
    per-channel), loads interictal HFO participation, and classifies
    contacts into axis / non-axis-strict / ambiguous (``classify_contacts``,
    Task 2 frozen) plus the axis/non-axis geometry gate
    (``geometry_sufficient``, Task 2 frozen).
    """
    _, subj = ds_sid.split("_", 1)
    ctx = load_context(ds_sid, cohort)
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    cache_names = [str(x) for x in data["channels"]]
    meta = json.loads((CACHE / f"{ds_sid}.json").read_text())

    zt_keys = [f"bb_zt__{si}" for si in meta.get("eligible_idxs", []) if f"bb_zt__{si}" in data.files]
    all_clean = [
        name for i, name in enumerate(cache_names)
        if channel_is_valid(np.concatenate([data[k][i] for k in zt_keys]) if zt_keys else np.array([]))
    ]
    all_clean_set = set(all_clean)

    participation, skip_reason = _load_participation(subj, all_clean)
    if skip_reason:
        print(f"[warn] {ds_sid} ({cohort}): {skip_reason}", flush=True)

    axis_template_names = _axis_template_names(ctx, all_clean_set)
    cl = classify_contacts(
        all_clean, axis_template_names, participation,
        cfg["geometry"]["nonaxis_hfo_participation_max"],
    )
    shaft_by_name = {n: shaft_of(n) for n in all_clean}
    shafts_with_both = len(
        {shaft_by_name[n] for n in cl["is_axis"]} & {shaft_by_name[n] for n in cl["is_nonaxis_strict"]}
    )
    geom_ok, geom_reason = geometry_sufficient(cl["n_axis"], cl["n_nonaxis"], shafts_with_both, cfg)

    return {
        "ctx": ctx,
        "all_clean": all_clean,
        "participation": participation,
        "axis_template_names": axis_template_names,
        "is_axis": cl["is_axis"],
        "is_nonaxis_strict": cl["is_nonaxis_strict"],
        "is_ambiguous": cl["is_ambiguous_hfo"],
        "n_axis": cl["n_axis"],
        "n_nonaxis": cl["n_nonaxis"],
        "n_ambiguous": cl["n_ambiguous"],
        "shaft_by_name": shaft_by_name,
        "shafts_with_both": shafts_with_both,
        "geometry_sufficient": geom_ok,
        "geometry_reason": geom_reason,
        "cache_names": cache_names,
        "meta": meta,
    }
