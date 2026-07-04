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
from src.topic5_v3_mode_transition import (  # noqa: E402
    classify_contacts,
    geometry_sufficient,
    i1_range,
    phase_bin_range,
)

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


def load_subject_phase_envelopes(
    ds_sid: str,
    cohort: str,
    cfg: dict,
    phases: list,
    onset_shift: float = 0.0,
    cls: dict | None = None,
) -> dict:
    """Shared eeg-onset-anchored phase-envelope loader (plan's DRY ``load_subject_full_span``).

    Reused by the H3b avalanche run (Task 6) and the H3c dynamics run (Task 8)
    so the contact pool, the axis/non-axis indexing, and the eeg-onset window
    anchoring are defined in ONE place. Pass a precomputed ``cls`` (the
    ``classify_subject_contacts`` dict) to avoid re-running the expensive
    context + lagPat load — e.g. when reloading the same subject at onset
    jitter shifts.

    Returns
    -------
    dict with:
      - ``cls``          : the ``classify_subject_contacts`` dict (source of truth).
      - ``axis_idx``     : int array of indices into ``all_clean`` for ``is_axis``.
      - ``nonaxis_idx``  : int array of indices into ``all_clean`` for ``is_nonaxis_strict``.
      - ``seizures``     : list of ``{"idx": si, "i1_eligible": bool,
                           "phases": {phase: env}}`` where ``env`` is a
                           ``(n_all_clean, n_t)`` bb-envelope slice with ROWS
                           ORDERED BY ``all_clean`` (so ``axis_idx``/``nonaxis_idx``
                           index its rows directly).

    Window contract (matches ``src.topic5_v3_mode_transition``):
      - windows anchored on ``eeg_onset_rel + onset_shift`` via ``phase_bin_range``;
      - ``i1_eligible`` is evaluated at the UNSHIFTED onset (``i1_range(...)[2]``)
        — the run-script gate on ``i1_eligible`` is fixed while jitter only
        translates the window (plan cross-task contract: run scripts gate on
        ``i1_eligible``, ``phase_bin_range`` does not);
      - phase ``"I1"`` is emitted only when ``i1_eligible`` (else omitted for
        that seizure); any phase whose window range is ``None`` is skipped.
    """
    if cls is None:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
    all_clean = cls["all_clean"]
    cache_names = cls["cache_names"]
    meta = cls["meta"]

    name_to_row = {n: i for i, n in enumerate(cache_names)}
    all_clean_rows = np.array([name_to_row[n] for n in all_clean], dtype=int)
    axis_set = set(cls["is_axis"])
    nonaxis_set = set(cls["is_nonaxis_strict"])
    axis_idx = np.array([i for i, n in enumerate(all_clean) if n in axis_set], dtype=int)
    nonaxis_idx = np.array([i for i, n in enumerate(all_clean) if n in nonaxis_set], dtype=int)

    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    seizures: list = []
    for si in meta.get("eligible_idxs", []):
        zt_key = f"bb_zt__{si}"
        relt_key = f"bb_relt__{si}"
        sz = meta.get("seizure", {}).get(str(si))
        if zt_key not in data.files or relt_key not in data.files or sz is None:
            continue
        onset = float(sz["eeg_onset_rel"])
        offset = float(sz["eeg_offset_rel"])
        dur = float(sz["eeg_duration_sec"])
        relt = np.asarray(data[relt_key], dtype=float)
        bb_zt = np.asarray(data[zt_key], dtype=float)[all_clean_rows]  # rows ordered by all_clean
        i1_elig = bool(i1_range(onset, offset, dur, cfg)[2])  # UNSHIFTED gate

        ph_env: dict = {}
        for phase in phases:
            if phase == "I1" and not i1_elig:
                continue  # run-script gate on i1_eligible (phase_bin_range does not)
            rng = phase_bin_range(relt, onset, offset, dur, phase, cfg, onset_shift)
            if rng is None:
                continue
            start, stop = rng
            ph_env[phase] = bb_zt[:, start:stop]
        seizures.append({"idx": si, "i1_eligible": i1_elig, "phases": ph_env})

    return {"cls": cls, "axis_idx": axis_idx, "nonaxis_idx": nonaxis_idx, "seizures": seizures}
