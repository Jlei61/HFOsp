"""Shared Phase-2 (criticality/state layer) data plumbing.

DRY foundation for the four ``run_topic5_v2_crit_*`` scripts. READ-ONLY reuse of
the Phase-1 field pipeline: ``load_context`` (G_HFO geometry + matched contacts)
and the substrate-independent ``ictal_field_long_cache`` (per-subject peri-ictal
baseline-robust-z envelope). Phase 2 reads only the PREICTAL segment (``relt<0``).

Nothing here rebuilds ``G_HFO``; it is the fixed interictal ``typical_rank``.
Every consumer is EXPLORATORY, peri-ictal susceptibility, NOT forecasting.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_topic5_ictal_field_dynamics import load_context  # noqa: E402

CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"

# state_band name -> long-cache key prefix (the cache stores bb=1-45 and hfa=60-100)
_STATE_PREFIX = {"legacy_bb_1_45": "bb", "legacy_hfa_60_100": "hfa"}


def get_contact_alignment() -> tuple[object, str]:
    """Return ``(contact_alignment, source)``.

    Prefer the real Phase-1 ``src.topic5_v2_band_scan.contact_alignment``; fall
    back to the verbatim contract shim while Phase 1 is in flight. Deterministic
    correlation → identical output either way (source recorded for provenance).
    """
    try:
        from src.topic5_v2_band_scan import contact_alignment  # type: ignore
        return contact_alignment, "band_scan"
    except (ImportError, AttributeError):
        from src._topic5_v2_p1_contract_shim import contact_alignment
        return contact_alignment, "contract_shim"


def get_null_fns() -> tuple[dict | None, str]:
    """Return ``(fns, source)`` for the science-critical Phase-1 null builders.

    ``fns`` = ``{spatial_constrained_permute, order_null_rank_pair,
    rebuild_typical_rank}`` from ``src.topic5_v2_band_scan`` once Phase 1 lands
    Tasks 8/9, else ``None`` (these are NOT shimmed — the null construction is the
    contract). Consumers with ``None`` emit observed stats + a ``pending_phase1``
    null status; they never fabricate a null.
    """
    try:
        from src.topic5_v2_band_scan import (  # type: ignore
            spatial_constrained_permute, order_null_rank_pair, rebuild_typical_rank,
        )
        return ({
            "spatial_constrained_permute": spatial_constrained_permute,
            "order_null_rank_pair": order_null_rank_pair,
            "rebuild_typical_rank": rebuild_typical_rank,
        }, "band_scan")
    except (ImportError, AttributeError):
        return None, "pending_phase1"


def state_prefix(state_band: str) -> str:
    """Map a state_band name to its ``ictal_field_long_cache`` key prefix."""
    try:
        return _STATE_PREFIX[str(state_band)]
    except KeyError as exc:
        raise ValueError(
            f"Phase-2 state_band {state_band!r} not in long cache "
            f"(available: {sorted(_STATE_PREFIX)})"
        ) from exc


def shaft_of(name: str) -> str:
    """Electrode shaft id = contact name minus its trailing contact number.

    ``HL10 -> HL``, ``TLA1 -> TLA``, ``GA'1 -> GA'``. Falls back to the whole
    name when no trailing digits exist.
    """
    stripped = re.sub(r"\d+$", "", str(name))
    return stripped if stripped else str(name)


def ghfo_ranks(ctx: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Fixed interictal ``typical_rank`` maps for templates A and B (G_HFO)."""

    def _ranks(rec) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in rec["channels"]:
            r = c.get("typical_rank", np.nan)
            if c["name"] in ctx["pos"] and np.isfinite(r):
                out[c["name"]] = float(r)
        return out

    return _ranks(ctx["ta"]), _ranks(ctx["tb"])


def window_index_range(relt: np.ndarray, lo: float, hi: float) -> tuple[int, int] | None:
    """Half-open ``(start, stop)`` sample indices where ``lo <= relt <= hi``.

    ``relt`` is monotone increasing, so the mask is contiguous. Returns ``None``
    if the window is empty.
    """
    relt = np.asarray(relt, dtype=float)
    mask = (relt >= float(lo)) & (relt <= float(hi))
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return int(idx[0]), int(idx[-1] + 1)


def load_subject_preictal(ds_sid: str, substrate: str, cfg: dict) -> dict:
    """Load a subject's preictal state envelopes + fixed G_HFO geometry.

    Returns a dict with ``status`` in ``{ok, skipped}``. On ``ok`` it carries,
    for each eligible seizure that has preictal data in ``span_rel``, the
    matched-contact envelope ``E`` (``n_mapped, n_t``) and its ``relt`` axis.

    Skip contract (``skipped`` != negative): a subject whose best seizure has
    ``available_pre_sec < min_required_pre_sec`` yields one skipped record with
    no features — never silently dropped.
    """
    prefix = state_prefix(cfg["state_band"])
    span_lo, span_hi = (float(v) for v in cfg["preictal"]["span_rel"])
    required = float(cfg["preictal"]["min_required_pre_sec"])
    hop_default = float(cfg["preictal"].get("hop_sec", 0.1))

    ctx = load_context(ds_sid, substrate)
    mapped = list(ctx["mapped"])
    ta_rank, tb_rank = ghfo_ranks(ctx)
    pos = {n: ctx["pos"][n] for n in mapped if n in ctx["pos"]}
    shaft_by_name = {n: shaft_of(n) for n in mapped}

    meta = json.loads((CACHE / f"{ds_sid}.json").read_text())
    hop_sec = float(meta.get("hop_sec", hop_default))
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    cache_names = [str(x) for x in data["channels"]]
    idx_of = {n: i for i, n in enumerate(cache_names)}
    used_names = [n for n in mapped if n in idx_of]
    rows = [idx_of[n] for n in used_names]

    base = dict(
        ds_sid=ds_sid, substrate=substrate, ctx=ctx,
        ta_rank=ta_rank, tb_rank=tb_rank,
        mapped=used_names, pos=pos, coord_by_name=pos, shaft_by_name=shaft_by_name,
        hop_sec=hop_sec, required_pre_sec=required, state_band=cfg["state_band"],
        n_contacts=len(used_names),
    )

    if not used_names:
        return {**base, "status": "skipped", "skip_reason": "no_matched_contacts_in_cache",
                "available_pre_sec": 0.0, "seizures": [], "n_seizures": 0}

    seizures: list[dict] = []
    available = 0.0
    for si in meta.get("eligible_idxs", []):
        zk, rk = f"{prefix}_zt__{si}", f"{prefix}_relt__{si}"
        if zk not in data.files or rk not in data.files:
            continue
        relt_raw = np.asarray(data[rk], dtype=float)
        pre = relt_raw[relt_raw < 0.0]
        if pre.size:
            available = max(available, float(-pre.min()))
        span = (relt_raw >= span_lo) & (relt_raw <= span_hi)
        if span.sum() < 2:
            continue
        E = np.asarray(data[zk], dtype=float)[np.ix_(rows, np.flatnonzero(span))]
        seizures.append({"idx": int(si), "E": E, "relt": relt_raw[span]})

    if available < required or not seizures:
        reason = ("insufficient_preictal"
                  f"(avail={available:.1f}<req={required:.1f})" if seizures
                  else "no_eligible_preictal_window")
        return {**base, "status": "skipped", "skip_reason": reason,
                "available_pre_sec": float(available),
                "seizures": [], "n_seizures": 0, "n_seizures_total": len(seizures)}

    # Tractability cap for per-perm refit nulls: keep the first `max_seizures`
    # (deterministic, by seizure idx); n_seizures_total keeps the cap transparent.
    n_total = len(seizures)
    max_sz = int(cfg["preictal"].get("max_seizures", 0) or 0)
    if max_sz > 0 and n_total > max_sz:
        seizures = seizures[:max_sz]

    return {**base, "status": "ok", "skip_reason": "",
            "available_pre_sec": float(available),
            "seizures": seizures, "n_seizures": len(seizures), "n_seizures_total": n_total}
