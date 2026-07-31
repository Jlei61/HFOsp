"""Phase-C phenotype analysis correction for whole-sheet runaway scope.

The production-locked v1 classifier uses the pathology-core source trace for
both temporal morphology and the 250-Hz runaway gate.  The latter has
whole-sheet semantics.  Production C1 observables already contain the required
all-sheet E-rate trace at its own cadence, so this analysis-only v2 adapter
keeps core morphology unchanged while applying runaway/trend detection to that
separate trace.

The v1 module is deliberately left byte-identical because it is part of the
immutable production manifest.  This adapter is single-process analysis code;
it delegates every non-runaway phenotype calculation to v1.
"""
from __future__ import annotations

from threading import RLock

import numpy as np

from src import topic4_zm_phasec_phenotype as V1


PHASEC_PHENOTYPE_VERSION = (
    "zm_phasec_phenotype_v2_all_sheet_runaway_scope_2026-07-31"
)
_V1_COMMON_BOUNDED_GATE = V1.common_bounded_gate
_PATCH_LOCK = RLock()


def common_bounded_gate(
    source_rate_hz,
    *,
    bin_ms,
    active_area_fraction,
    rest_mask=None,
    all_sheet_rate_hz=None,
    all_sheet_bin_ms=None,
    runaway_early_stop_ms=None,
    saturation_fraction=None,
    refractory_fraction=None,
    thresholds=None,
):
    """Run v1 occupancy/rest logic but gate runaway on all-sheet E rate.

    ``source_rate_hz`` remains the pathology-core morphology trace.  The
    all-sheet series may have a different cadence and length; its own early,
    tail and percentile statistics are therefore evaluated independently.
    Calls without a separate all-sheet trace preserve the v1 source fallback
    used by synthetic and historical tests.
    """
    th = dict(V1.DEFAULTS)
    if thresholds:
        th.update(thresholds)

    source = np.asarray(source_rate_hz, float).ravel()
    if all_sheet_rate_hz is None:
        runaway_rate = source
        runaway_bin_ms = float(bin_ms)
        scope = "source_fallback"
    else:
        runaway_rate = np.asarray(all_sheet_rate_hz, float).ravel()
        if runaway_rate.size < 4 or not np.all(np.isfinite(runaway_rate)):
            raise ValueError(
                "all_sheet_rate_hz must be finite and contain >=4 bins"
            )
        if all_sheet_bin_ms is None:
            raise ValueError(
                "all_sheet_bin_ms is required with all_sheet_rate_hz"
            )
        runaway_bin_ms = float(all_sheet_bin_ms)
        if not np.isfinite(runaway_bin_ms) or runaway_bin_ms <= 0:
            raise ValueError("all_sheet_bin_ms must be finite and positive")
        scope = "all_sheet_E"

    # Preserve every v1 decision except rate-derived runaway.  Explicit online
    # runaway and the registered saturation conjunction retain their original
    # higher-priority ordering.
    no_rate_runaway = dict(th)
    no_rate_runaway["runaway_rate_hz"] = float("inf")
    out = _V1_COMMON_BOUNDED_GATE(
        source,
        bin_ms=bin_ms,
        active_area_fraction=active_area_fraction,
        rest_mask=rest_mask,
        runaway_early_stop_ms=runaway_early_stop_ms,
        saturation_fraction=saturation_fraction,
        refractory_fraction=refractory_fraction,
        thresholds=no_rate_runaway,
    )

    n_quarter = max(4, runaway_rate.size // 4)
    runaway_early = float(np.median(runaway_rate[:n_quarter]))
    runaway_tail = float(np.median(runaway_rate[-n_quarter:]))
    runaway_p95 = float(np.percentile(runaway_rate, 95))
    tail_escalating = bool(
        runaway_tail - runaway_early >= 25.0
        and runaway_tail
        >= 1.5 * max(runaway_early, float(th["active_floor_hz"]))
    )
    rate_runaway = bool(
        runaway_tail >= float(th["runaway_rate_hz"])
        or (
            tail_escalating
            and runaway_p95 >= float(th["runaway_rate_hz"])
        )
    )
    if (
        runaway_early_stop_ms is None
        and out["status"] != "saturation"
        and rate_runaway
    ):
        out["status"] = "runaway"

    source_n_quarter = max(4, source.size // 4)
    out.update({
        "runaway_rate_scope": scope,
        "runaway_rate_bin_ms": runaway_bin_ms,
        "runaway_rate_mean_hz": float(np.mean(runaway_rate)),
        "runaway_rate_p95_hz": runaway_p95,
        "runaway_early_median_hz": runaway_early,
        "runaway_tail_median_hz": runaway_tail,
        "tail_escalating": tail_escalating,
        # Keep the historical generic names aligned with the trace that now
        # actually adjudicates runaway, and expose source-only values
        # explicitly for temporal-morphology diagnostics.
        "early_median_hz": runaway_early,
        "tail_median_hz": runaway_tail,
        "source_early_median_hz": float(
            np.median(source[:source_n_quarter])
        ),
        "source_tail_median_hz": float(
            np.median(source[-source_n_quarter:])
        ),
    })
    return out


def classify_phasec_run(
    E_rate_grid,
    I_rate_grid,
    *,
    bin_ms,
    source_rate_hz,
    rest_mask=None,
    active_area_fraction=None,
    kymograph=None,
    axis_positions=None,
    readout_kernel_width_mm=None,
    all_sheet_rate_hz=None,
    all_sheet_bin_ms=None,
    runaway_early_stop_ms=None,
    saturation_fraction=None,
    refractory_fraction=None,
    thresholds=None,
    relay_n_perm=999,
    relay_rng_seed=0,
):
    """Delegate v1 morphology while injecting the corrected runaway gate."""

    def scoped_gate(source, **kwargs):
        return common_bounded_gate(
            source,
            all_sheet_rate_hz=all_sheet_rate_hz,
            all_sheet_bin_ms=all_sheet_bin_ms,
            **kwargs,
        )

    # V1 resolves ``common_bounded_gate`` dynamically from its module globals.
    # Protect the temporary analysis-only substitution from concurrent calls.
    with _PATCH_LOCK:
        previous = V1.common_bounded_gate
        V1.common_bounded_gate = scoped_gate
        try:
            result = V1.classify_phasec_run(
                E_rate_grid,
                I_rate_grid,
                bin_ms=bin_ms,
                source_rate_hz=source_rate_hz,
                rest_mask=rest_mask,
                active_area_fraction=active_area_fraction,
                kymograph=kymograph,
                axis_positions=axis_positions,
                readout_kernel_width_mm=readout_kernel_width_mm,
                runaway_early_stop_ms=runaway_early_stop_ms,
                saturation_fraction=saturation_fraction,
                refractory_fraction=refractory_fraction,
                thresholds=thresholds,
                relay_n_perm=relay_n_perm,
                relay_rng_seed=relay_rng_seed,
            )
        finally:
            V1.common_bounded_gate = previous

    result["phasec_phenotype_version"] = PHASEC_PHENOTYPE_VERSION
    result["analysis_correction"] = {
        "scope": "all_sheet_E_rate_for_runaway_only",
        "source_morphology": "pathology_core_rate_unchanged",
        "threshold_changed": False,
    }
    return result

