"""Contract tests for the kick-calibration selector (_select_calibration) and the
spatial-extent helper (_spatial_extent).

The selector picks the MINIMUM kick_boost whose response is LOCAL, RETURNED, and
non-global — NOT the biggest downstream wave. (Domain fact, Stage3: a LOCAL event
is STRONG ignition followed by propagation-relay FAILURE, so "maximize downstream"
systematically picks the wrong end — the global runaway.) These tests drive the
selector with synthetic per-bin records carrying the run-property + spatial fields
(no SNN); they cover that a big global wave can never be selected, that the minimum
qualifying kick wins, that the selector FAILS LOUDLY (go/no-go) rather than freezing
a default when nothing is local+returned, and (P1 reviewer fixes) that locality is
judged robustly per-seed / per-rep-bin and that r95 is actually gated.
"""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_m3_kick_calibration import (  # noqa: E402
    _select_calibration, _spatial_extent, _diagnose, _GATE_ORDER,
    _core_only_quiet, _event_onset, _event_aligned_window,
    SOURCE_FLOOR, DOWNSTREAM_FLOOR, FAR_FRAC_CAP, FRAC_TIME_ON_MAX,
    SEED_PASS_FRAC, BIN_PASS_FRAC, R95_CAP_MM_FALLBACK,
    CORE_BG_RATIO, CORE_BG_MARGIN,
    ONSET_MIN_MASS, EA_DELTA1, EA_DELTA2,
)


def _rec(kb, win, *, downstream=10.0, source=10.0, returned=True, runaway=False,
         frac_time_on_post=0.0, pass_frac_seeds=1.0, n_activated_bins=1, r95_mm=1.0,
         far_field_frac=0.0, after=True, core_only_quiet=True,
         no_core_no_kick_downstream=1.0, no_core_no_kick_source=1.0):
    """Build one per-(bin, boost, win) record with all selector-consumed fields.

    Defaults describe a healthy LOCAL+RETURNED point (qualifies). Override individual
    fields to construct disqualified / big-wave records.

    ``core_only_quiet`` defaults True so bare-sheet records (no confound) still
    qualify — the confound gate is a no-op for the legacy path. Override to False to
    construct a CONFOUNDED core record (core self-ignites around the window). The
    ``no_core_no_kick_*`` bare-sheet backgrounds are carried so a record stays
    consistent with the RELATIVE confound gate (they are not re-derived by the
    selector, which reads the precomputed ``core_only_quiet`` flag).
    """
    return {
        "kick_boost": kb,
        "win_ms": list(win),
        "source_resp": source,
        "downstream_resp": downstream,
        "returned": returned,
        "runaway": runaway,
        "frac_time_on_post": frac_time_on_post,
        "pass_frac_seeds": pass_frac_seeds,
        "n_activated_bins": n_activated_bins,
        "r95_mm": r95_mm,
        "far_field_frac": far_field_frac,
        "window_after_dur_kick": after,
        "core_only_quiet": core_only_quiet,
        "no_core_no_kick_downstream": no_core_no_kick_downstream,
        "no_core_no_kick_source": no_core_no_kick_source,
    }


# ---------------------------------------------------------------------------
# Existing selector contract (kept working)
# ---------------------------------------------------------------------------

def test_big_wave_cannot_be_selected():
    # The MAX-downstream record is a global wave (not returned, far-field high) and
    # must NOT be selected; a smaller local+returned record at a lower boost wins.
    rbb = {0: [
        _rec(0.2, (18, 24), downstream=8.0, returned=True, runaway=False,
             far_field_frac=0.0, n_activated_bins=1),                       # local, returned
        _rec(4.0, (18, 24), downstream=999.0, returned=False, runaway=True,
             far_field_frac=0.9, n_activated_bins=14),                      # biggest wave, global
    ]}
    out = _select_calibration(rbb)
    assert out["kick_boost"] == 0.2                 # the local one, NOT the biggest
    assert out["win_ms"] == [18, 24]


def test_minimum_kick_selected():
    # Several qualifying (local + returned + downstream>=floor) records at boosts
    # {0.1, 0.2, 0.5}: the 0.1 (minimum) is chosen.
    rbb = {0: [
        _rec(0.1, (18, 24), downstream=5.0),
        _rec(0.2, (18, 24), downstream=8.0),
        _rec(0.5, (18, 24), downstream=20.0),
    ]}
    out = _select_calibration(rbb)
    assert out["kick_boost"] == 0.1


def test_no_qualifying_raises():
    # All records are big-wave / not-returned -> raise (go/no-go), never a default.
    rbb = {0: [
        _rec(1.0, (18, 24), downstream=50.0, returned=False, runaway=True,
             far_field_frac=0.9, n_activated_bins=14),
        _rec(2.0, (18, 24), downstream=99.0, returned=False, runaway=True,
             far_field_frac=0.95, n_activated_bins=15),
    ]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_returned_false_alone_disqualifies():
    # A record identical to a qualifying one EXCEPT returned=False is disqualified;
    # with no other qualifying record -> raise.
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, returned=False)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_far_field_frac_above_cap_alone_disqualifies():
    # A record that is returned + local count but dumps mass far away (far_field_frac
    # above cap) is NOT local -> disqualified -> raise (no other candidate).
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0,
                    far_field_frac=FAR_FRAC_CAP + 0.2)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_select_raises_when_no_post_kick_window():
    # All windows fall during the kick drive (window_after_dur_kick=False) -> no
    # artifact-free window -> raise (kept from the original contract).
    rbb = {0: [_rec(1.0, (2, 6), after=False),
               _rec(2.0, (2, 6), after=False)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


# ---------------------------------------------------------------------------
# FIX 1: _spatial_extent must EXCLUDE the source bin from every metric
# ---------------------------------------------------------------------------

def test_spatial_extent_source_artifact_is_nonlocal():
    # Construct: source bin (index 0) at the origin with a HUGE direct-stim mass,
    # plus ten far bins (radius > far_radius) each with a modest mass, all else 0.
    # With the OLD source-inclusive math the huge source inflates the floor and
    # dilutes the far-field denominator -> looks (0, 0.0, ~0.09) = falsely local.
    # The fixed non-source math must report all ten far bins as activated and
    # far_field_frac == 1.0 (every NON-source bin is far) = NON-local.
    far_radius_mm = 5.0
    n_far = 10
    # bin 0 = source at origin; bins 1..10 = far bins on a line at x=10mm (> far_radius);
    # plus a couple of quiet near bins (mass 0) to make the geometry realistic.
    centers = [[0.0, 0.0]]                      # source
    centers += [[10.0, float(i)] for i in range(n_far)]  # ten far bins, radius ~10mm
    centers += [[1.0, 0.0], [2.0, 0.0]]         # two quiet near bins (radius < far_radius)
    bin_centers = np.array(centers, dtype=float)

    net_bins = np.zeros(len(bin_centers), dtype=float)
    net_bins[0] = 10000.0                        # source direct-stim footprint
    for i in range(1, 1 + n_far):
        net_bins[i] = 100.0                      # ten far bins

    n_act, r95, far_frac = _spatial_extent(net_bins, bin_centers, src_bin_idx=0,
                                           far_radius_mm=far_radius_mm)
    assert n_act == 10                           # ten far bins, NOT 0
    assert far_frac == 1.0                        # all NON-source mass is far, NOT ~0.09
    assert r95 > far_radius_mm                    # r95 over non-source activated bins


# ---------------------------------------------------------------------------
# FIX 2: r95_mm is now gated
# ---------------------------------------------------------------------------

def test_r95_above_cap_alone_disqualifies():
    # A record returned + far low + small n_activated, but r95_mm huge (50mm).
    # Pass L=20 so r95_cap = 0.30 * 20 = 6mm is active -> disqualified -> raise.
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, r95_mm=50.0,
                    far_field_frac=0.0, n_activated_bins=1)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb, L=20.0)


def test_r95_gated_by_fallback_cap_when_L_omitted():
    # When L is omitted the fixed fallback cap (6mm) still gates r95. A 50mm r95
    # record is disqualified even without L.
    assert R95_CAP_MM_FALLBACK == 6.0
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, r95_mm=50.0)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


# ---------------------------------------------------------------------------
# FIX 3 + per-condition single-gate disqualification
# ---------------------------------------------------------------------------

def test_source_floor_not_met_alone_disqualifies():
    # source_resp below SOURCE_FLOOR (kick did not ignite locally) -> disqualified.
    rbb = {0: [_rec(0.3, (18, 24), source=SOURCE_FLOOR - 1.0)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_downstream_floor_not_met_alone_disqualifies():
    # downstream_resp below DOWNSTREAM_FLOOR (no first-gen response) -> disqualified.
    rbb = {0: [_rec(0.3, (18, 24), downstream=DOWNSTREAM_FLOOR - 1.0)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_frac_time_on_post_above_max_alone_disqualifies():
    # frac_time_on_post above FRAC_TIME_ON_MAX (mostly-on, never settles) -> disqualified.
    rbb = {0: [_rec(0.3, (18, 24), frac_time_on_post=FRAC_TIME_ON_MAX + 0.1)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_pass_frac_seeds_below_threshold_alone_disqualifies():
    # A record qualifying on every numeric/spatial gate but with too few seeds
    # locally-returned (pass_frac_seeds below SEED_PASS_FRAC) -> disqualified.
    assert SEED_PASS_FRAC == pytest.approx(2.0 / 3.0)
    rbb = {0: [_rec(0.3, (18, 24), pass_frac_seeds=SEED_PASS_FRAC - 0.2)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_half_failing_boost_does_not_qualify_over_a_clean_boost():
    # Multi-rep-bin robustness: boost X has 1 qualifying bin + 2 failing bins
    # (pass_frac_bins = 1/3 < BIN_PASS_FRAC) so it does NOT qualify despite a LOWER
    # boost; boost Y has all 3 bins qualifying (pass_frac_bins = 1.0) and is selected.
    # A half/mostly-failing boost cannot be rescued by averaging.
    assert BIN_PASS_FRAC == 0.5  # boundary: 1/3 < 0.5 fails, 1.0 passes
    rbb = {
        # rep bin 0
        0: [_rec(0.2, (18, 24), downstream=8.0),                       # X bin0 qualifies
            _rec(0.5, (18, 24), downstream=8.0)],                      # Y bin0 qualifies
        # rep bin 1
        1: [_rec(0.2, (18, 24), downstream=8.0, returned=False),       # X bin1 FAILS
            _rec(0.5, (18, 24), downstream=8.0)],                      # Y bin1 qualifies
        # rep bin 2
        2: [_rec(0.2, (18, 24), downstream=8.0, runaway=True,
                 returned=False),                                       # X bin2 FAILS
            _rec(0.5, (18, 24), downstream=8.0)],                      # Y bin2 qualifies
    }
    out = _select_calibration(rbb)
    assert out["kick_boost"] == 0.5                # Y, NOT the lower half-failing X=0.2
    assert out["pass_frac_bins"] == 1.0


# ---------------------------------------------------------------------------
# CORE mode: confound gate + the metric-sourcing split (differenced vs raw)
# ---------------------------------------------------------------------------

def test_core_only_confound_alone_disqualifies():
    # A record whose DIFFERENCED response (core_kick - core_only) looks perfectly
    # local + returned on every numeric/spatial gate, BUT the raw core_only run has an
    # active-fraction event in the measurement window -> core_only_quiet=False ->
    # CONFOUNDED. The (core_mean, kick, win) can never qualify; with no other
    # candidate -> raise (go/no-go). This is the core-mode-specific gate: the
    # differenced response could look local+returned only because the core was
    # already firing, so it must be vetoed.
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, source=10.0,
                    returned=True, runaway=False, far_field_frac=0.0,
                    n_activated_bins=1, r95_mm=1.0,
                    core_only_quiet=False)]}
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


def test_raw_returned_false_disqualifies_even_if_differenced_is_local():
    # The metric-sourcing split: returned is a RAW core_kick field, NOT recoverable
    # from the differenced (core_kick - core_only) response. A record whose DIFFERENCED
    # spatial metrics are perfectly local (small n_activated, low far_field, small r95,
    # core_only quiet) but whose RAW core_kick run did NOT return (returned=False) is
    # disqualified -> raise. This guards that a "differenced-returned" can never stand
    # in for the raw-returned gate (differencing two persistently-active runs can fall
    # to 0 and FAKE 'returned'; the gate must read the raw run).
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, source=10.0,
                    returned=False,            # RAW core_kick did NOT return
                    runaway=False, far_field_frac=0.0,
                    n_activated_bins=1, r95_mm=1.0,
                    core_only_quiet=True)]}    # differenced/confound side all clean
    with pytest.raises(RuntimeError):
        _select_calibration(rbb)


# ---------------------------------------------------------------------------
# DIAGNOSE: explore-mode no-go does NOT raise; strict-mode no-go still raises
# ---------------------------------------------------------------------------

def test_explore_mode_no_go_does_not_raise():
    # Nothing qualifies (a single big-wave / not-returned record). explore mode must
    # NOT raise — it returns status=='NO_GO' with a non-empty candidate table, each
    # candidate reporting a first_failed_gate (so the L20-explore phase keeps the data).
    rbb = {0: [_rec(1.0, (18, 24), downstream=50.0, returned=False, runaway=True,
                    far_field_frac=0.9, n_activated_bins=14)]}
    diag = _diagnose(rbb, mode="explore")          # MUST NOT raise
    assert diag["status"] == "NO_GO"
    assert len(diag["candidates"]) == 1
    assert diag["selected"] is None
    for c in diag["candidates"]:
        assert c["first_failed_gate"] is not None   # every candidate reports where it died


def test_strict_mode_no_go_still_raises():
    # Same no-go records but strict mode keeps the GO/NO-GO RuntimeError.
    rbb = {0: [_rec(1.0, (18, 24), downstream=50.0, returned=False, runaway=True,
                    far_field_frac=0.9, n_activated_bins=14)]}
    with pytest.raises(RuntimeError):
        _diagnose(rbb, mode="strict")


def test_explore_mode_go_selects_and_does_not_raise():
    # A qualifying record -> explore mode returns status=='GO' and a selected candidate.
    rbb = {0: [_rec(0.2, (18, 24), downstream=8.0)]}
    diag = _diagnose(rbb, mode="explore")
    assert diag["status"] == "GO"
    assert diag["selected"]["kick_boost"] == 0.2


# ---------------------------------------------------------------------------
# DIAGNOSE: first_failed_gate correctness
# ---------------------------------------------------------------------------

def test_first_failed_gate_is_pass_local_when_only_far_field_high():
    # A candidate failing ONLY pass_local (far_field high; everything else clean) reports
    # first_failed_gate == 'pass_local'.
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, source=10.0, returned=True,
                    far_field_frac=FAR_FRAC_CAP + 0.2)]}
    diag = _diagnose(rbb, mode="explore")
    c = diag["candidates"][0]
    assert c["first_failed_gate"] == "pass_local"


def test_first_failed_gate_is_core_quiet_and_class_confounded():
    # A candidate whose core self-ignites (core_only_quiet=False) reports
    # first_failed_gate == 'pass_core_quiet' (it precedes every numeric gate in the
    # waterfall) AND candidate_class == 'confounded'.
    rbb = {0: [_rec(0.3, (18, 24), downstream=10.0, source=10.0, returned=True,
                    core_only_quiet=False)]}
    diag = _diagnose(rbb, mode="explore")
    c = diag["candidates"][0]
    assert c["first_failed_gate"] == "pass_core_quiet"
    assert c["candidate_class"] == "confounded"


# ---------------------------------------------------------------------------
# DIAGNOSE: 3-class candidate labels (one record per class)
# ---------------------------------------------------------------------------

def test_candidate_class_labels():
    # Build one record per class and assert candidate_class is right.
    rbb = {0: [
        # confounded: NOT core_only_quiet (core self-ignites).
        _rec(0.1, (18, 24), core_only_quiet=False),
        # silent: core quiet, but source-excluded early response ~0 (< DOWNSTREAM_FLOOR).
        _rec(0.2, (18, 24), downstream=DOWNSTREAM_FLOOR - 1.0),
        # escape: core quiet, downstream present, but NOT returned (runaway / sustained).
        _rec(0.3, (18, 24), downstream=10.0, returned=False, runaway=True),
        # linear_probe: returned + local + not runaway (the small-kick W_small candidate).
        _rec(0.4, (18, 24), downstream=10.0, returned=True, runaway=False,
             far_field_frac=0.0, n_activated_bins=1, r95_mm=1.0),
        # finite_event: returned + not runaway BUT NOT local (far_field above cap).
        _rec(0.5, (18, 24), downstream=10.0, returned=True, runaway=False,
             far_field_frac=FAR_FRAC_CAP + 0.2),
    ]}
    diag = _diagnose(rbb, mode="explore")
    by_kb = {c["kick_boost"]: c["candidate_class"] for c in diag["candidates"]}
    assert by_kb[0.1] == "confounded"
    assert by_kb[0.2] == "silent"
    assert by_kb[0.3] == "escape"
    assert by_kb[0.4] == "linear_probe"
    assert by_kb[0.5] == "finite_event"


# ---------------------------------------------------------------------------
# DIAGNOSE: gate waterfall is monotonic non-increasing; final == n selected
# ---------------------------------------------------------------------------

def test_waterfall_monotonic_and_final_equals_selected():
    # Mixed candidates dying at different gates + one clean qualifier.
    rbb = {0: [
        _rec(0.1, (18, 24), core_only_quiet=False),                  # dies pass_core_quiet
        _rec(0.2, (18, 24), downstream=DOWNSTREAM_FLOOR - 1.0),      # dies pass_early
        _rec(0.3, (18, 24), far_field_frac=FAR_FRAC_CAP + 0.2),     # dies pass_local
        _rec(0.4, (18, 24), returned=False, runaway=True),          # dies pass_return
        _rec(0.5, (18, 24), downstream=8.0),                        # qualifies
    ]}
    diag = _diagnose(rbb, mode="explore")
    counts = [n for (_stage, n) in diag["waterfall"]]
    # Non-increasing along the gate order.
    for a, b in zip(counts, counts[1:]):
        assert b <= a
    # The 'SELECTED' stage label is last and equals the number of qualifying candidates.
    last_stage, last_n = diag["waterfall"][-1]
    assert last_stage == "SELECTED"
    n_selected = sum(1 for c in diag["candidates"] if c["qualifies"])
    assert last_n == n_selected == 1
    # Waterfall stages are: total + every gate (in order) + SELECTED.
    stages = [s for (s, _n) in diag["waterfall"]]
    assert stages == ["total", *list(_GATE_ORDER), "SELECTED"]


# ---------------------------------------------------------------------------
# DIAGNOSE: artifact dump writes all files even on no-go (offline, no SNN)
# ---------------------------------------------------------------------------

def test_write_diagnostics_dumps_all_files_on_no_go(tmp_path):
    from run_m3_kick_calibration import _write_diagnostics
    rbb = {0: [
        _rec(0.2, (18, 24), far_field_frac=FAR_FRAC_CAP + 0.2),     # finite_event
        _rec(0.5, (18, 24), core_only_quiet=False),                 # confounded
    ]}
    diag = _diagnose(rbb, mode="explore")
    assert diag["status"] == "NO_GO"
    out = str(tmp_path)
    _write_diagnostics(diag, rbb, thresholds={"SOURCE_FLOOR": SOURCE_FLOOR},
                       config={"mode": "explore"}, out_dir=out)
    for fname in ("config.json", "thresholds.json", "git_sha.txt",
                  "candidate_table.csv", "per_seed_metrics.csv",
                  "gate_waterfall.csv", "best_failed_candidate.json", "STATUS.md"):
        assert os.path.exists(os.path.join(out, fname)), fname
    # candidate_table has the two candidates (plus a header).
    rows = open(os.path.join(out, "candidate_table.csv")).read().strip().splitlines()
    assert len(rows) == 1 + 2
    # STATUS.md reports the NO_GO verdict.
    assert "NO_GO" in open(os.path.join(out, "STATUS.md")).read()


# ---------------------------------------------------------------------------
# FIXED core_quiet gate: RELATIVE-to-bare-background (not an absolute floor)
# ---------------------------------------------------------------------------

def test_relative_gate_passes_quiet_but_warm_core():
    # A core whose core_only window activity ≈ the bare sheet (ratio ~1.0) is QUIET
    # under the relative gate, even though its raw spike counts (≈15–25) sit far above
    # the OLD absolute floor (< 2). Real data: narrow cores core_only_downstream≈bare.
    # Under the old absolute floor this was core_only_quiet=False; the relative gate
    # must report True.
    quiet = _core_only_quiet(
        co_src=0.7, co_downstream=14.7, co_frac_on_post=0.0,
        co_event_in_win=False,
        nc_src=0.7, nc_downstream=14.7,                 # bare sheet == core_only here
    )
    assert quiet is True


def test_relative_gate_fails_self_igniting_core():
    # A core whose core_only downstream is 5× the bare sheet (genuine self-ignition,
    # like the WIDE core: 71–443 vs bare 25) is NOT quiet — it produces materially more
    # activity than the bare sheet, so the difference can't be trusted.
    bare = 25.0
    quiet = _core_only_quiet(
        co_src=100.0, co_downstream=5.0 * bare, co_frac_on_post=0.0,
        co_event_in_win=False,
        nc_src=0.7, nc_downstream=bare,
    )
    assert quiet is False
    # Sanity: 5×bare is well above the CORE_BG_RATIO*bare + CORE_BG_MARGIN bar.
    assert 5.0 * bare > CORE_BG_RATIO * bare + CORE_BG_MARGIN


def test_offline_reclassify_flips_narrow_keeps_wide_confounded(tmp_path):
    # Offline-reclassify unit test on a tiny synthetic candidate_table.csv: a NARROW-like
    # row (core_only ≈ bare) flips from old NO_GO (confounded) to qualifying under the
    # relative gate; a WIDE-like row (core_only ≫ bare) stays confounded.
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from reclassify_m3_calibration_gate import _reclassify_run

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cols = ["kick_boost", "win_lo", "win_hi",
            "source_resp", "downstream_resp", "n_activated_bins", "r95_mm",
            "far_field_frac", "frac_time_on_post", "pass_frac_seeds", "pass_frac_bins",
            "core_only_source_resp", "core_only_downstream_resp",
            "core_only_frac_time_on_post", "no_core_no_kick_downstream",
            "window_after_dur_kick", "returned", "runaway", "core_only_quiet",
            "qualifies", "candidate_class", "first_failed_gate"]
    # NARROW-like: core_only≈bare (14.7≈14.7), a clean LOCAL+RETURNED differenced
    # response (small r95, low far field, returned). Old dump = confounded / NO_GO.
    narrow = [0.75, 18.0, 24.0,
              10.0, 8.0, 1, 1.0,
              0.0, 0.0, 1.0, 1.0,
              0.7, 14.7,
              0.0, 14.7,
              1, 1, 0, 0,          # window_after, returned, runaway, OLD core_only_quiet=0
              0, "confounded", "pass_core_quiet"]
    # WIDE-like: core_only ≫ bare (200 vs 25) — genuine self-ignition.
    wide = [0.75, 18.0, 24.0,
            10.0, 8.0, 1, 1.0,
            0.0, 0.0, 1.0, 1.0,
            150.0, 200.0,
            0.0, 25.0,
            1, 1, 0, 0,
            0, "confounded", "pass_core_quiet"]
    with open(run_dir / "candidate_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(narrow)
        w.writerow(wide)

    res = _reclassify_run(str(run_dir))
    by_cls = {(c["core_only_source_resp"], c["core_only_downstream_resp"]): c
              for c in res["candidates"]}
    narrow_c = by_cls[(0.7, 14.7)]
    wide_c = by_cls[(150.0, 200.0)]
    # Narrow flips: now core_quiet, qualifies (NOT confounded any more).
    assert narrow_c["core_only_quiet"] is True
    assert narrow_c["candidate_class"] != "confounded"
    assert narrow_c["qualifies"] is True
    # Wide stays confounded: core_quiet stays False, still dies at pass_core_quiet.
    assert wide_c["core_only_quiet"] is False
    assert wide_c["candidate_class"] == "confounded"
    assert wide_c["first_failed_gate"] == "pass_core_quiet"
    # Run-level: old NO_GO -> new GO (the narrow row now qualifies).
    assert res["old_status"] == "NO_GO"
    assert res["new_status"] == "GO"


# ---------------------------------------------------------------------------
# EVENT-ALIGNED window (B-branch finite-event operator): pure onset + slicer.
# These exercise the onset-detection + event-aligned window helpers WITHOUT the
# engine (synthetic down_diff traces). They cover: (1) onset is found at the known
# jump time, (2) flat noise never crosses -> no event, (3) the event-aligned window
# is anchored at t0 (NOT at a fixed delay).
# ---------------------------------------------------------------------------

def test_event_onset_detects_jump_at_known_time():
    # Synthetic differenced downstream trace: ~0 baseline, then a clear jump that begins
    # at t=130 ms (well after t_kick+dur_kick=100+18=118ms). _event_onset must return
    # t0=130 ms (the jump bin's left edge), detected=True. bin_ms=2ms => bin index of
    # the jump = 130/2 = 65.
    bin_ms = 2.0
    t_kick, dur_kick = 100.0, 18.0
    n = 250                                   # covers t in [0, 500) ms
    down_diff = np.zeros(n)
    down_diff[: int(t_kick / bin_ms)] = 0.0   # quiet pre-kick baseline (exactly 0)
    jump_bin = int(130.0 / bin_ms)            # = 65 -> t=130 ms
    down_diff[jump_bin:] = 10.0 * ONSET_MIN_MASS   # well above the onset bar
    t0_ms, detected = _event_onset(
        down_diff, bin_ms, t_kick, dur_kick,
        baseline_lo=t_kick - 50.0, baseline_hi=t_kick)
    assert detected is True
    assert t0_ms == pytest.approx(130.0)


def test_event_onset_no_event_on_flat_subthreshold_trace():
    # A flat trace whose values never exceed the ONSET_MIN_MASS absolute floor (here
    # well below it, with tiny noise) -> no crossing -> detected=False, t0=NaN.
    bin_ms = 2.0
    t_kick, dur_kick = 100.0, 18.0
    n = 250
    rng = np.random.default_rng(0)
    # Noise amplitude kept far below ONSET_MIN_MASS so the absolute floor is never met.
    down_diff = np.abs(rng.normal(0.0, 0.1 * ONSET_MIN_MASS, size=n))
    t0_ms, detected = _event_onset(
        down_diff, bin_ms, t_kick, dur_kick,
        baseline_lo=t_kick - 50.0, baseline_hi=t_kick)
    assert detected is False
    assert np.isnan(t0_ms)


def test_event_aligned_window_is_anchored_at_t0_not_fixed_delay():
    # The event-aligned window is [t0+EA_DELTA1, t0+EA_DELTA2] — it follows the detected
    # onset, NOT a fixed delay from t_kick. Detect t0 on a trace whose onset is LATE
    # (t=160 ms) and assert the window slides with it.
    bin_ms = 2.0
    t_kick, dur_kick = 100.0, 18.0
    n = 250
    down_diff = np.zeros(n)
    onset_t = 160.0
    down_diff[int(onset_t / bin_ms):] = 10.0 * ONSET_MIN_MASS
    t0_ms, detected = _event_onset(
        down_diff, bin_ms, t_kick, dur_kick,
        baseline_lo=t_kick - 50.0, baseline_hi=t_kick)
    assert detected is True
    assert t0_ms == pytest.approx(onset_t)
    lo, hi = _event_aligned_window(t0_ms, EA_DELTA1, EA_DELTA2)
    # Window anchored at the (late) onset, NOT at a fixed t_kick-relative delay.
    assert lo == pytest.approx(onset_t + EA_DELTA1)
    assert hi == pytest.approx(onset_t + EA_DELTA2)
    # Sanity: an EARLIER onset produces an EARLIER window (the window tracks t0).
    lo_early, hi_early = _event_aligned_window(130.0, EA_DELTA1, EA_DELTA2)
    assert lo_early < lo and hi_early < hi


def test_event_onset_ignores_crossing_before_dur_kick():
    # A spike DURING the kick drive (t in [t_kick, t_kick+dur_kick)) must NOT be picked
    # as the onset — direct-stim contamination. The genuine event begins later; onset is
    # reported at the later, artifact-free crossing.
    bin_ms = 2.0
    t_kick, dur_kick = 100.0, 18.0
    n = 250
    down_diff = np.zeros(n)
    # Direct-stim blip during [100, 118): a single high bin at t=104.
    down_diff[int(104.0 / bin_ms)] = 10.0 * ONSET_MIN_MASS
    # Genuine event onset at t=140 (>= t_kick+dur_kick=118).
    down_diff[int(140.0 / bin_ms):] = 10.0 * ONSET_MIN_MASS
    t0_ms, detected = _event_onset(
        down_diff, bin_ms, t_kick, dur_kick,
        baseline_lo=t_kick - 50.0, baseline_hi=t_kick)
    assert detected is True
    assert t0_ms == pytest.approx(140.0)   # NOT 104 (during-kick blip is skipped)
