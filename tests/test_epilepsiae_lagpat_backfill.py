"""Tests for scripts/run_epilepsiae_lagpat_backfill.py.

Plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md

Stage A: skeleton + record discovery (dry-print, no writes).
"""
from __future__ import annotations

from pathlib import Path


def test_output_dir_constant_is_results_subtree():
    """OUTPUT_ROOT must live under results/, never under /mnt.

    Hard guard against accidentally writing back to /mnt/epilepsia_data/...
    (legacy lagPat is part of Topic 1's current contract; never overwrite).
    """
    from scripts.run_epilepsiae_lagpat_backfill import OUTPUT_ROOT

    assert OUTPUT_ROOT == Path("results/epilepsiae_lagpat_backfill")
    assert "/mnt" not in str(OUTPUT_ROOT.resolve())


def test_smoke_lists_first_record():
    """_discover_records('253') yields at least one record with valid metadata.

    Subject 253 is the smallest pure-512Hz cohort (268 records); used for
    smoke ladder step B.4.b.
    """
    from scripts.run_epilepsiae_lagpat_backfill import _discover_records

    recs = _discover_records("253")
    assert len(recs) >= 1
    first = recs[0]
    assert first["stem"].startswith("253")
    assert first["sfreq"] in (256.0, 512.0, 1024.0)
    assert first["new_gpu_path"].exists()
    assert first["raw_data_path"].exists() and first["raw_head_path"].exists()


def test_refine_path_is_subject_not_per_record():
    """Refine artifact lives at <gpu_dir>/<subject>/_refineGpu.npz (subject-level).

    The new pipeline writes ONE refine artifact per subject (no per-record
    suffix, no `sub_` prefix). Schema: {chns_names, events_count}.
    """
    from scripts.run_epilepsiae_lagpat_backfill import _refine_path_for_subject

    p = _refine_path_for_subject("253")
    assert p.name == "_refineGpu.npz"
    assert p.parent.name == "253"


def test_load_refine_chns_for_subject_returns_tuple():
    """load_refine_chns_for_subject returns a hashable Tuple[str, ...].

    Note: deviates from the plan draft which asserted `isinstance(chns, list)`.
    The implementation uses lru_cache, which requires hashable returns; tuple
    is the correct contract.
    """
    from scripts.run_epilepsiae_lagpat_backfill import load_refine_chns_for_subject

    chns = load_refine_chns_for_subject("253")
    assert isinstance(chns, tuple)
    assert len(chns) > 0
    assert all(isinstance(c, str) for c in chns)
    # subject-level: two calls return identical content
    chns2 = load_refine_chns_for_subject("253")
    assert chns == chns2


def test_select_core_indices_mean_plus_std_matches_legacy_formula():
    """Core selector implements legacy `events_count > mean + 1*std`.

    Legacy reference:
        all_chnCounts = refine_counts
        pickChns_index = np.where(all_chnCounts > mean + 1 * std)[0]
    """
    import numpy as np

    from scripts.run_epilepsiae_lagpat_backfill import (
        _select_core_indices_mean_plus_std,
    )

    ec = np.array([1.0, 1.0, 1.0, 1.0, 100.0], dtype=float)
    # mean = 20.8, std ≈ 39.5; threshold ≈ 60.3; only the 100 passes.
    out = _select_core_indices_mean_plus_std(ec)
    assert out.tolist() == [4]

    # All-equal: nothing exceeds mean+std.
    ec2 = np.array([7.0] * 10, dtype=float)
    assert _select_core_indices_mean_plus_std(ec2).tolist() == []

    # Empty array yields empty index array (no crash).
    assert _select_core_indices_mean_plus_std(np.array([], dtype=float)).size == 0


def test_load_refine_chns_strategy_default_is_mean_plus_std():
    """Default strategy must apply core selector; refine_all is opt-in only.

    Regression guard: a previous version returned the raw _refineGpu chns_names
    (full set), which inflated chnNames cohort-wide and made packing thresholds
    too strict (Stage C audit 2026-04-30 surfaced count_ratio_med = 0.056).
    """
    import numpy as np

    import scripts.run_epilepsiae_lagpat_backfill as mod

    mod.load_refine_chns_for_subject.cache_clear()
    mod._load_subject_pack_params.cache_clear()
    # Pick a subject with pick_k=1.0 so the test stays anchored to the legacy
    # mean+std formula (subject 1084 has legacy pick_k=1.0).
    default = mod.load_refine_chns_for_subject("1084")
    refine_all = mod.load_refine_chns_for_subject("1084", strategy="refine_all")
    assert len(default) <= len(refine_all)
    z = np.load(mod._refine_path_for_subject("1084"), allow_pickle=True)
    all_names = np.asarray([str(c) for c in z["chns_names"]])
    ec = np.asarray(z["events_count"], dtype=float)
    expected = tuple(
        str(c) for c in all_names[mod._select_core_indices_mean_plus_std(ec)]
    )
    assert default == expected


def test_load_subject_pack_params_reads_per_subject_overrides():
    """Per-subject pick_k and pack_win_sec must reflect subject_params.json.

    Legacy contract: epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py:459
    sub_pickT_list = {'253':0.2, '1073':1.5, ...}
    sub_packWL_list = {'253':0.300, '1073':0.110, ...}
    """
    import scripts.run_epilepsiae_lagpat_backfill as mod

    mod._load_subject_pack_params.cache_clear()
    pick_k_253, pack_win_253 = mod._load_subject_pack_params("253")
    assert pick_k_253 == 0.2
    assert abs(pack_win_253 - 0.3) < 1e-9

    pick_k_1073, pack_win_1073 = mod._load_subject_pack_params("1073")
    assert pick_k_1073 == 1.5
    assert abs(pack_win_1073 - 0.11) < 1e-9


def test_select_core_indices_uses_per_subject_pick_k_for_subject_253():
    """Subject 253 uses pick_k=0.2 -> picks more channels than the global k=1.0
    default. Anchors Δ2 of legacy_replication_audit_2026-05-03.md."""
    import numpy as np

    import scripts.run_epilepsiae_lagpat_backfill as mod

    mod.load_refine_chns_for_subject.cache_clear()
    mod._load_subject_pack_params.cache_clear()
    default = mod.load_refine_chns_for_subject("253")  # k=0.2
    z = np.load(mod._refine_path_for_subject("253"), allow_pickle=True)
    ec = np.asarray(z["events_count"], dtype=float)
    n_at_k_02 = int(np.sum(ec > ec.mean() + 0.2 * ec.std()))
    n_at_k_10 = int(np.sum(ec > ec.mean() + 1.0 * ec.std()))
    assert len(default) == n_at_k_02
    assert len(default) >= n_at_k_10  # k=0.2 must pick at least as many as k=1.0


def test_whole_dets_units_are_seconds():
    """Defensive: probe whole_dets max value is in seconds, not samples.

    A 1h block at 512 Hz has up to ~1.84M samples but only ~3600 seconds.
    If max ever exceeds 4000, the upstream contract drifted (probably to
    samples) and pack_record would explode. Auto-guard for hfo_detector.py:82.
    """
    import numpy as np

    z = np.load(
        "results/hfo_detection/253/25300102_0000_gpu.npz", allow_pickle=True
    )
    found = False
    for arr in z["whole_dets"]:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        found = True
        assert float(a.max()) < 4000.0, "whole_dets unit drift detected (samples?)"
    assert found, "no non-empty whole_dets in the smoke record"


def test_pack_record_returns_packed_times_2d():
    """pack_record yields (n_events, 2) [start_sec, end_sec] with end >= start.

    Smoke verification on 253/25300102_0000 (pure 512 Hz, 1h block).
    """
    from scripts.run_epilepsiae_lagpat_backfill import pack_record

    pt = pack_record("253", "25300102_0000")
    assert pt.ndim == 2
    assert pt.shape[1] == 2
    if pt.shape[0] > 0:
        assert (pt[:, 1] >= pt[:, 0]).all()
        assert pt[:, 1].max() < 4000.0  # SECONDS contract


def test_compute_lagpat_record_shapes():
    """compute_lagpat_record yields the legacy-compatible schema for one record.

    Shape contract: lagPatRaw (n_pick, n_ev), lagPatRank (n_pick, n_ev),
    eventsBool (n_pick, n_ev), chnNames (n_pick,), start_t scalar Unix epoch.
    Also asserts packedTimes is returned aligned with kept events, and that
    no event column has n_participating == 0 (regression guard against
    boundary-window leakage).
    """
    from scripts.run_epilepsiae_lagpat_backfill import compute_lagpat_record

    out = compute_lagpat_record("253", "25300102_0000")
    assert set(out.keys()) >= {
        "lagPatRaw",
        "lagPatRank",
        "eventsBool",
        "chnNames",
        "start_t",
        "packedTimes",
    }
    n_pick = len(out["chnNames"])
    n_ev = out["lagPatRaw"].shape[1]
    assert out["lagPatRaw"].shape == (n_pick, n_ev)
    assert out["lagPatRank"].shape == (n_pick, n_ev)
    assert out["eventsBool"].shape == (n_pick, n_ev)
    assert out["packedTimes"].shape == (n_ev, 2)
    # start_t is Unix epoch; Epilepsiae 253 was recorded 2004-2012.
    assert 1e9 < float(out["start_t"]) < 2e9

    # Regression guard: previously a packed window with negative start
    # (boundary-centered run) was excluded by the segment rule but its
    # column was kept, leaking n_participating=0 events into downstream
    # stats. The empty column now must never be saved.
    if out["eventsBool"].size > 0:
        per_event_n_part = (out["eventsBool"] > 0).sum(axis=0)
        assert (per_event_n_part > 0).all(), (
            f"empty event columns leaked: per_event = {per_event_n_part.tolist()}"
        )


def test_process_subject_writes_log_with_required_keys(tmp_path, monkeypatch):
    """process_subject writes _backfill_log.json with the required schema.

    Uses an empty discovery list so the test runs in milliseconds and does
    not touch real data; the log writer is the unit under test.
    """
    import scripts.run_epilepsiae_lagpat_backfill as mod

    monkeypatch.setattr(mod, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_discover_records", lambda subject: [])

    log = mod.process_subject("FAKE")
    log_path = tmp_path / "FAKE" / "_backfill_log.json"
    assert log_path.exists()
    import json

    with open(log_path) as fh:
        data = json.load(fh)
    assert set(data.keys()) >= {
        "subject",
        "started_at",
        "completed_at",
        "n_records_total",
        "n_records_done",
        "n_skipped_existing",
        "n_failed",
        "failures",
        "per_record_seconds",
        "median_record_seconds",
    }
    assert data["subject"] == "FAKE"
    assert data["n_records_total"] == 0
    assert data["completed_at"] is not None
    assert log["n_records_done"] == 0


def test_process_subject_skip_existing_log_accounting(tmp_path, monkeypatch):
    """Skip-existing default: pre-existing record is logged as skipped, not done.

    Patches ``process_one_record`` so the test does not touch real Epilepsiae
    data; the unit under test is ``process_subject``'s log accounting.
    """
    import scripts.run_epilepsiae_lagpat_backfill as mod

    monkeypatch.setattr(mod, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "_discover_records",
        lambda subject: [{"stem": "FAKE_0000"}, {"stem": "FAKE_0001"}],
    )

    def fake_one(subject, stem, *, force):
        if stem == "FAKE_0000":
            return {"stem": stem, "skipped": True, "reason": "outputs exist (loadable)"}
        return {
            "stem": stem,
            "skipped": False,
            "n_events": 0,
            "n_channels": 0,
            "n_packed": 0,
            "start_t": 1e9,
            "runtime_sec": 0.01,
            "pt_path": "x",
            "lag_path": "y",
        }

    monkeypatch.setattr(mod, "process_one_record", fake_one)
    log = mod.process_subject("FAKE", max_record_sec=0)
    assert log["n_records_done"] == 1
    assert log["n_skipped_existing"] == 1
    assert log["n_failed"] == 0
    assert "FAKE_0001" in log["per_record_seconds"]


def test_process_subject_force_flag_propagates(tmp_path, monkeypatch):
    """``--force`` reaches ``process_one_record`` so existing files do get re-run."""
    import scripts.run_epilepsiae_lagpat_backfill as mod

    monkeypatch.setattr(mod, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        mod, "_discover_records", lambda subject: [{"stem": "FAKE_0000"}]
    )

    seen_force = []

    def fake_one(subject, stem, *, force):
        seen_force.append(force)
        return {
            "stem": stem,
            "skipped": False,
            "n_events": 0,
            "n_channels": 0,
            "n_packed": 0,
            "start_t": 1e9,
            "runtime_sec": 0.01,
            "pt_path": "x",
            "lag_path": "y",
        }

    monkeypatch.setattr(mod, "process_one_record", fake_one)
    mod.process_subject("FAKE", force=True, max_record_sec=0)
    assert seen_force == [True]


def test_process_subject_per_record_timeout(tmp_path, monkeypatch):
    """A record that exceeds ``max_record_sec`` is marked TimeoutError, run continues.

    Uses SIGALRM (Linux main-thread); pytest does not pre-install a SIGALRM
    handler, so the patched ``process_one_record`` sleeping for 2 s with a 1 s
    cap reliably trips the alarm.
    """
    import time as _time
    import scripts.run_epilepsiae_lagpat_backfill as mod

    monkeypatch.setattr(mod, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "_discover_records",
        lambda subject: [{"stem": "SLOW_0000"}, {"stem": "FAST_0001"}],
    )

    def fake_one(subject, stem, *, force):
        if stem == "SLOW_0000":
            _time.sleep(2)  # exceeds 1 s timeout
        return {
            "stem": stem,
            "skipped": False,
            "n_events": 0,
            "n_channels": 0,
            "n_packed": 0,
            "start_t": 1e9,
            "runtime_sec": 0.01,
            "pt_path": "x",
            "lag_path": "y",
        }

    monkeypatch.setattr(mod, "process_one_record", fake_one)
    log = mod.process_subject("FAKE", max_record_sec=1)
    assert log["n_failed"] == 1
    assert log["n_records_done"] == 1
    assert log["failures"][0]["stem"] == "SLOW_0000"
    assert log["failures"][0]["type"] == "TimeoutError"
    assert "FAST_0001" in log["per_record_seconds"]


def test_aggregate_cohort_summary_emits_one_row_per_canonical_subject(tmp_path):
    """``cohort_summary.csv`` always has 20 rows (one per canonical subject).

    Subjects with no log show ``status='not_started'`` so a partial run does
    not silently undercount. Plan §3 B.5 Step 6 contract: "20 行".
    """
    import csv as _csv
    import json as _json
    import scripts.run_epilepsiae_lagpat_backfill as mod

    seeded = {"253", "548"}
    for subj in seeded:
        d = tmp_path / subj
        d.mkdir(parents=True)
        with open(d / "_backfill_log.json", "w") as fh:
            _json.dump(
                {
                    "subject": subj,
                    "started_at": "2026-04-30T00:00:00",
                    "completed_at": "2026-04-30T01:00:00",
                    "n_records_total": 10,
                    "n_records_done": 10,
                    "n_skipped_existing": 0,
                    "n_failed": 0,
                    "failures": [],
                    "median_record_seconds": 12.3,
                },
                fh,
            )

    out = mod._aggregate_cohort_summary(output_root=tmp_path)
    assert out.name == "cohort_summary.csv"
    with open(out) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == len(mod.COHORT_SUBJECTS)
    by_subject = {r["subject"]: r for r in rows}
    assert by_subject["253"]["status"] == "completed"
    assert by_subject["253"]["n_records_done"] == "10"
    not_started = {s for s in mod.COHORT_SUBJECTS if s not in seeded}
    for s in not_started:
        assert by_subject[s]["status"] == "not_started"


def test_smoke_does_not_create_output_files():
    """--smoke must not create any files under OUTPUT_ROOT.

    Snapshots OUTPUT_ROOT contents before and after _smoke_print(); equality
    means zero new files written. Auto-guard: if a future change accidentally
    introduces a write inside _smoke_print (logs, partial artifacts, etc.),
    this test catches it instead of relying on a manual `find` audit.
    """
    import scripts.run_epilepsiae_lagpat_backfill as mod

    def _snapshot(p: Path) -> set:
        if not p.exists():
            return set()
        return set(p.rglob("*"))

    before = _snapshot(mod.OUTPUT_ROOT)
    mod._smoke_print("253")
    after = _snapshot(mod.OUTPUT_ROOT)
    assert before == after, (
        f"--smoke must not write to OUTPUT_ROOT={mod.OUTPUT_ROOT}; "
        f"new entries: {sorted(after - before)}"
    )
