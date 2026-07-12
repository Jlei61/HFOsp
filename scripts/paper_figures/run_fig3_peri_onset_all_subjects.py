#!/usr/bin/env python3
"""Fail-closed batch driver for Fig3-B peri-onset field similarity across all subjects.

For every subject that has an interictal propagation template axis
(``results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/<subject>_t_a.json``)
this driver reproduces the locked Fig3-B pipeline:

  1. ``scripts/plot_topic5_signed_broadband_similarity_timecourse.py``
     (1-150 Hz summed spectrogram log power on notch-filtered input — no extra
     FFT-bin line mask — per-channel baseline robust-z,
     onset-aligned ``[-120,+20]s``, 10 s window, 2 s step) -> per-seizure CSV.
  2. ``scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py``
     -> paper-ready two-panel PNG/PDF + summary JSON (panel a = maxAB sign-free
     scaffold similarity, panel b = signed template A/B polarity sidecar).

Each subject runs in its own subprocess wrapped in try/except: one subject's
failure never aborts the batch, and its ``drop_reason`` is recorded instead. A
subject index CSV/JSON summarises the whole cohort.

This driver does **not** change the Fig3-B figure style and does **not** compute
a formal cohort statistic. It only assembles the per-subject material pool that
supports Fig3-B; the formal cohort shift remains the Fig3-A Data-vs-Null panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REAL_DIR = ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
FIELD_DIR = ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
PAPER_DIR = ROOT / "results/paper-ready-figure/fig3_peri_onset_field_similarity"
FIG_DIR = PAPER_DIR / "figures"
INDEX_CSV = PAPER_DIR / "fig3_peri_onset_subject_index.csv"
INDEX_JSON = PAPER_DIR / "fig3_peri_onset_subject_index.json"

TIMECOURSE_SCRIPT = ROOT / "scripts/plot_topic5_signed_broadband_similarity_timecourse.py"
PAPER_SCRIPT = ROOT / "scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py"

# Locked Fig3-B contract (docs/figure_style_guide.md 5a). Do not change.
TIMECOURSE_ARGS = [
    "--start-sec", "-120", "--stop-sec", "20",
    "--band-lo", "1", "--band-hi", "150",
    "--window-sec", "10", "--step-sec", "2",
]

TIMECOURSE_SUMMARY = "{sid}_signed_broadband_1_150Hz_similarity_timecourse_m120_p20_10s_step2s_summary.json"
PAPER_SUMMARY = "{sid}_peri_onset_field_similarity_paper_ready_summary.json"

# Canonical index columns (docs plan requires at least these). Extra debugging
# fields (_stage / _detail) live only in the JSON records, never the CSV.
INDEX_COLUMNS = [
    "subject",
    "status",
    "drop_reason",
    "n_eligible",
    "n_seizures",
    "n_seizure_drops",
    "n_windows",
    "maxAB_median_of_window_medians",
    "maxAB_median_of_window_variances",
    "signed_A_median_of_window_medians",
    "signed_B_median_of_window_medians",
    "source_csv",
    "figure_png",
    "figure_pdf",
]


def _discover_subjects() -> list[str]:
    subs = sorted({p.name[: -len("_t_a.json")] for p in REAL_DIR.glob("*_t_a.json")})
    if not subs:
        raise FileNotFoundError(f"no *_t_a.json under {REAL_DIR}")
    return subs


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def _tail(text: str | None, n: int = 1) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def _blank_record(ds_sid: str) -> dict:
    rec = {c: "" for c in INDEX_COLUMNS}
    rec["subject"] = ds_sid
    return rec


def _process_subject(ds_sid: str) -> dict:
    rec = _blank_record(ds_sid)

    # Step 1: upstream 1-150 Hz peri-onset timecourse -> per-seizure CSV.
    tc = _run([sys.executable, str(TIMECOURSE_SCRIPT), "--subject", ds_sid, *TIMECOURSE_ARGS])
    if tc.returncode != 0:
        rec["status"] = "drop"
        rec["drop_reason"] = _tail(tc.stderr) or _tail(tc.stdout) or "timecourse failed (no output)"
        rec["_stage"] = "timecourse"
        rec["_detail"] = _tail((tc.stdout or "") + "\n" + (tc.stderr or ""), 10)
        return rec

    # Step 2: paper-ready two-panel figure from that CSV.
    pf = _run([sys.executable, str(PAPER_SCRIPT), "--subject", ds_sid])
    if pf.returncode != 0:
        rec["status"] = "drop"
        rec["drop_reason"] = _tail(pf.stderr) or "paper figure failed (no output)"
        rec["_stage"] = "paper_figure"
        rec["_detail"] = _tail((pf.stdout or "") + "\n" + (pf.stderr or ""), 10)
        return rec

    # Success: pull the locked readouts from the paper-ready summary JSON.
    paper_fp = FIG_DIR / PAPER_SUMMARY.format(sid=ds_sid)
    if not paper_fp.exists():
        rec["status"] = "drop"
        rec["drop_reason"] = f"paper summary missing: {paper_fp.name}"
        rec["_stage"] = "paper_figure"
        return rec
    paper = json.loads(paper_fp.read_text())
    r = paper["readouts"]
    rec["status"] = "ok"
    rec["n_seizures"] = paper["n_seizures"]
    rec["n_windows"] = paper["n_windows"]
    rec["maxAB_median_of_window_medians"] = r["maxAB_abs"]["median_of_window_medians"]
    rec["maxAB_median_of_window_variances"] = r["maxAB_abs"]["median_of_window_variances"]
    rec["signed_A_median_of_window_medians"] = r["signed_A"]["median_of_window_medians"]
    rec["signed_B_median_of_window_medians"] = r["signed_B"]["median_of_window_medians"]
    rec["source_csv"] = paper["source_csv"]
    rec["figure_png"] = paper["outputs"]["png"]
    rec["figure_pdf"] = paper["outputs"]["pdf"]

    # Partial-drop context (eligible vs processed seizures) from the upstream summary.
    tc_fp = FIELD_DIR / TIMECOURSE_SUMMARY.format(sid=ds_sid)
    if tc_fp.exists():
        tc_sum = json.loads(tc_fp.read_text())
        rec["n_eligible"] = tc_sum.get("n_eligible_requested", "")
        rec["n_seizure_drops"] = len(tc_sum.get("drops", []))
    return rec


def _write_index(records: list[dict]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({c: rec.get(c, "") for c in INDEX_COLUMNS})

    n_ok = sum(1 for r in records if r.get("status") == "ok")
    payload = {
        "figure": "Fig3-B peri-onset field similarity — per-subject material pool",
        "generated_by": "scripts/paper_figures/run_fig3_peri_onset_all_subjects.py",
        "tier": "per-subject material pool for Fig3-B; NOT a formal cohort statistic",
        "contract": {
            "band_hz": [1.0, 150.0],
            "time_range_sec": [-120.0, 20.0],
            "window_sec": 10.0,
            "step_sec": 2.0,
            "normalization": "per-channel baseline robust-z (1-150 Hz summed spectrogram log power, "
                             "notch-filtered input at 50/100/150/200 Hz; no extra FFT-bin line mask)",
            "panel_a": "max(|r_A|, |r_B|) sign-free scaffold similarity",
            "panel_b": "signed template A/B similarity (polarity sidecar)",
        },
        "n_subjects": len(records),
        "n_ok": n_ok,
        "n_drop": len(records) - n_ok,
        "subjects": records,
    }
    INDEX_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="explicit subject list (e.g. epilepsiae_1146); default = all *_t_a.json subjects",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="run subjects but do not write the cohort index (use for single-subject validation)",
    )
    args = ap.parse_args()

    subjects = args.subjects if args.subjects else _discover_subjects()
    print(f"processing {len(subjects)} subject(s)", flush=True)

    records: list[dict] = []
    for i, ds_sid in enumerate(subjects, start=1):
        print(f"[{i}/{len(subjects)}] {ds_sid} ...", flush=True)
        rec = _process_subject(ds_sid)
        records.append(rec)
        if rec["status"] == "ok":
            print(
                f"    ok  n_seizures={rec['n_seizures']} n_windows={rec['n_windows']} "
                f"maxAB_medmed={rec['maxAB_median_of_window_medians']:.4f} "
                f"A={rec['signed_A_median_of_window_medians']:.4f} "
                f"B={rec['signed_B_median_of_window_medians']:.4f}",
                flush=True,
            )
        else:
            print(f"    DROP [{rec.get('_stage', '?')}] {rec['drop_reason']}", flush=True)
        if not args.no_index:
            _write_index(records)  # incremental: index reflects progress if interrupted

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\nDONE: {n_ok}/{len(records)} ok, {len(records) - n_ok} drop", flush=True)
    if not args.no_index:
        print(f"index CSV : {INDEX_CSV}", flush=True)
        print(f"index JSON: {INDEX_JSON}", flush=True)


if __name__ == "__main__":
    main()
