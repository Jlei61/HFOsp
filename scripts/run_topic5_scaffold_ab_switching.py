#!/usr/bin/env python3
"""Topic5 V3d — scaffold A/B lateral-switching producer (per subject).

Builds the fixed-orientation interictal A/B contrast coordinate ``C_AB(t)`` on the
peri-onset broadband energy of every eligible seizure, gates each window by a
within-shaft-shuffle null (``axis_present``), scores per-seizure near-onset lateral
polarization (``locking``) against an exhaustive circular-shift time null, and
combines the valid seizures into a subject-level combinatorial null (H1).

This is the go/no-go real-data producer for spec
``docs/superpowers/specs/2026-07-09-topic5-v3d-scaffold-ab-lateral-switching-design.md``
(§7 tables 1/2/3, §6.1 locked constants, §10 fail-closed). It REUSES the Fig3-B
window machinery (``_compute_values`` / ``_seizure_args`` / ``_keep_window``) — it does
NOT re-load EDF and it NEVER touches any mirror-invariant path (§2.2/§10). The
numeric core lives in ``src.topic5_scaffold_ab_contrast`` (frozen, tested).

Outputs to ``results/topic5_ictal_recruitment/scaffold_ab_switching/per_subject/``:
  <ds_sid>_scaffold_ab_per_window.csv    (table 1: per window continuous)
  <ds_sid>_scaffold_ab_per_seizure.csv   (table 2: per seizure state)
  <ds_sid>_scaffold_ab_summary.json      (table 3: per-subject summary + drops)
  <ds_sid>_scaffold_ab_contrast.npz      (per-seizure C_AB/present arrays for the plotter)

CLI: ``--subject <ds_sid> [--gate-nperm N] [--seed S]`` (single subject) or
``--all-ok [--gate-nperm N] [--seed S]`` (batch over every Fig3-B paper-index
status==ok subject; writes ``cohort_index.json``/``.csv``). Fail-closed per
seizure and per subject; if a subject has no template-B axis, no usable
seizure, or raises unexpectedly, a drop is recorded and the batch continues
(no crash).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _compute_values,
    _load_axis,
)
from scripts.plot_topic5_signed_broadband_similarity_timecourse import (  # noqa: E402
    _eligible_idxs,
)
from scripts.run_topic5_fig3b_maxab_spatial_null import (  # noqa: E402
    START_SEC,
    STEP_SEC,
    STOP_SEC,
    WINDOW_SEC,
    _keep_window,
    _ok_subjects,
    _seizure_args,
)
from src.topic5_scaffold_ab_contrast import (  # noqa: E402
    _signed_mean,  # authoritative near-onset signed mean (matches classify_event.near_side)
    axis_present,
    circular_shift_null_seizure,
    classify_event,
    contrast_timecourse,
    derive_joint_contacts,
    label_sides,
    locking_statistic,
    subject_locking_null,
    template_pair_tier,
)

OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/scaffold_ab_switching"
SUB_DIR = OUT_DIR / "per_subject"

# Locked ranges (spec §6.1). WINDOW/STEP/START/STOP reused from the Fig3-B contract.
FAR_PRE = (-120.0, -60.0)
NEAR_ONSET = (-30.0, 10.0)
NEAR_PRE = (-30.0, 0.0)
EARLY_ICTAL = (0.0, 10.0)
DELTA_SIDE = 0.2
N_VALID_SEIZURE_MIN = 3
N_PERM_SUBJECT = 1000

# Single common window-center grid (spec/plan: center = window_start + WINDOW/2;
# window_start in {-120,...,+10} -> center in {-115,...,+15} = 66 windows).
GRID_CENTERS = np.arange(START_SEC + WINDOW_SEC / 2.0, STOP_SEC - WINDOW_SEC / 2.0 + 1e-9, STEP_SEC)


def _f(v) -> float:
    """Plain float (NaN preserved) for CSV cells."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _j(v):
    """JSON-safe scalar: non-finite floats -> None (Python json would emit bare NaN)."""
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if np.isfinite(f) else None


def _grid_col(center: float) -> int:
    return int(round((float(center) - GRID_CENTERS[0]) / STEP_SEC))


def _drop_summary(ds_sid: str, reason: str, drops: list) -> dict:
    """Fail-closed subject summary (spec §10): explicit drop, no silent defaults."""
    return {
        "subject": ds_sid,
        "status": "drop",
        "drop_reason": reason,
        "template_pair_tier": None,
        "rho_AB": None,
        "n_joint": None,
        "axis_present": {"testable": False, "low_dof": None, "qc": None},
        "H1": {"L_obs": None, "L_null_p95": None, "subject_locked": False,
               "p": None, "n_valid_seizures": 0, "H1_eligible": False},
        "drops": drops,
    }


def compute_subject(ds_sid: str, gate_nperm: int, seed: int):
    """Return (per_window_rows, per_seizure_rows, summary, npz_dict|None)."""
    axis_b = _load_axis(ds_sid, "t_b")
    if axis_b is None:
        return [], [], _drop_summary(ds_sid, "no_t_b_axis", []), None
    try:
        idxs = _eligible_idxs(ds_sid)
    except Exception as exc:  # no eligibility metadata -> nothing to do, fail-closed
        return [], [], _drop_summary(ds_sid, f"no_eligible_cache: {type(exc).__name__}: {exc}", []), None

    per_window_rows: list[dict] = []
    per_seizure_rows: list[dict] = []
    seiz: list[dict] = []
    drops: list[dict] = []

    for si in idxs:
        si = int(si)
        try:
            _ds, _i, sw, offset, bl, matched, names, starts, window_vals, _onset = \
                _compute_values(_seizure_args(ds_sid, si))
        except Exception as exc:  # fail-closed per seizure
            drops.append({"seizure_idx": si, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        keep = [k for k, lo in enumerate(starts) if _keep_window(float(lo))]
        if not keep:
            drops.append({"seizure_idx": si, "reason": "no_kept_windows"})
            continue
        centers = np.array([float(starts[k]) + WINDOW_SEC / 2.0 for k in keep])
        W = np.asarray(window_vals, float)[keep]

        jc = derive_joint_contacts(matched, axis_b, W)
        if jc["status"] != "ok":
            drops.append({"seizure_idx": si, "reason": jc["status"], "n_joint": int(jc.get("n_joint", 0))})
            continue

        Ej = W[:, jc["idx"]]
        ct = contrast_timecourse(Ej, jc["D_AB"], jc["eA"], jc["eB"])
        ap = axis_present(Ej, jc["names"], jc["eA"], jc["eB"],
                          np.random.default_rng(seed + si), n_perm=gate_nperm)
        C = ct["C_AB"]
        present = ap["present"]
        ls = locking_statistic(C, present, centers, FAR_PRE, NEAR_ONSET)
        cev = classify_event(C, present, centers, FAR_PRE, NEAR_ONSET, NEAR_PRE, EARLY_ICTAL, DELTA_SIDE)
        csn = circular_shift_null_seizure(C, present, centers, FAR_PRE, NEAR_ONSET)
        side_lbl = label_sides(C, present, DELTA_SIDE)
        side01 = 0.5 + 0.5 * np.clip(C, -1.0, 1.0)

        for w, cc in enumerate(centers):
            per_window_rows.append({
                "subject": ds_sid, "seizure_idx": si,
                "window_center_sec": float(cc),
                "C_AB": _f(C[w]), "r_A": _f(ct["r_A"][w]), "r_B": _f(ct["r_B"][w]),
                "maxAB": _f(ct["maxAB"][w]),
                "axis_present": bool(present[w]),
                "within_shaft_p": _f(ap["within_shaft_p"][w]),
                "side01": _f(side01[w]),
                "side_label": str(side_lbl[w]),
            })

        # per-seizure H1 contribution gate (spec §6.3 ③④): testable & shufflable, >=3 present
        # windows each side (== finite polars), and an exhaustive time null with >=40 valid shifts.
        far_near_ok = bool(np.isfinite(ls["polar_far"]) and np.isfinite(ls["polar_near"]))
        h1_valid = bool(ap["testable"] and not ap["low_dof"] and far_near_ok and csn["status"] == "ok")

        per_seizure_rows.append({
            "subject": ds_sid, "seizure_idx": si,
            "n_axis_present_win": int(np.asarray(present, bool).sum()),
            "polar_far": _f(ls["polar_far"]), "polar_near": _f(ls["polar_near"]),
            "locking": _f(ls["locking"]),
            "locking_shift_p": _f(csn["locking_shift_p"]), "n_valid_shift": int(csn["n_valid_shift"]),
            "far_side": cev["far_side"], "near_side": cev["near_side"], "event_class": cev["event_class"],
        })

        # near-onset-side alignment sign for the plot (flip each seizure so its near side points up).
        sn = _signed_mean(C, present, centers, *NEAR_ONSET)
        align = float(np.sign(sn)) if (np.isfinite(sn) and sn != 0.0) else 1.0

        seiz.append({
            "seizure_idx": si, "centers": centers, "C_AB": np.asarray(C, float),
            "present": np.asarray(present, bool), "align_sign": align, "h1_valid": h1_valid,
            "csn": csn, "n_joint": int(jc["n_joint"]), "rho_AB": float(jc["rho_AB"]),
            "tier": jc["tier"], "testable": bool(ap["testable"]), "low_dof": bool(ap["low_dof"]),
            "qc": ap["qc"], "event_class": cev["event_class"], "locking": _f(ls["locking"]),
            "locking_shift_p": _f(csn["locking_shift_p"]),
        })

    if not seiz:
        return per_window_rows, per_seizure_rows, _drop_summary(ds_sid, "no_usable_seizure", drops), None

    valid = [r for r in seiz if r["h1_valid"]]
    if valid:
        h1 = subject_locking_null([r["csn"] for r in valid], n_perm=N_PERM_SUBJECT, seed=seed)
    else:
        h1 = {"L_obs": float("nan"), "L_null_p95": float("nan"), "subject_locked": False,
              "p": float("nan"), "n_valid_seizures": 0}
    h1_eligible = len(valid) >= N_VALID_SEIZURE_MIN

    # reference seizure for subject-level pair-tier / axis-present QC = most complete joint set.
    ref = max(seiz, key=lambda r: r["n_joint"])
    tier = template_pair_tier(ref["rho_AB"])

    summary = {
        "subject": ds_sid,
        "status": "ok",
        "template_pair_tier": tier,
        "rho_AB": _j(ref["rho_AB"]),
        "n_joint": int(ref["n_joint"]),
        "axis_present": {
            "testable": bool(ref["testable"]),
            "low_dof": bool(ref["low_dof"]),
            "qc": {k: _j(v) for k, v in ref["qc"].items()},
        },
        "H1": {
            "L_obs": _j(h1["L_obs"]),
            "L_null_p95": _j(h1["L_null_p95"]),
            "subject_locked": bool(h1["subject_locked"]),
            "p": _j(h1["p"]),
            "n_valid_seizures": int(h1["n_valid_seizures"]),
            "H1_eligible": bool(h1_eligible),
        },
        "n_eligible": len(idxs),
        "n_kept_seizures": len(seiz),
        "gate_nperm": int(gate_nperm),
        "n_perm_subject": int(N_PERM_SUBJECT),
        "seed": int(seed),
        "ranges_sec": {"far_pre": list(FAR_PRE), "near_onset": list(NEAR_ONSET),
                       "near_pre": list(NEAR_PRE), "early_ictal": list(EARLY_ICTAL),
                       "delta_side": DELTA_SIDE},
        "per_seizure_jc": [
            {"seizure_idx": r["seizure_idx"], "n_joint": r["n_joint"], "rho_AB": _j(r["rho_AB"]),
             "tier": r["tier"], "testable": r["testable"], "low_dof": r["low_dof"],
             "h1_valid": r["h1_valid"], "event_class": r["event_class"],
             "locking": _j(r["locking"]), "locking_shift_p": _j(r["locking_shift_p"])}
            for r in seiz
        ],
        "drops": drops,
        "caveats": [
            "跨 seizure 的 median C_AB 只能看趋势，不能独自证切换——相反侧发作互相抵消（图1 的粗线是按各 seizure "
            "near-onset 主侧对齐后的 median，才对应 H1 的 |mean|）。",
            "H1 = 近-onset 侧向极化/选择（locking = |mean_near| − |mean_far|），不是已证明的 preictal switching。",
            f"gate 用 axis_present within-shaft null n_perm={gate_nperm}（preview 质量；locked 默认 1000）；"
            "C_AB 本身是精确的（无置换）。",
        ],
    }

    # npz for the plotter: per-seizure arrays aligned onto the common center grid.
    n_sz = len(seiz)
    cab = np.full((n_sz, GRID_CENTERS.size), np.nan)
    pres = np.zeros((n_sz, GRID_CENTERS.size), bool)
    for i, r in enumerate(seiz):
        for cc, cval, pv in zip(r["centers"], r["C_AB"], r["present"]):
            jcol = _grid_col(cc)
            if 0 <= jcol < GRID_CENTERS.size:
                cab[i, jcol] = cval
                pres[i, jcol] = bool(pv)
    npz_dict = {
        "grid_centers": GRID_CENTERS,
        "cab": cab,
        "present": pres,
        "align_sign": np.array([r["align_sign"] for r in seiz], float),
        "h1_valid": np.array([r["h1_valid"] for r in seiz], bool),
        "seizure_idx": np.array([r["seizure_idx"] for r in seiz], int),
    }
    return per_window_rows, per_seizure_rows, summary, npz_dict


_PER_WINDOW_COLS = ["subject", "seizure_idx", "window_center_sec", "C_AB", "r_A", "r_B", "maxAB",
                    "axis_present", "within_shaft_p", "side01", "side_label"]
_PER_SEIZURE_COLS = ["subject", "seizure_idx", "n_axis_present_win", "polar_far", "polar_near",
                     "locking", "locking_shift_p", "n_valid_shift", "far_side", "near_side", "event_class"]


def _write_csv(fp: Path, cols: list[str], rows: list[dict]) -> None:
    with fp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_subject(ds_sid: str, gate_nperm: int, seed: int) -> dict:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    per_window_rows, per_seizure_rows, summary, npz_dict = compute_subject(ds_sid, gate_nperm, seed)

    _write_csv(SUB_DIR / f"{ds_sid}_scaffold_ab_per_window.csv", _PER_WINDOW_COLS, per_window_rows)
    _write_csv(SUB_DIR / f"{ds_sid}_scaffold_ab_per_seizure.csv", _PER_SEIZURE_COLS, per_seizure_rows)
    (SUB_DIR / f"{ds_sid}_scaffold_ab_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if npz_dict is not None:
        np.savez_compressed(SUB_DIR / f"{ds_sid}_scaffold_ab_contrast.npz", **npz_dict)
    return summary


# --------------------------------------------------------------------------
# --all-ok batch (mirrors run_topic5_fig3b_maxab_spatial_null's fail-closed
# batch + cohort index pattern; reuses its _ok_subjects() paper-index reader).
# --------------------------------------------------------------------------
COHORT_JSON = OUT_DIR / "cohort_index.json"
COHORT_CSV = OUT_DIR / "cohort_index.csv"
COHORT_COLS = ["subject", "status", "drop_reason", "template_pair_tier", "rho_AB",
               "n_seizures", "n_valid_seizures", "H1_eligible", "testable", "low_dof",
               "subject_locked", "H1_p", "L_obs", "n_joint"]


def _record_from_summary(summ: dict) -> dict:
    """Cohort-index row from a per-subject summary dict (ok or drop status).

    `_drop_summary` always populates `axis_present`/`H1` sub-dicts (with None/False
    placeholders) but omits `n_kept_seizures`/`n_joint` entirely, so those two use
    `.get(...)` -> None for drop rows.
    """
    ax = summ.get("axis_present") or {}
    h1 = summ.get("H1") or {}
    return {
        "subject": summ["subject"],
        "status": summ["status"],
        "drop_reason": summ.get("drop_reason") or "",
        "template_pair_tier": summ.get("template_pair_tier"),
        "rho_AB": summ.get("rho_AB"),
        "n_seizures": summ.get("n_kept_seizures"),
        "n_valid_seizures": h1.get("n_valid_seizures"),
        "H1_eligible": h1.get("H1_eligible"),
        "testable": ax.get("testable"),
        "low_dof": ax.get("low_dof"),
        "subject_locked": h1.get("subject_locked"),
        "H1_p": h1.get("p"),
        "L_obs": h1.get("L_obs"),
        "n_joint": summ.get("n_joint"),
    }


def _write_cohort_index(records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with COHORT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COHORT_COLS, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow({c: rec.get(c, "") for c in COHORT_COLS})
    n_ok = sum(1 for r in records if r["status"] == "ok")
    COHORT_JSON.write_text(json.dumps({
        "generated_by": "scripts/run_topic5_scaffold_ab_switching.py",
        "spec": "docs/superpowers/specs/2026-07-09-topic5-v3d-scaffold-ab-lateral-switching-design.md",
        "tier": "per-subject H1 verdicts (pre-registered primary hypothesis test, spec §3/§6); "
                "subject_locked is a per-subject result, not yet a cohort-level claim -- see spec "
                "§4 for the two-axis tier layering (template_pair_tier x axis_present testability) "
                "that must accompany any subject_locked count.",
        "n_subjects": len(records), "n_ok": n_ok, "n_drop": len(records) - n_ok,
        "subjects": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_batch(gate_nperm: int, seed: int) -> None:
    """Fail-closed batch over every Fig3-B paper-index status==ok subject (spec §12 step 5)."""
    subjects = _ok_subjects()
    print(f"processing {len(subjects)} subject(s)", flush=True)
    records = []
    for i, ds_sid in enumerate(subjects, 1):
        t0 = time.time()
        print(f"[{i}/{len(subjects)}] {ds_sid} ...", flush=True)
        try:
            summ = run_subject(ds_sid, gate_nperm, seed)
            rec = _record_from_summary(summ)
            if summ["status"] == "ok":
                print(f"    ok   tier={rec['template_pair_tier']} rho_AB={rec['rho_AB']} "
                      f"H1_eligible={rec['H1_eligible']} locked={rec['subject_locked']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            else:
                print(f"    drop {rec['drop_reason']} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:  # fail-closed: unexpected per-subject failure -> drop, batch continues
            rec = {c: None for c in COHORT_COLS}
            rec["subject"] = ds_sid
            rec["status"] = "drop"
            rec["drop_reason"] = f"{type(exc).__name__}: {exc}"
            print(f"    DROP (exception) {rec['drop_reason']} ({time.time() - t0:.0f}s)", flush=True)
        records.append(rec)
        _write_cohort_index(records)  # incremental: survives a mid-batch interruption
    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\nDONE: {n_ok}/{len(records)} ok -> {COHORT_CSV}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--all-ok", action="store_true",
                    help="batch: every subject with an observed Fig3-B figure (paper index "
                         "status==ok); writes cohort_index.json/.csv, fail-closed per subject")
    ap.add_argument("--gate-nperm", type=int, default=1000,
                    help="within-shaft null permutations for axis_present (locked default 1000; "
                         "use 200 for a fast preview render)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.all_ok:
        run_batch(args.gate_nperm, args.seed)
        return

    t0 = time.time()
    summ = run_subject(args.subject, args.gate_nperm, args.seed)
    if summ["status"] == "drop":
        print(json.dumps({"subject": summ["subject"], "status": "drop",
                          "drop_reason": summ["drop_reason"]}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "subject": summ["subject"], "template_pair_tier": summ["template_pair_tier"],
            "rho_AB": summ["rho_AB"], "n_joint": summ["n_joint"],
            "axis_present": summ["axis_present"], "H1": summ["H1"],
            "n_kept_seizures": summ["n_kept_seizures"], "n_eligible": summ["n_eligible"],
            "n_seizure_drops": len(summ["drops"]),
        }, ensure_ascii=False, indent=2))
    print(f"[{args.subject}] done in {time.time() - t0:.1f}s -> {SUB_DIR}", flush=True)


if __name__ == "__main__":
    main()
