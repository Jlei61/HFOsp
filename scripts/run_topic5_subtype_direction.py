#!/usr/bin/env python3
"""Topic 5 C-line runner — subtype x activation-direction (per-subject, exploratory).

PILOT-FIRST (plan §8): runs the alignment-contract check + per-subject Q_axis / Q_pol /
geometry-gate / direction-clustering for a subject set, writes per-subject JSON + figure A,
then HARD STOP for human inspection before the full cohort. Reuses the tested pure functions
in src.topic5_subtype_direction and the A-line rose loaders (no reinvented I/O).

Two layers kept separate (review Point 1):
  Q_axis (mode=axis): subtype separation of the seizure AXIS angle — the only layer eligible
                      to explain the sign-free A-line split-half instability.
  Q_pol  (mode=pol) : subtype separation of the endpoint POLARITY — descriptive direction
                      read-out only; cannot explain the A-line.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_axis_direction_rose import (_load_frame, _electrode_kind,
                                                     CACHE_DIR, ACTIVATION_KEY)
from src.topic5_axis_direction import gradient_angle, axial_mean, rotate_to_reference
from src.topic1_topic5_bridge import load_topic5_subtype_labels
from src.topic5_axis_alignment import seizure_parity_subsets
from src.topic5_subtype_direction import (within_subject_perm_p, direction_clustering,
                                          coord_aspect_ratio, align_subtype_to_direction,
                                          oddeven_subtype_imbalance)

AUDIT = _ROOT / "results/topic5_ictal_recruitment/t0_eligibility_audit.csv"
RESULTS_ROOT = _ROOT / "results"
OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/subtype_direction"
FIG_DIR = OUT_DIR / "figures"
HFA_JOINT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/hfa_joint_confirm.json"
ASPECT_MIN = 0.15           # locked (plan §3.5): coord_aspect < this → near-1D caveat
PERM_B = 2000
PERM_SEED = 20260615
PILOT_SUBJECTS = ["epilepsiae_590", "epilepsiae_958", "epilepsiae_922"]
SUBTYPE_COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def _all_epilepsiae_with_frame():
    """Full C-line universe: every epilepsiae cache subject that has a display frame."""
    out = []
    for p in sorted(CACHE_DIR.glob("epilepsiae_*.npz")):
        ds_sid = p.stem
        if _load_frame(ds_sid) is not None:
            out.append(ds_sid)
    return out


def _idx_to_seizure_id(ds_sid):
    m = {}
    with open(AUDIT) as fh:
        for r in csv.DictReader(fh):
            if r["subject_id"] == ds_sid:
                m[int(r["seizure_idx"])] = r["seizure_id"]
    return m


def _angles_by_idx(ds_sid, x, y, names_rec, activation):
    """{eligible_idx: θ (radians, [0,2π), NaN if not computable)} + the eligible_idxs list."""
    data = np.load(CACHE_DIR / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.load(open(CACHE_DIR / f"{ds_sid}.json"))
    cidx = {str(n): i for i, n in enumerate(data["channels"])}
    key = ACTIVATION_KEY[activation]
    out = {}
    for sz in meta.get("eligible_idxs", []):
        k = f"{key}__{sz}"
        if k not in data.files:
            continue
        arr = np.asarray(data[k], float)
        vals = np.array([arr[cidx[n]] if n in cidx else np.nan for n in names_rec])
        out[int(sz)] = gradient_angle(x, y, vals)
    return out, [int(i) for i in meta.get("eligible_idxs", [])]


def run_subject(ds_sid, band, activation, verbose=True):
    ds, subj = ds_sid.split("_", 1)
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return {"subject": ds_sid, "status": "no_frame"}
    _rec, x, y, names_rec = loaded
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    try:
        sub = load_topic5_subtype_labels(subj, band, RESULTS_ROOT, dataset=ds)
    except FileNotFoundError:
        return {"subject": ds_sid, "status": "no_zer_json", "band": band}
    if sub["status"] != "ok":                       # plan §2 step 1: hard status gate
        return {"subject": ds_sid, "status": f"zer_status_{sub['status']}", "band": band}
    sid2sub = sub["seizure_id_to_subtype"]

    angles_by_idx, eligible_idxs = _angles_by_idx(ds_sid, x, y, names_rec, activation)
    idx2sid = _idx_to_seizure_id(ds_sid)
    aligned = align_subtype_to_direction(sid2sub, idx2sid, eligible_idxs, angles_by_idx)

    # --- alignment contract diagnostics (plan §2 step 3: print, fail-loud on zero overlap) ---
    elig_ids = {idx2sid[i] for i in eligible_idxs if i in idx2sid}
    contract = {
        "n_zer_ids": len(sid2sub),
        "n_t0_eligible_ids": len(elig_ids),
        "n_id_overlap": len(elig_ids & set(sid2sub)),
        "n_aligned_with_finite_angle": len(aligned),
        "subtype_dist_aligned": {},
    }
    ang = np.array([r["theta"] for r in aligned], float)
    lab = np.array([r["subtype"] for r in aligned], int)
    for s in sorted(set(lab.tolist())):
        contract["subtype_dist_aligned"][int(s)] = int((lab == s).sum())

    # --- geometry gate (plan §3.5) ---
    kind, detail = _electrode_kind(ds, subj, names_rec)
    aspect = coord_aspect_ratio(x, y)
    geom_caveat = bool((not np.isfinite(aspect)) or aspect < ASPECT_MIN or kind == "SEEG")

    # --- Step-0 clustering + two-layer permutation ---
    clustering = direction_clustering(ang) if ang.size else {"clustered": False, "R_axial": float("nan"), "R_dir": float("nan"), "n": 0}
    q_axis = within_subject_perm_p(ang, lab, mode="axis", B=PERM_B, rng=np.random.default_rng(PERM_SEED))
    q_pol = within_subject_perm_p(ang, lab, mode="pol", B=PERM_B, rng=np.random.default_rng(PERM_SEED))

    cohort_eligible = bool(q_axis["eligibility"] == "ok" and clustering["clustered"] and not geom_caveat)
    reasons = []
    if q_axis["eligibility"] != "ok":
        reasons.append(f"perm:{q_axis['eligibility']}")
    if not clustering["clustered"]:
        reasons.append("not_clustered")
    if geom_caveat:
        reasons.append(f"geometry:{kind}/aspect={aspect:.2f}")

    out = {
        "subject": ds_sid, "status": "ok", "band": band, "activation": activation,
        "alignment_contract": contract,
        "geometry": {"electrode_kind": kind, "detail": detail, "coord_aspect": float(aspect),
                     "geometry_caveat": geom_caveat},
        "direction_clustering": clustering,
        "Q_axis": q_axis, "Q_pol": q_pol,
        "cohort_eligible": cohort_eligible,
        "tier": "cohort-test" if cohort_eligible else "case-series",
        "case_series_reasons": reasons,
        "eligible_idxs": eligible_idxs,
        "idx_to_subtype": {int(r["idx"]): int(r["subtype"]) for r in aligned},
        "aligned": aligned,
    }
    if verbose:
        print(f"\n=== {ds_sid}  band={band} activation={activation} ===")
        print(f"  alignment: z-ER ids={contract['n_zer_ids']}  T0-eligible ids={contract['n_t0_eligible_ids']}  "
              f"overlap={contract['n_id_overlap']}  aligned(finite θ)={contract['n_aligned_with_finite_angle']}")
        print(f"  subtype dist (aligned): {contract['subtype_dist_aligned']}")
        print(f"  geometry: {kind} ({detail}) aspect={aspect:.3f} caveat={geom_caveat}")
        print(f"  clustering: R_axial={clustering['R_axial']:.3f} R_dir={clustering['R_dir']:.3f} clustered={clustering['clustered']}")
        print(f"  Q_axis: elig={q_axis['eligibility']} k={q_axis['k']} dropped={q_axis['dropped_subtypes']} "
              f"T_obs={q_axis['T_obs']} p={q_axis['p']}")
        print(f"  Q_pol : elig={q_pol['eligibility']} T_obs={q_pol['T_obs']} p={q_pol['p']}")
        print(f"  --> tier={out['tier']}  reasons={reasons}")
    return out


def plot_subject_rose(res, activation):
    """Figure A: per-seizure direction ticks colored by subtype, vs the seizure axial axis."""
    if res.get("status") != "ok" or not res["aligned"]:
        return None
    ang = np.array([r["theta"] for r in res["aligned"]], float)
    lab = np.array([r["subtype"] for r in res["aligned"]], int)
    ref = axial_mean(ang)
    if not np.isfinite(ref):
        return None
    rot = rotate_to_reference(ang, ref)
    subs = sorted(set(lab.tolist()))
    fig = plt.figure(figsize=(7.4, 8.0))
    axp = fig.add_subplot(111, projection="polar")
    for j, s in enumerate(subs):
        a = rot[lab == s]
        color = SUBTYPE_COLORS[j % len(SUBTYPE_COLORS)]
        for k, ai in enumerate(a):
            axp.plot([ai, ai], [0, 0.92], color=color, lw=2.0, alpha=0.8,
                     label=(f"subtype {s} (n={a.size})" if k == 0 else None))
    axp.plot([0, 0], [0, 1.12], color="black", lw=3.0, zorder=4, label="seizure axis (axial mean)")
    axp.plot([np.pi, np.pi], [0, 1.12], color="black", lw=3.0, zorder=4)
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.set_yticklabels([])
    g = res["geometry"]
    qa, qp = res["Q_axis"], res["Q_pol"]
    cl = res["direction_clustering"]
    pretty = res["subject"].replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    title = (f"Patient {pretty} — {g['electrode_kind']} (aspect={g['coord_aspect']:.2f}"
             f"{', CAVEAT' if g['geometry_caveat'] else ''})\n"
             f"per-seizure activation direction by z-ER subtype  ({activation})\n"
             f"clustered={cl['clustered']} (R_axial={cl['R_axial']:.2f})  |  "
             f"Q_axis p={qa['p']}  Q_pol p={qp['p']}  [{res['tier']}]")
    axp.set_title(title, fontsize=10, pad=12)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.08, right=0.92)
    axp.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
    cap = ("ticks = individual seizures (NOT interictal templates), colored by subtype; "
           "black = seizure axial mean set to 0°/180°; a subtype clustered near one tick band "
           "= that subtype lights up a distinct direction")
    if g["geometry_caveat"]:
        cap = "GEOMETRY CAVEAT (SEEG / near-1D): direction angle is fragile.  " + cap
    fig.text(0.5, 0.04, cap, ha="center", fontsize=7.6, color="0.35", wrap=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{res['subject']}_subtype_direction_{activation}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _load_aline_instability():
    """{ds_sid: |corr_even − corr_odd|} from the hfa×joint split-half arms (plan §4)."""
    if not HFA_JOINT.exists():
        return {}
    d = json.load(open(HFA_JOINT))
    def per_arm(arm):
        return {r["subject_id"]: r["real_median_abs_corr"] for r in d["arms"][arm]["per_subject"]}
    even, odd = per_arm("split_half_even"), per_arm("split_half_odd")
    return {s: abs(even[s] - odd[s]) for s in even if s in odd}


def aline_connection(results):
    """Per-subject 3-scalar A-line connection table (plan §4). Activation-independent except
    `subtype_axis_sep`, which is read from each result's Q_axis (so call per activation)."""
    instab = _load_aline_instability()
    rows = []
    for res in results:
        if res.get("status") != "ok":
            continue
        sid = res["subject"]
        even, odd = seizure_parity_subsets(res["eligible_idxs"])
        idx2sub = {int(k): int(v) for k, v in res["idx_to_subtype"].items()}
        imbalance = oddeven_subtype_imbalance(even, odd, idx2sub)
        rows.append({
            "subject": sid,
            "subtype_axis_sep_T": res["Q_axis"]["T_obs"],
            "subtype_axis_sep_p": res["Q_axis"]["p"],
            "q_axis_eligibility": res["Q_axis"]["eligibility"],
            "oddeven_subtype_imbalance": (None if (imbalance != imbalance) else imbalance),
            "oddeven_aline_instability": instab.get(sid),
            "tier": res["tier"],
        })
    return rows


def cohort_summary(results):
    eligible = [r for r in results if r.get("tier") == "cohort-test"]
    pos = [r for r in eligible if (r["Q_axis"]["p"] is not None and r["Q_axis"]["p"] < 0.05)]
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    binom_p = None
    if eligible:
        from scipy.stats import binomtest
        binom_p = float(binomtest(len(pos), len(eligible), 0.05, alternative="greater").pvalue)
    return {
        "n_subjects_run": len(results),
        "n_status_ok": n_ok,
        "n_cohort_eligible": len(eligible),
        "cohort_eligible_subjects": [r["subject"] for r in eligible],
        "n_qaxis_positive": len(pos),
        "qaxis_positive_subjects": [r["subject"] for r in pos],
        "binom_p_reference": binom_p,
        "verdict_plain": (
            "no cohort-eligible subject" if not eligible else
            f"{len(pos)}/{len(eligible)} cohort-eligible subjects show subtype axis-angle "
            f"separation at p<0.05 (binom-vs-5% reference p={binom_p:.3g}); exploratory, descriptive."
        ),
    }


def plot_connection(conn_rows, activation, out_path):
    """Figure C: A-line connection — subtype axis separation vs A-line split-half instability,
    point size = odd/even subtype imbalance. Descriptive (plan §4)."""
    pts = [r for r in conn_rows if r["subtype_axis_sep_T"] is not None
           and r["oddeven_aline_instability"] is not None]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    for r in pts:
        imb = r["oddeven_subtype_imbalance"] or 0.0
        size = 40 + 400 * imb
        sig = (r["subtype_axis_sep_p"] is not None and r["subtype_axis_sep_p"] < 0.05)
        eligible = (r["tier"] == "cohort-test")
        ax.scatter(r["subtype_axis_sep_T"], r["oddeven_aline_instability"], s=size,
                   facecolor=(("#d62728" if sig else "#1f77b4") if eligible else "none"),
                   edgecolor=("#d62728" if sig else "#1f77b4"), lw=1.6, alpha=0.85)
        ax.annotate(r["subject"].replace("epilepsiae_", "E"),
                    (r["subtype_axis_sep_T"], r["oddeven_aline_instability"]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=9,
               label="cohort-eligible (geometry+clustering OK)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#1f77b4",
               markersize=9, label="case-series (geometry/clustering caveat — T_axis unreliable)"),
    ], loc="best", fontsize=8, frameon=False)
    ax.set_xlabel("subtype axis-angle separation  T_axis  (rad; higher = subtypes use more different axes)")
    ax.set_ylabel("A-line split-half instability\n|corr_even − corr_odd|  (hfa×joint)")
    ax.set_title(f"C↔A connection ({activation} direction): does subtype axis heterogeneity\n"
                 "track A-line odd/even instability?  point size = odd/even subtype imbalance\n"
                 "filled = cohort-eligible; red = subtype axis sep p<0.05 (exploratory, descriptive)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="default = full epilepsiae cohort (all cache subjects with a frame)")
    ap.add_argument("--pilot", action="store_true", help="run only the 3 pilot subjects")
    ap.add_argument("--band", default="gamma_ER", choices=["gamma_ER", "broad_ER"])
    ap.add_argument("--activations", nargs="*", default=["broadband", "hfa"],
                    choices=list(ACTIVATION_KEY))
    args = ap.parse_args()
    if args.subjects:
        subs = args.subjects
    elif args.pilot:
        subs = PILOT_SUBJECTS
    else:
        subs = _all_epilepsiae_with_frame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for activation in args.activations:
        print(f"\n########## activation = {activation} ##########")
        results = []
        for ds_sid in subs:
            res = run_subject(ds_sid, args.band, activation)
            results.append(res)
            with (OUT_DIR / f"{ds_sid}__subtype_direction_{activation}.json").open("w") as fh:
                json.dump(res, fh, indent=2, default=str)
            fig = plot_subject_rose(res, activation)
            if fig:
                print(f"  wrote figure {fig.name}")
        summ = cohort_summary(results)
        conn = aline_connection(results)
        with (OUT_DIR / f"cohort_summary_{activation}.json").open("w") as fh:
            json.dump({"activation": activation, "band": args.band,
                       "cohort": summ, "aline_connection": conn}, fh, indent=2, default=str)
        cfig = plot_connection(conn, activation, FIG_DIR / f"cohort_C_to_A_connection_{activation}.png")
        print(f"\n  [{activation}] {summ['verdict_plain']}")
        if cfig:
            print(f"  wrote figure {cfig.name}")
    print("\n[COHORT DONE] inspect cohort_summary_*.json + figures, then archive.")


if __name__ == "__main__":
    main()
