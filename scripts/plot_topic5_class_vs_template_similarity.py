#!/usr/bin/env python3
"""Topic 5 — cohort similarity of the EVENT-AGGREGATED class field vs the AGGREGATE-TEMPLATE field.

Two ways to build a subject's interictal propagation field for a class:
  (1) project the aggregate template (the t_a / t_b record's typical_rank), or
  (2) aggregate that class's actual events (weight-normalized masked ranks).
This figure asks, cohort-level: are the two fields the same? If yes, the cheap template projection is
a faithful stand-in for the per-event aggregation (and the difference between classes lives in the
per-event dispersion, not the aggregate field).

Two panels (each one independent question, CLAUDE.md §7):
  (a) per-subject spatial similarity |r| of class-field vs template-field (mirror-invariant field
      correlation, same plane), one point per class A/B — the cohort statistic.
  (b) pooled contact-level agreement: aggregate-template order vs event-aggregated order, every
      contact of every subject, with the identity line — the construct-level "why".

Broad substrate (where the per-event class field is dense). EXPLORATORY secondary; descriptive
construct-equivalence, not a cohort hypothesis test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import src.topic5_event_resolved_alignment as erm
from src.propagation_contact_plane_readout import make_plane_grid, corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN

GEOM = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
OUT = _ROOT / "results/topic5_ictal_recruitment/event_resolved_alignment"
COL_A, COL_B = "#5b8c5a", "#8c5a8c"   # class A (green) / class B (purple) — neutral, not the swap red/blue


def _subject_sim(ds_sid):
    ds, subj = ds_sid.split("_", 1)
    ta_f, tb_f = GEOM / f"{ds_sid}_t_a.json", GEOM / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        return None
    try:
        b = erm.load_event_labels_ranks(ds, subj)
    except Exception:
        return None
    pa = json.loads(ta_f.read_text()); pb = json.loads(tb_f.read_text())
    order = b["channel_names"]
    tar = np.array([{c["name"]: c.get("typical_rank") for c in pa["channels"]}.get(n, np.nan) for n in order], float)
    tbr = np.array([{c["name"]: c.get("typical_rank") for c in pb["channels"]}.get(n, np.nan) for n in order], float)
    cm = erm.map_clusters_to_templates(np.array(b["cluster_template_ranks"][0], float),
                                       np.array(b["cluster_template_ranks"][1], float), tar, tbr)
    if cm["ambiguous"]:
        return None
    lab = {"t_a": [k for k, t in cm["map"].items() if t == "t_a"][0],
           "t_b": [k for k, t in cm["map"].items() if t == "t_b"][0]}
    X, Y = make_plane_grid()
    out = {"subject_id": ds_sid, "dataset": ds}
    contact = {"A": [], "B": []}
    for tid, plane, cls in (("t_a", pa, "A"), ("t_b", pb, "B")):
        sig = erm.class_template_sigma(plane, X=X, Y=Y)
        Ftpl = erm.field_from_contact_values(plane, {c["name"]: c["typical_rank"] for c in plane["channels"]},
                                             sigma=sig, X=X, Y=Y)
        cv = erm.class_aggregate_contact_values(b, lab[tid])
        Fcls = erm.field_from_contact_values(plane, {n: d["value"] for n, d in cv.items()},
                                             support_by_name={n: d["support"] for n, d in cv.items()},
                                             sigma=sig, X=X, Y=Y)
        if Ftpl is None or Fcls is None:
            out[f"sim_{cls}"] = None; continue
        r = corr_pair_mirror_invariant(Ftpl["T"], Ftpl["S"], Fcls["T"], Fcls["S"], S_THRESH, OVERLAP_MIN)
        out[f"sim_{cls}"] = (abs(r["corr"]) if (not r["insufficient_overlap"] and r["corr"] is not None
                             and np.isfinite(r["corr"])) else None)
        tmap = {c["name"]: c["typical_rank"] for c in plane["channels"]}
        contact[cls] = [(tmap[n], cv[n]["value"]) for n in tmap
                        if n in cv and cv[n]["value"] is not None and np.isfinite(cv[n]["value"])
                        and tmap[n] is not None and np.isfinite(tmap[n])]
    out["contact"] = contact
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    subs = sorted(p.stem[:-len("_t_a")] for p in GEOM.glob("*_t_a.json"))
    rows = [r for r in (_subject_sim(s) for s in subs) if r is not None]
    simA = [r["sim_A"] for r in rows if r.get("sim_A") is not None]
    simB = [r["sim_B"] for r in rows if r.get("sim_B") is not None]
    cA = np.array([p for r in rows for p in r["contact"]["A"]], float).reshape(-1, 2)
    cB = np.array([p for r in rows for p in r["contact"]["B"]], float).reshape(-1, 2)
    allc = np.vstack([cA, cB]) if cA.size and cB.size else (cA if cA.size else cB)
    rho_all = spearmanr(allc[:, 0], allc[:, 1]).correlation

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.6))
    # Panel A — per-subject field similarity by class
    data = [simA, simB]
    parts = ax[0].violinplot(data, positions=[0, 1], widths=0.8, showmedians=True)
    for pc, c in zip(parts["bodies"], (COL_A, COL_B)):
        pc.set_facecolor(c); pc.set_alpha(0.45)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("0.3")
    rng = np.random.default_rng(0)
    for x, d, c in ((0, simA, COL_A), (1, simB, COL_B)):
        ax[0].scatter(x + (rng.random(len(d)) - 0.5) * 0.22, d, s=34, color=c, edgecolors="white",
                      linewidths=0.5, zorder=3)
        ax[0].text(x, 1.012, f"median {np.median(d):.3f}\nmin {np.min(d):.3f}  (n={len(d)})",
                   ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["propagation class A", "propagation class B"])
    ax[0].set_ylabel("spatial similarity of the two interictal fields  |r|")
    ax[0].set_ylim(min(0.55, min(simA + simB) - 0.05), 1.06)
    ax[0].axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax[0].set_title("per subject: event-aggregated class field vs aggregate-template field", fontsize=10.5)

    # Panel B — pooled contact-level agreement
    ax[1].plot([0, 1], [0, 1], color="0.6", lw=1.0, ls="--", zorder=1)
    ax[1].scatter(cA[:, 0], cA[:, 1], s=16, color=COL_A, alpha=0.5, label="class A contacts", zorder=2)
    ax[1].scatter(cB[:, 0], cB[:, 1], s=16, color=COL_B, alpha=0.5, label="class B contacts", zorder=2)
    ax[1].set_xlabel("aggregate-template propagation order (early 0 → late 1)")
    ax[1].set_ylabel("event-aggregated propagation order (early 0 → late 1)")
    ax[1].set_xlim(-0.03, 1.03); ax[1].set_ylim(-0.03, 1.03); ax[1].set_aspect("equal", adjustable="box")
    ax[1].text(0.03, 0.97, f"all contacts, all subjects\nSpearman r = {rho_all:.3f}  (n={allc.shape[0]})",
               va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax[1].set_title("contact level: do the two methods order contacts the same?", fontsize=10.5)
    ax[1].legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    med = np.median(simA + simB)
    fig.suptitle(f"Building the interictal field from a class's events ≈ projecting its aggregate template "
                 f"(broad cohort, N={len(rows)}; median field |r| = {med:.3f})", fontsize=12)
    fig.text(0.5, 0.005, "EXPLORATORY construct-equivalence: the two field-construction methods agree, so the "
             "aggregate template is a faithful stand-in; the A/B difference lives in per-event dispersion, not the "
             "aggregate field.", ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    figdir = Path(args.out) / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    fp = figdir / "class_vs_template_field_similarity_cohort.png"
    fig.savefig(fp, dpi=140); plt.close(fig)
    json.dump({"note": "cohort similarity: event-aggregated class field vs aggregate-template field; "
               "EXPLORATORY construct-equivalence (descriptive, near 1 = methods agree).",
               "n_subjects": len(rows), "median_sim_field": float(med),
               "median_sim_A": float(np.median(simA)), "median_sim_B": float(np.median(simB)),
               "min_sim": float(np.min(simA + simB)), "contact_spearman_all": float(rho_all),
               "rows": [{k: r.get(k) for k in ("subject_id", "dataset", "sim_A", "sim_B")} for r in rows]},
              open(Path(args.out) / "class_vs_template_field_similarity_cohort.json", "w"), indent=2)
    print(f"[done] N={len(rows)} subjects, median field |r|={med:.3f}, contact Spearman={rho_all:.3f} -> {fp}")


if __name__ == "__main__":
    main()
