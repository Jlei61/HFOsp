"""M3B Round-1 Task 4 — figures (each panel answers ONE question, §7 discipline).

Fig 1 (already produced by the Task-1 pilot): task1_pilot/figures/task1_axis_recovery.png
  Q: does the virtual readout recover the known 45deg axis? (instrument gate)
Fig 2 (here) bridge_interictal.png — INSTRUMENT-PROBE interictal bridge:
  panel A: does the model field land in the real interictal cohort? (placement vs real-to-real dist)
  panel B: does the match beat geometry? (model real-corr vs the rank-shuffle null, channel/within_shaft)
Fig 3 (here) bridge_ictal_and_gain.png:
  panel A: ictal leg = PLACEMENT-ONLY (model<->ictal corr vs its geometry null — does NOT beat)
  panel B: gain sweep — axis stays ~45deg while recruitment extent (n_part) rises (same field, two gains)

All bridges are kick-rate-field instrument-probe (P1-1); no mechanism claim.
Run from worktree root (AFTER task2/task3 JSONs exist): python scripts/run_m3b_figures.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src import propagation_contact_plane_readout as R                            # noqa: E402
from run_m3b_task2_geometry_null import (_load_reals, _field, _subject_first_median_corr,   # noqa: E402
                                         _permuted_record, MODEL_REC, REAL_DIR)
from run_m3b_task2_part3_ictal_axis import build_ictal_fields, _field as _ifield, _corr     # noqa: E402

OUT = "results/topic4_sef_hfo/m3b_bridge/task2_bridge/figures"
B = 500   # viz histograms only; authoritative p-values live in the B=2000 task2 JSONs


def _bridge_interictal(model_rec, real_items):
    mf = _field(model_rec)
    real_corr, _ = _subject_first_median_corr(mf[0], mf[1], real_items)
    # real-to-real interictal placement distribution (each subject vs the others, subject-first)
    by_subj = {}
    for ds, subj, fld in real_items:
        by_subj.setdefault((ds, subj), []).append(fld)
    subj_keys = list(by_subj.keys())
    r2r = []
    for i, k in enumerate(subj_keys):
        fi = by_subj[k][0]
        others = [(d, s, f) for (d, s), fs in by_subj.items() if (d, s) != k for f in fs]
        v, _ = _subject_first_median_corr(fi[0], fi[1], others)
        if np.isfinite(v):
            r2r.append(v)
    r2r = np.array(r2r)
    # geometry null (channel + within_shaft)
    n_ch = len(model_rec["channels"])
    from collections import defaultdict
    shafts = defaultdict(list)
    for i, c in enumerate(model_rec["channels"]):
        shafts[c.get("shaft", "?")].append(i)
    rng = np.random.default_rng(0)

    def _null(groups):
        v = []
        for _ in range(B):
            mr = _permuted_record(model_rec, groups, rng)
            mf2 = _field(mr)
            x, _n = _subject_first_median_corr(mf2[0], mf2[1], real_items)
            if np.isfinite(x):
                v.append(x)
        return np.array(v)
    null_ch = _null([list(range(n_ch))])
    null_sh = _null(list(shafts.values()))
    return real_corr, r2r, null_ch, null_sh


def main():
    os.makedirs(OUT, exist_ok=True)
    model_rec = json.loads(open(MODEL_REC).read())
    reals = _load_reals(REAL_DIR)                          # records (dicts)
    real_items = [(r["dataset"], r["subject"], _field(r)) for r in reals]

    real_corr, r2r, null_ch, null_sh = _bridge_interictal(model_rec, real_items)

    # ---- Fig 2: interictal bridge ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    axA.hist(r2r, bins=15, color="0.7", edgecolor="k", alpha=0.8)
    axA.axvline(real_corr, color="navy", lw=2.5,
                label=f"model (kick-rate-field)\ncorr={real_corr:.2f} (upper tail;\nPart-1 placement = 74th pct)")
    axA.set_xlabel("field-corr to interictal cohort"); axA.set_ylabel("# real subjects")
    axA.set_title("A. model lands inside real interictal cohort"); axA.legend(fontsize=8)
    axB.hist(null_ch, bins=30, color="lightcoral", edgecolor="none", alpha=0.65, label="channel-shuffle null")
    axB.hist(null_sh, bins=30, color="khaki", edgecolor="none", alpha=0.6, label="within-shaft null")
    axB.axvline(real_corr, color="navy", lw=2.5, label=f"model real corr={real_corr:.2f}")
    p_ch = (1 + np.sum(null_ch >= real_corr)) / (1 + null_ch.size)
    axB.set_xlabel("field-corr to interictal cohort"); axB.set_ylabel("# shuffles")
    axB.set_title(f"B. beats geometry null (channel p={p_ch:.3f})"); axB.legend(fontsize=8)
    fig.suptitle("M3B interictal bridge (instrument-probe): lands in cohort AND beats geometry → shared scaffold",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bridge_interictal.png"), dpi=130)
    plt.close(fig)

    # ---- Fig 3: ictal placement-only + gain sweep ----
    p3 = json.loads(open("results/topic4_sef_hfo/m3b_bridge/task2_bridge/task2_part3_ictal_axis.json").read())
    gs = json.loads(open("results/topic4_sef_hfo/m3b_bridge/task3_gain_sweep/task3_gain_sweep.json").read())
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel A: ictal placement-only — recompute null hist for the figure
    ictal = build_ictal_fields()
    epi = [s for s in sorted(ictal) if s.startswith("epilepsiae_")]
    ictal_fields = [_ifield(ictal[s]) for s in epi]
    mf_i = _ifield(model_rec)
    ic_corr = float(np.median([c for c in [_corr(mf_i, f) for f in ictal_fields] if np.isfinite(c)]))
    from collections import defaultdict
    shafts = defaultdict(list)
    for i, c in enumerate(model_rec["channels"]):
        shafts[c.get("shaft", "?")].append(i)
    rng = np.random.default_rng(0)
    null_i = []
    for _ in range(B):
        mr = _permuted_record(model_rec, [list(range(len(model_rec["channels"])))], rng)
        mfp = _ifield(mr)
        v = np.median([c for c in [_corr(mfp, f) for f in ictal_fields] if np.isfinite(c)])
        if np.isfinite(v):
            null_i.append(v)
    null_i = np.array(null_i)
    p_i = (1 + np.sum(null_i >= ic_corr)) / (1 + null_i.size)
    axA.hist(null_i, bins=30, color="lightcoral", edgecolor="none", alpha=0.65, label="channel-shuffle null")
    axA.axvline(ic_corr, color="darkred", lw=2.5, label=f"model real corr={ic_corr:.2f}")
    axA.set_xlabel("field-corr to ICTAL cohort (Epi n=18)"); axA.set_ylabel("# shuffles")
    axA.set_title(f"A. ictal = PLACEMENT-ONLY (does NOT beat geom, p={p_i:.2f})"); axA.legend(fontsize=8)
    # panel B: gain sweep
    rows = [r for r in gs["per_gain"] if r.get("axis_err_deg") is not None]
    ratios = [r["ratio"] for r in rows]; axerr = [r["axis_err_deg"] for r in rows]
    supp = [r["support_frac"] for r in rows]
    axB2 = axB.twinx()
    axB.plot(ratios, axerr, "o-", color="crimson", label="axis err vs 45° (left)")
    axB.axhline(25, ls="--", color="crimson", alpha=0.4)
    axB2.plot(ratios, supp, "s-", color="navy", label="event extent: support frac (right)")
    axB.set_xlabel("excitability (operating-point ratio)"); axB.set_ylabel("axis err (°)", color="crimson")
    axB2.set_ylabel("event extent (field support frac)", color="navy")
    axB2.set_ylim(0, 1)
    axB.set_title("B. event size ~FIXED (gain range inaccessible); axis stays ~45°")
    axB.set_ylim(0, 30)
    fig.suptitle("M3B ictal leg (placement-only) + gain readout (instrument-probe)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bridge_ictal_and_gain.png"), dpi=130)
    plt.close(fig)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
