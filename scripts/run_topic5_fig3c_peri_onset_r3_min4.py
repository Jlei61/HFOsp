#!/usr/bin/env python3
"""Fig3-C peri-onset R3 — pure within-shaft null with locked min_group = 4.

Method-sensitivity fix for the reviewer finding that the accepted Fig3-C tree
(`peri_onset_r3/`) used the permissive engine within-shaft null (any shaft with
>= 2 contacts is shuffled, singletons fixed). This driver instead uses the locked
`gg.within_shaft_permutations(..., min_group=4)` contract with NO fallback:

* all-contact null: unchanged (full channel shuffle, per-seizure fixed mapping
  reused across all 66 windows);
* pure within-shaft null: only permutes contacts inside shafts that have >= 4
  finite contacts, and a subject with ANY shaft below 4 is marked
  ``pure_within_shaft = unavailable`` (observed R3 + all-contact null kept;
  within-shaft null / maxT / cluster reported NA; reason / small_shafts /
  shaft_sizes saved; no fallback, no small-shaft fixing that is still called
  "pure within-shaft").

Eligibility is COMPUTED from the frozen source contacts, never hardcoded.
Old `peri_onset_r3/` is preserved as audit evidence; results go to a new tree:

    results/paper-ready-figure/
      fig3_ictal_field_concordance_grid_method_sensitivity/peri_onset_r3_min4/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_gradient_grid_field as gg
from scripts.run_topic5_figure3_ictal_grid_rebuild import event_seed
from scripts.run_topic5_fig3c_peri_onset_r3 import (
    SHARED_ONLY, build_r3_engine, r3_detail, r3_maxab,
)
from scripts.run_topic5_fig3b_maxab_spatial_null import (
    COL_AC, COL_OBS, SIG_ALPHA, START_SEC, STOP_SEC, WINDOW_SEC, STEP_SEC, BAND,
    _compute_shared_values, _finalize, _keep_window, _permutation_indices,
    _plot, _seizure_args, _shaft_structure, _trajectory_source_provenance,
)

MIN_GROUP = 4
# Fig3-C is raised to the cohort primary resolution N=161 (P1-3 / P1-4).
FIG3C_GRID_N = 161


def _seizure_scorer_n(sf, source_indices, grid_n=FIG3C_GRID_N):
    finite = np.zeros(len(sf.contact_order), bool)
    finite[source_indices] = True
    return sf.build_event_scorers(finite, grid_n)
OUT = _ROOT / ("results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/"
               "peri_onset_r3_min4")
FIGDIR = OUT / "figures"
CONTRACT = "fig3c_peri_onset_shared_gradient_R3_pure_within_shaft_min4_v1"


def within_shaft_eligibility(names, min_group=MIN_GROUP):
    ws = gg.within_shaft_permutations(list(names), np.ones(len(names), bool),
                                      n_perm=1, seed=1, min_group=min_group)
    return ws


def _plot_unavailable(ds_sid, rows, summary, ws_status, out_png, out_pdf):
    """Observed R3 + all-contact null only; explicit within-shaft unavailability."""
    x = np.array([r["window_center_sec"] for r in rows], float)
    obs = np.array([r["obs_median_maxAB"] for r in rows], float)
    oq25 = np.array([r["obs_q25"] for r in rows], float)
    oq75 = np.array([r["obs_q75"] for r in rows], float)
    ac_med = np.array([r["all_contact_null_median"] for r in rows], float)
    label = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(x, ac_med, color=COL_AC, lw=1.1, ls=":", zorder=2, label="all-contact null median")
    ax.fill_between(x, oq25, oq75, color=COL_OBS, alpha=0.13, lw=0, zorder=3, label="observed IQR")
    ax.plot(x, obs, color=COL_OBS, lw=2.2, zorder=5, label="observed median")
    ax.axvline(0, color="0.30", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("window center from onset (s)", fontsize=11)
    ax.set_ylabel("maxAB field similarity |r|", fontsize=11)
    ax.set_title("shared-gradient maxAB vs all-contact null", fontsize=11, pad=16)
    small = ", ".join(f"{k}:{v}" for k, v in sorted(ws_status["small_shafts"].items()))
    ax.text(0.5, 1.015,
            f"{label} · {summary['n_seizures']} sz · pure within-shaft unavailable "
            f"(min group = {MIN_GROUP}); shafts below 4: {small}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, color="0.35")
    ax.text(0.5, 0.5, "pure within-shaft unavailable\n(min group = 4)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12, color="0.55",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
    ax.legend(frameon=False, loc="lower left", fontsize=6.8, handlelength=1.5)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.85)
    for out in (out_png, out_pdf):
        fig.savefig(out, dpi=300)
    plt.close(fig)


def compute_subject_min4(ds_sid, n_perm, seed):
    source_csv, source, traj_prov = _trajectory_source_provenance(ds_sid)
    idxs = sorted(int(v) for v in source["seizure_idx"].unique())
    sf, order_index = build_r3_engine(ds_sid)
    contact_order = sf.contact_order
    rng = np.random.default_rng(seed)
    obs_by_win: dict[float, list[float]] = {}
    null_by_win = {"all_contact": {}, "within_shaft": {}}
    seizure_rows = []
    n_seizures = 0
    shaft = None
    drops = []
    ws_status = None
    for seizure_idx in idxs:
        try:
            (_ds, _i, sw, offset, bl, field_record, names, starts,
             window_vals, _onset) = _compute_shared_values(_seizure_args(ds_sid, seizure_idx))
        except Exception as exc:
            drops.append({"seizure_idx": int(seizure_idx), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        source_indices = np.asarray([order_index[n] for n in names], int)
        cur_shaft = _shaft_structure(names)
        if shaft is None:
            shaft = cur_shaft
        elif cur_shaft != shaft:
            raise RuntimeError(f"{ds_sid}: mixed shaft structure across seizures")
        # eligibility computed from the actual source contacts (constant per subject)
        elig = within_shaft_eligibility(names, MIN_GROUP)
        if ws_status is None:
            ws_status = elig
        kept = [(float(lo), np.asarray(vals, float)) for lo, vals in zip(starts, window_vals)
                if _keep_window(float(lo))]
        if len(kept) != 66:
            raise RuntimeError(f"{ds_sid} seizure {seizure_idx}: {len(kept)} windows != 66")
        n_seizures += 1
        ev = _seizure_scorer_n(sf, source_indices)
        for lo, vals in kept:
            nf = int(np.isfinite(vals).sum())
            if 0 < nf < len(names):
                raise RuntimeError(f"{ds_sid} sz{seizure_idx} window {lo}: partial window")
        window_los = [lo for lo, _ in kept]
        window_vals_arr = np.array([v for _, v in kept], float)
        obs_all = r3_maxab(ev, contact_order, source_indices, window_vals_arr)
        for lo, o in zip(window_los, obs_all):
            obs_by_win.setdefault(lo, []).append(float(o))
        for lo, vals in kept:
            det = r3_detail(ev, contact_order, source_indices, vals)
            seizure_rows.append({
                "subject": ds_sid, "seizure_idx": int(seizure_idx),
                "window_start_sec": lo, "window_center_sec": lo + WINDOW_SEC / 2.0,
                "maxab_r3": float(obs_all[window_los.index(lo)]), "best_template": det["best_template"],
                "signed_a": det["signed_a"], "signed_b": det["signed_b"],
                "n_finite_contacts": int(np.isfinite(vals).sum())})
        # all-contact perms (per-seizure fixed mapping across 66 windows)
        ac_perm = _permutation_indices(names, rng, "all_contact", n_perm)
        big = np.stack([vals[ac_perm] for _, vals in kept])
        res = r3_maxab(ev, contact_order, source_indices,
                       big.reshape(len(kept) * n_perm, big.shape[-1])).reshape(len(kept), n_perm)
        for j, lo in enumerate(window_los):
            null_by_win["all_contact"].setdefault(lo, []).append(res[j])
        # pure within-shaft (min-4) only when eligible
        if ws_status["eligible"]:
            ws = gg.within_shaft_permutations(list(names), np.ones(len(names), bool),
                                              n_perm=n_perm, seed=event_seed(ds_sid, seizure_idx),
                                              min_group=MIN_GROUP)
            wperm = ws["permutations"]
            bigw = np.stack([vals[wperm] for _, vals in kept])
            resw = r3_maxab(ev, contact_order, source_indices,
                            bigw.reshape(len(kept) * n_perm, bigw.shape[-1])).reshape(len(kept), n_perm)
            for j, lo in enumerate(window_los):
                null_by_win["within_shaft"].setdefault(lo, []).append(resw[j])
    if drops or n_seizures != len(idxs):
        raise RuntimeError(f"{ds_sid}: incomplete recompute ({n_seizures}/{len(idxs)}, drops={drops})")

    los = sorted(obs_by_win)
    centers = np.array([lo + WINDOW_SEC / 2.0 for lo in los])
    n_sz = np.array([len(obs_by_win[lo]) for lo in los])
    obs = np.array([float(np.nanmedian(obs_by_win[lo])) for lo in los])
    obs_q25 = np.array([float(np.nanpercentile(obs_by_win[lo], 25)) for lo in los])
    obs_q75 = np.array([float(np.nanpercentile(obs_by_win[lo], 75)) for lo in los])
    models = ["all_contact"] + (["within_shaft"] if ws_status["eligible"] else [])
    null_mats = {}
    for m in models:
        M = np.empty((n_perm, len(los)))
        for j, lo in enumerate(los):
            M[:, j] = np.nanmedian(np.vstack(null_by_win[m][lo]), axis=0)
        null_mats[m] = M
    meta = {"n_perm": int(n_perm), "seed": int(seed), "n_seizures": int(n_seizures),
            "n_seizure_drops": len(drops), "seizure_drops": drops,
            "readout": "R3_dense_grid_maxab", "r3_formula_version": gg.R3_FORMULA_VERSION,
            "shaft_structure": shaft, "sigma_common": sf.sigma_common,
            "fidelity_max_abs_err": None, "source_observed_max_abs_err": None,
            "pure_within_shaft": {"min_group": MIN_GROUP, "eligible": bool(ws_status["eligible"]),
                                  "reason": ws_status.get("reason"),
                                  "small_shafts": ws_status.get("small_shafts", {}),
                                  "shaft_sizes": shaft.get("shaft_sizes")},
            "provenance": {"field_plane": "shared", "trajectory": traj_prov}}
    rows, summary = _finalize(ds_sid, centers, n_sz, obs, obs_q25, obs_q75, null_mats, meta)
    return {"rows": rows, "summary": summary, "centers": centers, "n_sz": n_sz,
            "obs": obs, "obs_q25": obs_q25, "obs_q75": obs_q75, "null_mats": null_mats,
            "seizure_rows": seizure_rows, "n_seizures": n_seizures, "meta": meta,
            "ws_status": ws_status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=SHARED_ONLY)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    index_rows, window_rows, null_stat_rows, summary_rows, avail_rows, npz_payload = [], [], [], [], [], {}

    def _flush():
        pd.DataFrame(index_rows).to_csv(OUT / "subject_index.csv", index=False)
        pd.DataFrame(window_rows).to_csv(OUT / "per_seizure_window.csv", index=False)
        pd.DataFrame(null_stat_rows).to_csv(OUT / "spatial_null_stats.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(OUT / "subject_summary.csv", index=False)
        pd.DataFrame(avail_rows).to_csv(OUT / "within_shaft_availability.csv", index=False)
        np.savez_compressed(OUT / "spatial_null_matrices.npz", **npz_payload)

    t0 = time.time()
    for ds_sid in args.subjects:
        print(f"[fig3c-min4] {ds_sid} ...", flush=True)
        res = compute_subject_min4(ds_sid, args.n_perm, args.seed)
        elig = res["ws_status"]["eligible"]
        if elig:
            # eligible figure MUST go to the min-4 tree (P1-3 fix); do NOT reuse the
            # old-tree plot_subject_r3 which writes into peri_onset_r3/.
            _plot(ds_sid, res["rows"], res["summary"],
                  FIGDIR / f"{ds_sid}_peri_onset_r3_min4.png",
                  FIGDIR / f"{ds_sid}_peri_onset_r3_min4.pdf", maxt_style="line")
        else:
            _plot_unavailable(ds_sid, res["rows"], res["summary"], res["meta"]["pure_within_shaft"],
                              FIGDIR / f"{ds_sid}_peri_onset_r3_min4.png",
                              FIGDIR / f"{ds_sid}_peri_onset_r3_min4.pdf")
        window_rows.extend(res["seizure_rows"])
        for r in res["rows"]:
            null_stat_rows.append({"subject": ds_sid, **r})
        for model, s in res["summary"].get("nulls", {}).items():
            summary_rows.append({"subject": ds_sid, "null_model": model, "n_seizures": res["n_seizures"],
                                 "n_pointwise_p05": s.get("n_pointwise_p05"),
                                 "n_maxt_p05": s.get("n_maxt_p05"),
                                 "n_cluster_sig_windows": s.get("n_cluster_sig_windows"),
                                 "n_clusters": s.get("n_clusters")})
        pw = res["meta"]["pure_within_shaft"]
        avail_rows.append({"subject": ds_sid, "pure_within_shaft_eligible": pw["eligible"],
                           "reason": pw["reason"], "small_shafts": json.dumps(pw["small_shafts"]),
                           "shaft_sizes": json.dumps(pw["shaft_sizes"]), "n_seizures": res["n_seizures"]})
        index_rows.append({"subject": ds_sid, "n_seizures": res["n_seizures"],
                           "n_windows": int(res["centers"].size), "sigma_common": res["meta"]["sigma_common"],
                           "pure_within_shaft_eligible": pw["eligible"], "status": "complete_ok"})
        npz_payload[f"{ds_sid}__centers"] = res["centers"]
        npz_payload[f"{ds_sid}__obs"] = res["obs"]
        for model, M in res["null_mats"].items():
            npz_payload[f"{ds_sid}__{model}_null"] = M
        _flush()
        wsn = res["summary"].get("nulls", {}).get("within_shaft")
        print(f"  {ds_sid}: {res['n_seizures']} sz, eligible={elig}, "
              f"within-shaft maxT={wsn.get('n_maxt_p05') if wsn else 'NA'} "
              f"cluster={wsn.get('n_cluster_sig_windows') if wsn else 'NA'} ({time.time()-t0:.0f}s)", flush=True)

    manifest = {"contract": CONTRACT, "min_group": MIN_GROUP,
                "subjects": args.subjects, "n_perm": args.n_perm, "seed": args.seed,
                "window_contract": {"start_sec": START_SEC, "stop_sec": STOP_SEC,
                                    "window_sec": WINDOW_SEC, "step_sec": STEP_SEC, "n_windows": 66},
                "supersedes_within_shaft_of": "peri_onset_r3/ (legacy min-2, kept as audit evidence)",
                "note": "pure within-shaft null uses gg.within_shaft_permutations(min_group=4) no fallback; "
                        "ineligible subjects report within-shaft NA."}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[fig3c-min4] wrote {len(index_rows)} subjects -> {OUT}")


if __name__ == "__main__":
    main()
