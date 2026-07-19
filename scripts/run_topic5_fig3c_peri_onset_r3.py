#!/usr/bin/env python3
"""Stage 2B/C — Fig3-C peri-onset field similarity migrated to the R3 dense-grid readout.

Keeps the entire Fig3-C contract fixed and only replaces the evaluation layer:
the observed and null maxAB per 66-window `[-120,+20] s` trajectory are now the
R3 dense-grid support-gated field concordance (handoff §6.2), not the R2
contact-evaluated readout. The 7 shared-only subjects, the canonical successful
seizure set, the `[-120,+20] s` / 10 s / 2 s grid, the per-seizure per-replicate
spatial mapping reused across all 66 windows, and the all-contact + within-shaft
null pair are all taken unchanged from the accepted engine. maxT and cluster
correction are recomputed from the NEW R3 null matrices via the shared
`_finalize`; the figure reuses the accepted `_plot` grammar.

Outputs to a parallel R3 tree (old R2 spatial-null results are never overwritten):

    results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild/peri_onset_r3/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_gradient_grid_field as gg
from scripts.run_topic5_figure3_ictal_grid_rebuild import SubjectField
from scripts.run_topic5_fig3b_maxab_spatial_null import (
    NULL_MODELS, START_SEC, STOP_SEC, WINDOW_SEC, STEP_SEC, BAND,
    _compute_shared_values, _seizure_args, _keep_window, _permutation_indices,
    _shaft_structure, _finalize, _plot, _trajectory_source_provenance,
)

SHARED_ONLY = ["epilepsiae_1084", "epilepsiae_1146", "epilepsiae_384",
               "epilepsiae_548", "epilepsiae_583", "epilepsiae_590", "epilepsiae_958"]
STAGE = _ROOT / "results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild"
OUT = STAGE / "peri_onset_r3"
FIGDIR = OUT / "figures"
CONTRACT = "fig3c_peri_onset_shared_gradient_R3_dense_grid_v1"


def plot_subject_r3(ds_sid: str, rows: list[dict], summary: dict) -> None:
    """Render the R3 trajectory without altering any stored statistic.

    Fig3-C uses black horizontal segments for contiguous maxT-significant
    windows. Cluster significance remains encoded by the background shading.
    """
    _plot(
        ds_sid,
        rows,
        summary,
        FIGDIR / f"{ds_sid}_peri_onset_r3.png",
        FIGDIR / f"{ds_sid}_peri_onset_r3.pdf",
        maxt_style="line",
    )


def rebuild_figures_from_stats(subjects: list[str]) -> None:
    """Re-render Fig3-C from the frozen CSV statistics; never recompute nulls."""
    stats_path = OUT / "spatial_null_stats.csv"
    summary_path = OUT / "subject_summary.csv"
    manifest_path = OUT / "run_manifest.json"
    for path in (stats_path, summary_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"cannot rebuild figures: missing {path}")

    stats = pd.read_csv(stats_path)
    summaries = pd.read_csv(summary_path)
    manifest = json.loads(manifest_path.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ds_sid in subjects:
        rows_df = stats.loc[stats["subject"] == ds_sid].copy()
        subject_summaries = summaries.loc[summaries["subject"] == ds_sid]
        if len(rows_df) != 66 or len(subject_summaries) != len(NULL_MODELS):
            raise RuntimeError(
                f"{ds_sid}: cached figure inputs incomplete "
                f"({len(rows_df)} windows, {len(subject_summaries)} null rows)"
            )
        rows_df = rows_df.sort_values("window_center_sec")
        nulls = {}
        for rec in subject_summaries.to_dict("records"):
            nulls[rec["null_model"]] = {
                "n_pointwise_p05": int(rec["n_pointwise_p05"]),
                "n_maxt_p05": int(rec["n_maxt_p05"]),
                "n_cluster_sig_windows": int(rec["n_cluster_sig_windows"]),
                "n_clusters": int(rec["n_clusters"]),
            }
        summary = {
            "n_seizures": int(subject_summaries["n_seizures"].iloc[0]),
            "n_perm": int(manifest["n_perm"]),
            "nulls": nulls,
        }
        plot_subject_r3(ds_sid, rows_df.to_dict("records"), summary)
        print(f"[fig3c-r3] rebuilt figure from cached stats: {ds_sid}", flush=True)


def build_r3_engine(ds_sid: str):
    sf = SubjectField(ds_sid)
    if sf.route != "shared":
        raise RuntimeError(f"{ds_sid}: expected shared route, got {sf.route}")
    order_index = {n: i for i, n in enumerate(sf.contact_order)}
    return sf, order_index


def seizure_scorer(sf, source_indices):
    """Build the R3 event scorer with the seizure's fixed contact set as finite.

    A complete window keeps every source contact finite, and a spatial permutation
    of a complete window keeps all of them finite, so the ictal-support mask is
    constant across all windows/replicates -> the fixed-mask fast path is exact
    (the driver asserts no partial windows).
    """
    finite = np.zeros(len(sf.contact_order), bool)
    finite[source_indices] = True
    return sf.build_event_scorers(finite, gg.GRID_N)


def r3_maxab(ev, contact_order, source_indices, vals_matrix):
    aligned = np.full((len(vals_matrix), len(contact_order)), np.nan)
    aligned[:, source_indices] = vals_matrix
    return gg.score_event_maxab_batch(ev, aligned)   # fixed-mask fast path


def r3_detail(ev, contact_order, source_indices, vals):
    aligned = np.full(len(contact_order), np.nan)
    aligned[source_indices] = vals
    return gg.score_event_detail_single(ev, aligned)


def compute_subject_r3(ds_sid: str, n_perm: int, seed: int):
    source_csv, source, traj_prov = _trajectory_source_provenance(ds_sid)
    idxs = sorted(int(v) for v in source["seizure_idx"].unique())
    sf, order_index = build_r3_engine(ds_sid)
    contact_order = sf.contact_order
    rng = np.random.default_rng(seed)
    obs_by_win: dict[float, list[float]] = {}
    null_by_win = {m: {} for m in NULL_MODELS}
    seizure_rows = []
    n_seizures = 0
    shaft = None
    drops = []
    for seizure_idx in idxs:
        try:
            (_ds, _i, sw, offset, bl, field_record, names, starts,
             window_vals, _onset) = _compute_shared_values(_seizure_args(ds_sid, seizure_idx))
        except Exception as exc:
            drops.append({"seizure_idx": int(seizure_idx), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        missing = [n for n in names if n not in order_index]
        if missing:
            raise RuntimeError(f"{ds_sid}: names outside contact order: {missing}")
        source_indices = np.asarray([order_index[n] for n in names], int)
        cur_shaft = _shaft_structure(names)
        if shaft is None:
            shaft = cur_shaft
        elif cur_shaft != shaft:
            raise RuntimeError(f"{ds_sid}: mixed shaft structure across seizures")
        kept = [(float(lo), np.asarray(vals, float)) for lo, vals in zip(starts, window_vals)
                if _keep_window(float(lo))]
        if len(kept) != 66:
            raise RuntimeError(f"{ds_sid} seizure {seizure_idx}: {len(kept)} windows != 66")
        n_seizures += 1
        ev = seizure_scorer(sf, source_indices)
        # fixed-mask fast path is exact only if every window is fully complete or
        # fully missing over the source set (no partial windows). Guard it.
        for lo, vals in kept:
            nf = int(np.isfinite(vals).sum())
            if 0 < nf < len(names):
                raise RuntimeError(
                    f"{ds_sid} sz{seizure_idx} window {lo}: partial window "
                    f"({nf}/{len(names)}) — fixed-mask fast path not exact")
        perm_idx = {m: _permutation_indices(names, rng, m, n_perm) for m in NULL_MODELS}
        window_los = [lo for lo, _ in kept]
        window_vals = np.array([v for _, v in kept], float)                 # (66, n_source)
        # observed: one batched R3 call for all 66 windows
        obs_all = r3_maxab(ev, contact_order, source_indices, window_vals)
        for lo, o in zip(window_los, obs_all):
            obs_by_win.setdefault(lo, []).append(float(o))
        # signed A/B detail per window (cheap; observed only)
        for lo, vals in kept:
            det = r3_detail(ev, contact_order, source_indices, vals)
            seizure_rows.append({
                "subject": ds_sid, "seizure_idx": int(seizure_idx),
                "window_start_sec": lo, "window_center_sec": lo + WINDOW_SEC / 2.0,
                "maxab_r3": float(obs_all[window_los.index(lo)]), "best_template": det["best_template"],
                "signed_a": det["signed_a"], "signed_b": det["signed_b"],
                "abs_a": det["abs_a"], "abs_b": det["abs_b"],
                "mirror_a": det["mirror_a"], "mirror_b": det["mirror_b"],
                "overlap_a": det["overlap_a"], "overlap_b": det["overlap_b"],
                "n_finite_contacts": int(np.isfinite(vals).sum())})
        # null: one batched R3 call per model over all (window, replicate) rows,
        # preserving the per-seizure fixed mapping (perm_idx same across windows)
        for m in NULL_MODELS:
            big = np.stack([vals[perm_idx[m]] for _, vals in kept])          # (66, n_perm, n_source)
            flat = big.reshape(len(kept) * n_perm, big.shape[-1])
            res = r3_maxab(ev, contact_order, source_indices, flat)
            res = res.reshape(len(kept), n_perm)
            for j, lo in enumerate(window_los):
                null_by_win[m].setdefault(lo, []).append(res[j])
    if drops or n_seizures != len(idxs):
        raise RuntimeError(f"{ds_sid}: incomplete recompute ({n_seizures}/{len(idxs)}, drops={drops})")

    los = sorted(obs_by_win)
    centers = np.array([lo + WINDOW_SEC / 2.0 for lo in los])
    n_sz = np.array([len(obs_by_win[lo]) for lo in los])
    obs = np.array([float(np.nanmedian(obs_by_win[lo])) for lo in los])
    obs_q25 = np.array([float(np.nanpercentile(obs_by_win[lo], 25)) for lo in los])
    obs_q75 = np.array([float(np.nanpercentile(obs_by_win[lo], 75)) for lo in los])
    null_mats = {}
    for m in NULL_MODELS:
        M = np.empty((n_perm, len(los)))
        for j, lo in enumerate(los):
            M[:, j] = np.nanmedian(np.vstack(null_by_win[m][lo]), axis=0)
        null_mats[m] = M
    meta = {"n_perm": int(n_perm), "seed": int(seed), "n_seizures": int(n_seizures),
            "n_seizure_drops": len(drops), "seizure_drops": drops,
            "readout": "R3_dense_grid_maxab", "r3_formula_version": gg.R3_FORMULA_VERSION,
            "shaft_structure": shaft, "sigma_common": sf.sigma_common,
            "fidelity_max_abs_err": None, "source_observed_max_abs_err": None,
            "provenance": {"field_plane": "shared", "trajectory": traj_prov}}
    rows, summary = _finalize(ds_sid, centers, n_sz, obs, obs_q25, obs_q75, null_mats, meta)
    return {"rows": rows, "summary": summary, "centers": centers, "n_sz": n_sz,
            "obs": obs, "obs_q25": obs_q25, "obs_q75": obs_q75, "null_mats": null_mats,
            "seizure_rows": seizure_rows, "n_seizures": n_seizures, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=SHARED_ONLY)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument(
        "--rebuild-figures-only",
        action="store_true",
        help="re-render PNG/PDF from cached stats; do not recompute observed or null values",
    )
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if args.rebuild_figures_only:
        rebuild_figures_from_stats(args.subjects)
        return

    index_rows, window_rows, null_stat_rows, summary_rows, npz_payload = [], [], [], [], {}

    def _flush():
        pd.DataFrame(index_rows).to_csv(OUT / "subject_index.csv", index=False)
        pd.DataFrame(window_rows).to_csv(OUT / "per_seizure_window.csv", index=False)
        pd.DataFrame(null_stat_rows).to_csv(OUT / "spatial_null_stats.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(OUT / "subject_summary.csv", index=False)
        np.savez_compressed(OUT / "spatial_null_matrices.npz", **npz_payload)

    t0 = time.time()
    for ds_sid in args.subjects:
        print(f"[fig3c-r3] {ds_sid} ...", flush=True)
        res = compute_subject_r3(ds_sid, args.n_perm, args.seed)
        plot_subject_r3(ds_sid, res["rows"], res["summary"])
        window_rows.extend(res["seizure_rows"])
        for r in res["rows"]:
            null_stat_rows.append({"subject": ds_sid, **r})
        for model, s in res["summary"].get("nulls", {}).items():
            summary_rows.append({"subject": ds_sid, "null_model": model,
                                 "n_seizures": res["n_seizures"],
                                 "n_pointwise_p05": s.get("n_pointwise_p05"),
                                 "n_maxt_p05": s.get("n_maxt_p05"),
                                 "n_cluster_sig_windows": s.get("n_cluster_sig_windows"),
                                 "n_clusters": s.get("n_clusters")})
        index_rows.append({"subject": ds_sid, "n_seizures": res["n_seizures"],
                           "n_windows": int(res["centers"].size),
                           "sigma_common": res["meta"]["sigma_common"], "status": "complete_ok"})
        npz_payload[f"{ds_sid}__centers"] = res["centers"]
        npz_payload[f"{ds_sid}__obs"] = res["obs"]
        npz_payload[f"{ds_sid}__obs_q25"] = res["obs_q25"]
        npz_payload[f"{ds_sid}__obs_q75"] = res["obs_q75"]
        for model, M in res["null_mats"].items():
            npz_payload[f"{ds_sid}__{model}_null"] = M
        _flush()   # incremental: persist after every subject so partials survive
        print(f"  {ds_sid}: {res['n_seizures']} seizures, "
              f"within-shaft clusters={res['summary'].get('nulls',{}).get('within_shaft',{}).get('n_cluster_sig_windows')}, "
              f"({time.time()-t0:.0f}s)", flush=True)

    manifest = {"contract": CONTRACT, "readout": "R3_dense_grid_maxab",
                "r3_formula_version": gg.R3_FORMULA_VERSION,
                "subjects": args.subjects, "n_perm": args.n_perm, "seed": args.seed,
                "window_contract": {"start_sec": START_SEC, "stop_sec": STOP_SEC,
                                    "window_sec": WINDOW_SEC, "step_sec": STEP_SEC,
                                    "band": list(BAND), "n_windows": 66},
                "null_models": list(NULL_MODELS), "grid_n": gg.GRID_N,
                "old_r2_reused": False, "note": "R3 migration of the shared-only peri-onset trajectory; "
                "old R2 spatial-null tree not overwritten."}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[fig3c-r3] wrote {len(index_rows)} subjects -> {OUT}")


if __name__ == "__main__":
    main()
