#!/usr/bin/env python3
"""Decisive (last) variant: aligned-template source stability vs rate within SOZ∩U.

The per-event KMeans label is arbitrary (0/1 = random which physical template), so
max(fwd,rev) is winner's-curse biased UP and mean(fwd,rev) is diluted DOWN. This test
removes both: each subject's two clusters are aligned to the full-recording template via
rank_a_dense (centroid best-correlated with rank_a = "template A / source template", a
WINDOW-INDEPENDENT choice), and only that template's source stability is tested vs rate.
Pre-specified primary: M=100, Jaccard, epilepsiae-primary. If null here too, the
hypothesis (geometry source more sampling-stable than rate within SOZ) is unsupported.

Forking-paths note: this is the ~5th operationalization tested; no p here is confirmatory
(hypothesis-generating only — would need pre-registration + held-out cohort).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import KMeans

from src.sef_hfo_soz_localization import soz_internal_source_stability
from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import build_masked_kmeans_features

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "soz_localization"
GEOM_DIR = _ROOT / "results" / "interictal_propagation_masked" / "rank_displacement" / "per_subject"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
M_GRID = [50, 100, 200]
ALIGN_MARGIN = 0.20  # |corr_A - corr_B| must exceed this, else clusters don't separate -> exclude


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _align_template_cluster(labels, feats, ev_channels, pair):
    """Return (fwd_cluster_id, corrA, corrB) or None if clusters don't separate on rank_a.

    Centroid of each cluster (mean masked rank per channel) is correlated with rank_a_dense
    over shared geometry-valid channels; the cluster correlating MOST POSITIVELY with rank_a
    (both 0=source) is template A. Window-independent (uses full-recording rank_a)."""
    pair_names = pair["channel_names"]
    ra = np.asarray(pair["rank_a_dense_full"], dtype=float)
    valid = np.asarray(pair["joint_valid"], dtype=bool)
    # shared channels present in features AND geometry-valid with finite rank_a
    name_to_feat = {n: i for i, n in enumerate(ev_channels)}
    shared = [(name_to_feat[pair_names[j]], j) for j in range(len(pair_names))
              if valid[j] and np.isfinite(ra[j]) and pair_names[j] in name_to_feat]
    if len(shared) < 3:
        return None
    feat_idx = [s[0] for s in shared]
    ra_vec = np.array([ra[s[1]] for s in shared], dtype=float)
    corrs = []
    for c in (0, 1):
        centroid = feats[labels == c].mean(axis=0)[feat_idx]
        rho = spearmanr(centroid, ra_vec).correlation
        corrs.append(rho if rho == rho else 0.0)
    if abs(corrs[0] - corrs[1]) < ALIGN_MARGIN:
        return None  # ambiguous: do not force-assign
    fwd = int(np.argmax(corrs))  # most positively correlated with rank_a = source template
    return fwd, float(corrs[fwd]), float(corrs[1 - fwd])


def main() -> int:
    cohort = json.loads((OUT_DIR / "cohort.json").read_text())
    per_subject, excluded = [], []
    for rec in cohort["kept"]:
        soz_u = rec["soz_in_universe"]
        if len(soz_u) < 3:
            excluded.append({"subject": rec["subject"], "reason": "|SOZ∩U|<3"})
            continue
        ev = load_subject_propagation_events(_subject_dir(rec["dataset"], rec["subject"]))
        chan = ev["channel_names"]
        soz_u = [c for c in soz_u if c in chan]
        if len(soz_u) < 3:
            excluded.append({"subject": rec["subject"], "reason": "soz_u not in events"})
            continue
        pair = json.loads((GEOM_DIR / f"{rec['dataset']}_{rec['subject']}.json").read_text())["pairs"][0]
        feats = build_masked_kmeans_features(ev["ranks"], np.asarray(ev["bools"]) > 0)
        labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(feats)
        aligned = _align_template_cluster(labels, feats, chan, pair)
        if aligned is None:
            excluded.append({"subject": rec["subject"], "reason": "clusters do not separate on rank_a"})
            continue
        fwd, corrA, corrB = aligned
        times = ev["event_abs_times"]
        if not np.all(np.isfinite(times)):
            times = np.where(np.isfinite(times), times, ev["event_rel_times"])
        k = max(1, min(3, len(soz_u) // 2))
        res = soz_internal_source_stability(
            ev["ranks"], ev["bools"], times, chan, soz_u, M_grid=M_GRID, k=k,
            n_null=100, n_starts=12, seed=0, labels=labels, fwd_id=fwd, rev_id=1 - fwd,
            readouts=["rate", "source_fwd"])
        per_subject.append({"dataset": rec["dataset"], "subject": rec["subject"],
                            "n_soz_u": len(soz_u), "k": k, "corrA": corrA, "corrB": corrB,
                            "curves": res["curves"]})

    def paired(sel, M):
        g, rt = [], []
        for p in per_subject:
            if not sel(p):
                continue
            c = p["curves"][M]
            gv, rv = c["source_fwd"]["jaccard_obs"], c["rate"]["jaccard_obs"]
            if gv == gv and rv == rv:
                g.append(gv); rt.append(rv)
        g, rt = np.array(g), np.array(rt); d = g - rt
        try:
            pv = float(wilcoxon(g, rt, alternative="greater").pvalue) if len(d) and np.any(d != 0) else float("nan")
        except ValueError:
            pv = float("nan")
        return {"n": len(g), "median_aligned_source": float(np.median(g)) if len(g) else float("nan"),
                "median_rate": float(np.median(rt)) if len(rt) else float("nan"),
                "median_delta": float(np.median(d)) if len(d) else float("nan"),
                "n_source_ge_rate": int(np.sum(d >= 0)), "wilcoxon_p_source_gt_rate": pv}

    agg = {"ALL": {M: paired(lambda p: True, M) for M in M_GRID},
           "EPILEPSIAE": {M: paired(lambda p: p["dataset"] == "epilepsiae", M) for M in M_GRID},
           "YUQUAN": {M: paired(lambda p: p["dataset"] == "yuquan", M) for M in M_GRID}}
    out = {"meta": {"test": "aligned-template source vs rate (decisive, window-independent rank_a alignment)",
                    "primary": "M=100, Jaccard, epilepsiae", "align_margin": ALIGN_MARGIN,
                    "n_used": len(per_subject), "forking_paths": "~5th operationalization; not confirmatory"},
           "excluded": excluded, "aggregate": agg, "per_subject": per_subject}
    (OUT_DIR / "aligned_source_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"=== Aligned-template source vs rate (n_used={len(per_subject)}, excluded={len(excluded)}) ===")
    print("    (aligned source = source within the rank_a-matched template; Δ>0 = source MORE stable than rate)")
    for label in ("ALL", "EPILEPSIAE", "YUQUAN"):
        for M in M_GRID:
            a = agg[label][M]
            star = " *PRIMARY*" if (label == "EPILEPSIAE" and M == 100) else ""
            print(f"  {label:<11} M={M:>3}(n={a['n']:>2}): aligned_source={a['median_aligned_source']:.2f} "
                  f"rate={a['median_rate']:.2f} Δ={a['median_delta']:+.2f} "
                  f"src≥rate={a['n_source_ge_rate']}/{a['n']} p={a['wilcoxon_p_source_gt_rate']:.3f}{star}")
    print(f"  excluded: {[(e['subject'], e['reason']) for e in excluded]}")
    print(f"\nwrote {OUT_DIR / 'aligned_source_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
