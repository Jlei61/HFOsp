"""Stage B: Per-seizure feature extraction for Q1' overnight exploration.

Produces a single CSV results/topic1_topic5_bridge/q1prime_features.csv
with one row per (dataset, subject, seizure_id), including:
  - delta_rho_full, delta_rho_swap, assignment_full, assignment_swap
  - n_active, n_total, active_fraction, fast_recruit_fraction
  - onset_spread_sec, median_onset_latency_sec
  - subtype_label, swap_class, topic5_status

Sources:
  - results/topic1_topic5_bridge/q1prime_per_subject/{ds}_{sid}__q1prime.json
    → swap subset rho_a/rho_b (delta_rho_swap), subtype_label, swap_class
  - results/interictal_propagation/per_subject/{ds}_{sid}.json + rank_displacement
    → template ranks (t0_rank, t1_rank) for computing delta_rho_full
  - results/data_driven_soz/layer_a_ictal_er_rank/per_subject/{ds}_{sid}.json
    → n_active, n_total, fast_recruit_fraction, onset_spread_sec, median_onset_latency_sec,
       and per-seizure channel_onsets for full alignment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

from src.topic1_topic5_bridge import (
    load_atlas_seizure_channel_onsets,
    load_template_ranks_with_t0t1,
    load_swap_channel_subset,
    compute_seizure_template_alignment,
)

RESULTS = REPO / "results"
Q1PRIME_DIR = RESULTS / "topic1_topic5_bridge" / "q1prime_per_subject"
ATLAS_DIR = RESULTS / "data_driven_soz" / "layer_a_ictal_er_rank" / "per_subject"
BAND = "gamma_ER"
TAU_MIN = 0.10

COHORT_EPI = [
    "1073","1077","1084","1096","1146","1150","139","253",
    "442","548","583","590","635","916","922","958"
]
COHORT_YUQ = [
    "gaolan","huanghanwen","litengsheng","pengzihang","sunyuanxin",
    "xuxinyi","zhangjinhan","zhangkexuan","zhaojinrui"
]


def load_atlas_features(dataset: str, sid: str) -> dict[str, dict]:
    """Load n_active, n_total, fast_recruit_fraction, onset features per seizure."""
    p = ATLAS_DIR / f"{dataset}_{sid}.json"
    if not p.exists():
        return {}
    with p.open() as fh:
        d = json.load(fh)
    band_d = d.get("per_er", {}).get(BAND, {})
    sr_list = band_d.get("seizure_records", [])
    out = {}
    for sr in sr_list:
        sz_id = str(sr["seizure_id"])
        n_active = sr.get("n_active", 0)
        n_total = sr.get("n_total", 0)
        fast_recruit = sr.get("fast_recruit_fraction", 0.0)
        co = sr.get("channel_onsets") or {}
        onset_times = [
            v["t_onset_sec"] for v in co.values()
            if isinstance(v, dict) and v.get("t_onset_sec") is not None
        ]
        if onset_times:
            spread = float(max(onset_times) - min(onset_times))
            med_lat = float(np.median(onset_times))
        else:
            spread = float("nan")
            med_lat = float("nan")
        out[sz_id] = {
            "n_active": int(n_active),
            "n_total": int(n_total),
            "active_fraction": float(n_active / n_total) if n_total > 0 else float("nan"),
            "fast_recruit_fraction": float(fast_recruit) if fast_recruit is not None else float("nan"),
            "onset_spread_sec": spread,
            "median_onset_latency_sec": med_lat,
        }
    return out


def compute_full_alignment_per_subject(
    dataset: str,
    sid: str,
) -> dict[str, dict]:
    """Compute delta_rho_full for all atlas seizures of a subject.

    Full = use ALL topic1 lagPat channels as swap_subset (no endpoint restriction).
    Returns dict[seizure_id → {rho_a_full, rho_b_full, delta_rho_full, assignment_full}].
    """
    try:
        tmpl = load_template_ranks_with_t0t1(sid, RESULTS, Path("/dev/null"), dataset=dataset)
        atlas = load_atlas_seizure_channel_onsets(sid, BAND, RESULTS, dataset=dataset)
    except Exception as e:
        print(f"  [WARN] full alignment load failed {dataset}_{sid}: {e}")
        return {}

    # "Full" swap_subset = all topic1 channels
    all_topic1_chs = list(tmpl["channel_names"])
    out = {}
    for sz_id, sz_onsets in atlas.items():
        result = compute_seizure_template_alignment(
            seizure_onsets=sz_onsets,
            t0_rank=tmpl["t0_rank"],
            t1_rank=tmpl["t1_rank"],
            swap_subset=all_topic1_chs,
            channel_names_topic1=all_topic1_chs,
            channel_names_atlas=list(sz_onsets.keys()),
            tau_min=TAU_MIN,
        )
        rho_a = result.get("rho_a")
        rho_b = result.get("rho_b")
        if rho_a is not None and rho_b is not None and np.isfinite(rho_a) and np.isfinite(rho_b):
            delta_rho = float(rho_a - rho_b)
        else:
            delta_rho = float("nan")
        out[sz_id] = {
            "rho_a_full": float(rho_a) if rho_a is not None and np.isfinite(rho_a) else float("nan"),
            "rho_b_full": float(rho_b) if rho_b is not None and np.isfinite(rho_b) else float("nan"),
            "delta_rho_full": delta_rho,
            "assignment_full": result.get("assignment", "insufficient_n"),
            "n_full_channels_used": result.get("n_swap_channels_used", 0),
        }
    return out


def main() -> None:
    all_rows = []
    skipped = []

    for ds, sids in [("epilepsiae", COHORT_EPI), ("yuquan", COHORT_YUQ)]:
        for sid in sids:
            q1_path = Q1PRIME_DIR / f"{ds}_{sid}__q1prime.json"
            if not q1_path.exists():
                skipped.append(f"{ds}_{sid}: no q1prime JSON")
                continue
            with q1_path.open() as fh:
                d = json.load(fh)
            if d.get("status") == "failed":
                skipped.append(f"{ds}_{sid}: q1prime status=failed")
                continue

            swap_class = d.get("swap_class", "unknown")
            topic5_status = d.get("topic5_status", "unknown")
            per_seizure = d.get("per_seizure", [])
            if not per_seizure:
                skipped.append(f"{ds}_{sid}: empty per_seizure")
                continue

            # Load atlas features
            atlas_feats = load_atlas_features(ds, sid)

            # Compute full-intersection alignment
            full_align = compute_full_alignment_per_subject(ds, sid)

            for s in per_seizure:
                sz_id = str(s["seizure_id"])
                rho_a = s.get("rho_a")
                rho_b = s.get("rho_b")
                assignment_swap = s.get("assignment", "unknown")
                subtype_label = s.get("subtype_label")

                if rho_a is not None and rho_b is not None and np.isfinite(rho_a) and np.isfinite(rho_b):
                    delta_rho_swap = float(rho_a - rho_b)
                else:
                    delta_rho_swap = float("nan")

                af = atlas_feats.get(sz_id, {})
                fa = full_align.get(sz_id, {})

                all_rows.append({
                    "dataset": ds,
                    "subject": sid,
                    "seizure_id": sz_id,
                    # Swap-subset alignment
                    "rho_a_swap": float(rho_a) if rho_a is not None and np.isfinite(rho_a) else float("nan"),
                    "rho_b_swap": float(rho_b) if rho_b is not None and np.isfinite(rho_b) else float("nan"),
                    "delta_rho_swap": delta_rho_swap,
                    "assignment_swap": assignment_swap,
                    # Full-intersection alignment
                    "rho_a_full": fa.get("rho_a_full", float("nan")),
                    "rho_b_full": fa.get("rho_b_full", float("nan")),
                    "delta_rho_full": fa.get("delta_rho_full", float("nan")),
                    "assignment_full": fa.get("assignment_full", "unknown"),
                    "n_full_channels_used": fa.get("n_full_channels_used", 0),
                    # Metadata
                    "subtype_label": int(subtype_label) if subtype_label is not None else None,
                    "swap_class": swap_class,
                    "topic5_status": topic5_status,
                    # Atlas features
                    "n_active": af.get("n_active"),
                    "n_total": af.get("n_total"),
                    "active_fraction": af.get("active_fraction"),
                    "fast_recruit_fraction": af.get("fast_recruit_fraction"),
                    "onset_spread_sec": af.get("onset_spread_sec"),
                    "median_onset_latency_sec": af.get("median_onset_latency_sec"),
                })

    df = pd.DataFrame(all_rows)
    out_path = RESULTS / "topic1_topic5_bridge" / "q1prime_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows ({df['subject'].nunique()} subjects, {df['dataset'].nunique()} datasets)")
    print(f"  Epilepsiae: {len(df[df['dataset']=='epilepsiae'])} rows, {df[df['dataset']=='epilepsiae']['subject'].nunique()} subjects")
    print(f"  Yuquan:     {len(df[df['dataset']=='yuquan'])} rows, {df[df['dataset']=='yuquan']['subject'].nunique()} subjects")
    print(f"  Skipped: {skipped}")
    print()

    # Feature summary
    feat_cols = [
        "delta_rho_swap","delta_rho_full","rho_a_swap","rho_b_swap","rho_a_full","rho_b_full",
        "n_active","n_total","active_fraction","fast_recruit_fraction",
        "onset_spread_sec","median_onset_latency_sec"
    ]
    print("Feature summary (non-NaN stats):")
    for fc in feat_cols:
        if fc not in df.columns:
            continue
        col = df[fc]
        col_f = col.dropna()
        col_f = col_f[col_f.apply(lambda x: np.isfinite(float(x)) if isinstance(x, (int, float)) else False)]
        if len(col_f) == 0:
            print(f"  {fc}: ALL NaN (n=0)")
        else:
            print(f"  {fc}: n={len(col_f)}, median={float(col_f.median()):.3f}, IQR=[{float(col_f.quantile(0.25)):.3f},{float(col_f.quantile(0.75)):.3f}]")

    print(f"\nassignment_swap value counts:")
    print(df["assignment_swap"].value_counts().to_string())

    print(f"\nassignment_full value counts:")
    print(df["assignment_full"].value_counts().to_string())

    print(f"\nsubtype_label value counts:")
    print(df["subtype_label"].value_counts(dropna=False).to_string())

    print(f"\nswap_class distribution:")
    print(df["swap_class"].value_counts().to_string())

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
