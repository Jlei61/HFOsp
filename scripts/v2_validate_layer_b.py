"""HFO detector v2 — Layer B validation (group-event quality).

Computes per-subject metrics from packed lagPat artifacts:
  - n_participating distribution (p10, p50, p90)
  - pack_window_width distribution (p50, p90)
  - splithalf_event_rank_corr (Spearman, channel-mean-rank, half-by-half)
  - oddeven_event_rank_corr (Spearman, channel-mean-rank, even-vs-odd)
  - chunk_boundary_event_frac (descriptive proxy for merge_overhead)

Inputs: results/hfo_detector_v2/lagpat/<subject>/<stem>_lagPat.npz
        + (optional) results/hfo_detector_v2/lagpat/<subject>/<stem>_packedTimes.npy

Schema (canonical, matches src/interictal_propagation.py loader):
  - lagPatRank, eventsBool: shape (n_chn, n_events)
  - chnNames: shape (n_chn,)
  - packedTimes.npy: shape (n_events, 2), columns = [start_rel, end_rel]
  - 'channel_names' is supported as fallback for cross-dataset npz files.

Output: results/hfo_detector_v2/validation/layer_b_<subject>.json

Scope: this is *pipeline internal self-consistency*, NOT biological validity.
See docs/archive/hfo_detector_v2/v2_validation_contract.md Layer B row.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def compute_n_participating_stats(n_part: np.ndarray) -> dict:
    return {
        "p10": float(np.percentile(n_part, 10)),
        "p50": float(np.percentile(n_part, 50)),
        "p90": float(np.percentile(n_part, 90)),
        "n_total": int(len(n_part)),
    }


def compute_pack_width_stats(starts: np.ndarray, ends: np.ndarray) -> dict:
    widths = ends - starts
    return {
        "p50": float(np.percentile(widths, 50)),
        "p90": float(np.percentile(widths, 90)),
    }


def compute_splithalf_rank_corr(rank_first: np.ndarray, rank_second: np.ndarray) -> float:
    """Spearman rank correlation between two channel-rank vectors (legacy helper)."""
    from scipy.stats import spearmanr
    if len(rank_first) != len(rank_second):
        raise ValueError("rank arrays must align")
    rho, _ = spearmanr(rank_first, rank_second)
    return float(rho)


def compute_subset_rank_corr(
    lag_pat_rank: np.ndarray,    # (n_events, n_chn) per-event ranks
    valid_mask: np.ndarray,      # (n_events, n_chn) boolean: channel participated
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> float:
    """For two disjoint event subsets a, b of the same channel set, compute the
    Spearman correlation between (mean rank in subset a) and (mean rank in subset b).

    Inputs are event-major (n_events, n_chn). Channels with zero participation in
    either subset are dropped before the corr. Returns NaN if fewer than 3
    channels survive (Spearman undefined).
    """
    from scipy.stats import spearmanr
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    rk_a = lag_pat_rank[idx_a]
    rk_b = lag_pat_rank[idx_b]
    vm_a = valid_mask[idx_a]
    vm_b = valid_mask[idx_b]
    cnt_a = vm_a.sum(axis=0).astype(float)
    cnt_b = vm_b.sum(axis=0).astype(float)
    keep = (cnt_a > 0) & (cnt_b > 0)
    if keep.sum() < 3:
        return float("nan")
    sum_a = (rk_a * vm_a).sum(axis=0)
    sum_b = (rk_b * vm_b).sum(axis=0)
    mean_a = np.where(cnt_a > 0, sum_a / np.maximum(cnt_a, 1), np.nan)
    mean_b = np.where(cnt_b > 0, sum_b / np.maximum(cnt_b, 1), np.nan)
    rho, _ = spearmanr(mean_a[keep], mean_b[keep])
    return float(rho)


def compute_chunk_boundary_event_frac(starts: np.ndarray, chunk_sec: float = 200.0,
                                      tol_sec: float = 2.0) -> float:
    """Fraction of events whose start_time falls within +/- tol_sec of any chunk
    boundary k * chunk_sec (k>=0). Descriptive proxy for merge_overhead."""
    if len(starts) == 0:
        return 0.0
    nearest_k = np.round(starts / chunk_sec)
    distance = np.abs(starts - nearest_k * chunk_sec)
    return float(np.mean(distance <= tol_sec))


def extract_layer_b_per_subject(subject_lagpat_root: Path) -> dict:
    """Walk *_lagPat.npz + (optional) *_packedTimes.npy under subject_lagpat_root.

    Aggregate group event quality across all records of one subject. Uses union
    channel set across records; channels missing from a given record's lagPat
    contribute zero to that record's rank/valid rows for the union columns.

    Canonical npz layout (per src/interictal_propagation.py):
      - lagPatRank, eventsBool: shape (n_chn, n_events) — chn-major
      - chnNames: (n_chn,) array of strings
      - packedTimes.npy: (n_events, 2) — rows = [start_rel, end_rel]

    We transpose npz arrays to (n_events, n_chn) before stacking so that the
    rank-corr helper sees event-major matrices (matches its docstring).
    """
    npz_files = sorted(subject_lagpat_root.glob("*_lagPat.npz"))
    if not npz_files:
        return {"error": f"no lagpat npz found in {subject_lagpat_root}"}

    # First pass: build union channel ordering. Backfill writes 'chnNames';
    # 'channel_names' is the cross-dataset fallback.
    union_chs: list[str] = []
    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        ch_key = "chnNames" if "chnNames" in d.files else (
            "channel_names" if "channel_names" in d.files else None
        )
        if ch_key is None:
            continue
        for ch in [str(c) for c in d[ch_key]]:
            if ch not in union_chs:
                union_chs.append(ch)
    n_ch = len(union_chs)
    if n_ch < 3:
        return {"error": f"too few channels for rank corr ({n_ch})"}

    rank_rows = []
    valid_rows = []
    n_parts_all = []
    widths_all = []
    starts_all = []

    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        if "eventsBool" not in d.files:
            continue
        ch_key = "chnNames" if "chnNames" in d.files else "channel_names"
        if ch_key not in d.files:
            continue
        chs_local = [str(c) for c in d[ch_key]]
        col_to_union = [union_chs.index(c) for c in chs_local]

        evb = np.asarray(d["eventsBool"])  # (n_chn, n_events) canonical
        if evb.ndim != 2 or evb.shape[0] != len(chs_local):
            continue
        evb_em = evb.T.astype(bool)  # event-major (n_events, n_chn_local)
        n_evts_rec = evb_em.shape[0]

        evb_full = np.zeros((n_evts_rec, n_ch), dtype=bool)
        evb_full[:, col_to_union] = evb_em

        if "lagPatRank" in d.files:
            rk = np.asarray(d["lagPatRank"])  # (n_chn, n_events) canonical
            if rk.ndim == 2 and rk.shape[0] == len(chs_local) and rk.shape[1] == n_evts_rec:
                rk_em = rk.T.astype(float)  # (n_events, n_chn_local)
                rk_full = np.zeros((n_evts_rec, n_ch), dtype=float)
                rk_full[:, col_to_union] = rk_em
                rank_rows.append(rk_full)
                valid_rows.append(evb_full)

        n_parts_all.append(evb_em.sum(axis=1))

        # Backfill writes *_packedTimes.npy alongside *_lagPat.npz (no withFreqCent).
        packed_path = npz_path.parent / (
            npz_path.stem.replace('_lagPat', '_packedTimes') + '.npy'
        )
        if packed_path.exists():
            packed = np.load(packed_path, allow_pickle=True)
            packed = np.asarray(packed)
            # Canonical layout: (n_events, 2), columns [start_rel, end_rel].
            if packed.ndim == 2 and packed.shape[1] >= 2:
                starts = packed[:, 0].astype(float, copy=False)
                ends = packed[:, 1].astype(float, copy=False)
                widths_all.append(ends - starts)
                starts_all.append(starts)

    if not n_parts_all:
        return {"error": "no usable npz files"}

    n_part_concat = np.concatenate(n_parts_all)
    res = {
        "channel_names": union_chs,
        "n_participating": compute_n_participating_stats(n_part_concat),
    }
    if widths_all:
        widths_concat = np.concatenate(widths_all)
        res["pack_width_sec"] = compute_pack_width_stats(
            np.zeros_like(widths_concat), widths_concat
        )
    if starts_all:
        starts_concat = np.concatenate(starts_all)
        res["chunk_boundary_event_frac"] = compute_chunk_boundary_event_frac(
            starts_concat, chunk_sec=200.0, tol_sec=2.0
        )

    if rank_rows:
        rank_all = np.vstack(rank_rows)
        valid_all = np.vstack(valid_rows)
        n_evts = rank_all.shape[0]
        if n_evts >= 6:
            half = n_evts // 2
            idx_first = np.arange(0, half)
            idx_second = np.arange(half, n_evts)
            res["splithalf_event_rank_corr"] = compute_subset_rank_corr(
                rank_all, valid_all, idx_first, idx_second
            )
            idx_odd = np.arange(0, n_evts, 2)
            idx_even = np.arange(1, n_evts, 2)
            res["oddeven_event_rank_corr"] = compute_subset_rank_corr(
                rank_all, valid_all, idx_odd, idx_even
            )
        else:
            res["splithalf_event_rank_corr"] = float("nan")
            res["oddeven_event_rank_corr"] = float("nan")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--lagpat-root", required=True,
                   help="Path to <subject>'s lagpat directory (NOT the parent root)")
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res = extract_layer_b_per_subject(Path(args.lagpat_root))
    out = out_dir / f"layer_b_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    if "error" in res:
        import sys
        print(f"WARN: {res['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
