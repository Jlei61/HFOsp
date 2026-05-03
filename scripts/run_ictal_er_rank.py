"""PR-T3-1 v2.1 Layer A driver — ictal ER-rank producer.

CLI modes (per pivot plan v2.1 §9):

  --sentinel    Step A.3: sentinel sanity on epilepsiae/548 + 916.
                Produces per-sentinel JSON + sanity_report.json. If
                sanity FAIL → archive note, do NOT touch cohort.

  --per-subject Step A.4: cohort run (audit_eligible 24). Blocked
                until --sentinel passes. (Not yet implemented.)

Sentinel sanity gate (plan §3.4, three pass criteria, all required):

  1. focal r_sz median earlier than non-focal (Wilcoxon one-sided
     p < 0.1) per ER config
  2. ≥ 3 ok-status seizures per (subject × ER config)
  3. focal mean r_sz_valid_count ≥ 0.5 × non-focal mean

Each sentinel subject is processed for both gamma_ER and broad_ER
independently. Output:

  results/data_driven_soz/layer_a_ictal_er_rank/_sentinel/
    epilepsiae_548.json
    epilepsiae_916.json
    sanity_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ictal_er_rank import (  # noqa: E402
    calibrate_lambda_per_subject,
    compute_cusum_n_d,
    compute_per_channel_rsz_coverage,
    compute_per_subject_r_sz,
    compute_seizure_status,
    compute_stability_s_sz,
    rank_channels_by_n_d,
)
from src.ictal_onset_extraction import (  # noqa: E402
    BASELINE_PRE_SEC,
    BROAD_ER_BANDS,
    GAMMA_ER_BANDS,
    baseline_zscore_er,
    compute_er,
    extract_seizure_window,
    resolve_baseline_window,
)


SENTINEL_SUBJECTS = ["epilepsiae/548", "epilepsiae/916"]
ER_CONFIGS = (GAMMA_ER_BANDS, BROAD_ER_BANDS)
HOP_SEC = 0.1
WIN_SEC = 1.0
PRE_SEC = float(BASELINE_PRE_SEC)              # 300 s pre-onset
POST_SEC = 30.0                                  # 30 s post-onset
DETECTION_PRE_SEC = 5.0                          # plan §3.3: detection [-5s, +30s]
DETECTION_POST_SEC = 30.0
LAMBDA_FPR_PER_HOUR = 1.0
SENTINEL_OUT_DIR = ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank" / "_sentinel"

# Sentinel sanity thresholds (plan §3.4).
SANITY_WILCOXON_ALPHA = 0.1
SANITY_MIN_OK_SEIZURES = 3
SANITY_FOCAL_COVERAGE_RATIO_MIN = 0.5


def _focus_rel_path() -> Path:
    return ROOT / "results" / "epilepsiae_electrode_focus_rel.json"


def _seizure_inventory_path() -> Path:
    return ROOT / "results" / "epilepsiae_seizure_inventory.csv"


def _load_focus_rel() -> Dict:
    with _focus_rel_path().open() as f:
        return json.load(f)


def _focal_channels(subject: str, focus_rel: Dict) -> List[str]:
    sid = subject.split("/", 1)[1]
    return list(focus_rel.get(sid, {}).get("i", []))


def _count_seizures(subject: str) -> int:
    sid = subject.split("/", 1)[1]
    n = 0
    with _seizure_inventory_path().open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("subject") == sid and row.get("clin_onset_epoch"):
                n += 1
    return n


def _detection_idx_window(n_time_frames: int) -> Tuple[int, int]:
    """Detection window indices in ER frames, plan §3.3.

    With ``pre_sec=300`` and ``hop_sec=0.1``, the spectrogram time axis
    is centered such that frame ``round(pre_sec / hop_sec)`` is at
    clinical onset. The detection window is then
    ``[onset - 5s, onset + 30s]``.

    Caps both endpoints to ``[0, n_time_frames]`` to handle boundary
    cases (very short signals).
    """
    onset_frame = int(round(PRE_SEC / HOP_SEC))
    start = max(0, onset_frame - int(round(DETECTION_PRE_SEC / HOP_SEC)))
    end = min(n_time_frames, onset_frame + int(round(DETECTION_POST_SEC / HOP_SEC)))
    return start, end


def _process_one_seizure_all_ers(
    subject: str,
    seizure_idx: int,
    er_configs: Tuple[Dict, ...],
) -> Optional[Dict]:
    """Load seizure window once, compute every ER config from the same
    signal. Cuts signal-load I/O in half compared with separate calls
    per ER config.

    Returns ``None`` on extract_seizure_window failure. Otherwise dict
    with ``per_er[er_key] = {z_er, baseline_idx, status_so_far}`` plus
    shared metadata (``seizure_id``, ``ch_names``, ``n_t_frames`` may
    differ slightly per ER config due to spectrogram framing — kept
    inside the per_er dict).
    """
    try:
        sw = extract_seizure_window(
            subject, seizure_idx,
            pre_sec=PRE_SEC, post_sec=POST_SEC,
            results_root=ROOT / "results", reference="car",
        )
    except (ValueError, IndexError) as exc:
        return {"skipped_reason": str(exc)}

    eeg_rel = (
        sw.eeg_onset_epoch - sw.clin_onset_epoch
        if sw.eeg_onset_epoch is not None else None
    )
    out: Dict = {
        "seizure_id": sw.seizure_id,
        "seizure_idx": seizure_idx,
        "ch_names": list(sw.ch_names),
        "fs": float(sw.fs),
        "per_er": {},
    }
    for er_cfg in er_configs:
        er = compute_er(
            sw.signal, sw.fs,
            fast_band=er_cfg["fast"], slow_band=er_cfg["slow"],
            win_sec=WIN_SEC, hop_sec=HOP_SEC,
        )
        n_t = er.shape[1]
        bw = resolve_baseline_window(
            n_t, hop_sec=HOP_SEC,
            pre_sec=PRE_SEC, eeg_onset_rel_sec=eeg_rel,
        )
        if not bw.valid:
            out["per_er"][er_cfg["key"]] = {
                "status_so_far": "baseline_invalid",
                "baseline_window_sec": [bw.start_sec, bw.end_sec],
                "baseline_valid_sec": float(bw.valid_sec),
                "n_t_frames": n_t,
            }
            continue
        z_er = baseline_zscore_er(
            er, (bw.start_idx, bw.end_idx), hop_sec=HOP_SEC,
        )
        out["per_er"][er_cfg["key"]] = {
            "status_so_far": "ok",
            "baseline_window_sec": [bw.start_sec, bw.end_sec],
            "baseline_valid_sec": float(bw.valid_sec),
            "z_er": z_er,
            "baseline_idx": (bw.start_idx, bw.end_idx),
            "n_t_frames": n_t,
        }
    return out




def _run_subject_all_ers(
    subject: str,
    *,
    n_seizures: int,
    er_configs: Tuple[Dict, ...] = ER_CONFIGS,
) -> Dict[str, Dict]:
    """Layer A pipeline for one subject across every ER config. Plan §3.3 - §3.5.

    Loads each seizure ONCE, computes both ERs from the same signal,
    then per-ER calibrates λ on pooled baselines and runs CUSUM in the
    detection window per seizure.

    Returns ``{er_key: per_er_record_dict}``.
    """
    print(f"  loading {n_seizures} seizures (ER computed for both configs in one pass)...", flush=True)

    # Pass 1: load + compute_er (per ER config) + baseline z-score.
    pass1: List[Dict] = []
    for sz_idx in range(n_seizures):
        t0 = time.time()
        rec = _process_one_seizure_all_ers(subject, sz_idx, er_configs)
        if rec is None or "skipped_reason" in (rec or {}):
            print(
                f"    seizure {sz_idx}: skipped ({(rec or {}).get('skipped_reason', 'unknown')})",
                flush=True,
            )
            continue
        pass1.append(rec)
        per_er_status = {
            k: v.get("status_so_far") for k, v in rec["per_er"].items()
        }
        print(
            f"    seizure {sz_idx} ({rec['seizure_id']}): {per_er_status} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    out: Dict[str, Dict] = {}
    onset_idx = int(round(PRE_SEC / HOP_SEC))

    for er_cfg in er_configs:
        er_key = er_cfg["key"]
        # Pool baseline frames for this ER config.
        baseline_chunks = [
            r["per_er"][er_key]["z_er"][:, r["per_er"][er_key]["baseline_idx"][0]:r["per_er"][er_key]["baseline_idx"][1]]
            for r in pass1
            if r["per_er"].get(er_key, {}).get("status_so_far") == "ok"
        ]
        if not baseline_chunks:
            out[er_key] = {
                "er_config": er_key,
                "n_seizures_total": n_seizures,
                "n_seizures_loaded": len(pass1),
                "n_seizures_baseline_invalid": sum(
                    1 for r in pass1
                    if r["per_er"].get(er_key, {}).get("status_so_far") == "baseline_invalid"
                ),
                "fail_reason": "no valid baselines for λ calibration",
            }
            continue

        n_ch_min = min(c.shape[0] for c in baseline_chunks)
        pooled = np.concatenate([c[:n_ch_min] for c in baseline_chunks], axis=1)
        print(
            f"  [{er_key}] pooled baseline {pooled.shape} "
            f"({pooled.shape[1] * HOP_SEC / 60:.1f} min); calibrating λ...",
            flush=True,
        )
        lam = calibrate_lambda_per_subject(
            pooled, fpr_target_per_hour=LAMBDA_FPR_PER_HOUR,
            bias=0.5, hop_sec=HOP_SEC,
        )
        print(f"  [{er_key}] λ = {lam:.2f}", flush=True)

        per_seizure_ranks: List[Dict[str, Optional[float]]] = []
        seizure_statuses: List[str] = []
        seizure_records: List[Dict] = []

        for r in pass1:
            er_rec = r["per_er"].get(er_key, {})
            ch_names = r["ch_names"]
            if er_rec.get("status_so_far") == "baseline_invalid":
                seizure_statuses.append("baseline_invalid")
                per_seizure_ranks.append({})
                seizure_records.append({
                    "seizure_id": r["seizure_id"],
                    "seizure_idx": r["seizure_idx"],
                    "status": "baseline_invalid",
                    "baseline_window_sec": er_rec["baseline_window_sec"],
                    "baseline_valid_sec": er_rec["baseline_valid_sec"],
                })
                continue
            if "z_er" not in er_rec:
                continue

            z_er = er_rec["z_er"]
            n_t = er_rec["n_t_frames"]
            det_window = _detection_idx_window(n_t)
            n_d_per_ch: Dict[str, Optional[float]] = {}
            for ch_idx, ch in enumerate(ch_names):
                if ch_idx >= z_er.shape[0]:
                    continue
                n_d = compute_cusum_n_d(
                    z_er[ch_idx], lambda_thresh=lam,
                    bias=0.5, detection_idx_window=det_window,
                )
                n_d_per_ch[ch] = float(n_d) if n_d is not None else None

            ranks = rank_channels_by_n_d(n_d_per_ch)
            status_res = compute_seizure_status(
                n_d_per_ch, n_total=len(ch_names), onset_idx=onset_idx,
            )
            per_seizure_ranks.append(ranks)
            seizure_statuses.append(status_res.status)
            seizure_records.append({
                "seizure_id": r["seizure_id"],
                "seizure_idx": r["seizure_idx"],
                "status": status_res.status,
                "n_active": status_res.n_active,
                "n_total": status_res.n_total,
                "fast_recruit_fraction": status_res.fast_recruit_fraction,
                "baseline_window_sec": er_rec["baseline_window_sec"],
            })

        r_sz = compute_per_subject_r_sz(
            per_seizure_ranks, seizure_statuses=seizure_statuses,
        )
        s_sz = compute_stability_s_sz(
            per_seizure_ranks, seizure_statuses=seizure_statuses,
        )
        coverage = compute_per_channel_rsz_coverage(
            per_seizure_ranks, seizure_statuses=seizure_statuses,
        )
        n_ok = sum(1 for s in seizure_statuses if s == "ok")
        out[er_key] = {
            "er_config": er_key,
            "lambda": float(lam),
            "n_seizures_total": n_seizures,
            "n_seizures_loaded": len(pass1),
            "n_seizures_ok": n_ok,
            "n_seizures_baseline_invalid": sum(
                1 for s in seizure_statuses if s == "baseline_invalid"
            ),
            "n_seizures_onset_tied": sum(
                1 for s in seizure_statuses if s == "onset_tied"
            ),
            "n_seizures_onset_unreached": sum(
                1 for s in seizure_statuses if s == "onset_unreached"
            ),
            "r_sz": {ch: (None if v is None else float(v)) for ch, v in r_sz.items()},
            "r_sz_valid_count": {ch: int(v) for ch, v in coverage.items()},
            "s_sz": (None if s_sz is None else float(s_sz)),
            "seizure_records": seizure_records,
        }
    return out


def _wilcoxon_focal_vs_nonfocal(
    r_sz: Dict[str, Optional[float]],
    coverage: Dict[str, int],
    focal_set: set,
    *,
    coverage_floor: int = 1,
) -> Dict:
    """Wilcoxon one-sided test: focal r_sz < non-focal r_sz.

    Restricts to channels with coverage >= coverage_floor (default 1)
    so that channels that never had a finite rank in any qualifying
    seizure don't enter the comparison.
    """
    focal_vals = []
    nonfocal_vals = []
    for ch, v in r_sz.items():
        if v is None:
            continue
        if coverage.get(ch, 0) < coverage_floor:
            continue
        if ch in focal_set:
            focal_vals.append(float(v))
        else:
            nonfocal_vals.append(float(v))

    out: Dict = {
        "n_focal_with_finite_rsz": len(focal_vals),
        "n_nonfocal_with_finite_rsz": len(nonfocal_vals),
        "focal_rsz_median": (float(np.median(focal_vals)) if focal_vals else None),
        "nonfocal_rsz_median": (
            float(np.median(nonfocal_vals)) if nonfocal_vals else None
        ),
    }
    if len(focal_vals) >= 1 and len(nonfocal_vals) >= 1:
        # Use Mann-Whitney U one-sided (focal < nonfocal); Wilcoxon
        # rank-sum is the equivalent test here since focal and non-focal
        # are independent samples, not paired.
        from scipy.stats import mannwhitneyu
        try:
            stat, p = mannwhitneyu(focal_vals, nonfocal_vals, alternative="less")
            out["mannwhitney_u_stat"] = float(stat)
            out["mannwhitney_p_one_sided"] = float(p)
        except ValueError:
            out["mannwhitney_u_stat"] = None
            out["mannwhitney_p_one_sided"] = None
    else:
        out["mannwhitney_u_stat"] = None
        out["mannwhitney_p_one_sided"] = None
    return out


def _coverage_focal_vs_nonfocal(
    coverage: Dict[str, int],
    focal_set: set,
) -> Dict:
    focal = [coverage[ch] for ch in coverage if ch in focal_set]
    nonfocal = [coverage[ch] for ch in coverage if ch not in focal_set]
    return {
        "focal_coverage_mean": float(np.mean(focal)) if focal else 0.0,
        "nonfocal_coverage_mean": float(np.mean(nonfocal)) if nonfocal else 0.0,
        "n_focal_channels": len(focal),
        "n_nonfocal_channels": len(nonfocal),
    }


def _evaluate_sanity(per_subject: Dict) -> Dict:
    """Apply 3 sentinel sanity criteria per (subject × ER config)."""
    focal_set = set(per_subject.get("focal_channels", []))
    sanity: Dict = {}
    for er_key in ("gamma_ER", "broad_ER"):
        if er_key not in per_subject["per_er"]:
            continue
        rec = per_subject["per_er"][er_key]
        if "fail_reason" in rec:
            sanity[er_key] = {"pass": False, "fail_reason": rec["fail_reason"]}
            continue

        wilcox = _wilcoxon_focal_vs_nonfocal(
            rec["r_sz"], rec["r_sz_valid_count"], focal_set,
        )
        cov = _coverage_focal_vs_nonfocal(
            rec["r_sz_valid_count"], focal_set,
        )
        c1 = (
            wilcox["mannwhitney_p_one_sided"] is not None
            and wilcox["mannwhitney_p_one_sided"] < SANITY_WILCOXON_ALPHA
        )
        c2 = rec["n_seizures_ok"] >= SANITY_MIN_OK_SEIZURES
        c3 = (
            cov["nonfocal_coverage_mean"] == 0
            or cov["focal_coverage_mean"] >= SANITY_FOCAL_COVERAGE_RATIO_MIN * cov["nonfocal_coverage_mean"]
        )
        sanity[er_key] = {
            "pass": bool(c1 and c2 and c3),
            "criterion_1_wilcoxon_p_lt_alpha": bool(c1),
            "criterion_2_min_ok_seizures": bool(c2),
            "criterion_3_focal_coverage_ratio": bool(c3),
            "wilcoxon_focal_lt_nonfocal": wilcox,
            "coverage_focal_vs_nonfocal": cov,
        }
    return sanity


def _run_sentinel(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel = _load_focus_rel()
    summary: Dict = {
        "step": "Layer A Step A.3 sentinel sanity",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "alpha_wilcoxon": SANITY_WILCOXON_ALPHA,
        "min_ok_seizures": SANITY_MIN_OK_SEIZURES,
        "focal_coverage_ratio_min": SANITY_FOCAL_COVERAGE_RATIO_MIN,
        "lambda_fpr_per_hour": LAMBDA_FPR_PER_HOUR,
        "subjects": [],
    }
    any_fail = False

    for subject in SENTINEL_SUBJECTS:
        focal = _focal_channels(subject, focus_rel)
        n_sz = _count_seizures(subject)
        print(
            f"\n=== {subject}  focal(i)={len(focal)}  n_seizures={n_sz} ===",
            flush=True,
        )
        per_subject: Dict = {
            "subject": subject,
            "n_seizures_total": n_sz,
            "focal_channels": sorted(focal),
            "per_er": {},
        }
        t_sub = time.time()
        per_er = _run_subject_all_ers(subject, n_seizures=n_sz, er_configs=ER_CONFIGS)
        per_subject["per_er"] = per_er
        elapsed = time.time() - t_sub
        for er_key, rec in per_er.items():
            print(
                f"  [{er_key}] ok={rec.get('n_seizures_ok')} "
                f"bi={rec.get('n_seizures_baseline_invalid')} "
                f"tied={rec.get('n_seizures_onset_tied')} "
                f"ur={rec.get('n_seizures_onset_unreached')} "
                f"s_sz={rec.get('s_sz')} λ={rec.get('lambda')}",
                flush=True,
            )
        print(f"  total subject elapsed: {elapsed:.1f}s", flush=True)

        per_subject["sanity"] = _evaluate_sanity(per_subject)
        # Per-sentinel JSON.
        sid_under = subject.replace("/", "_")
        out_path = out_dir / f"{sid_under}.json"
        with out_path.open("w") as f:
            json.dump(per_subject, f, indent=2, default=_json_default)
        print(f"  wrote {out_path}", flush=True)

        sanity_brief = {
            er_key: {
                "pass": s["pass"],
                "p_one_sided": s.get("wilcoxon_focal_lt_nonfocal", {}).get(
                    "mannwhitney_p_one_sided"
                ),
                "n_ok": per_subject["per_er"][er_key].get("n_seizures_ok"),
                "focal_cov_mean": s.get("coverage_focal_vs_nonfocal", {}).get(
                    "focal_coverage_mean"
                ),
                "nonfocal_cov_mean": s.get("coverage_focal_vs_nonfocal", {}).get(
                    "nonfocal_coverage_mean"
                ),
            }
            for er_key, s in per_subject["sanity"].items()
        }
        summary["subjects"].append({
            "subject": subject,
            "sanity": sanity_brief,
        })
        if any(not s["pass"] for s in per_subject["sanity"].values()):
            any_fail = True

    summary["overall_pass"] = not any_fail
    summary["next_step_unblocked"] = not any_fail and len(summary["subjects"]) == len(SENTINEL_SUBJECTS)
    sanity_path = out_dir / "sanity_report.json"
    with sanity_path.open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\n[sentinel] sanity report → {sanity_path}", flush=True)
    if not summary["overall_pass"]:
        print(
            "[sentinel] OVERALL FAIL — Step A.4 cohort run blocked. "
            "Review sentinel JSON + sanity_report.json before proceeding.",
            flush=True,
        )
        return 1
    print("[sentinel] OVERALL PASS — Step A.4 cohort run unblocked.", flush=True)
    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"unserializable {type(obj).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentinel", action="store_true",
        help="Run sentinel sanity (Step A.3) on epilepsiae/548 + 916.",
    )
    parser.add_argument(
        "--per-subject", action="store_true",
        help="Cohort run (Step A.4). Blocked until sentinel passes.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SENTINEL_OUT_DIR,
        help="Output directory (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.sentinel:
        return _run_sentinel(args.output_dir)
    if args.per_subject:
        raise NotImplementedError("Step A.4 cohort run not yet implemented")
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
