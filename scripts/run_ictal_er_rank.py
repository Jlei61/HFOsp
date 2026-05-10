"""PR-T3-1 v2.2 Layer A driver — ictal ER-rank producer.

CLI modes (per pivot plan v2.2 §9):

  --sentinel    Step A.3: sentinel sanity on epilepsiae/548 + 916.
                Produces per-sentinel JSON + sanity_report.json. If
                no sentinel cell reaches `producer_health == "stable"`
                → archive note, do NOT touch cohort.

  --per-subject Step A.4: cohort run (audit_eligible 24). Blocked
                until --sentinel passes. (Not yet implemented.)

v2.2 sentinel gate (plan §3.4, rewritten 2026-05-04):

  Two **independent** classifications per (subject × ER config),
  computed via :func:`src.ictal_er_rank.classify_producer_health` /
  :func:`classify_clinical_concordance`:

  - `producer_health` ∈ {stable, moderate, unstable, insufficient}:
    decides whether the data-driven r_sz is usable on this cell.
  - `clinical_concordance` ∈ {concordant, partial, discordant,
    not_assessable}: METADATA-ONLY; does NOT gate any pipeline
    decision. Reports whether r_sz aligns with clinical i-label.

  A.4 unblock condition (v2.2): at least 1 sentinel × ER cell is
  `producer_health == "stable"`. Old v2.1 three-criterion hard gate
  (Wilcoxon p<0.1 + n_ok≥3 + focal_cov_ratio) is superseded; see
  plan §3.4.3 for the strikethrough.

Output schema (per-subject JSON + sanity_report.json) follows
plan §3.5 v2.2 — `producer_health` / `clinical_concordance` blocks
nested as `{gamma_ER, broad_ER, details}`.

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

from src.atomic_io import purge_stale_tmp, write_json_atomic  # noqa: E402
from src.ictal_er_rank import (  # noqa: E402
    calibrate_lambda_per_subject,
    classify_clinical_concordance,
    classify_producer_health,
    compute_cusum_n_d,
    compute_cusum_n_d_with_time,
    compute_focal_in_topk,
    compute_per_channel_rsz_coverage,
    compute_per_subject_r_sz,
    compute_seizure_status,
    compute_stability_s_sz,
    compute_top10_coverage_list,
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
# v2.3 (2026-05-08): detection window [-120s, +30s] (was [-5s, +30s] in v2.2).
# This is a science contract change — r_sz / s_sz / tags will be recomputed
# across the cohort. v2.2 JSON is incompatible and lives in per_subject_v2_2/.
DETECTION_PRE_SEC = 120.0                        # was 5.0 in v2.2
DETECTION_POST_SEC = 30.0
SCHEMA_VERSION = "pr_t3_1_layer_a_v2_3_timing"   # v2.3 (2026-05-08)
LAMBDA_FPR_PER_HOUR = 1.0
# v2.2.3 (2026-05-06) — module-level overrides settable from CLI for the
# Phase 2 λ_max relaxation sweep. Kept as module globals so the existing
# _run_subject_all_ers signature stays stable.
LAMBDA_MAX_OVERRIDE = 100.0    # default: same as v2.2.2
LAMBDA_N_GRID_OVERRIDE = 199
SENTINEL_OUT_DIR = ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank" / "_sentinel"
COHORT_OUT_DIR = ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank" / "per_subject"
AUDIT_CSV_PATH = ROOT / "results" / "spatial_modulation" / "data_driven_soz" / "audit.csv"
SENTINEL_ONLY_SUBJECTS = ["epilepsiae/916"]  # 916 NOT in audit_eligible 24; sentinel only

# v2.3 PR-0.1 (2026-05-10) — yuquan dataset now supported.
# extract_seizure_window in src/ictal_onset_extraction.py routes by
# dataset prefix; yuquan path consumes yuquan_block_inventory.csv +
# yuquan_seizure_inventory.csv (Tasks 1–3 of topic5 PR-0.1 plan).
SUPPORTED_DATASETS = frozenset({"epilepsiae", "yuquan"})

# v2.2 unblock threshold (plan §3.4.1, 2026-05-04). At least 1 cell
# (sentinel × ER) must reach `producer_health == "stable"` to unblock A.4.
SENTINEL_MIN_STABLE_CELLS = 1

# Top-K used by classify_producer_health / classify_clinical_concordance.
TOPK_FOR_TAGS = 10


def _focus_rel_path(dataset: str = "epilepsiae") -> Path:
    if dataset == "epilepsiae":
        return ROOT / "results" / "epilepsiae_electrode_focus_rel.json"
    if dataset == "yuquan":
        return ROOT / "results" / "yuquan_soz_core_channels.json"
    raise ValueError(f"Unsupported dataset for focus_rel: {dataset}")


def _seizure_inventory_path(dataset: str = "epilepsiae") -> Path:
    candidates = (
        ROOT / "results" / "dataset_inventory" / f"{dataset}_seizure_inventory.csv",
        ROOT / "results" / f"{dataset}_seizure_inventory.csv",
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"{dataset}_seizure_inventory.csv not found")


def _load_focus_rel(dataset: str = "epilepsiae") -> Dict:
    path = _focus_rel_path(dataset)
    with path.open() as f:
        d = json.load(f)
    if dataset == "yuquan":
        # Normalize flat {sid: [channels]} to {sid: {"i": [channels]}} so the
        # caller surface matches the epilepsiae 3-tier focus_rel JSON.
        return {sid: {"i": list(chans), "l": [], "e": []} for sid, chans in d.items()}
    return d


def _focal_channels(subject: str, focus_rel: Dict) -> List[str]:
    sid = subject.split("/", 1)[1]
    return list(focus_rel.get(sid, {}).get("i", []))


def _count_seizures(subject: str) -> int:
    dataset, sid = subject.split("/", 1)
    inv = _seizure_inventory_path(dataset)
    onset_col = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    n = 0
    with inv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("subject") == sid and row.get(onset_col):
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
            lambda_max=LAMBDA_MAX_OVERRIDE,
            lambda_n_grid=LAMBDA_N_GRID_OVERRIDE,
        )
        print(f"  [{er_key}] λ = {lam:.2f} (lambda_max={LAMBDA_MAX_OVERRIDE})", flush=True)

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
            channel_onsets: Dict[str, Dict[str, Optional[float]]] = {}
            for ch_idx, ch in enumerate(ch_names):
                if ch_idx >= z_er.shape[0]:
                    continue
                # v2.3: get both frame_idx (for ranks) AND t_onset_sec (for atlas).
                onset = compute_cusum_n_d_with_time(
                    z_er[ch_idx], lambda_thresh=lam,
                    bias=0.5, detection_idx_window=det_window,
                    hop_sec=HOP_SEC, win_sec=WIN_SEC, pre_sec=PRE_SEC,
                )
                n_d_per_ch[ch] = (
                    float(onset.frame_idx) if onset.frame_idx is not None else None
                )
                channel_onsets[ch] = {
                    "frame_idx": int(onset.frame_idx) if onset.frame_idx is not None else None,
                    "t_onset_sec": float(onset.t_onset_sec) if onset.t_onset_sec is not None else None,
                }

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
                "channel_onsets": channel_onsets,   # v2.3: per-channel onset for atlas
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


def _build_v2_2_tags(per_subject: Dict) -> Dict[str, Dict]:
    """v2.2 tags per (subject × ER config) — plan §3.4.1 / §3.4.2 / §3.5.

    Returns a 2-key dict matching the per-subject schema in plan §3.5:

        {
          "producer_health": {
            "gamma_ER": "stable" | "moderate" | "unstable" | "insufficient",
            "broad_ER": ...,
            "details": {"gamma_ER": {...}, "broad_ER": {...}},
          },
          "clinical_concordance": {
            "gamma_ER": "concordant" | "partial" | "discordant" | "not_assessable",
            "broad_ER": ...,
            "details": {"gamma_ER": {...}, "broad_ER": {...}},
          },
        }

    Old v2.1 :func:`_evaluate_sanity` (3 hard criteria) is superseded;
    `_wilcoxon_focal_vs_nonfocal` + `_coverage_focal_vs_nonfocal` are
    still used here as inputs to ``classify_clinical_concordance`` and
    for the human-readable sanity_brief output, but they no longer
    decide a hard pass/fail.
    """
    focal_set = set(per_subject.get("focal_channels", []))
    ph_tags: Dict[str, str] = {}
    ph_details: Dict[str, Dict] = {}
    cc_tags: Dict[str, str] = {}
    cc_details: Dict[str, Dict] = {}

    for er_key in ("gamma_ER", "broad_ER"):
        if er_key not in per_subject.get("per_er", {}):
            continue
        rec = per_subject["per_er"][er_key]
        if "fail_reason" in rec:
            ph_tags[er_key] = "insufficient"
            ph_details[er_key] = {"fail_reason": rec["fail_reason"]}
            cc_tags[er_key] = "not_assessable"
            cc_details[er_key] = {"fail_reason": rec["fail_reason"]}
            continue

        n_ok = int(rec.get("n_seizures_ok") or 0)
        s_sz = rec.get("s_sz")
        r_sz: Dict[str, Optional[float]] = rec.get("r_sz") or {}
        coverage: Dict[str, int] = rec.get("r_sz_valid_count") or {}

        # producer_health (tie-inclusive top-K, v2.2.2)
        cov_threshold = max(3.0, 0.5 * float(n_ok))
        topk_cov, ph_expanded = compute_top10_coverage_list(
            r_sz, coverage, top_k=TOPK_FOR_TAGS
        )
        cov_pass = sum(1 for c in topk_cov if c >= cov_threshold)
        ph_tag, ph_d = classify_producer_health(
            n_ok=n_ok,
            s_sz=s_sz,
            topk_cov_pass_count=cov_pass,
            topk_total=len(topk_cov),
            topk_cov_threshold=cov_threshold,
            boundary_tie_inclusive=ph_expanded,
        )
        ph_tags[er_key] = ph_tag
        ph_details[er_key] = ph_d

        # clinical_concordance — METADATA ONLY (not a gate)
        focal_top, topk_total_cc, cc_expanded = compute_focal_in_topk(
            r_sz, focal_set, top_k=TOPK_FOR_TAGS
        )
        wilcox = _wilcoxon_focal_vs_nonfocal(r_sz, coverage, focal_set)
        cc_tag, cc_d = classify_clinical_concordance(
            mannwhitney_u_p_one_sided=wilcox["mannwhitney_p_one_sided"],
            focal_in_topk_count=focal_top,
            topk_total=topk_total_cc,
            n_focal_with_finite_rsz=wilcox["n_focal_with_finite_rsz"],
            n_nonfocal_with_finite_rsz=wilcox["n_nonfocal_with_finite_rsz"],
            focal_rsz_median=wilcox["focal_rsz_median"],
            nonfocal_rsz_median=wilcox["nonfocal_rsz_median"],
            producer_health=ph_tag,
            boundary_tie_inclusive=cc_expanded,
        )
        cc_tags[er_key] = cc_tag
        cc_details[er_key] = cc_d

    return {
        "producer_health": {**ph_tags, "details": ph_details},
        "clinical_concordance": {**cc_tags, "details": cc_details},
    }


def _count_stable_cells(per_subject_list: List[Dict]) -> Tuple[int, int, int, int]:
    """Cohort-level tag counter — count cells per producer_health tag."""
    n_stable = n_moderate = n_unstable = n_insufficient = 0
    for s in per_subject_list:
        ph = s.get("producer_health", {})
        for er_key in ("gamma_ER", "broad_ER"):
            if er_key not in ph:
                continue
            t = ph[er_key]
            if t == "stable":
                n_stable += 1
            elif t == "moderate":
                n_moderate += 1
            elif t == "unstable":
                n_unstable += 1
            elif t == "insufficient":
                n_insufficient += 1
    return n_stable, n_moderate, n_unstable, n_insufficient


def _run_sentinel(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel_per_dataset = {
        "epilepsiae": _load_focus_rel("epilepsiae"),
        "yuquan": _load_focus_rel("yuquan"),
    }
    summary: Dict = {
        "step": "Layer A Step A.3 sentinel sanity (v2.3 timing)",
        "schema_version": SCHEMA_VERSION,
        "detection_window_sec": [-DETECTION_PRE_SEC, DETECTION_POST_SEC],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lambda_fpr_per_hour": LAMBDA_FPR_PER_HOUR,
        "topk_for_tags": TOPK_FOR_TAGS,
        "unblock_threshold": {
            "rule": "at least N (subject × ER) cells with producer_health == 'stable'",
            "min_stable_cells": SENTINEL_MIN_STABLE_CELLS,
        },
        "subjects": [],
    }

    for subject in SENTINEL_SUBJECTS:
        ds = subject.split("/", 1)[0]
        focal = _focal_channels(subject, focus_rel_per_dataset[ds])
        n_sz = _count_seizures(subject)
        print(
            f"\n=== {subject}  focal(i)={len(focal)}  n_seizures={n_sz} ===",
            flush=True,
        )
        per_subject: Dict = {
            "schema_version": SCHEMA_VERSION,
            "detection_window_sec": [-DETECTION_PRE_SEC, DETECTION_POST_SEC],
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

        # v2.2 tags — producer_health + clinical_concordance.
        v2_2 = _build_v2_2_tags(per_subject)
        per_subject.update(v2_2)

        # Per-sentinel JSON.
        sid_under = subject.replace("/", "_")
        out_path = out_dir / f"{sid_under}.json"
        write_json_atomic(out_path, per_subject, default=_json_default)
        print(f"  wrote {out_path}", flush=True)

        # Brief for sanity_report (one row per ER cell).
        brief = {}
        for er_key in ("gamma_ER", "broad_ER"):
            if er_key not in per_subject["per_er"]:
                continue
            ph = per_subject["producer_health"].get(er_key)
            cc = per_subject["clinical_concordance"].get(er_key)
            ph_d = per_subject["producer_health"]["details"].get(er_key, {})
            cc_d = per_subject["clinical_concordance"]["details"].get(er_key, {})
            brief[er_key] = {
                "producer_health": ph,
                "clinical_concordance": cc,
                "n_ok": per_subject["per_er"][er_key].get("n_seizures_ok"),
                "s_sz": per_subject["per_er"][er_key].get("s_sz"),
                "top10_cov_pass": ph_d.get("top10_cov_pass_count"),
                "top10_cov_threshold": ph_d.get("top10_cov_threshold"),
                "wilcoxon_p_one_sided": cc_d.get("wilcoxon_one_sided_p"),
                "focal_in_top10": cc_d.get("focal_in_top10_count"),
                "focal_in_top10_fraction": cc_d.get("focal_in_top10_fraction"),
            }
            print(
                f"  [{er_key}] producer_health={ph}  clinical_concordance={cc}",
                flush=True,
            )
        summary["subjects"].append({"subject": subject, "tags": brief})

    # v2.2 unblock decision: count `stable` cells across all sentinels × ER.
    n_stable, n_moderate, n_unstable, n_insufficient = _count_stable_cells([
        {
            "producer_health": {
                er_key: row["producer_health"]
                for er_key, row in s["tags"].items()
            }
        }
        for s in summary["subjects"]
    ])
    summary["cohort_distribution"] = {
        "n_stable_cells": n_stable,
        "n_moderate_cells": n_moderate,
        "n_unstable_cells": n_unstable,
        "n_insufficient_cells": n_insufficient,
    }
    unblock = n_stable >= SENTINEL_MIN_STABLE_CELLS
    summary["next_step_unblocked"] = bool(unblock)
    summary["unblock_decision"] = (
        f"{n_stable} stable cell(s) ≥ {SENTINEL_MIN_STABLE_CELLS} → "
        f"{'UNBLOCK' if unblock else 'BLOCK'} A.4 cohort run"
    )

    sanity_path = out_dir / "sanity_report.json"
    write_json_atomic(sanity_path, summary, default=_json_default)
    print(f"\n[sentinel] sanity report → {sanity_path}", flush=True)
    print(
        f"[sentinel] producer_health distribution: "
        f"stable={n_stable} moderate={n_moderate} "
        f"unstable={n_unstable} insufficient={n_insufficient}",
        flush=True,
    )
    if not unblock:
        print(
            f"[sentinel] BLOCK — fewer than {SENTINEL_MIN_STABLE_CELLS} "
            f"stable cell(s). Review sentinel JSON + sanity_report.json "
            f"before changing parameters.",
            flush=True,
        )
        return 1
    print(
        f"[sentinel] UNBLOCK — {n_stable} stable cell(s) ≥ "
        f"{SENTINEL_MIN_STABLE_CELLS}. Step A.4 cohort run unblocked.",
        flush=True,
    )
    return 0


def _load_audit_eligible_subjects() -> List[str]:
    """Read audit.csv; return list of "<dataset>/<subject>" with audit_eligible=1."""
    out: List[str] = []
    with AUDIT_CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if int(row.get("audit_eligible") or 0):
                out.append(f"{row['dataset']}/{row['subject']}")
    return out


def _cohort_subject_list() -> Tuple[List[str], List[str]]:
    """v2.3 cohort = audit_eligible ∩ SUPPORTED_DATASETS + sentinel-only (916).

    SUPPORTED_DATASETS now includes both ``epilepsiae`` and ``yuquan``
    (topic5 PR-0.1, 2026-05-10). ``excluded`` should be empty under the
    current contract; it is retained as a defensive log so any future
    dataset still in audit.csv but absent from SUPPORTED_DATASETS is
    auditable instead of silent.

    Returns
    -------
    (included, excluded)
        ``included`` is the cohort to actually run.
        ``excluded`` is the audit_eligible subjects skipped due to
        unsupported dataset (logged + recorded in cohort_summary so any
        scope constraint is auditable, not silent).
    """
    eligible = _load_audit_eligible_subjects()
    included: List[str] = []
    excluded: List[str] = []
    for s in eligible:
        ds = s.split("/", 1)[0]
        if ds in SUPPORTED_DATASETS:
            included.append(s)
        else:
            excluded.append(s)
    extra = [s for s in SENTINEL_ONLY_SUBJECTS if s not in included]
    return included + extra, excluded


def _per_subject_json_path(out_dir: Path, subject: str) -> Path:
    sid_under = subject.replace("/", "_")
    return out_dir / f"{sid_under}.json"


def _is_current_schema_per_subject_json(path: Path) -> bool:
    """v2.3: skip-existing only matches the active schema_version.

    Stale v2.2 JSONs are NOT accepted; backup them to per_subject_v2_2/
    before rerunning so this check correctly forces a recompute.
    """
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return d.get("schema_version") == SCHEMA_VERSION


def _run_cohort(out_dir: Path, *, skip_existing: bool = True) -> int:
    """Step A.4 — cohort run on audit_eligible 24 + sentinel-only 916.

    Reuses the same _run_subject_all_ers + _build_v2_2_tags pipeline as
    sentinel; per-subject JSONs land in ``out_dir`` with v2.2 schema.

    Per plan §3.4 unblock contract, this should ONLY be invoked after
    sentinel sanity passed (sentinel sanity_report.json
    ``next_step_unblocked == True``). The CLI checks that automatically
    unless ``--force`` is passed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rel_per_dataset = {
        "epilepsiae": _load_focus_rel("epilepsiae"),
        "yuquan": _load_focus_rel("yuquan"),
    }
    subjects, excluded = _cohort_subject_list()
    if excluded:
        print(
            f"\n[cohort] EXCLUDED (dataset not in SUPPORTED_DATASETS): "
            f"{len(excluded)} subjects → {', '.join(excluded)}",
            flush=True,
        )
    print(
        f"\n[cohort] starting v2.2 cohort run on {len(subjects)} subjects "
        f"(skip_existing={skip_existing})",
        flush=True,
    )

    n_done = 0
    n_skipped = 0
    n_new = 0
    t_start = time.time()
    for subject in subjects:
        out_path = _per_subject_json_path(out_dir, subject)
        if skip_existing and _is_current_schema_per_subject_json(out_path):
            n_skipped += 1
            n_done += 1
            continue

        ds = subject.split("/", 1)[0]
        focal = _focal_channels(subject, focus_rel_per_dataset[ds])
        n_sz = _count_seizures(subject)
        if n_sz == 0:
            print(f"\n=== {subject}  SKIP (no seizures in inventory) ===", flush=True)
            n_done += 1
            continue
        print(
            f"\n=== {subject}  focal(i)={len(focal)}  n_seizures={n_sz} "
            f"  [{n_done+1}/{len(subjects)}] ===",
            flush=True,
        )
        per_subject: Dict = {
            "schema_version": SCHEMA_VERSION,
            "detection_window_sec": [-DETECTION_PRE_SEC, DETECTION_POST_SEC],
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

        v2_2 = _build_v2_2_tags(per_subject)
        per_subject.update(v2_2)
        write_json_atomic(out_path, per_subject, default=_json_default)
        for er_key in ("gamma_ER", "broad_ER"):
            if er_key in per_subject["producer_health"]:
                ph = per_subject["producer_health"][er_key]
                cc = per_subject["clinical_concordance"][er_key]
                print(f"  [{er_key}] producer_health={ph}  clinical_concordance={cc}", flush=True)
        print(f"  total subject elapsed: {elapsed:.1f}s  → {out_path}", flush=True)
        n_done += 1
        n_new += 1

    elapsed_total = time.time() - t_start
    print(
        f"\n[cohort] DONE — {n_done}/{len(subjects)} subjects "
        f"({n_new} newly computed, {n_skipped} skip_existing) "
        f"in {elapsed_total/60:.1f} min",
        flush=True,
    )
    _write_cohort_summary(out_dir)
    return 0


def _write_cohort_summary(out_dir: Path) -> None:
    """Build cohort_summary.json from per-subject JSONs in ``out_dir``."""
    rows: List[Dict] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "cohort_summary.json":
            continue
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "producer_health" not in d:
            continue
        rows.append(d)

    n_subjects = len(rows)
    # 2D table: (producer_health × clinical_concordance) per ER config
    table: Dict[str, Dict[Tuple[str, str], int]] = {
        "gamma_ER": {},
        "broad_ER": {},
    }
    # v2.2.3 fix (2026-05-07): "at-cap" must reference the *actual* λ_max in this run,
    # not the historical 100. Compare against LAMBDA_MAX_OVERRIDE (the cap used in
    # the just-finished run). Use a small tolerance for float-grid endpoint.
    lambda_at_cap_count = {"gamma_ER": 0, "broad_ER": 0}
    cap_value = float(LAMBDA_MAX_OVERRIDE)
    cap_tol = max(1e-6, cap_value * 1e-4)
    producer_fail_subjects = {"gamma_ER": [], "broad_ER": []}
    for d in rows:
        for er_key in ("gamma_ER", "broad_ER"):
            if er_key not in d.get("producer_health", {}):
                continue
            ph = d["producer_health"][er_key]
            cc = d["clinical_concordance"][er_key]
            key = (ph, cc)
            table[er_key][key] = table[er_key].get(key, 0) + 1
            rec = d.get("per_er", {}).get(er_key, {})
            lam = rec.get("lambda")
            if isinstance(lam, (int, float)) and abs(lam - cap_value) <= cap_tol:
                lambda_at_cap_count[er_key] += 1
            if ph in ("insufficient", "unstable"):
                producer_fail_subjects[er_key].append(d["subject"])

    # Convert tuple keys to strings for JSON
    table_serializable = {
        er_key: {f"{ph}|{cc}": cnt for (ph, cc), cnt in cells.items()}
        for er_key, cells in table.items()
    }

    _, excluded = _cohort_subject_list()
    summary = {
        "step": "Layer A Step A.4 cohort summary (v2.3 timing)",
        "schema_version": SCHEMA_VERSION,
        "detection_window_sec": [-DETECTION_PRE_SEC, DETECTION_POST_SEC],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_subjects": n_subjects,
        "subjects": [d["subject"] for d in rows],
        "excluded_subjects": {
            "list": excluded,
            "reason": "dataset prefix not in SUPPORTED_DATASETS (currently {epilepsiae, yuquan}); list expected to be empty under v2.3 PR-0.1",
        },
        "two_d_distribution": table_serializable,
        "lambda_max_cap": cap_value,
        "lambda_at_cap_count": lambda_at_cap_count,
        "producer_fail_subjects": producer_fail_subjects,
    }
    summary_path = out_dir / "cohort_summary.json"
    write_json_atomic(summary_path, summary, default=_json_default)
    print(f"\n[cohort] summary → {summary_path}", flush=True)
    print(f"[cohort] (producer_health, clinical_concordance) 2D table:", flush=True)
    for er_key in ("gamma_ER", "broad_ER"):
        print(f"  {er_key}:", flush=True)
        for key, cnt in sorted(table_serializable[er_key].items()):
            print(f"    {key:<40} {cnt}", flush=True)
    print(
        f"[cohort] λ-at-cap count (cells with λ == cap={cap_value:.1f}): "
        f"gamma={lambda_at_cap_count['gamma_ER']} broad={lambda_at_cap_count['broad_ER']}",
        flush=True,
    )


def _check_sentinel_unblock() -> bool:
    """Read sanity_report.json; return next_step_unblocked flag."""
    sanity_path = SENTINEL_OUT_DIR / "sanity_report.json"
    if not sanity_path.exists():
        return False
    try:
        d = json.loads(sanity_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(d.get("next_step_unblocked"))


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
        help="Cohort run (Step A.4). Blocked until sentinel sanity passes.",
    )
    parser.add_argument(
        "--cohort-summary-only", action="store_true",
        help="(Re)build cohort_summary.json from existing per-subject JSONs.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip sentinel-unblock check before cohort run.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-run subjects even if a v2.2-schema per-subject JSON exists.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir override (sentinel default: %(default)s; cohort default: per_subject/).",
    )
    parser.add_argument(
        "--lambda-max", type=float, default=100.0,
        help="Cap for λ calibration grid (v2.2.3 sweep mode; default 100). "
             "Phase 2: try 500 to find cells where current cap is artificially restrictive.",
    )
    parser.add_argument(
        "--lambda-n-grid", type=int, default=199,
        help="λ grid resolution (default 199). Scale up with --lambda-max to keep step size.",
    )
    args = parser.parse_args()

    # v2.2.3: plumb λ overrides via module globals (keeps inner functions stable)
    global LAMBDA_MAX_OVERRIDE, LAMBDA_N_GRID_OVERRIDE
    LAMBDA_MAX_OVERRIDE = float(args.lambda_max)
    LAMBDA_N_GRID_OVERRIDE = int(args.lambda_n_grid)

    # v2.3: scrub leftover *.json.tmp from a previous interrupted run
    for candidate in (SENTINEL_OUT_DIR, COHORT_OUT_DIR, args.output_dir):
        if candidate is not None:
            n_purged = purge_stale_tmp(candidate)
            if n_purged:
                print(f"[startup] purged {n_purged} stale .json.tmp in {candidate}", flush=True)

    if args.sentinel:
        out_dir = args.output_dir or SENTINEL_OUT_DIR
        return _run_sentinel(out_dir)

    if args.per_subject:
        out_dir = args.output_dir or COHORT_OUT_DIR
        if not args.force and not _check_sentinel_unblock():
            print(
                "[cohort] BLOCKED — sentinel sanity_report.json missing or "
                "next_step_unblocked=False. Run --sentinel first or pass --force.",
                flush=True,
            )
            return 1
        return _run_cohort(out_dir, skip_existing=not args.no_skip_existing)

    if args.cohort_summary_only:
        out_dir = args.output_dir or COHORT_OUT_DIR
        _write_cohort_summary(out_dir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
