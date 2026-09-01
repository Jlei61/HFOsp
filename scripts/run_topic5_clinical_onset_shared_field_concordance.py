#!/usr/bin/env python3
"""Clinical-onset 0--10 s adaptive-broadband/shared-field concordance.

This is the matched old-onset counterpart of
``run_topic5_eeg_onset_shared_field_concordance.py``.  It changes only the
readout window anchor to clinical onset.  The distal baseline remains the same
true EEG-referenced [-120,-90] s interval, and the frozen shared field, raw
spectral extraction, channel-shuffle null and subject-first folding are kept.

To retain low-sampling-rate events, broadband extends from 1 Hz through the
highest non-Nyquist FFT bin no higher than 150 Hz (for fs=256 Hz: 1--127 Hz).
This adaptive readout is explicitly labelled and is not called exact 1--150 Hz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (  # noqa: E402
    plot_paired_data_null_groups,
)
from scripts.run_topic5_eeg_onset_shared_field_concordance import (  # noqa: E402
    BASE_SEED,
    CONTRACT as EEG_CONTRACT,
    OLD_CACHE,
    _eeg_offset_from_inventory,
    _inventory_for_subject,
    _sha256,
    select_shared_scorers,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE  # noqa: E402
from scripts.run_topic5_tspectral_field_concordance import (  # noqa: E402
    FIELD_ROOT,
    MIN_CONTACTS,
    SPECTRAL_WINDOW_SEC,
    _extract_log_band_power,
    _field_quality,
    _seed,
)
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    DISTAL_BASELINE_EEG_SEC,
    aggregate_complete_windows,
    distal_baseline_robust_z,
    exact_name_align_matrix,
    fold_seizure_null_draws,
    jsonable,
    make_contact_permutations,
    paired_sign_flip_p,
    score_observed_bundle,
    score_permutation_matrix,
)


ARTIFACT_ROOT = Path(os.environ.get("HFOSP_ARTIFACT_ROOT", ROOT)).resolve()
OUT = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
PAPER = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance"
PAPER_FIGURES = PAPER / "figures"
CHECKPOINT_DIR = OUT / "per_subject/clinical_onset_shared_field"

CONTRACT = "topic5_clinical_onset_0_10_shared_field_adaptive_broadband_v1"
CLINICAL_WINDOW = (0.0, 10.0)
BAND = "adaptive_broadband_1_to_min150_nyquist"
BAND_LABEL = "Adaptive BB 1–max valid ≤150 Hz"
MIN_BASELINE_FRAMES = 50
MIN_PERM = 1000

STRATA = {
    "strict_reversed": {
        "axis_quality_tier": "strict_2d",
        "axis_relation": "reversed",
        "label": "Strict reversed",
    },
    "non_strict_reversed": {
        "axis_quality_tier": "non_strict_2d",
        "axis_relation": "reversed",
        "label": "Non-strict reversed",
    },
}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clinical_relative_times(times_from_crop: Sequence[float], pre_sec: float) -> np.ndarray:
    """Spectral centers with genuine clinical onset at zero."""
    return np.asarray(times_from_crop, float) - float(pre_sec)


def highest_valid_broadband_upper(fs: float, *, window_sec: float = 1.0,
                                  requested_hi: float = 150.0) -> float:
    """Highest non-Nyquist FFT bin at or below the requested upper edge."""
    nperseg = int(round(float(fs) * float(window_sec)))
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))
    valid = freqs[(freqs >= 1.0) & (freqs < float(fs) / 2.0)
                  & (freqs <= float(requested_hi))]
    if valid.size == 0:
        raise ValueError(f"no_valid_broadband_bins_for_fs:{fs:g}")
    return float(valid[-1])


def clinical_stratum(field_record: Mapping[str, object]) -> str | None:
    quality = _field_quality(field_record)
    if not quality.get("shared_field_available"):
        return None
    for stratum, contract in STRATA.items():
        if (quality.get("axis_quality_tier") == contract["axis_quality_tier"]
                and quality.get("axis_relation") == contract["axis_relation"]):
            return stratum
    return None


def _event_seed(subject: str, seizure_idx: int, seed: int) -> int:
    return int((zlib.crc32(f"{subject}:{seizure_idx}:clinical-shared".encode()) + seed)
               % (2**32 - 1))


def _extract_bounds(eeg_rel_clinical: float) -> tuple[float, float]:
    """Cover the EEG distal baseline and clinical [0,10] with a guard."""
    pre = max(121.0, 121.0 - float(eeg_rel_clinical))
    return pre, 11.0


def _checkpoint(subject: str, seizure_idx: int) -> Path:
    return CHECKPOINT_DIR / subject / f"seizure_{seizure_idx:03d}.json"


def _checkpoint_valid(path: Path, *, field_hash: str, source_hash: str,
                      n_perm: int, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        payload.get("contract") == CONTRACT
        and payload.get("field_sha256") == field_hash
        and payload.get("source_cache_sha256") == source_hash
        and int(payload.get("n_perm", -1)) == int(n_perm)
        and int(payload.get("seed", -1)) == int(seed)
    )


def _process_event(subject: str, dataset: str, seizure_idx: int,
                   inventory_row: Mapping[str, object], field_record,
                   shared_scorers, *, n_perm: int, seed: int) -> dict:
    sid = subject.split("_", 1)[1]
    eeg_rel = _eeg_offset_from_inventory(dataset, inventory_row)
    pre_sec, post_sec = _extract_bounds(eeg_rel)
    sw = extract_seizure_window(
        f"{dataset}/{sid}", seizure_idx, pre_sec=pre_sec, post_sec=post_sec,
        reference=ICTAL_REFERENCE[dataset],
        results_root=ARTIFACT_ROOT / "results",
    )
    if dataset == "epilepsiae":
        if sw.eeg_onset_epoch is None:
            raise ValueError("missing_eeg_onset")
        actual_eeg_rel = float(sw.eeg_onset_epoch) - float(sw.clin_onset_epoch)
    else:
        actual_eeg_rel = 0.0
    if not np.isclose(actual_eeg_rel, eeg_rel, atol=1e-6):
        raise ValueError(f"inventory_loader_eeg_offset_mismatch:{eeg_rel}:{actual_eeg_rel}")

    target_names = [str(v) for v in field_record["interictal_field"]["contact_order"]]
    raw_names = [recruit.bipolar_alias_label(v) for v in sw.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("raw_channel_aliases_not_unique")
    raw_index = {name: i for i, name in enumerate(raw_names)}
    matched_names = [name for name in target_names if name in raw_index]
    if len(matched_names) < MIN_CONTACTS:
        raise ValueError(f"fewer_than_6_exact_name_contacts:{len(matched_names)}")
    signal = sw.signal[[raw_index[name] for name in matched_names]]

    upper = highest_valid_broadband_upper(sw.fs)
    powers, times_crop = _extract_log_band_power(
        signal, sw.fs, [BAND], band_hz_override={BAND: (1.0, upper)},
    )
    rel_clinical = clinical_relative_times(times_crop, sw.pre_sec)
    baseline_clinical = (
        actual_eeg_rel + DISTAL_BASELINE_EEG_SEC[0],
        actual_eeg_rel + DISTAL_BASELINE_EEG_SEC[1],
    )
    robust = distal_baseline_robust_z(
        powers[BAND], rel_clinical, baseline_clinical,
        min_frames=MIN_BASELINE_FRAMES,
    )
    aligned = exact_name_align_matrix(field_record, matched_names, robust["delta"])
    grid = np.asarray([[CLINICAL_WINDOW[0], CLINICAL_WINDOW[1], 5.0]], float)
    activation_rows, complete = aggregate_complete_windows(
        aligned["values"], rel_clinical, grid,
        spectral_window_sec=SPECTRAL_WINDOW_SEC,
    )
    if not bool(complete[0]):
        raise ValueError("incomplete_clinical_onset_0_10_window")
    activation = np.asarray(activation_rows[0], float)
    n_finite = int(np.isfinite(activation).sum())
    if n_finite < MIN_CONTACTS:
        raise ValueError(f"fewer_than_6_finite_contacts:{n_finite}")

    observed = score_observed_bundle(shared_scorers, activation)
    data = observed.get("shared_maxab")
    if data is None or not np.isfinite(data):
        raise ValueError("nonfinite_shared_maxab")
    matched_mask = np.asarray([name in raw_index for name in target_names], bool)
    perm_seed = _event_seed(subject, seizure_idx, seed)
    permutations = make_contact_permutations(
        target_names, matched_mask, n_perm, perm_seed, mode="all_contact",
    )
    null = score_permutation_matrix(
        shared_scorers, activation[None, :], permutations, chunk_draws=100,
    )["shared_maxab"][:, 0]
    if len(null) != n_perm or not np.isfinite(null).all():
        raise ValueError("nonfinite_or_incomplete_channel_null")

    return {
        "dataset": dataset, "subject": subject, "seizure_idx": int(seizure_idx),
        "seizure_id": sw.seizure_id, "status": "included",
        "time_reference": "clinical_onset", "window_sec": list(CLINICAL_WINDOW),
        "baseline_reference": "eeg_onset", "baseline_eeg_sec": list(DISTAL_BASELINE_EEG_SEC),
        "baseline_clinical_sec": list(map(float, baseline_clinical)),
        "eeg_onset_minus_clinical_sec": float(actual_eeg_rel),
        "fs": float(sw.fs), "band_low_hz": 1.0, "band_high_hz": upper,
        "is_exact_1_150": bool(np.isclose(upper, 150.0)),
        "shared_a_signed": observed.get("shared_a_signed"),
        "shared_a_abs": observed.get("shared_a_abs"),
        "shared_b_signed": observed.get("shared_b_signed"),
        "shared_b_abs": observed.get("shared_b_abs"),
        "shared_best_template": observed.get("shared_best_template"),
        "shared_maxab": float(data), "channel_null": np.asarray(null, float).tolist(),
        "channel_null_median": float(np.median(null)),
        "channel_null_p95": float(np.percentile(null, 95)),
        "n_target_contacts": int(aligned["n_target"]),
        "n_matched_contacts": int(aligned["n_matched"]),
        "n_finite_contacts": n_finite,
        "missing_contacts": list(aligned["missing_names"]),
        "permutation_mode": "all_contact", "permutation_seed": int(perm_seed),
        "mirror_reselected_each_draw": True, "shared_ab_max_reselected_each_draw": True,
    }


def _fold_subject(subject: str, dataset: str, stratum: str, quality,
                  events: list[dict], n_perm: int) -> dict | None:
    if not events:
        return None
    data = float(np.median([event["shared_maxab"] for event in events]))
    folded = fold_seizure_null_draws([
        np.asarray(event["channel_null"], float)[:, None] for event in events
    ])[:, 0]
    null = float(np.median(folded))
    upper = np.asarray([event["band_high_hz"] for event in events], float)
    return {
        "dataset": dataset, "subject": subject, "stratum": stratum,
        "stratum_label": STRATA[stratum]["label"],
        "band": BAND, "band_label": BAND_LABEL,
        "field_plane": "shared_only",
        "field_statistic": "shared_maxab=max(shared_a_abs,shared_b_abs)",
        "time_reference": "clinical_onset", "window_start_sec": 0.0,
        "window_end_sec": 10.0, "shared_maxab": data,
        "channel_null_median_folded": null,
        "channel_null_p2p5_folded": float(np.percentile(folded, 2.5)),
        "channel_null_p95_folded": float(np.percentile(folded, 95)),
        "margin_vs_channel_null_median": data - null,
        "subject_empirical_one_sided_p": float(
            (1 + np.sum(folded >= data - 1e-15)) / (len(folded) + 1)
        ),
        "n_seizures": len(events),
        "n_exact_1_150_seizures": int(np.sum(np.isclose(upper, 150.0))),
        "n_adaptive_lower_upper_seizures": int(np.sum(upper < 150.0)),
        "minimum_band_high_hz": float(np.min(upper)),
        "maximum_band_high_hz": float(np.max(upper)),
        "seizure_idxs": ";".join(str(event["seizure_idx"]) for event in events),
        "n_channel_shuffle_draws": int(n_perm),
        "axis_quality_tier": quality.get("axis_quality_tier"),
        "axis_relation": quality.get("axis_relation"),
        "shared_field_available": quality.get("shared_field_available"),
        "field_fingerprint_algorithm": quality.get("field_fingerprint_algorithm"),
        "field_fingerprint_sha256": quality.get("field_fingerprint_sha256"),
    }


def _cohort(subject: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for stratum in STRATA:
        frame = subject[subject.stratum == stratum].sort_values("subject")
        if frame.empty:
            continue
        data = frame.shared_maxab.to_numpy(float)
        null = frame.channel_null_median_folded.to_numpy(float)
        margin = data - null
        # n=1 is retained as the requested descriptive case, not called a
        # cohort replication.  Wilcoxon is shown only to preserve the accepted
        # paired-figure grammar; inferential status is explicit in the table.
        wilcoxon_p = float(wilcoxon(data, null, alternative="greater").pvalue)
        signflip = (float(paired_sign_flip_p(
            margin, n_perm=100000, seed=_seed(f"clinical:{stratum}", seed)
        )) if len(frame) >= 2 else np.nan)
        rows.append({
            "stratum": stratum, "stratum_label": STRATA[stratum]["label"],
            "inference_status": "cohort" if len(frame) >= 2 else "single_subject_descriptive",
            "n_subjects": len(frame), "n_seizures": int(frame.n_seizures.sum()),
            "data_median": float(np.median(data)),
            "data_iqr_low": float(np.percentile(data, 25)),
            "data_iqr_high": float(np.percentile(data, 75)),
            "null_median": float(np.median(null)),
            "null_iqr_low": float(np.percentile(null, 25)),
            "null_iqr_high": float(np.percentile(null, 75)),
            "margin_median": float(np.median(margin)),
            "margin_iqr_low": float(np.percentile(margin, 25)),
            "margin_iqr_high": float(np.percentile(margin, 75)),
            "n_data_gt_null": int(np.sum(margin > 0)),
            "wilcoxon_one_sided_data_gt_null_p": wilcoxon_p,
            "two_sided_subject_sign_flip_p": signflip,
        })
    return pd.DataFrame(rows)


def _plot(subject: pd.DataFrame, cohort: pd.DataFrame) -> tuple[Path, Path]:
    groups = []
    stats = cohort.set_index("stratum")
    for stratum in STRATA:
        frame = subject[subject.stratum == stratum].sort_values("subject")
        if frame.empty:
            continue
        p = float(stats.loc[stratum, "wilcoxon_one_sided_data_gt_null_p"])
        groups.append({
            "label": STRATA[stratum]["label"],
            "rows": [{"subject_id": row.subject, "data": float(row.shared_maxab),
                      "null": float(row.channel_null_median_folded),
                      "n_seizures": int(row.n_seizures)} for row in frame.itertuples()],
            "summary": {"n": len(frame)}, "display_p": p,
            "p_label": "one-sided p" if len(frame) >= 2 else "case p",
        })
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    png = PAPER_FIGURES / "clinical_onset_shared_field_strict_vs_non_strict_reversed.png"
    pdf = png.with_suffix(".pdf")
    plot_paired_data_null_groups(
        groups, png, pdf, ylabel="Clinical-onset shared-field concordance |r|",
        seed=BASE_SEED + 1,
    )
    return png, pdf


def _write_readme(cohort: pd.DataFrame) -> None:
    readme = PAPER_FIGURES / "README.md"
    existing = readme.read_text() if readme.exists() else "# Fig3 supplement figures\n"
    marker = "### clinical_onset_shared_field_strict_vs_non_strict_reversed.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    stats = "; ".join(
        f"{r.stratum_label}: n={r.n_subjects} subjects/{r.n_seizures} seizures, "
        f"one-sided Wilcoxon p={r.wilcoxon_one_sided_data_gt_null_p:.4g}"
        for r in cohort.itertuples()
    )
    addition = f"""### clinical_onset_shared_field_strict_vs_non_strict_reversed.png / clinical_onset_shared_field_strict_vs_non_strict_reversed.pdf

以 clinical onset（旧 onset）为 0 s，从原始 SEEG 重提 `[0,10] s` 能量，并和冻结 shared A/B fields 对齐。宽带上限逐事件取不超过150 Hz的最高非-Nyquist FFT bin，因此256 Hz记录使用1–127 Hz；Null仍是接触点层1000次channel shuffle，并在每个draw重新选择mirror和shared A/B max。绘图严格复用既有 Fig3 成对 Data–Null 函数。

**关注点**：{stats}。`Non-strict reversed` 当前只有E384一名患者，只是case/sensitivity，不作为独立cohort复现。
"""
    readme.write_text(existing + addition)


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < MIN_PERM:
        raise ValueError(f"n_perm must be >= {MIN_PERM}")
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(OLD_CACHE.glob("*.npz"))
    hashes_before = {_display_path(p): _sha256(p) for p in source_paths}

    event_rows, subject_rows, audit = [], [], []
    candidates = []
    for field_path in sorted(FIELD_ROOT.glob("*.json")):
        if args.subjects and field_path.stem not in set(args.subjects):
            continue
        record = json.loads(field_path.read_text())
        stratum = clinical_stratum(record)
        if stratum:
            candidates.append((field_path.stem, field_path, record, stratum))

    for number, (subject, field_path, record, stratum) in enumerate(candidates, 1):
        dataset, sid = subject.split("_", 1)
        meta_path, source_npz = OLD_CACHE / f"{subject}.json", OLD_CACHE / f"{subject}.npz"
        quality = _field_quality(record)
        if not meta_path.exists() or not source_npz.exists():
            audit.append({"subject": subject, "stratum": stratum, "status": "drop",
                          "drop_reason": "missing_old_eligibility_cache",
                          "axis_quality_tier": quality.get("axis_quality_tier"),
                          "axis_relation": quality.get("axis_relation")})
            continue
        eligible = [int(v) for v in json.loads(meta_path.read_text()).get("eligible_idxs", [])]
        print(f"[{number}/{len(candidates)}] {subject} {stratum}: {len(eligible)} events",
              flush=True)
        field_hash, source_hash = _sha256(field_path), _sha256(source_npz)
        try:
            shared_scorers = select_shared_scorers(scorers_from_interictal_record(record))
        except Exception as exc:
            audit.append({"subject": subject, "stratum": stratum, "status": "drop",
                          "drop_reason": f"field_unavailable:{type(exc).__name__}:{exc}"})
            continue
        inventory = _inventory_for_subject(dataset, sid)
        events = []
        for j, seizure_idx in enumerate(eligible, 1):
            checkpoint = _checkpoint(subject, seizure_idx)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and _checkpoint_valid(
                    checkpoint, field_hash=field_hash, source_hash=source_hash,
                    n_perm=args.n_perm, seed=args.seed):
                event = json.loads(checkpoint.read_text())["event"]
                print(f"  seizure {seizure_idx} [{j}/{len(eligible)}] resume", flush=True)
            else:
                print(f"  seizure {seizure_idx} [{j}/{len(eligible)}]", flush=True)
                try:
                    if not (0 <= seizure_idx < len(inventory)):
                        raise IndexError(f"seizure_idx_out_of_inventory:{len(inventory)}")
                    event = _process_event(
                        subject, dataset, seizure_idx, inventory[seizure_idx], record,
                        shared_scorers, n_perm=args.n_perm, seed=args.seed,
                    )
                except Exception as exc:
                    event = {"dataset": dataset, "subject": subject,
                             "seizure_idx": seizure_idx, "status": "drop",
                             "drop_reason": f"{type(exc).__name__}:{exc}"}
                    print(f"    DROP {event['drop_reason']}", flush=True)
                checkpoint.write_text(json.dumps(jsonable({
                    "contract": CONTRACT, "eeg_counterpart_contract": EEG_CONTRACT,
                    "field_sha256": field_hash,
                    "field_fingerprint_sha256": (
                        record.get("interictal_field") or {}
                    ).get("fingerprint_sha256"),
                    "source_cache_sha256": source_hash,
                    "n_perm": args.n_perm, "seed": args.seed, "event": event,
                }), ensure_ascii=False) + "\n")
            if event.get("status") == "included": events.append(event)
            event_rows.append({
                "dataset": dataset, "subject": subject, "stratum": stratum,
                "seizure_idx": seizure_idx, "status": event.get("status"),
                "drop_reason": event.get("drop_reason", ""),
                "time_reference": "clinical_onset", "window_start_sec": 0.0,
                "window_end_sec": 10.0, "fs": event.get("fs"),
                "band_low_hz": event.get("band_low_hz"),
                "band_high_hz": event.get("band_high_hz"),
                "is_exact_1_150": event.get("is_exact_1_150"),
                "n_finite_contacts": event.get("n_finite_contacts"),
            })
        row = _fold_subject(subject, dataset, stratum, quality, events, args.n_perm)
        if row is not None: subject_rows.append(row)
        audit.append({
            "subject": subject, "stratum": stratum,
            "status": "included" if row is not None else "drop",
            "drop_reason": "" if row is not None else "no_resolvable_events",
            "n_contract_events": len(eligible), "n_included_events": len(events),
            "axis_quality_tier": quality.get("axis_quality_tier"),
            "axis_relation": quality.get("axis_relation"),
            "scorers_used": "shared_a;shared_b", "own_scorers_used": False,
        })

    event_frame, subject_frame = pd.DataFrame(event_rows), pd.DataFrame(subject_rows)
    cohort = _cohort(subject_frame, args.seed)
    audit_frame = pd.DataFrame(audit)
    paths = {
        "event_inventory": OUT / "clinical_onset_shared_field_event_inventory.csv",
        "subject": OUT / "clinical_onset_shared_field_subject.csv",
        "cohort": OUT / "clinical_onset_shared_field_cohort.csv",
        "audit": OUT / "clinical_onset_shared_field_subject_audit.csv",
    }
    event_frame.to_csv(paths["event_inventory"], index=False)
    subject_frame.to_csv(paths["subject"], index=False)
    cohort.to_csv(paths["cohort"], index=False)
    audit_frame.to_csv(paths["audit"], index=False)
    hashes_after = {_display_path(p): _sha256(p) for p in source_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("historical source cache NPZ changed during analysis")

    png, pdf = _plot(subject_frame, cohort)
    _write_readme(cohort)
    summary = {
        "contract": CONTRACT, "eeg_counterpart_contract": EEG_CONTRACT,
        "time_reference": "clinical_onset", "window_sec": list(CLINICAL_WINDOW),
        "baseline_reference": "eeg_onset", "baseline_sec": list(DISTAL_BASELINE_EEG_SEC),
        "field_plane": "shared_only", "scorers_used": ["shared_a", "shared_b"],
        "own_scorers_used": False,
        "observed_statistic": "shared_maxab=max(shared_a_abs,shared_b_abs)",
        "adaptive_band_contract": (
            "1 Hz to highest non-Nyquist 1-s FFT bin <=150 Hz; fs=256 -> 127 Hz"
        ),
        "null": {"mode": "all_contact_channel_shuffle", "n_draws": args.n_perm,
                 "mirror_reselected_each_draw": True,
                 "shared_ab_max_reselected_each_draw": True,
                 "folding": "seizure median within subject for every draw"},
        "cohort_statistics": cohort.to_dict(orient="records"),
        "source_cache_npz_unchanged": True,
        "outputs": {key: _display_path(path) for key, path in paths.items()}
                   | {"figure_png": _display_path(png),
                      "figure_pdf": _display_path(pdf)},
    }
    summary_path = OUT / "clinical_onset_shared_field_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    for key in ("subject", "cohort"):
        (PAPER / paths[key].name).write_text(paths[key].read_text())
    (PAPER / summary_path.name).write_text(summary_path.read_text())
    print(cohort.to_string(index=False), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    global ARTIFACT_ROOT, FIELD_ROOT, OLD_CACHE, OUT, PAPER, PAPER_FIGURES, CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--field-root", type=Path, default=FIELD_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--old-cache-dir", type=Path, default=OLD_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ARTIFACT_ROOT = args.artifact_root.resolve()
    FIELD_ROOT = args.field_root.resolve()
    OLD_CACHE = args.old_cache_dir.resolve()
    OUT = args.out_dir.resolve()
    PAPER = args.paper_dir.resolve()
    PAPER_FIGURES = PAPER / "figures"
    CHECKPOINT_DIR = OUT / "per_subject/clinical_onset_shared_field"
    run(args)


if __name__ == "__main__":
    main()
