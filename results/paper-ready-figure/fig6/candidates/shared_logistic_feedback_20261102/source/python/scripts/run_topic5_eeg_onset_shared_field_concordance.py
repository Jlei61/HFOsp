#!/usr/bin/env python3
"""EEG-onset 0--10 s concordance with the frozen shared interictal field.

This focused analysis answers one narrow question: among subjects whose two
interictal template axes are high-quality and reverse-collinear, does the
contact-wise early-ictal broadband energy pattern (0--10 s after the genuine
electrographic onset) align with their single frozen shared propagation plane
more strongly than an all-contact channel-shuffle null?

The event denominator is held to the historical ``t0_feature_cache_bb150``
eligibility list, but no activation value or time zero is read from that cache.
Raw SEEG is re-extracted around the true EEG onset.  Only ``shared_a`` and
``shared_b`` scorers are loaded after the frozen-field fingerprint succeeds;
``own_a`` and ``own_b`` are never scored.  Every null draw permutes contact
identity before smoothing/correlation and therefore repeats mirror choice and
the shared A/B max selection.  Events are median-folded within subject before
cohort inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE, _inventory_rows  # noqa: E402
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


OUT = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
PAPER = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance"
PAPER_FIGURES = PAPER / "figures"
OLD_CACHE = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
CHECKPOINT_DIR = OUT / "per_subject/eeg_onset_shared_field"

CONTRACT = "topic5_eeg_onset_0_10_shared_field_strict_reversed_v2"
EEG_WINDOW = (0.0, 10.0)
N_PERM_MIN = 1000
MIN_BASELINE_FRAMES = 50
BASE_SEED = 20260717

BAND_HZ = {
    "broadband_1_150": (1.0, 150.0),
    "broadband_1_45_sensitivity": (1.0, 45.0),
    "broadband_subject_nyquist_fallback": (1.0, 150.0),
}
BAND_LABEL = {
    "broadband_1_150": "Shared BB 1–150 Hz",
    "broadband_1_45_sensitivity": "Shared BB 1–45 Hz sensitivity",
    "broadband_subject_nyquist_fallback": "Shared BB ≤150 Hz (Nyquist-limited)",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_seed(subject: str, seizure_idx: int, seed: int) -> int:
    return int((zlib.crc32(f"{subject}:{seizure_idx}:eeg-shared".encode()) + seed)
               % (2**32 - 1))


def true_eeg_relative_times(times_from_crop: Sequence[float], pre_sec: float,
                            eeg_onset_minus_clinical_sec: float) -> np.ndarray:
    """Convert crop-relative spectral centers to genuine EEG-onset seconds."""
    rel_clinical = np.asarray(times_from_crop, float) - float(pre_sec)
    return rel_clinical - float(eeg_onset_minus_clinical_sec)


def available_bands_for_fs(fs: float, *, subject_fallback_upper_hz: float = 150.0
                           ) -> list[str]:
    """Return independently available bands without losing low-Nyquist sensitivity."""
    nyquist = float(fs) / 2.0
    output = [
        key for key in ("broadband_1_150", "broadband_1_45_sensitivity")
        if BAND_HZ[key][1] < nyquist
    ]
    if 1.0 < float(subject_fallback_upper_hz) < nyquist:
        output.append("broadband_subject_nyquist_fallback")
    return output


def select_shared_scorers(scorers: Mapping[str, Mapping[str, object]]):
    """Fail closed unless both frozen shared-template fields are available."""
    selected = {key: scorers[key] for key in ("shared_a", "shared_b") if key in scorers}
    if set(selected) != {"shared_a", "shared_b"}:
        raise ValueError("missing_shared_a_or_shared_b_field")
    if any(key.startswith("own_") for key in selected):
        raise AssertionError("own field leaked into shared-only scorer set")
    return selected


def is_strict_reversed(field_record: Mapping[str, object]) -> bool:
    quality = _field_quality(field_record)
    return bool(
        quality.get("axis_quality_tier") == "strict_2d"
        and quality.get("axis_relation") == "reversed"
        and quality.get("shared_field_available")
    )


def _inventory_for_subject(dataset: str, sid: str) -> list[dict]:
    rows, onset_field = _inventory_rows(dataset, sid)
    return sorted(rows, key=lambda row: float(row[onset_field]))


def _subject_common_broadband_upper_hz(dataset: str, sid: str,
                                       eligible: Sequence[int],
                                       inventory: Sequence[Mapping[str, object]]) -> float:
    """Highest 1-s FFT bin supported by every contract event for a subject."""
    if dataset != "epilepsiae":
        return 150.0
    candidates = (
        ROOT / "results/dataset_inventory/epilepsiae_block_inventory.csv",
        ROOT / "results/epilepsiae_block_inventory.csv",
    )
    block_path = next((path for path in candidates if path.exists()), None)
    if block_path is None:
        raise FileNotFoundError("missing_epilepsiae_block_inventory")
    with block_path.open() as handle:
        block_rows = [row for row in csv.DictReader(handle) if row.get("subject") == sid]
    fs_by_block = {}
    for row in block_rows:
        value = row.get("head_sample_rate") or row.get("sample_rate_sql")
        if value not in (None, ""):
            fs_by_block[str(row["block_id"])] = float(value)
    sample_rates = []
    for seizure_idx in eligible:
        if not (0 <= int(seizure_idx) < len(inventory)):
            raise IndexError(f"seizure_idx_out_of_inventory:{seizure_idx}:{len(inventory)}")
        block_id = str(inventory[int(seizure_idx)].get("block_id", ""))
        if block_id not in fs_by_block:
            raise ValueError(f"missing_sample_rate_for_block:{block_id}")
        sample_rates.append(fs_by_block[block_id])
    if not sample_rates:
        raise ValueError("no_sample_rates_for_subject_fallback")
    # A 1-s FFT has ~1-Hz bins.  Exclude the Nyquist bin itself, matching the
    # exact-band extractor's strict ``upper < fs/2`` rule.
    highest_common_bin = float(np.floor(min(sample_rates) / 2.0 - 1e-9))
    return min(150.0, highest_common_bin)


def _eeg_offset_from_inventory(dataset: str, row: Mapping[str, object]) -> float:
    """Require a genuine EEG annotation; never substitute the clinical onset."""
    if dataset == "yuquan":
        # Yuquan's sole onset reference is electrographic by construction.
        return 0.0
    eeg = row.get("eeg_onset_epoch")
    clinical = row.get("clin_onset_epoch")
    if eeg in (None, ""):
        raise ValueError("missing_eeg_onset")
    if clinical in (None, ""):
        raise ValueError("missing_clinical_reference_for_eeg_offset")
    return float(eeg) - float(clinical)


def _extract_bounds(eeg_rel_clinical: float) -> tuple[float, float]:
    """Cover EEG [-120,-90] and [0,10] plus a one-second spectral guard."""
    pre = max(121.0, 121.0 - float(eeg_rel_clinical))
    post = max(11.0, 11.0 + float(eeg_rel_clinical))
    return pre, post


def _event_checkpoint(subject: str, seizure_idx: int) -> Path:
    return CHECKPOINT_DIR / subject / f"seizure_{seizure_idx:03d}.json"


def _checkpoint_valid(path: Path, *, field_hash: str, source_hash: str,
                      subject_fallback_upper_hz: float,
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
        and np.isclose(float(payload.get("subject_fallback_upper_hz", np.nan)),
                       float(subject_fallback_upper_hz))
    )


def _process_event(subject: str, dataset: str, seizure_idx: int,
                   inventory_row: Mapping[str, object],
                   field_record: Mapping[str, object], shared_scorers,
                   *, subject_fallback_upper_hz: float,
                   n_perm: int, seed: int) -> dict:
    sid = subject.split("_", 1)[1]
    eeg_rel_inventory = _eeg_offset_from_inventory(dataset, inventory_row)
    pre_sec, post_sec = _extract_bounds(eeg_rel_inventory)
    sw = extract_seizure_window(
        f"{dataset}/{sid}", seizure_idx, pre_sec=pre_sec, post_sec=post_sec,
        reference=ICTAL_REFERENCE[dataset],
    )
    if dataset == "epilepsiae":
        if sw.eeg_onset_epoch is None:
            raise ValueError("missing_eeg_onset")
        eeg_rel_actual = float(sw.eeg_onset_epoch) - float(sw.clin_onset_epoch)
    else:
        # Do not create a fictitious Yuquan clinical annotation.  The loader's
        # sole reference is already the EEG onset.
        eeg_rel_actual = 0.0
    if not np.isclose(eeg_rel_actual, eeg_rel_inventory, atol=1e-6):
        raise ValueError(
            f"inventory_loader_eeg_offset_mismatch:{eeg_rel_inventory}:{eeg_rel_actual}"
        )

    target_names = [str(value) for value in
                    field_record["interictal_field"]["contact_order"]]
    raw_names = [recruit.bipolar_alias_label(value) for value in sw.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("raw_channel_aliases_not_unique")
    raw_index = {name: idx for idx, name in enumerate(raw_names)}
    matched_names = [name for name in target_names if name in raw_index]
    if len(matched_names) < MIN_CONTACTS:
        raise ValueError(f"fewer_than_6_exact_name_contacts:{len(matched_names)}")
    signal = sw.signal[[raw_index[name] for name in matched_names]]

    band_keys = available_bands_for_fs(
        sw.fs, subject_fallback_upper_hz=subject_fallback_upper_hz
    )
    if "broadband_1_45_sensitivity" not in band_keys:
        raise ValueError(f"nyquist_below_45_hz:{sw.fs / 2.0:g}")
    band_hz_override = dict(BAND_HZ)
    band_hz_override["broadband_subject_nyquist_fallback"] = (
        1.0, float(subject_fallback_upper_hz)
    )
    powers, times_crop = _extract_log_band_power(
        signal, sw.fs, band_keys, band_hz_override=band_hz_override,
    )
    rel_eeg = true_eeg_relative_times(times_crop, sw.pre_sec, eeg_rel_actual)

    matched_mask = np.asarray([name in raw_index for name in target_names], bool)
    perm_seed = _event_seed(subject, seizure_idx, seed)
    permutations = make_contact_permutations(
        target_names, matched_mask, n_perm, perm_seed, mode="all_contact",
    )
    fixed_grid = np.asarray([[EEG_WINDOW[0], EEG_WINDOW[1], 5.0]], float)

    band_results = {}
    for band_key in band_keys:
        robust = distal_baseline_robust_z(
            powers[band_key], rel_eeg, DISTAL_BASELINE_EEG_SEC,
            min_frames=MIN_BASELINE_FRAMES,
        )
        aligned = exact_name_align_matrix(field_record, matched_names, robust["delta"])
        activation_rows, complete = aggregate_complete_windows(
            aligned["values"], rel_eeg, fixed_grid,
            spectral_window_sec=SPECTRAL_WINDOW_SEC,
        )
        if not bool(complete[0]):
            raise ValueError("incomplete_eeg_onset_0_10_window")
        activation = np.asarray(activation_rows[0], float)
        n_finite = int(np.isfinite(activation).sum())
        if n_finite < MIN_CONTACTS:
            raise ValueError(f"fewer_than_6_finite_contacts:{n_finite}")

        observed = score_observed_bundle(shared_scorers, activation)
        shared_maxab = observed.get("shared_maxab")
        if shared_maxab is None or not np.isfinite(shared_maxab):
            raise ValueError("nonfinite_shared_maxab")
        null_scores = score_permutation_matrix(
            shared_scorers, activation[None, :], permutations, chunk_draws=100,
        )
        null = np.asarray(null_scores["shared_maxab"][:, 0], float)
        if len(null) != n_perm or not np.isfinite(null).all():
            raise ValueError("nonfinite_or_incomplete_channel_null")
        base_center = np.asarray(robust["baseline_z_center"], float)
        band_results[band_key] = {
            "band_hz": list(band_hz_override[band_key]),
            "shared_a_signed": observed.get("shared_a_signed"),
            "shared_a_abs": observed.get("shared_a_abs"),
            "shared_b_signed": observed.get("shared_b_signed"),
            "shared_b_abs": observed.get("shared_b_abs"),
            "shared_best_template": observed.get("shared_best_template"),
            "shared_maxab": float(shared_maxab),
            "channel_null": null.tolist(),
            "channel_null_median": float(np.median(null)),
            "channel_null_p95": float(np.percentile(null, 95)),
            "n_target_contacts": int(aligned["n_target"]),
            "n_matched_contacts": int(aligned["n_matched"]),
            "n_finite_contacts": n_finite,
            "missing_contacts": list(aligned["missing_names"]),
            "spatial_median_delta_energy": float(np.nanmedian(activation)),
            "spatial_mean_delta_energy": float(np.nanmean(activation)),
            "distal_baseline_z_center_max_abs": float(
                np.nanmax(np.abs(base_center))
            ),
        }

    return {
        "dataset": dataset,
        "subject": subject,
        "seizure_idx": int(seizure_idx),
        "seizure_id": sw.seizure_id,
        "status": "included",
        "time_reference": "genuine_eeg_onset",
        "window_start_sec": EEG_WINDOW[0],
        "window_end_sec": EEG_WINDOW[1],
        "distal_baseline_start_sec": DISTAL_BASELINE_EEG_SEC[0],
        "distal_baseline_end_sec": DISTAL_BASELINE_EEG_SEC[1],
        "eeg_onset_minus_clinical_sec": eeg_rel_actual,
        "clinical_onset_epoch": None if dataset == "yuquan" else sw.clin_onset_epoch,
        "eeg_onset_epoch": (sw.clin_onset_epoch if dataset == "yuquan"
                            else sw.eeg_onset_epoch),
        "fs": float(sw.fs),
        "subject_fallback_upper_hz": float(subject_fallback_upper_hz),
        "permutation_mode": "all_contact",
        "permutation_seed": int(perm_seed),
        "same_permutation_reused_across_bands": True,
        "n_perm": int(n_perm),
        "bands": band_results,
    }


def _fold_subject(subject: str, dataset: str, quality: Mapping[str, object],
                  events: list[dict], n_perm: int) -> list[dict]:
    rows = []
    for band_key in BAND_HZ:
        usable = [event for event in events if band_key in event.get("bands", {})]
        if not usable:
            continue
        ranges = {
            tuple(map(float, event["bands"][band_key]["band_hz"])) for event in usable
        }
        if len(ranges) != 1:
            raise ValueError(f"within_subject_band_range_drift:{subject}:{band_key}:{ranges}")
        band_lo, band_hi = next(iter(ranges))
        observed = np.asarray(
            [event["bands"][band_key]["shared_maxab"] for event in usable], float
        )
        event_null = [
            np.asarray(event["bands"][band_key]["channel_null"], float)[:, None]
            for event in usable
        ]
        folded_null = fold_seizure_null_draws(event_null)[:, 0]
        if len(folded_null) != n_perm:
            raise AssertionError("subject null draw count drifted")
        data = float(np.median(observed))
        null_median = float(np.median(folded_null))
        rows.append({
            "dataset": dataset,
            "subject": subject,
            "band": band_key,
            "band_label": BAND_LABEL[band_key],
            "band_lower_hz": band_lo,
            "band_upper_hz": band_hi,
            "field_plane": "shared_only",
            "field_statistic": "shared_maxab=max(shared_a_abs,shared_b_abs)",
            "time_reference": "genuine_eeg_onset",
            "window_start_sec": EEG_WINDOW[0],
            "window_end_sec": EEG_WINDOW[1],
            "shared_maxab": data,
            "channel_null_median_folded": null_median,
            "channel_null_p2p5_folded": float(np.percentile(folded_null, 2.5)),
            "channel_null_p95_folded": float(np.percentile(folded_null, 95)),
            "margin_vs_channel_null_median": data - null_median,
            "subject_empirical_one_sided_p": float(
                (1 + np.sum(folded_null >= data - 1e-15)) / (len(folded_null) + 1)
            ),
            "n_seizures": len(usable),
            "seizure_idxs": ";".join(str(event["seizure_idx"]) for event in usable),
            "n_channel_shuffle_draws": int(n_perm),
            "axis_quality_tier": quality.get("axis_quality_tier"),
            "axis_relation": quality.get("axis_relation"),
            "shared_field_available": quality.get("shared_field_available"),
            "field_fingerprint_algorithm": quality.get("field_fingerprint_algorithm"),
            "field_fingerprint_sha256": quality.get("field_fingerprint_sha256"),
        })
    return rows


def _cohort_statistics(subject_rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for band_key, frame in subject_rows.groupby("band", sort=False):
        frame = frame.sort_values("subject")
        data = frame["shared_maxab"].to_numpy(float)
        null = frame["channel_null_median_folded"].to_numpy(float)
        margin = data - null
        if len(frame) >= 2:
            one_sided = float(wilcoxon(data, null, alternative="greater").pvalue)
            two_sided = float(paired_sign_flip_p(
                margin, n_perm=100000, seed=_seed(f"eeg-shared:{band_key}", seed)
            ))
        else:
            one_sided = np.nan
            two_sided = np.nan
        rows.append({
            "band": band_key,
            "band_label": BAND_LABEL[band_key],
            "n_subjects": len(frame),
            "n_seizures": int(frame["n_seizures"].sum()),
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
            "wilcoxon_one_sided_data_gt_null_p": one_sided,
            "two_sided_subject_sign_flip_p": two_sided,
        })
    return pd.DataFrame(rows)


def _plot(subject_rows: pd.DataFrame, cohort: pd.DataFrame) -> tuple[Path, Path]:
    groups = []
    stats = cohort.set_index("band")
    for band_key in ("broadband_1_150", "broadband_1_45_sensitivity"):
        frame = subject_rows[subject_rows.band == band_key].sort_values("subject")
        if frame.empty:
            continue
        p_value = float(stats.loc[band_key, "wilcoxon_one_sided_data_gt_null_p"])
        rows = [{
            "subject_id": row.subject,
            "data": float(row.shared_maxab),
            "null": float(row.channel_null_median_folded),
            "n_seizures": int(row.n_seizures),
        } for row in frame.itertuples()]
        groups.append({
            "label": BAND_LABEL[band_key],
            "rows": rows,
            "summary": {"n": len(rows)},
            "display_p": p_value,
            "p_label": "one-sided p",
        })
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    png = PAPER_FIGURES / "strict_reversed_eeg_onset_shared_field_data_vs_channel_null.png"
    pdf = png.with_suffix(".pdf")
    plot_paired_data_null_groups(
        groups, png, pdf, ylabel="Shared-field concordance |r|", seed=BASE_SEED,
    )
    return png, pdf


def _plot_nyquist_fallback(subject_rows: pd.DataFrame,
                           cohort: pd.DataFrame) -> tuple[Path, Path]:
    band_key = "broadband_subject_nyquist_fallback"
    frame = subject_rows[subject_rows.band == band_key].sort_values("subject")
    if frame.empty:
        raise ValueError("no subject-level Nyquist-fallback rows")
    stats = cohort.set_index("band").loc[band_key]
    rows = [{
        "subject_id": row.subject,
        "data": float(row.shared_maxab),
        "null": float(row.channel_null_median_folded),
        "n_seizures": int(row.n_seizures),
    } for row in frame.itertuples()]
    groups = [{
        "label": BAND_LABEL[band_key],
        "rows": rows,
        "summary": {"n": len(rows)},
        "display_p": float(stats["wilcoxon_one_sided_data_gt_null_p"]),
        "p_label": "one-sided p",
    }]
    png = (PAPER_FIGURES /
           "strict_reversed_eeg_onset_shared_field_nyquist_fallback_data_vs_channel_null.png")
    pdf = png.with_suffix(".pdf")
    plot_paired_data_null_groups(
        groups, png, pdf, ylabel="Shared-field concordance |r|", seed=BASE_SEED,
    )
    return png, pdf


def _write_readme(cohort: pd.DataFrame) -> None:
    readme = PAPER_FIGURES / "README.md"
    existing = readme.read_text() if readme.exists() else "# Fig3 supplement figures\n"
    marker = "### strict_reversed_eeg_onset_shared_field_data_vs_channel_null.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    summaries = []
    for row in cohort[cohort.band != "broadband_subject_nyquist_fallback"].itertuples():
        summaries.append(
            f"{row.band_label}: n={row.n_subjects} subjects/{row.n_seizures} seizures, "
            f"one-sided Wilcoxon p={row.wilcoxon_one_sided_data_gt_null_p:.4g}"
        )
    fallback = cohort[cohort.band == "broadband_subject_nyquist_fallback"].iloc[0]
    addition = f"""### strict_reversed_eeg_onset_shared_field_data_vs_channel_null.png / strict_reversed_eeg_onset_shared_field_data_vs_channel_null.pdf

只纳入预先存在的 `strict_2d + reversed` 患者，以真实 EEG onset 为 0 s，从原始 SEEG 重提 `[0,10] s` 早期能量。共线患者只使用冻结的 `shared_a/shared_b` 场，Data 为患者内 seizure 中位数的 `shared_maxAB`；Null 为每次 seizure 在接触点层做 all-contact channel shuffle、重新平滑、重新选择 mirror 与 shared A/B 后，再按 draw 折叠至患者的中位数。绘图严格复用 Fig3 的 Data–Null violin、box、subject 配对线、统计括号和显著性标注函数。

**关注点**：{'; '.join(summaries)}。精确 1–150 Hz 逐事件执行 Nyquist 合同，因此同一患者进入该频带的事件数可能少于 1–45 Hz；具体分母见 event inventory。1–45 Hz 只作为保留低采样率事件的敏感性，不冒充 1–150 Hz。

### strict_reversed_eeg_onset_shared_field_nyquist_fallback_data_vs_channel_null.png / strict_reversed_eeg_onset_shared_field_nyquist_fallback_data_vs_channel_null.pdf

与上一张图使用相同的真实 EEG onset `[0,10] s`、冻结 shared field、subject-first folding 和 all-contact channel-shuffle null，但为每名患者预先固定其全部可用事件共同支持的最高宽带上限。E1084、E1146、E590和E958使用1–150 Hz；E583因最低采样率为256 Hz，全部21次可用事件统一使用1–127 Hz，不在同一患者内混用频带。

**关注点**：n={int(fallback.n_subjects)} subjects/{int(fallback.n_seizures)} seizures，Data median={fallback.data_median:.4f}，Null median={fallback.null_median:.4f}，one-sided Wilcoxon p={fallback.wilcoxon_one_sided_data_gt_null_p:.4g}。这是显式标注的 Nyquist-limited sensitivity，不等同于所有患者均使用精确1–150 Hz。
"""
    readme.write_text(existing + addition)


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < N_PERM_MIN:
        raise ValueError(f"n_perm must be >= {N_PERM_MIN}")
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(OLD_CACHE.glob("*.npz"))
    source_hashes_before = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    inventory_rows, subject_rows, subject_audit = [], [], []

    candidates = []
    for field_path in sorted(FIELD_ROOT.glob("*.json")):
        field_record = json.loads(field_path.read_text())
        if is_strict_reversed(field_record):
            candidates.append((field_path.stem, field_path, field_record))

    for subject_number, (subject, field_path, field_record) in enumerate(candidates, 1):
        dataset, sid = subject.split("_", 1)
        meta_path = OLD_CACHE / f"{subject}.json"
        source_npz = OLD_CACHE / f"{subject}.npz"
        if not meta_path.exists() or not source_npz.exists():
            subject_audit.append({
                "subject": subject, "status": "drop", "reason": "missing_old_eligibility_cache",
                "axis_quality_tier": _field_quality(field_record).get("axis_quality_tier"),
                "axis_relation": _field_quality(field_record).get("axis_relation"),
                "scorers_used": "", "own_scorers_used": False,
            })
            continue
        meta = json.loads(meta_path.read_text())
        eligible = [int(value) for value in meta.get("eligible_idxs", [])]
        print(f"[{subject_number}/{len(candidates)}] {subject}: {len(eligible)} eligible events",
              flush=True)
        field_hash = _sha256(field_path)
        source_hash = _sha256(source_npz)
        quality = _field_quality(field_record)
        try:
            all_scorers = scorers_from_interictal_record(field_record)
            shared_scorers = select_shared_scorers(all_scorers)
        except Exception as exc:
            subject_audit.append({
                "subject": subject, "status": "drop",
                "reason": f"field_unavailable:{type(exc).__name__}:{exc}",
            })
            continue
        inv = _inventory_for_subject(dataset, sid)
        subject_fallback_upper_hz = _subject_common_broadband_upper_hz(
            dataset, sid, eligible, inv
        )
        print(f"  subject-common fallback band: 1-{subject_fallback_upper_hz:g} Hz",
              flush=True)
        events = []
        for event_number, seizure_idx in enumerate(eligible, 1):
            checkpoint = _event_checkpoint(subject, seizure_idx)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and _checkpoint_valid(
                checkpoint, field_hash=field_hash, source_hash=source_hash,
                subject_fallback_upper_hz=subject_fallback_upper_hz,
                n_perm=args.n_perm, seed=args.seed,
            ):
                event = json.loads(checkpoint.read_text())["event"]
                print(f"  seizure {seizure_idx} [{event_number}/{len(eligible)}] resume",
                      flush=True)
            else:
                print(f"  seizure {seizure_idx} [{event_number}/{len(eligible)}]",
                      flush=True)
                try:
                    if not (0 <= seizure_idx < len(inv)):
                        raise IndexError(f"seizure_idx_out_of_inventory:{len(inv)}")
                    event = _process_event(
                        subject, dataset, seizure_idx, inv[seizure_idx],
                        field_record, shared_scorers,
                        subject_fallback_upper_hz=subject_fallback_upper_hz,
                        n_perm=args.n_perm, seed=args.seed,
                    )
                except Exception as exc:
                    event = {
                        "dataset": dataset, "subject": subject,
                        "seizure_idx": seizure_idx, "status": "drop",
                        "drop_reason": f"{type(exc).__name__}:{exc}",
                    }
                    print(f"    DROP {event['drop_reason']}", flush=True)
                payload = {
                    "contract": CONTRACT, "field_sha256": field_hash,
                    "source_cache_sha256": source_hash, "n_perm": args.n_perm,
                    "seed": args.seed,
                    "subject_fallback_upper_hz": subject_fallback_upper_hz,
                    "event": event,
                }
                checkpoint.write_text(
                    json.dumps(jsonable(payload), ensure_ascii=False) + "\n"
                )
            if event.get("status") == "included":
                events.append(event)
                for band_key in BAND_HZ:
                    if band_key in event.get("bands", {}):
                        row = event["bands"][band_key]
                        inventory_rows.append({
                            "dataset": dataset, "subject": subject,
                            "seizure_idx": seizure_idx, "status": "included",
                            "drop_reason": "", "band": band_key,
                            "time_reference": event["time_reference"],
                            "window_start_sec": EEG_WINDOW[0],
                            "window_end_sec": EEG_WINDOW[1],
                            "eeg_onset_minus_clinical_sec": event[
                                "eeg_onset_minus_clinical_sec"
                            ],
                            "fs": event["fs"],
                            "band_lower_hz": row["band_hz"][0],
                            "band_upper_hz": row["band_hz"][1],
                            "n_target_contacts": row["n_target_contacts"],
                            "n_matched_contacts": row["n_matched_contacts"],
                            "n_finite_contacts": row["n_finite_contacts"],
                        })
                    else:
                        inventory_rows.append({
                            "dataset": dataset, "subject": subject,
                            "seizure_idx": seizure_idx, "status": "drop",
                            "drop_reason": "band_unavailable_nyquist",
                            "band": band_key, "time_reference": "genuine_eeg_onset",
                            "window_start_sec": EEG_WINDOW[0],
                            "window_end_sec": EEG_WINDOW[1], "fs": event["fs"],
                        })
            else:
                for band_key in BAND_HZ:
                    inventory_rows.append({
                        "dataset": dataset, "subject": subject,
                        "seizure_idx": seizure_idx, "status": "drop",
                        "drop_reason": event.get("drop_reason"), "band": band_key,
                        "time_reference": "genuine_eeg_onset",
                        "window_start_sec": EEG_WINDOW[0],
                        "window_end_sec": EEG_WINDOW[1],
                    })
        local_rows = _fold_subject(subject, dataset, quality, events, args.n_perm)
        subject_rows.extend(local_rows)
        subject_audit.append({
            "subject": subject, "status": "included" if local_rows else "drop",
            "reason": "" if local_rows else "no_resolvable_events",
            "n_contract_events": len(eligible), "n_events_any_band": len(events),
            "axis_quality_tier": quality.get("axis_quality_tier"),
            "axis_relation": quality.get("axis_relation"),
            "subject_fallback_upper_hz": subject_fallback_upper_hz,
            "scorers_used": "shared_a;shared_b",
            "own_scorers_used": False,
        })

    event_frame = pd.DataFrame(inventory_rows)
    subject_frame = pd.DataFrame(subject_rows)
    audit_frame = pd.DataFrame(subject_audit)
    if subject_frame.empty:
        raise RuntimeError("no strict-reversed shared-field subject result")
    cohort = _cohort_statistics(subject_frame, args.seed)

    event_path = OUT / "eeg_onset_shared_field_event_inventory.csv"
    subject_path = OUT / "eeg_onset_shared_field_subject.csv"
    cohort_path = OUT / "eeg_onset_shared_field_cohort.csv"
    audit_path = OUT / "eeg_onset_shared_field_subject_audit.csv"
    event_frame.to_csv(event_path, index=False)
    subject_frame.to_csv(subject_path, index=False)
    cohort.to_csv(cohort_path, index=False)
    audit_frame.to_csv(audit_path, index=False)

    source_hashes_after = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    unchanged = source_hashes_before == source_hashes_after
    if not unchanged:
        raise RuntimeError("historical source cache NPZ changed during analysis")
    png, pdf = _plot(subject_frame, cohort)
    fallback_png, fallback_pdf = _plot_nyquist_fallback(subject_frame, cohort)
    _write_readme(cohort)

    summary = {
        "contract": CONTRACT,
        "time_reference": "genuine_eeg_onset",
        "window_sec": list(EEG_WINDOW),
        "distal_baseline_sec": list(DISTAL_BASELINE_EEG_SEC),
        "subject_filter": "axis_quality_tier==strict_2d and axis_relation==reversed",
        "field_plane": "shared_only",
        "scorers_used": ["shared_a", "shared_b"],
        "own_scorers_used": False,
        "observed_statistic": "shared_maxab=max(shared_a_abs,shared_b_abs)",
        "null": {
            "mode": "all_contact_channel_shuffle",
            "n_draws": args.n_perm,
            "contact_level_recomputation": True,
            "mirror_reselected_each_draw": True,
            "shared_ab_max_reselected_each_draw": True,
            "same_permutation_reused_across_bands_within_event": True,
            "folding": "seizure_median_within_subject_for_each_draw",
        },
        "event_denominator": (
            "historical t0_feature_cache_bb150 eligible_idxs; activation and time zero "
            "re-extracted from raw SEEG"
        ),
        "bands": {key: {"hz": list(BAND_HZ[key]), "label": BAND_LABEL[key]}
                  for key in BAND_HZ},
        "subject_nyquist_fallback_contract": (
            "band upper bound is fixed per subject to the highest 1-s FFT bin "
            "supported by every contract event, capped at 150 Hz"
        ),
        "cohort_statistics": cohort.to_dict(orient="records"),
        "source_cache_npz_unchanged": unchanged,
        "outputs": {
            "event_inventory": str(event_path.relative_to(ROOT)),
            "subject": str(subject_path.relative_to(ROOT)),
            "cohort": str(cohort_path.relative_to(ROOT)),
            "subject_audit": str(audit_path.relative_to(ROOT)),
            "figure_png": str(png.relative_to(ROOT)),
            "figure_pdf": str(pdf.relative_to(ROOT)),
            "nyquist_fallback_figure_png": str(fallback_png.relative_to(ROOT)),
            "nyquist_fallback_figure_pdf": str(fallback_pdf.relative_to(ROOT)),
        },
    }
    summary_path = OUT / "eeg_onset_shared_field_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    (PAPER / "eeg_onset_shared_field_subject.csv").write_text(subject_path.read_text())
    (PAPER / "eeg_onset_shared_field_cohort.csv").write_text(cohort_path.read_text())
    (PAPER / "eeg_onset_shared_field_summary.json").write_text(summary_path.read_text())
    print(cohort.to_string(index=False), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
