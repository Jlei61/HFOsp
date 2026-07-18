#!/usr/bin/env python3
"""Run phenotype-gated, patient-specific T_spectral_best refinement."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_raw_spectral_context import _alias_index  # noqa: E402
from scripts.run_topic5_onset_energy_cohort import CACHE_ROOT  # noqa: E402
from scripts.run_topic5_spectral_onset_review import (  # noqa: E402
    _load_subject_events,
    _plot_review,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_spectral_onset import (  # noqa: E402
    SpectralOnsetConfig,
    TargetEpisodeAssignment,
    assign_target_episode,
    calibration_samples,
    detect_spectral_episodes,
    fit_spectral_calibration,
)
from src.topic5_subject_spectral_onset import (  # noqa: E402
    SeedSignature,
    SubjectOnsetConfig,
    config_to_dict,
    connected_episode_indices,
    extract_candidate,
    refine_event_onset,
)


ANALYSIS_VERSION = "topic5_tspectral_subject_v1p2"
SEED = 20260714
DEFAULT_MANIFEST = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/seed_v1p1/review_manifest.csv"
)
DEFAULT_OUT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2"
)
DEFAULT_SUBJECTS = (
    "epilepsiae_1084",
    "epilepsiae_1146",
    "epilepsiae_442",
    "epilepsiae_583",
    "epilepsiae_916",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _manifest_lookup(path: Path) -> dict[tuple[str, int], dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {(row["subject"], int(row["seizure_idx"])): row for row in rows}


def _event_key(subject: str, seizure_idx: int) -> str:
    return f"{subject}__seizure_{int(seizure_idx)}"


def _build_subject_contexts(
    subject: str,
    manifest: dict[tuple[str, int], dict],
    spectral_config: SpectralOnsetConfig,
) -> tuple[list[dict], list[str], list[SeedSignature]]:
    events, band_order = _load_subject_events(subject, spectral_config)
    samples = [calibration_samples(event["prepared"]) for event in events]
    contexts: list[dict] = []
    for event_index, event in enumerate(events):
        other = [sample for index, sample in enumerate(samples) if index != event_index]
        if not other:
            other = [samples[event_index]]
        calibration = fit_spectral_calibration(other, config=spectral_config)
        clinical = float(event["clinical_rel_eeg_sec"])
        search = (
            spectral_config.baseline[1] + spectral_config.max_gap_sec,
            min(
                float(event["prepared"].rel_t[-1]),
                max(0.0, clinical) + spectral_config.assignment_post_sec,
            ),
        )
        diagnostics = detect_spectral_episodes(
            event["prepared"],
            calibration,
            search=search,
            config=spectral_config,
            seed=SEED + int(event["seizure_idx"]),
        )
        assignment = assign_target_episode(
            diagnostics.episodes,
            eeg_onset_sec=0.0,
            clinical_onset_sec=clinical,
            config=spectral_config,
        )
        connected = connected_episode_indices(
            diagnostics.episodes,
            eeg_onset_sec=0.0,
            clinical_onset_sec=clinical,
            assignment_post_sec=spectral_config.assignment_post_sec,
        )
        old = manifest.get((subject, int(event["seizure_idx"])))
        if old is None:
            raise KeyError(f"missing v1.1 manifest row for {subject} seizure {event['seizure_idx']}")
        contexts.append(
            {
                "event": event,
                "diagnostics": diagnostics,
                "assignment": assignment,
                "connected": connected,
                "onset_episode": (
                    [int(assignment.target_index)]
                    if assignment.target_index is not None
                    else []
                ),
                "old": old,
            }
        )

    seeds: list[SeedSignature] = []
    for context in contexts:
        event = context["event"]
        old = context["old"]
        old_time = _float(old.get("auto_t_spectral_rel_eeg_sec"))
        if not _truth(old.get("auto_stable_candidate_time")) or not np.isfinite(old_time):
            continue
        diagnostics = context["diagnostics"]
        if not context["onset_episode"]:
            continue
        episode_index = int(context["onset_episode"][0])
        time_index = int(np.argmin(np.abs(event["prepared"].rel_t - old_time)))
        candidate = extract_candidate(
            event["prepared"],
            diagnostics.calibration,
            diagnostics.episodes[episode_index],
            episode_index=episode_index,
            time_index=time_index,
        )
        seeds.append(
            SeedSignature(
                event_key=_event_key(subject, event["seizure_idx"]),
                signature=candidate.signature,
                time_sec=candidate.time_sec,
                generic_score=candidate.generic_score,
            )
        )
    return contexts, band_order, seeds


def _row(context: dict, result, figure_path: Path | None) -> dict:
    event = context["event"]
    old = context["old"]
    diagnostics = context["diagnostics"]
    selected = None
    if result.has_candidate_time:
        selected = min(
            result.candidates,
            key=lambda candidate: (
                candidate.episode_index != result.episode_index,
                abs(candidate.time_sec - result.t_best_sec),
            ),
        )
    episode = (
        diagnostics.episodes[result.episode_index]
        if result.episode_index is not None
        else None
    )
    old_time = _float(old.get("auto_t_spectral_rel_eeg_sec"))
    t_candidate = result.t_candidate_sec
    t_best = result.t_best_sec
    clinical_available = bool(event.get("clinical_onset_available", True))
    eeg_rel_cache_zero = float(event["eeg_rel_clinical_sec"])
    return {
        "analysis_version": ANALYSIS_VERSION,
        "subject": event["subject"],
        "seizure_idx": int(event["seizure_idx"]),
        "seizure_id": event["seizure_id"],
        "annotation_mode": event.get("annotation_mode", "eeg_and_clinical"),
        "clinical_onset_available": clinical_available,
        "cache_zero_reference": event.get("cache_zero_reference", "clinical_onset"),
        "cache_tier": event.get("cache_tier", "primary_existing"),
        "eeg_onset_rel_cache_zero_sec": eeg_rel_cache_zero,
        "phenotype_status": result.phenotype_status,
        "timing_status": result.timing_status,
        "has_candidate_t": result.has_candidate_time,
        "has_accepted_t_best": result.has_accepted_time,
        "t_spectral_candidate_rel_eeg_sec": (
            t_candidate if result.has_candidate_time else ""
        ),
        "t_spectral_candidate_rel_cache_zero_sec": (
            t_candidate + eeg_rel_cache_zero if result.has_candidate_time else ""
        ),
        "t_spectral_candidate_rel_clinical_sec": (
            t_candidate + eeg_rel_cache_zero
            if result.has_candidate_time and clinical_available
            else ""
        ),
        "t_spectral_best_rel_eeg_sec": t_best if result.has_accepted_time else "",
        "t_spectral_best_rel_cache_zero_sec": (
            t_best + eeg_rel_cache_zero if result.has_accepted_time else ""
        ),
        "t_spectral_best_rel_clinical_sec": (
            t_best + eeg_rel_cache_zero
            if result.has_accepted_time and clinical_available
            else ""
        ),
        "selected_episode_start_rel_eeg_sec": episode.start_sec if episode else "",
        "selected_episode_end_rel_eeg_sec": episode.end_sec if episode else "",
        "selected_episode_index": result.episode_index if episode else "",
        "n_detected_episodes": len(diagnostics.episodes),
        "n_connected_episodes": len(context["connected"]),
        "n_candidates": result.n_candidates,
        "best_score": result.best_score if result.has_candidate_time else "",
        "second_score": result.second_score if np.isfinite(result.second_score) else "",
        "score_margin": result.score_margin if np.isfinite(result.score_margin) else "",
        "best_generic_score": selected.generic_score if selected else "",
        "best_prototype_similarity": (
            result.prototype_similarity if np.isfinite(result.prototype_similarity) else ""
        ),
        "prototype_available": result.prototype_available,
        "prototype_used": result.prototype_used,
        "prototype_n_training_events": result.n_training_events,
        "prototype_coherence": (
            result.prototype_coherence if np.isfinite(result.prototype_coherence) else ""
        ),
        "temporal_support_available": result.temporal_support_available,
        "temporal_support_radius_sec": (
            result.temporal_support_radius_sec
            if np.isfinite(result.temporal_support_radius_sec)
            else ""
        ),
        "temporal_n_supporting_events": result.temporal_n_supporting_events,
        "bootstrap_q05_rel_eeg_sec": (
            result.bootstrap_q05_sec if np.isfinite(result.bootstrap_q05_sec) else ""
        ),
        "bootstrap_q95_rel_eeg_sec": (
            result.bootstrap_q95_sec if np.isfinite(result.bootstrap_q95_sec) else ""
        ),
        "bootstrap_width_sec": (
            result.bootstrap_width_sec if np.isfinite(result.bootstrap_width_sec) else ""
        ),
        "selection_consistency_1s": (
            result.selection_consistency_1s
            if np.isfinite(result.selection_consistency_1s)
            else ""
        ),
        "v1p1_status": old["auto_status"],
        "v1p1_t_rel_eeg_sec": old.get("auto_t_spectral_rel_eeg_sec", ""),
        "delta_t_best_minus_v1p1_sec": (
            t_best - old_time
            if result.has_accepted_time and np.isfinite(old_time)
            else ""
        ),
        "delta_t_candidate_minus_v1p1_sec": (
            t_candidate - old_time
            if result.has_candidate_time and np.isfinite(old_time)
            else ""
        ),
        "candidate_times_rel_eeg_sec_json": json.dumps(
            [candidate.time_sec for candidate in result.candidates]
        ),
        "candidate_final_scores_json": json.dumps(
            [candidate.final_score for candidate in result.candidates]
        ),
        "candidate_prototype_similarity_json": json.dumps(
            [
                candidate.prototype_similarity
                if np.isfinite(candidate.prototype_similarity)
                else None
                for candidate in result.candidates
            ]
        ),
        "figure": str(figure_path.relative_to(ROOT)) if figure_path else "",
        "manual_accept_t_best": "",
        "manual_t_best_rel_eeg_sec": "",
        "manual_notes": "",
    }


def _render(context: dict, result, band_order: list[str], out_path: Path) -> None:
    event = context["event"]
    diagnostics = context["diagnostics"]
    base = context["assignment"]
    target_index = result.episode_index if result.has_candidate_time else base.target_index
    assignment = TargetEpisodeAssignment(
        status=result.phenotype_status,
        target_index=target_index,
        anchor_start_sec=base.anchor_start_sec,
        anchor_end_sec=base.anchor_end_sec,
        n_connected_episodes=base.n_connected_episodes,
        n_prior_episodes=base.n_prior_episodes,
    )
    clinical = float(event["clinical_rel_eeg_sec"])
    x0_eeg = max(float(diagnostics.rel_t[0]), -120.0)
    x1_eeg = min(float(diagnostics.rel_t[-1]), max(0.0, clinical) + 20.0)
    x0_clin = x0_eeg + float(event["eeg_rel_clinical_sec"])
    x1_clin = x1_eeg + float(event["eeg_rel_clinical_sec"])
    raw_sw = extract_seizure_window(
        f"{event['dataset']}/{event['sid']}",
        int(event["seizure_idx"]),
        pre_sec=max(10.0, -x0_clin),
        post_sec=max(10.0, x1_clin),
        results_root=ROOT / "results",
        reference=ICTAL_REFERENCE[event["dataset"]],
    )
    raw_lookup = _alias_index(raw_sw.ch_names)
    absent = [name for name in event["timing_channels"] if name not in raw_lookup]
    if absent:
        raise ValueError(f"raw timing contacts missing: {absent}")
    raw_idx = [int(raw_lookup[name]) for name in event["timing_channels"]]
    _plot_review(
        event,
        diagnostics,
        assignment,
        raw_sw,
        raw_idx,
        band_order,
        review_id=_event_key(event["subject"], event["seizure_idx"]),
        blind=False,
        out_path=out_path,
        refined_time_sec=result.t_candidate_sec if result.has_candidate_time else None,
        refined_q05_sec=(
            result.bootstrap_q05_sec if np.isfinite(result.bootstrap_q05_sec) else None
        ),
        refined_q95_sec=(
            result.bootstrap_q95_sec if np.isfinite(result.bootstrap_q95_sec) else None
        ),
        refined_status=result.timing_status,
        refined_label=(
            "patient-specific T_best"
            if result.has_accepted_time
            else "T_candidate (manual-only)"
        ),
    )


def _subject_summary(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for subject in sorted({row["subject"] for row in rows}):
        use = [row for row in rows if row["subject"] == subject]
        counts = Counter(row["phenotype_status"] for row in use)
        timing_counts = Counter(row["timing_status"] for row in use)
        candidate_times = np.asarray(
            [_float(row["t_spectral_candidate_rel_eeg_sec"]) for row in use],
            dtype=float,
        )
        candidate_times = candidate_times[np.isfinite(candidate_times)]
        times = np.asarray(
            [_float(row["t_spectral_best_rel_eeg_sec"]) for row in use], dtype=float
        )
        times = times[np.isfinite(times)]
        shifts = np.asarray(
            [_float(row["delta_t_best_minus_v1p1_sec"]) for row in use], dtype=float
        )
        shifts = shifts[np.isfinite(shifts)]
        widths = np.asarray([_float(row["bootstrap_width_sec"]) for row in use], dtype=float)
        widths = widths[np.isfinite(widths)]
        coherence = np.asarray(
            [
                _float(row["prototype_coherence"])
                for row in use
                if _truth(row["has_candidate_t"])
            ],
            dtype=float,
        )
        coherence = coherence[np.isfinite(coherence)]
        out.append(
            {
                "subject": subject,
                "n_seizures": len(use),
                "n_phenotype_present": counts["phenotype_present"],
                "n_phenotype_absent": counts["phenotype_absent"],
                "n_prior_candidate_manual_only": counts["prior_candidate_manual_only"],
                "fraction_phenotype_present": counts["phenotype_present"] / len(use),
                "n_candidate_t": int(candidate_times.size),
                "n_accepted_t_best": int(times.size),
                "n_candidate_no_subject_timing_template": timing_counts[
                    "candidate_no_subject_timing_template"
                ],
                "n_candidate_temporally_unanchored": timing_counts[
                    "candidate_temporally_unanchored"
                ],
                "n_accepted_subject_recurrent": timing_counts[
                    "accepted_subject_recurrent"
                ],
                # A LOSO prototype can be available even when the target event has
                # no eligible broadband episode.  Count prototype use only among
                # events that actually receive an accepted T_best; otherwise this
                # field misleadingly makes phenotype-absent events look timed.
                "n_accepted_t_best_with_prototype": sum(
                    _truth(row["prototype_used"])
                    and _truth(row["has_accepted_t_best"])
                    for row in use
                ),
                "prototype_coherence_median": (
                    float(np.median(coherence)) if coherence.size else float("nan")
                ),
                "candidate_t_q25_sec": (
                    float(np.quantile(candidate_times, 0.25))
                    if candidate_times.size
                    else float("nan")
                ),
                "candidate_t_median_sec": (
                    float(np.median(candidate_times))
                    if candidate_times.size
                    else float("nan")
                ),
                "candidate_t_q75_sec": (
                    float(np.quantile(candidate_times, 0.75))
                    if candidate_times.size
                    else float("nan")
                ),
                "t_best_q25_sec": (
                    float(np.quantile(times, 0.25)) if times.size else float("nan")
                ),
                "t_best_median_sec": float(np.median(times)) if times.size else float("nan"),
                "t_best_q75_sec": (
                    float(np.quantile(times, 0.75)) if times.size else float("nan")
                ),
                "median_abs_shift_from_v1p1_sec": (
                    float(np.median(np.abs(shifts))) if shifts.size else float("nan")
                ),
                "bootstrap_width_median_sec": (
                    float(np.median(widths)) if widths.size else float("nan")
                ),
            }
        )
    return out


def _write_figure_readmes(out_root: Path, subjects: list[str]) -> None:
    fig_root = out_root / "figures/per_seizure"
    fig_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures/README.md").write_text(
        "# Subject-specific T_spectral_best figures\n\n"
        "### per_seizure/\n\n每次 seizure 的原始波形、五频带能量、空间支持和变化门诊断。蓝线为患者内 LOSO 精修点；患者内时间模式复现时标为 T_best，否则明确标为 manual-only candidate。没有宽带 phenotype 的事件不画蓝线。\n\n"
        "**关注点**：检查蓝线是否落在真实的宽带上升沿，而不是后续更大的发作峰；同时确认 phenotype-negative 和时间上孤立的事件没有被强行升级为 onset。Yuquan 只有 EEG onset，不显示伪造的 clinical marker。\n",
        encoding="utf-8",
    )
    for subject in subjects:
        subject_dir = fig_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / "README.md").write_text(
            f"# {subject} T_spectral_best review\n\n"
            f"### {subject}_seizure_*_subject_tbest.png\n\n该患者逐 seizure 的患者内原型精修图；绿色为 v1.1 episode/change，蓝色为 v1.2 候选点与 90% 重采样区间。标题区分患者内复现的 accepted T_best 与时间上孤立的 manual-only candidate。\n\n"
            "**关注点**：比较患者内蓝线是否对应同一种多频带、空间扩展的起始形态，并检查孤立晚期状态是否被保守降级。\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Path:
    manifest = _manifest_lookup(args.manifest)
    spectral_config = SpectralOnsetConfig(n_boot=0)
    refinement_config = SubjectOnsetConfig(n_boot=int(args.n_boot))
    rows: list[dict] = []
    errors: list[dict] = []
    for subject in args.subjects:
        contexts, band_order, seeds = _build_subject_contexts(
            subject, manifest, spectral_config
        )
        for context in contexts:
            event = context["event"]
            try:
                result = refine_event_onset(
                    _event_key(subject, event["seizure_idx"]),
                    event["prepared"],
                    context["diagnostics"],
                    connected_indices=context["onset_episode"],
                    training_seeds=seeds,
                    config=refinement_config,
                    seed=SEED + int(event["seizure_idx"]),
                )
                figure_path = (
                    args.out_root
                    / "figures/per_seizure"
                    / subject
                    / f"{subject}_seizure_{int(event['seizure_idx']):02d}_subject_tbest.png"
                )
                if args.render and (args.force or not figure_path.exists()):
                    _render(context, result, band_order, figure_path)
                rows.append(_row(context, result, figure_path if args.render else None))
                print(
                    f"[subject-tbest] {subject} seizure {event['seizure_idx']} "
                    f"{result.timing_status} "
                    f"candidate={result.t_candidate_sec if result.has_candidate_time else 'NA'} "
                    f"accepted={result.t_best_sec if result.has_accepted_time else 'NA'}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - event-level provenance
                errors.append(
                    {
                        "subject": subject,
                        "seizure_idx": int(event["seizure_idx"]),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                print(f"[ERROR] {errors[-1]}", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "per_seizure_subject_refined_onset.csv"
    _write_csv(output, rows)
    _write_csv(args.out_root / "subject_summary.csv", _subject_summary(rows))
    (args.out_root / "contract.json").write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "status": "algorithmic_v1p2_manual_review_pending",
                "spectral_episode_config": {
                    **spectral_config.__dict__,
                    "note": "episode bootstrap disabled here; stable seed eligibility is read from frozen v1.1 manifest",
                },
                "subject_refinement_config": config_to_dict(refinement_config),
                "subjects": list(args.subjects),
                "n_rows": len(rows),
                "n_errors": len(errors),
                "seed_source": str(args.manifest),
                "rendered": bool(args.render),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_root / "processing_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.render:
        _write_figure_readmes(args.out_root, list(args.subjects))
    if errors:
        raise RuntimeError(f"{len(errors)} event(s) failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--all-epilepsiae", action="store_true")
    parser.add_argument("--all-yuquan-cache", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.out_root = args.out_root.resolve()
    if args.all_epilepsiae:
        args.subjects = sorted(path.stem for path in CACHE_ROOT.glob("epilepsiae_*.json"))
    if args.all_yuquan_cache:
        args.subjects = sorted(path.stem for path in CACHE_ROOT.glob("yuquan_*.json"))
    print(run(args))


if __name__ == "__main__":
    main()
