"""Tests for PR6 interictal synchrony analysis module."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from src.interictal_synchrony_analysis import (
    assign_fixed_window_positions,
    build_cohort_summary,
    compute_normalized_trajectory,
    load_event_rows,
    paired_window_test,
    plot_figure_a_subject_timeline,
    plot_figure_b_trajectory_ribbon,
    plot_figure_c_fixed_window_comparison,
    plot_figure_e_coverage_audit,
    run_pr6_analysis,
    trajectory_trend_test,
    within_interval_trend_test,
)


def _make_block(
    subject: str,
    stem: str,
    start: float,
    end: float,
    *,
    interval_id: str = "",
    prev_offset: float | None = None,
    next_onset: float | None = None,
    sync_legacy: float = 0.5,
    sync_phase: float = 0.4,
    sync_span: float = 0.8,
    assigned: bool = True,
    dataset: str = "test",
) -> Dict[str, object]:
    clean_sec = (
        (next_onset - prev_offset) if prev_offset is not None and next_onset is not None else None
    )
    return {
        "subject": subject,
        "patient_code": subject,
        "block_stem": stem,
        "recording_id": stem,
        "block_start_epoch": start,
        "block_end_epoch": end,
        "block_duration_sec": end - start,
        "block_start_day_night": "day",
        "block_end_day_night": "day",
        "timezone_name": "Europe/Berlin",
        "day_night_rule": "day=08:00-20:00 local",
        "interval_assignment_status": "assigned" if assigned else "outside_complete_intervals",
        "overlaps_complete_eeg_seizure": False,
        "phase_eligible": assigned,
        "diurnal_eligible": True,
        "seizure_interval_id": interval_id,
        "prev_seizure_id": f"{subject}_sz_prev" if interval_id else "",
        "next_seizure_id": f"{subject}_sz_next" if interval_id else "",
        "prev_eeg_onset_epoch": (prev_offset - 100.0) if prev_offset is not None else None,
        "prev_eeg_offset_epoch": prev_offset,
        "next_eeg_onset_epoch": next_onset,
        "next_eeg_offset_epoch": (next_onset + 100.0) if next_onset is not None else None,
        "clean_between_seizures_sec": clean_sec,
        "exclusion_reasons": "" if assigned else "outside_complete_intervals",
        "n_channels": 6.0,
        "n_core_channels": 6.0,
        "n_penumbra_channels": 0.0,
        "sync_phase_global": sync_phase,
        "sync_phase_core": sync_phase,
        "sync_phase_penumbra": None,
        "sync_legacy_global": sync_legacy,
        "sync_legacy_core": sync_legacy,
        "sync_legacy_penumbra": None,
        "sync_span_global": sync_span,
        "sync_span_core": sync_span,
        "sync_span_penumbra": None,
        "event_index": float(start),
        "event_start_epoch": start,
        "event_end_epoch": end,
        "event_center_epoch": 0.5 * (start + end),
        "event_duration_sec": end - start,
        "continuous_segment_id": 0,
        "n_participating": 6.0,
        "n_core": 6.0,
        "n_penumbra": 0.0,
        "event_stratum": "core_only",
        "dataset": dataset,
    }


def _make_5h_interval_blocks(subject: str = "A", interval_id: str = "A_szint_001") -> List[Dict]:
    """5h clean interval with 5 hourly blocks, sync increasing from 0.3 to 0.7."""
    clean_start = 1000.0
    clean_end = 19000.0  # 18000s = 5h
    blocks = []
    for i in range(5):
        bstart = clean_start + i * 3600
        bend = bstart + 3600
        sync = 0.3 + i * 0.1
        blocks.append(
            _make_block(
                subject, f"{subject}_b{i:02d}", bstart, bend,
                interval_id=interval_id,
                prev_offset=clean_start,
                next_onset=clean_end,
                sync_legacy=sync,
                sync_phase=sync * 0.8,
                sync_span=0.7 + i * 0.05,
            )
        )
    return blocks


def _make_short_interval_blocks(
    subject: str = "B", interval_id: str = "B_szint_001"
) -> List[Dict]:
    """2h clean interval (too short for fixed windows), 2 blocks."""
    clean_start = 500.0
    clean_end = 7700.0  # 7200s = 2h
    return [
        _make_block(
            subject, f"{subject}_b00", clean_start, clean_start + 3600,
            interval_id=interval_id,
            prev_offset=clean_start,
            next_onset=clean_end,
            sync_legacy=0.4,
        ),
        _make_block(
            subject, f"{subject}_b01", clean_start + 3600, clean_end,
            interval_id=interval_id,
            prev_offset=clean_start,
            next_onset=clean_end,
            sync_legacy=0.6,
        ),
    ]


def _make_25h_interval_blocks(subject: str = "L", interval_id: str = "L_szint_001") -> List[Dict]:
    clean_start = 1000.0
    clean_end = clean_start + 25 * 3600
    blocks = []
    for i in range(25):
        start = clean_start + i * 3600
        blocks.append(
            _make_block(
                subject,
                f"{subject}_b{i:02d}",
                start,
                start + 3600,
                interval_id=interval_id,
                prev_offset=clean_start,
                next_onset=clean_end,
                sync_legacy=0.45 + 0.01 * (i % 5),
            )
        )
    return blocks


# ── fixed-window assignment ────────────────────────────────────────────────


def test_assign_fixed_window_positions_5h_interval() -> None:
    blocks = _make_5h_interval_blocks()
    result = assign_fixed_window_positions(blocks)
    positions = {r["block_stem"]: r["window_position"] for r in result}

    assert positions["A_b00"] == "post"
    assert positions["A_b04"] == "pre"
    assert positions.get("A_b02") == "mid"


def test_assign_fixed_window_rejects_short_interval() -> None:
    blocks = _make_short_interval_blocks()
    result = assign_fixed_window_positions(blocks, min_interval_sec=10800.0)
    assert len(result) == 0


def test_assign_fixed_window_custom_thresholds() -> None:
    blocks = _make_short_interval_blocks()
    result = assign_fixed_window_positions(blocks, min_interval_sec=3600.0, window_sec=1800.0)
    assert len(result) > 0


def test_assign_fixed_window_skips_unassigned() -> None:
    blocks = _make_5h_interval_blocks()
    blocks.append(
        _make_block(
            "A", "A_outside", 50000.0, 53600.0, assigned=False, dataset="test"
        )
    )
    result = assign_fixed_window_positions(blocks)
    stems = {r["block_stem"] for r in result}
    assert "A_outside" not in stems


# ── normalized trajectory ─────────────────────────────────────────────────


def test_compute_normalized_trajectory_5h() -> None:
    blocks = _make_5h_interval_blocks()
    result = compute_normalized_trajectory(blocks)
    assert len(result) == 5

    result.sort(key=lambda r: r["norm_t"])
    assert result[0]["norm_t"] < 0.2
    assert result[-1]["norm_t"] > 0.8
    for r in result:
        assert 0.0 <= r["norm_t"] <= 1.0


def test_compute_normalized_trajectory_skips_unassigned() -> None:
    blocks = [
        _make_block("X", "x0", 100.0, 200.0, assigned=False),
    ]
    assert len(compute_normalized_trajectory(blocks)) == 0


# ── statistics ─────────────────────────────────────────────────────────────


def test_paired_window_test_increasing_sync() -> None:
    all_blocks = []
    for i in range(8):
        subj = f"S{i}"
        iv_id = f"{subj}_szint_001"
        clean_start = 1000.0
        clean_end = 19000.0
        post_sync = 0.30 + np.random.default_rng(i).uniform(-0.02, 0.02)
        pre_sync = 0.65 + np.random.default_rng(i + 100).uniform(-0.02, 0.02)
        mid_sync = (post_sync + pre_sync) / 2.0

        all_blocks.append(
            _make_block(
                subj, f"{subj}_b0", clean_start, clean_start + 3600,
                interval_id=iv_id, prev_offset=clean_start, next_onset=clean_end,
                sync_legacy=post_sync,
            )
        )
        all_blocks.append(
            _make_block(
                subj, f"{subj}_b2", clean_start + 7200, clean_start + 10800,
                interval_id=iv_id, prev_offset=clean_start, next_onset=clean_end,
                sync_legacy=mid_sync,
            )
        )
        all_blocks.append(
            _make_block(
                subj, f"{subj}_b4", clean_end - 3600, clean_end,
                interval_id=iv_id, prev_offset=clean_start, next_onset=clean_end,
                sync_legacy=pre_sync,
            )
        )

    fixed = assign_fixed_window_positions(all_blocks)
    assert len(fixed) > 0

    stat = paired_window_test(fixed, metric_col="sync_legacy_global", pair=("post", "pre"))
    assert stat["n_pairs"] == 8
    assert stat["p_value"] is not None
    assert stat["p_value"] < 0.05
    assert stat["median_diff"] > 0


def test_paired_window_test_insufficient_pairs() -> None:
    blocks = _make_5h_interval_blocks()
    fixed = assign_fixed_window_positions(blocks)
    stat = paired_window_test(fixed, pair=("post", "pre"))
    assert stat["n_pairs"] <= 1


def test_trajectory_trend_test_positive_trend() -> None:
    blocks = _make_5h_interval_blocks()
    traj = compute_normalized_trajectory(blocks)
    trend = trajectory_trend_test(traj, metric_col="sync_legacy_global")
    assert trend["n_events"] == 5
    assert trend["spearman_r"] is not None
    assert trend["spearman_r"] > 0.8
    assert len(trend["bins"]) == 10


def test_trajectory_trend_test_short_input() -> None:
    blocks = [
        _make_block(
            "Z", "z0", 100.0, 200.0,
            interval_id="Z_1", prev_offset=100.0, next_onset=200.0,
            sync_legacy=0.5,
        )
    ]
    traj = compute_normalized_trajectory(blocks)
    trend = trajectory_trend_test(traj)
    assert trend["spearman_r"] is None


def test_within_interval_trend_test_positive_trend() -> None:
    """Multiple intervals with consistently positive trends → significant."""
    all_blocks: list = []
    for i in range(8):
        subj = f"S{i}"
        iv_id = f"{subj}_szint_001"
        cs, ce = 1000.0, 37000.0  # 10h
        for j in range(10):
            bstart = cs + j * 3600
            sync = 0.3 + j * 0.04 + np.random.default_rng(i * 10 + j).uniform(-0.005, 0.005)
            all_blocks.append(
                _make_block(
                    subj, f"{subj}_b{j:02d}", bstart, bstart + 3600,
                    interval_id=iv_id, prev_offset=cs, next_onset=ce,
                    sync_legacy=sync,
                )
            )
    traj = compute_normalized_trajectory(all_blocks)
    result = within_interval_trend_test(traj, metric_col="sync_legacy_global")
    assert result["n_intervals_tested"] == 8
    assert result["median_rho"] is not None
    assert result["median_rho"] > 0.8
    assert result["n_positive"] == 8
    assert result["wilcoxon_p"] is not None
    assert result["wilcoxon_p"] < 0.05


def test_within_interval_trend_test_mixed_trends() -> None:
    """Intervals with opposing trends → non-significant."""
    all_blocks: list = []
    for i in range(6):
        subj = f"S{i}"
        iv_id = f"{subj}_szint_001"
        cs, ce = 1000.0, 19000.0
        direction = 1.0 if i % 2 == 0 else -1.0
        for j in range(5):
            bstart = cs + j * 3600
            sync = 0.5 + direction * j * 0.05
            all_blocks.append(
                _make_block(
                    subj, f"{subj}_b{j:02d}", bstart, bstart + 3600,
                    interval_id=iv_id, prev_offset=cs, next_onset=ce,
                    sync_legacy=sync,
                )
            )
    traj = compute_normalized_trajectory(all_blocks)
    result = within_interval_trend_test(traj, metric_col="sync_legacy_global")
    assert result["n_intervals_tested"] == 6
    assert result["n_positive"] == 3
    assert result["n_negative"] == 3


def test_within_interval_trend_test_insufficient_intervals() -> None:
    blocks = _make_5h_interval_blocks()
    traj = compute_normalized_trajectory(blocks)
    result = within_interval_trend_test(traj, metric_col="sync_legacy_global")
    assert result["n_intervals_tested"] == 1
    assert result["median_rho"] is None


# ── cohort summary ─────────────────────────────────────────────────────────


def test_build_cohort_summary() -> None:
    blocks = _make_5h_interval_blocks() + _make_short_interval_blocks()
    fixed = assign_fixed_window_positions(blocks)
    traj = compute_normalized_trajectory(blocks)
    summary = build_cohort_summary(blocks, fixed, traj)

    assert summary["n_events_total"] == 7
    assert summary["n_subjects_total"] == 2
    assert summary["n_trajectory_events"] == 7
    assert summary["n_intervals_fixed_window"] >= 1


# ── figures ────────────────────────────────────────────────────────────────


def test_plot_figure_a_returns_figure() -> None:
    blocks = _make_5h_interval_blocks()
    fig = plot_figure_a_subject_timeline(blocks, "A")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_figure_a_excludes_seizure_overlap_blocks() -> None:
    blocks = _make_5h_interval_blocks()
    overlap_block = _make_block(
        "A",
        "A_overlap",
        20000.0,
        23600.0,
        interval_id="A_szint_002",
        prev_offset=19000.0,
        next_onset=26000.0,
        sync_legacy=0.9,
    )
    overlap_block["overlaps_complete_eeg_seizure"] = True
    overlap_block["interval_assignment_status"] = "overlaps_seizure"
    blocks.append(overlap_block)

    fig = plot_figure_a_subject_timeline(blocks, "A")
    ax = fig.axes[0]
    scatter = ax.collections[0]
    offsets = scatter.get_offsets()
    assert len(offsets) == 5
    assert ax.get_xlabel() == "Hours from first clean assigned event"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_figure_a_facets_long_subject_into_10h_panels() -> None:
    blocks = _make_25h_interval_blocks()
    fig = plot_figure_a_subject_timeline(blocks, "L")
    assert len(fig.axes) == 3
    assert fig.axes[-1].get_xlabel() == "Hours from first clean assigned event"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_figure_a_uses_event_level_onset_windows_not_block_stem_fill() -> None:
    blocks = [
        _make_block(
            "A", "shared_block", 1200.0, 1800.0,
            interval_id="A_szint_001", prev_offset=1000.0, next_onset=10000.0,
        ),
        _make_block(
            "A", "shared_block", 5200.0, 5800.0,
            interval_id="A_szint_001", prev_offset=1000.0, next_onset=10000.0,
        ),
        _make_block(
            "A", "shared_block", 9200.0, 9800.0,
            interval_id="A_szint_001", prev_offset=1000.0, next_onset=10000.0,
        ),
    ]
    blocks[0]["event_index"] = 0.0
    blocks[1]["event_index"] = 1.0
    blocks[2]["event_index"] = 2.0

    fig = plot_figure_a_subject_timeline(blocks, "A")
    ax = fig.axes[0]
    scatter = ax.collections[0]
    facecolors = scatter.get_facecolors()
    assert np.allclose(facecolors[0], to_rgba("#4C72B0"))
    assert np.allclose(facecolors[1], to_rgba("#7f7f7f"))
    assert np.allclose(facecolors[2], to_rgba("#C44E52"))

    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert "Mid-window event" not in labels
    assert any("after onset" in label for label in labels)
    assert any("before onset" in label for label in labels)

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_figure_b_returns_figure() -> None:
    blocks = _make_5h_interval_blocks()
    traj = compute_normalized_trajectory(blocks)
    fig = plot_figure_b_trajectory_ribbon(traj)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_figure_c_returns_figure() -> None:
    all_blocks = []
    for i in range(4):
        subj = f"S{i}"
        iv_id = f"{subj}_szint_001"
        cs, ce = 1000.0, 19000.0
        for j, (bstart_off, sync) in enumerate([
            (0, 0.3 + i * 0.01), (7200, 0.5), (14400, 0.7 - i * 0.01)
        ]):
            all_blocks.append(
                _make_block(
                    subj, f"{subj}_b{j}", cs + bstart_off, cs + bstart_off + 3600,
                    interval_id=iv_id, prev_offset=cs, next_onset=ce,
                    sync_legacy=sync,
                )
            )
    fixed = assign_fixed_window_positions(all_blocks)
    fig = plot_figure_c_fixed_window_comparison(fixed)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_figure_e_returns_figure() -> None:
    blocks = _make_5h_interval_blocks()
    fixed = assign_fixed_window_positions(blocks)
    traj = compute_normalized_trajectory(blocks)
    fig = plot_figure_e_coverage_audit(blocks, fixed, traj)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── CSV round-trip ─────────────────────────────────────────────────────────


def test_load_event_rows_parses_types(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_annotations.csv"
    fieldnames = [
        "subject", "block_stem", "block_start_epoch", "block_end_epoch",
        "phase_eligible", "sync_legacy_global", "interval_assignment_status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "subject": "1073",
            "block_stem": "blk_0001",
            "block_start_epoch": "123.456",
            "block_end_epoch": "456.789",
            "phase_eligible": "True",
            "sync_legacy_global": "0.55",
            "interval_assignment_status": "assigned",
        })
        writer.writerow({
            "subject": "1073",
            "block_stem": "blk_0002",
            "block_start_epoch": "",
            "block_end_epoch": "nan",
            "phase_eligible": "False",
            "sync_legacy_global": "NaN",
            "interval_assignment_status": "outside_complete_intervals",
        })

    rows = load_event_rows(str(csv_path), dataset="epi")
    assert rows[0]["block_start_epoch"] == 123.456
    assert rows[0]["phase_eligible"] is True
    assert rows[0]["sync_legacy_global"] == 0.55
    assert rows[0]["dataset"] == "epi"
    assert rows[1]["block_start_epoch"] is None
    assert rows[1]["phase_eligible"] is False
    assert rows[1]["sync_legacy_global"] is None


# ── full pipeline ──────────────────────────────────────────────────────────


def test_run_pr6_analysis_end_to_end(tmp_path: Path) -> None:
    all_blocks = []
    for i in range(5):
        subj = f"S{i}"
        iv_id = f"{subj}_szint_001"
        cs, ce = 1000.0, 19000.0
        for j, (bstart_off, sync) in enumerate([
            (0, 0.3 + i * 0.02), (7200, 0.5), (14400, 0.7 - i * 0.02)
        ]):
            all_blocks.append(
                _make_block(
                    subj, f"{subj}_b{j}", cs + bstart_off, cs + bstart_off + 3600,
                    interval_id=iv_id, prev_offset=cs, next_onset=ce,
                    sync_legacy=sync, dataset="epilepsiae",
                )
            )

    csv_path = tmp_path / "annotations.csv"
    fieldnames = list(all_blocks[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_blocks)

    output_dir = tmp_path / "pr6_out"
    summary = run_pr6_analysis(
        epilepsiae_events_csv=str(csv_path),
        output_dir=str(output_dir),
    )

    assert summary["n_fixed_window_events"] > 0
    assert summary["n_trajectory_events"] == 15
    assert (output_dir / "pr6_fixed_window_events.csv").exists()
    assert (output_dir / "pr6_trajectory_events.csv").exists()
    assert (output_dir / "pr6_fixed_window_interval_means.csv").exists()
    assert (output_dir / "pr6_trajectory_interval_stats.csv").exists()
    assert (output_dir / "pr6_statistics_summary.json").exists()
    assert len(summary["figures"]) >= 4

    stats = json.loads(
        (output_dir / "pr6_statistics_summary.json").read_text(encoding="utf-8")
    )
    assert stats["analysis_contract"]["formal_statistical_unit"] == "seizure_interval"
    assert "post_vs_pre_legacy_all" in stats
    assert "trajectory_pooled_legacy_all" in stats
    assert "trajectory_within_interval_legacy_all" in stats
    assert stats["trajectory_pooled_legacy_all"]["is_exploratory"] is True
    assert stats["trajectory_pooled_legacy_all"]["spearman_r"] is not None
    assert stats["trajectory_within_interval_legacy_all"]["n_intervals_tested"] >= 1
