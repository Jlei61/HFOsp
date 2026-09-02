#!/usr/bin/env python3
"""Aggregate v0.3 pilot outputs into machine, plain-language and technical reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.subject import load_subject_timeline  # noqa: E402
from src.topic5_group_event_state.v03.pilot import nested_partition  # noqa: E402
from src.topic5_group_event_state.v03.partition import PHASE_NAMES  # noqa: E402


HORIZONS = ("300s", "1800s", "7200s")


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _med(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(median(values)) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _poisson_nll(count: np.ndarray, mean: np.ndarray) -> np.ndarray:
    y = np.asarray(count, dtype=np.float64)
    mu = np.clip(np.asarray(mean, dtype=np.float64), 1e-8, None)
    lgamma = np.vectorize(math.lgamma, otypes=[float])(y + 1.0)
    return mu - y * np.log(mu) + lgamma


def _paired_count_summary(
    horizon_rows: list[dict[str, Any]], a: str, b: str
) -> dict[str, Any]:
    """Keep every finite development-test pair; audits may flag, never delete.

    v0.3.1 originally used a post-hoc fitted-intercept threshold to decide
    whether a seed entered the reported contrast.  Because that threshold is
    computed from development-test performance, it changes the denominator as
    a function of the observed result.  The closeout preserves all finite raw
    pairs and keeps the old rule only as an explicitly deprecated diagnostic.
    """

    finite_rows = [
        row
        for row in horizon_rows
        if row["count_poisson_nll"].get(a) is not None
        and row["count_poisson_nll"].get(b) is not None
        and math.isfinite(float(row["count_poisson_nll"][a]))
        and math.isfinite(float(row["count_poisson_nll"][b]))
    ]
    def is_flagged(row: dict[str, Any]) -> bool:
        admissible = row["posthoc_intercept_audit"][
            "admissible_within_0p5_nats"
        ]
        return not admissible.get(a, False) or not admissible.get(b, False)

    flagged = [row for row in finite_rows if is_flagged(row)]
    legacy_admissible = [row for row in finite_rows if not is_flagged(row)]
    return {
        "raw_contrast": _med(
            row["count_poisson_nll"][a] - row["count_poisson_nll"][b]
            for row in finite_rows
        ),
        "n_scored_seeds": len(finite_rows),
        "n_posthoc_flagged_seeds": len(flagged),
        "legacy_filtered_contrast_deprecated": _med(
            row["count_poisson_nll"][a] - row["count_poisson_nll"][b]
            for row in legacy_admissible
        ),
        "legacy_n_admissible_seeds_deprecated": len(legacy_admissible),
    }


def _count_dispersion(count: np.ndarray) -> float | None:
    values = np.asarray(count, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all() or values.mean() <= 0:
        return None
    return float(values.var(ddof=1) / values.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3/pilot"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--date", default="2026-09-02")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=Path(
            "/data/hfosp_group_event_state_v0_3/source_audit/"
            "nested_source_audit.json"
        ),
    )
    args = parser.parse_args()
    source_audit = json.loads(args.source_audit.read_text())
    if not source_audit.get("model_layer_nested_contract", False):
        raise ValueError("model-layer nested source audit did not pass")
    manifest = json.loads((args.result_root / "task_manifest.json").read_text())
    incomplete = [t["id"] for t in manifest["tasks"] if t["status"] != "complete"]
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"pilot queue incomplete: {incomplete[:8]}")
    records = []
    timeline_cache = {}
    for subject in manifest["subjects"]:
        grammar_path = args.result_root / subject / "grammar/grammar_v03.json"
        grammar = json.loads(grammar_path.read_text()) if grammar_path.exists() else None
        for seed in manifest["seeds"]:
            root = args.result_root / subject / f"seed_{seed}"
            result_path = root / "result.json"
            open_path = root / "open_loop.json"
            if not (result_path.exists() and open_path.exists()):
                continue
            result = json.loads(result_path.read_text())
            opened = json.loads(open_path.read_text())
            if result.get("source_commit") != manifest["source_commit"]:
                raise ValueError(f"{subject}/{seed}: state source commit drift")
            if opened.get("source_commit") != manifest["source_commit"]:
                raise ValueError(f"{subject}/{seed}: evaluation source commit drift")
            if grammar and grammar.get("source_commit") != manifest["source_commit"]:
                raise ValueError(f"{subject}: grammar source commit drift")
            if grammar and not grammar.get("calibration_prefix_only", False):
                raise ValueError(f"{subject}: grammar is not calibration-prefix-only")
            # Every recorded second of the split must enter survival exactly
            # once (within float accumulation tolerance), and no gap may enter.
            for phase_name, phase_result in (
                ("state_train", result["history"][0]["train"]),
                ("dev_val", result["history"][0]["validation"]),
                ("dev_test", result["dev_test"]),
            ):
                expected = float(phase_result["expected_scoreable_seconds"])
                observed = float(phase_result["observed_seconds"])
                if abs(expected - observed) > 1.0:
                    raise ValueError(
                        f"{subject}/{seed}/{phase_name}: survival support {observed} != {expected}"
                    )
            if subject not in timeline_cache:
                timeline_cache[subject] = load_subject_timeline(subject)
            timeline = timeline_cache[subject]
            partition = nested_partition(timeline)
            phase_label = partition.labels_of(timeline.grid.t_anchor)
            for h_i, horizon in enumerate(timeline.config.horizons_seconds):
                key = f"{int(horizon)}s"
                if opened["horizons"][key].get("status") != "ok":
                    continue
                train_idx = np.flatnonzero(
                    (phase_label == 1)
                    & timeline.grid.eligible[:, h_i]
                    & (timeline.grid.t_anchor + horizon <= partition.boundary_epochs[1] + 1e-6)
                )
                test_idx = np.flatnonzero(
                    (phase_label == 3) & timeline.grid.eligible[:, h_i]
                )
                train_count = (
                    timeline.grid.window_hi[train_idx, h_i]
                    - timeline.grid.window_lo[train_idx, h_i]
                ).astype(np.float64)
                test_count = (
                    timeline.grid.window_hi[test_idx, h_i]
                    - timeline.grid.window_lo[test_idx, h_i]
                ).astype(np.float64)
                intercept_mean = max(float(train_count.mean()), 1e-8)
                intercept_nll = float(
                    _poisson_nll(test_count, np.full(test_count.shape, intercept_mean)).mean()
                )
                count_nll = opened["horizons"][key]["count_poisson_nll"]
                admissible = {
                    name: (
                        value is not None
                        and math.isfinite(float(value))
                        and float(value) <= intercept_nll + 0.5
                    )
                    for name, value in count_nll.items()
                }
                opened["horizons"][key]["posthoc_intercept_audit"] = {
                    "train_mean_count": intercept_mean,
                    "development_test_intercept_poisson_nll": intercept_nll,
                    "admissible_within_0p5_nats": admissible,
                }
            records.append({
                "subject": subject,
                "seed": seed,
                "grammar": grammar,
                "training": result,
                "open_loop": opened,
            })

    per_subject: dict[str, Any] = {}
    for subject in manifest["subjects"]:
        rows = [r for r in records if r["subject"] == subject]
        h_out = {}
        for horizon in HORIZONS:
            h_rows = [
                r["open_loop"]["horizons"][horizon]
                for r in rows
                if r["open_loop"]["horizons"][horizon].get("status") == "ok"
            ]
            def contrast(endpoint, a, b):
                return _med([
                    x[endpoint][a] - x[endpoint][b]
                    for x in h_rows
                    if x[endpoint].get(a) is not None and x[endpoint].get(b) is not None
                ])
            count_pairs = {
                "correct_vs_multiscale": _paired_count_summary(
                    h_rows, "correct_state", "multiscale_history"
                ),
                "correct_vs_shifted": _paired_count_summary(
                    h_rows, "correct_state", "block_shifted_state"
                ),
                "correct_vs_state_free": _paired_count_summary(
                    h_rows, "correct_state", "state_free"
                ),
            }
            h_out[horizon] = {
                "n_seeds": len(h_rows),
                "n_insufficient_coverage_seeds": len(rows) - len(h_rows),
                "count_correct_minus_multiscale": count_pairs[
                    "correct_vs_multiscale"
                ]["raw_contrast"],
                "count_correct_minus_shifted": count_pairs[
                    "correct_vs_shifted"
                ]["raw_contrast"],
                "count_correct_minus_state_free": count_pairs[
                    "correct_vs_state_free"
                ]["raw_contrast"],
                "legacy_posthoc_admissible_arm_seeds_deprecated": {
                    name: sum(
                        x["posthoc_intercept_audit"]["admissible_within_0p5_nats"].get(name, False)
                        for x in h_rows
                    )
                    for name in (
                        "correct_state", "multiscale_history", "block_shifted_state", "state_free"
                    )
                },
                "count_pair_scored_seeds": {
                    name: pair["n_scored_seeds"] for name, pair in count_pairs.items()
                },
                "count_pair_posthoc_flagged_seeds": {
                    name: pair["n_posthoc_flagged_seeds"]
                    for name, pair in count_pairs.items()
                },
                "legacy_posthoc_filtered_count_contrasts_deprecated": {
                    name: pair["legacy_filtered_contrast_deprecated"]
                    for name, pair in count_pairs.items()
                },
                "legacy_posthoc_admissible_pair_seeds_deprecated": {
                    name: pair["legacy_n_admissible_seeds_deprecated"]
                    for name, pair in count_pairs.items()
                },
                "intercept_poisson_nll": _med([
                    x["posthoc_intercept_audit"]["development_test_intercept_poisson_nll"] for x in h_rows
                ]),
                "continue_correct_minus_shifted": _med([
                    x["mark_nll"]["correct_state"]["continue"]
                    - x["mark_nll"]["block_shifted_state"]["continue"]
                    for x in h_rows if x["mark_nll"]["block_shifted_state"] is not None
                ]),
                "positive_size_correct_minus_shifted": _med([
                    x["mark_nll"]["correct_state"]["positive_size"]
                    - x["mark_nll"]["block_shifted_state"]["positive_size"]
                    for x in h_rows if x["mark_nll"]["block_shifted_state"] is not None
                ]),
                "subset_correct_minus_shifted": _med([
                    x["mark_nll"]["correct_state"]["subset"]
                    - x["mark_nll"]["block_shifted_state"]["subset"]
                    for x in h_rows if x["mark_nll"]["block_shifted_state"] is not None
                ]),
                "continue_correct_minus_state_free": _med([
                    x["mark_nll"]["correct_state"]["continue"]
                    - x["mark_nll"]["state_free"]["continue"] for x in h_rows
                ]),
                "positive_size_correct_minus_state_free": _med([
                    x["mark_nll"]["correct_state"]["positive_size"]
                    - x["mark_nll"]["state_free"]["positive_size"] for x in h_rows
                ]),
                "subset_correct_minus_state_free": _med([
                    x["mark_nll"]["correct_state"]["subset"]
                    - x["mark_nll"]["state_free"]["subset"] for x in h_rows
                ]),
                "n_development_test_anchors": int(
                    h_rows[0]["n_development_test_anchors"]
                ) if h_rows else 0,
                "n_shift_matched_anchors": int(h_rows[0]["n_shift_matched_anchors"]) if h_rows else 0,
                "multiscale_ridge_edge_seeds": sum(
                    bool(x["multiscale_count_fit"]["ridge_at_edge"]) for x in h_rows
                ),
            }
        timeline = timeline_cache[subject]
        partition = nested_partition(timeline)
        phase_labels = partition.labels_of(timeline.grid.t_anchor)
        dispersion_by_phase: dict[str, dict[str, Any]] = {}
        for phase in ("state_train", "dev_val", "dev_test"):
            phase_index = PHASE_NAMES.index(phase)
            _phase_lo, phase_hi = partition.bounds(phase)
            dispersion_by_phase[phase] = {}
            for horizon_index, horizon_seconds in enumerate(
                timeline.config.horizons_seconds
            ):
                anchor_index = np.flatnonzero(
                    (phase_labels == phase_index)
                    & timeline.grid.eligible[:, horizon_index]
                    & (
                        timeline.grid.t_anchor + float(horizon_seconds)
                        <= phase_hi + 1e-6
                    )
                )
                count = (
                    timeline.grid.window_hi[anchor_index, horizon_index]
                    - timeline.grid.window_lo[anchor_index, horizon_index]
                ).astype(np.float64)
                dispersion_by_phase[phase][f"{int(horizon_seconds)}s"] = {
                    "n_anchors": int(count.size),
                    "mean_count": float(count.mean()) if count.size else None,
                    "variance_to_mean": _count_dispersion(count),
                }
        per_subject[subject] = {
            "n_seeds": len(rows),
            "selected_epochs": [r["training"]["selected_epoch"] for r in rows],
            "n_history_epochs": [len(r["training"]["history"]) for r in rows],
            "optimization_status": (
                "first_trained_epoch_selected_all_seeds"
                if rows and all(r["training"]["selected_epoch"] == 0 for r in rows)
                else "budget_edge_all_seeds"
                if rows and all(
                    r["training"]["selected_epoch"] >= len(r["training"]["history"]) - 2
                    for r in rows
                )
                else "mixed_or_interior"
            ),
            "state_norm_validation": _med([
                r["training"]["history"][r["training"]["selected_epoch"]]["validation"]["state_norm_median"]
                for r in rows
            ]),
            "encoder_relative_update": _med([
                r["training"]["parameter_updates"]["event_encoder"] for r in rows
            ]),
            "state_relative_update": _med([
                r["training"]["parameter_updates"]["state"] for r in rows
            ]),
            "adapter_relative_update": _med([
                r["training"]["parameter_updates"]["grammar_adapter"] for r in rows
            ]),
            "n_events": rows[0]["training"]["n_events"] if rows else None,
            "count_dispersion": dispersion_by_phase,
            "horizons": h_out,
        }

    directions: dict[str, Any] = {}
    for horizon in HORIZONS:
        fields = (
            "count_correct_minus_multiscale",
            "count_correct_minus_shifted",
            "continue_correct_minus_shifted",
            "positive_size_correct_minus_shifted",
            "subset_correct_minus_shifted",
        )
        directions[horizon] = {}
        for field in fields:
            vals = [per_subject[s]["horizons"][horizon][field] for s in manifest["subjects"]]
            vals = [v for v in vals if v is not None]
            directions[horizon][field] = {
                "n_negative_favourable": sum(v < 0 for v in vals),
                "n_estimable": len(vals),
                "patient_median": _med(vals),
            }

    summary = {
        "format": "group_event_state_v0_3_1_closeout_summary",
        "status": (
            "V0_3_1_PILOT_CLOSED_MAJOR_REVISION"
            if not incomplete
            else "V0_3_1_PILOT_INCOMPLETE"
        ),
        "scientific_status": "instrument_complete_state_learning_unresolved",
        "source_commit": manifest["source_commit"],
        "subjects": manifest["subjects"],
        "seeds": manifest["seeds"],
        "n_completed_runs": len(records),
        "incomplete_tasks": incomplete,
        "per_subject": per_subject,
        "diagnostic_direction_summary": directions,
        "nested_source_audit": source_audit,
        "primary_estimand": {
            "required": [
                "H_plus_S_correct_vs_H",
                "H_plus_S_correct_vs_H_plus_S_shifted",
                "H_plus_S_dynamic_vs_H_plus_S_train_mean",
            ],
            "measured_in_v0_3_1": [],
            "claim_eligible": False,
            "reason": (
                "v0.3.1 compared S alone with H and compared S with a shifted "
                "S; it did not fit the nested residual H+S estimand"
            ),
        },
        "implementation_closeout_audit": {
            "adapter_exact_zero_dead_zone": False,
            "adapter_initial_gate_sigmoid_minus4": float(1.0 / (1.0 + math.exp(4.0))),
            "adapter_final_projection_initialisation_std": 1e-3,
            "effective_output_modulation_measured": False,
            "per_time_state_layernorm_present": True,
            "free_state_to_state_mixing_present": True,
            "tbptt_closes_on_first_event_or_time_limit": True,
            "tbptt_gradient_window_seconds": 1800.0,
            "longest_training_target_seconds": 7200.0,
            "checkpoint_specific_replay_from_segment_boundary": True,
            "segment_burnin_seconds": 300.0,
            "posthoc_test_performance_denominator_filter_removed_in_closeout": True,
        },
        "deprecated_interpretations": [
            "v0.3.1 establishes absence of a slow state",
            "state-alone versus multiscale history is the residual-state estimand",
            "near-zero mark contrasts from no-update checkpoints are biological nulls",
            "non-zero state norm proves that a functional state was learned",
            "the already-read 80-100 percent block is an independent final test",
        ],
        "scientific_scope": {
            "partition": "nested 16/4/50/10/20 physical-time development partition; formal sealed partition closed",
            "primary_input": "full interictal group-event waveform/multiband/participation/delay after scoring",
            "background_seeg": "not used in this primary pilot",
            "grammar": "calibration-prefix product-form tied-group grammar; legacy learned weights not loaded",
            "state": "16-dimensional fixed-timescale state at 5/30/120/360 min",
            "upstream_measurement": "legacy full-record contact selection remains transductive",
            "claim_level": "three-patient development instrument closeout; state learning unresolved",
            "development_test_reuse": (
                "the 80-100 percent block has informed architecture review and cannot "
                "serve as a future independent final test"
            ),
        },
    }
    machine_path = args.result_root / "summary_v0_3_1_closeout.json"
    _atomic(machine_path, json.dumps(summary, indent=2, sort_keys=True))

    dispersion_values = [
        cell["variance_to_mean"]
        for subject_row in per_subject.values()
        for phase_row in subject_row["count_dispersion"].values()
        for cell in phase_row.values()
        if cell["variance_to_mean"] is not None
    ]
    dispersion_range = (
        f"{min(dispersion_values):.1f}–{max(dispersion_values):.1f}"
        if dispersion_values
        else "NA"
    )

    plain_lines = [
        "# 群体间期事件状态 v0.3.1：审阅后收口报告（白话版）",
        "",
        "## 一句话",
        "",
        "v0.3.1 把 split、有效记录时间、冻结 contact grammar、连续时间扫描和 open-loop 评分接通了，但没有完成一次能够裁定慢状态的实验。最准确的状态是：**仪器跑通，state learning 未决。**",
        "",
        "## 为什么旧结论必须更正",
        "",
        "旧分析比较的是 `state S` 单独预测和显式历史 `H` 单独预测。真正的问题却应该是：在同一个 `H` 基础上，加动态状态是否还有增量，即 `H+S` 是否优于 `H`。这个主比较没有运行。",
        "",
        "旧 wrong-time 比较也只证明某些 state 数值与时刻有关；它没有回答这些数值是否在 `H` 之外仍有信息。旧图和旧报告因此不能再写成‘H1/H2a 阴性’。",
        "",
        "## 三位患者分别说明什么",
        "",
    ]
    for subject in manifest["subjects"]:
        row = per_subject[subject]
        if subject == "epilepsiae_1146":
            interpretation = (
                "5/30 分钟 correct-time 相对 shifted 有利，但 state 单独仍未超过 H；"
                "三个 seed 都在预算末端，最多支持‘存在时刻相关信息’，不能支持 residual state。"
            )
        elif subject == "yuquan_pengzihang":
            interpretation = (
                "三个 seed 都选择第一个训练 epoch，120 分钟无可评分 anchor；"
                "这是当前合同不适配/未形成有效更新，不是 state 阴性。"
            )
        else:
            interpretation = (
                "三个 seed 都选择第一个训练 epoch，correct 与 shifted 几乎相同，"
                "120 分钟严重失校准；更像训练塌缩，不是生物学阴性。"
            )
        plain_lines.append(
            f"- `{subject}`：选中 epoch {row['selected_epochs']}；event encoder / state / adapter 相对更新 "
            f"{_fmt(row['encoder_relative_update'], 5)} / {_fmt(row['state_relative_update'], 5)} / {_fmt(row['adapter_relative_update'], 5)}。{interpretation}"
        )
    plain_lines += [
        "",
        "## 新增审计发现",
        "",
        "1. adapter 不是数学上的全零死区：投影权重非零，初始 gate 为 sigmoid(-4)≈0.018，已有梯度记录也非零。但实际输出调制量没有被测量，而且全局 gradient clipping 在大多数 chunk 触发，所以有效 state path 是否真正学起来仍未知。",
        "2. grammar 在每个时刻对 state 做 LayerNorm，可能删除有意义的幅度；state update 又能任意混合旧 state，因此所谓固定时间尺度只是 nominal label。这两点使当前 latent 不能按 5/30/120/360 分钟生理时间常数解释。",
        "3. TBPTT 实现确实在事件数或30分钟任一先达到时切 chunk，旧技术报告写成 AND 是错误措辞；但120分钟 loss 的梯度只能回传30分钟，长 horizon 的差表现不能直接解释为没有长状态。",
        "4. validation/test 会从合法 segment 起点按当前 checkpoint 重放，没有发现 stale-state 复用；5分钟只是每个 segment 开头不评分，不是每个 chunk 都丢数据。但5分钟不足以单独初始化120分钟通道。",
        f"5. 未来事件数高度过度离散：三位患者各 split/horizon 的 variance/mean 为 {dispersion_range}，Poisson 明显不合适。",
        "6. 旧聚合器曾根据 development-test 上是否靠近 fitted-intercept 决定某个 seed 是否进入分母。现在已改成所有有限分数都保留，审计只加 flag，不再删除；旧 filtered 数值只留作 deprecated provenance。",
        "",
        "## 正式收口",
        "",
        "v0.3.1 的状态固定为 `V0_3_1_PILOT_CLOSED_MAJOR_REVISION`。允许写：nested split、有效 exposure、冻结 grammar、连续时间扫描和 open-loop 评分已完成端到端联调；1146 有有限的时刻相关诊断信号。",
        "",
        "不允许写：没有慢状态、H1/H2a 已被否定、当前 fixed-τ 对应真实生理尺度、非零 state norm 证明学到了状态、或者 H2b/H3 已经可以开始解释。",
        "",
        "已读取的80–100% development test 已参与架构审阅，今后不能再充当最终独立检验。正式/封存分区仍未打开。下一版的首要比较固定为 `H+S_correct` vs `H`、`H+S_correct` vs `H+S_shifted`、以及 dynamic `S` vs TRAIN mean `S`。",
        "",
        f"机器汇总：`{machine_path}`",
    ]

    technical_lines = [
        "# Group-Event State v0.3.1：审阅后技术收口",
        "",
        f"- source commit: `{manifest['source_commit']}`",
        f"- completed runs: {len(records)}/9",
        "- closeout status: `V0_3_1_PILOT_CLOSED_MAJOR_REVISION`",
        "- scientific status: `instrument_complete_state_learning_unresolved`",
        "- partition: 16/4/50/10/20 nested development physical-time split；formal sealed partition 未打开",
        "- grammar: calibration-prefix single K categorical, reported as continue + K|continue + product-form conditional K-subset likelihood",
        "- timing: trapezoidal marked point-process likelihood over valid exposure, including terminal censoring",
        "- state: 16 nominal dimensions at 300/1800/7200/21600 s, but learned state-to-state mixing means these are not identifiable physiological time constants",
        "- TBPTT: closes when either 1024 events or 1800 s is reached; carry+detach, no chunk reset; 300 s segment-level burn-in",
        "- slow objective: fixed-anchor future-count Poisson NLL at 300/1800/7200 s",
        "- source boundary: legacy learned decoder weights excluded, but legacy full-record contact selection remains upstream-transductive",
        "- open-loop: no future event update; horizons 300/1800/7200 s",
        "- missing primary estimand: H+S_correct vs H, H+S_correct vs H+S_shifted, dynamic S vs TRAIN-mean S",
        f"- count overdispersion audit: variance/mean across existing patient×phase×horizon cells = {dispersion_range}",
        "",
        "## Existing contrasts retained as diagnostics, not H1/H2a tests",
        "",
        "All finite development-test scores are retained below. The fitted-intercept audit is now a flag and never changes the denominator. `flagged` reports how many scored seeds would have been removed by the deprecated post-hoc rule.",
        "",
    ]
    for subject in manifest["subjects"]:
        technical_lines += [f"### {subject}", ""]
        for horizon in HORIZONS:
            h = per_subject[subject]["horizons"][horizon]
            technical_lines.append(
                f"- {horizon}: count−multiscale={_fmt(h['count_correct_minus_multiscale'])}; "
                f"count−shift={_fmt(h['count_correct_minus_shifted'])}; "
                f"continue−shift={_fmt(h['continue_correct_minus_shifted'])}; "
                f"positive-size−shift={_fmt(h['positive_size_correct_minus_shifted'])}; "
                f"subset−shift={_fmt(h['subset_correct_minus_shifted'])}; "
                f"anchors={h['n_development_test_anchors']}, matched={h['n_shift_matched_anchors']}, "
                f"pair-scored seeds={h['count_pair_scored_seeds']}; "
                f"posthoc-flagged seeds={h['count_pair_posthoc_flagged_seeds']}; "
                f"coverage-insufficient seeds={h['n_insufficient_coverage_seeds']}; "
                f"ridge-edge seeds={h['multiscale_ridge_edge_seeds']}/3."
            )
        technical_lines.append("")
    technical_lines += [
        "## Implementation audit",
        "",
        "- Adapter is not in an exact zero-gradient dead zone: final projections are N(0,1e-3), gates initialise at sigmoid(-4)=0.017986, and recorded gradients are non-zero. Effective logit modulation/Jacobian was not measured, so functional trainability remains unresolved.",
        "- Per-time state LayerNorm and free state-to-state mixing are present; both prevent interpreting latent amplitude or nominal tau labels physiologically.",
        "- Production TBPTT uses the first event/time limit reached, but 7200-s targets receive at most 1800 s of gradient credit assignment.",
        "- Each validation/development-test pass replays the current checkpoint from the segment boundary; no stale checkpoint trajectory path was found.",
        "- Poisson count is materially misspecified given the dispersion audit and must be replaced by a paired negative-binomial family before a new state experiment.",
        "- The 80–100% block is development-consumed and cannot be reused as a future independent final test.",
        "",
        "## Interpretation boundary",
        "",
        "Negative raw contrasts are favourable, but none of the above is the required residual H+S estimand. `epilepsiae_1146` provides a limited correct-time diagnostic at 5/30 min; the other two subjects are dominated by first-epoch selection, coverage limits or long-horizon miscalibration. Near-zero mark contrasts are not biological nulls. The pilot supports engineering integration only; state learning, H1 and H2a remain unresolved.",
        "",
        f"Machine report: `{machine_path}`",
    ]
    archive = args.repo / "docs/archive/topic5"
    plain_path = archive / f"group_event_state_v0_3_pilot_plain_{args.date}.md"
    technical_path = archive / f"group_event_state_v0_3_pilot_technical_{args.date}.md"
    _atomic(plain_path, "\n".join(plain_lines) + "\n")
    _atomic(technical_path, "\n".join(technical_lines) + "\n")
    print(json.dumps({
        "machine": str(machine_path),
        "plain": str(plain_path),
        "technical": str(technical_path),
        "status": summary["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
