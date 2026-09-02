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
            def admissible_count_contrast(a, b):
                return _med([
                    x["count_poisson_nll"][a] - x["count_poisson_nll"][b]
                    for x in h_rows
                    if x["posthoc_intercept_audit"]["admissible_within_0p5_nats"].get(a, False)
                    and x["posthoc_intercept_audit"]["admissible_within_0p5_nats"].get(b, False)
                ])
            h_out[horizon] = {
                "n_seeds": len(h_rows),
                "n_insufficient_coverage_seeds": len(rows) - len(h_rows),
                "count_correct_minus_multiscale": admissible_count_contrast(
                    "correct_state", "multiscale_history"
                ),
                "count_correct_minus_shifted": admissible_count_contrast(
                    "correct_state", "block_shifted_state"
                ),
                "count_correct_minus_state_free": admissible_count_contrast(
                    "correct_state", "state_free"
                ),
                "count_estimable_seeds": {
                    name: sum(
                        x["posthoc_intercept_audit"]["admissible_within_0p5_nats"].get(name, False)
                        for x in h_rows
                    )
                    for name in (
                        "correct_state", "multiscale_history", "block_shifted_state", "state_free"
                    )
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
        per_subject[subject] = {
            "n_seeds": len(rows),
            "selected_epochs": [r["training"]["selected_epoch"] for r in rows],
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
        "format": "group_event_state_v0_3_pilot_summary",
        "status": "complete" if not incomplete else "incomplete",
        "source_commit": manifest["source_commit"],
        "subjects": manifest["subjects"],
        "seeds": manifest["seeds"],
        "n_completed_runs": len(records),
        "incomplete_tasks": incomplete,
        "per_subject": per_subject,
        "direction_summary": directions,
        "nested_source_audit": source_audit,
        "scientific_scope": {
            "partition": "nested 16/4/50/10/20 physical-time development partition; formal sealed partition closed",
            "primary_input": "full interictal group-event waveform/multiband/participation/delay after scoring",
            "background_seeg": "not used in this primary pilot",
            "grammar": "calibration-prefix product-form tied-group grammar; legacy learned weights not loaded",
            "state": "16-dimensional fixed-timescale state at 5/30/120/360 min",
            "upstream_measurement": "legacy full-record contact selection remains transductive",
            "claim_level": "three-patient development instrument pilot only",
        },
    }
    machine_path = args.result_root / "summary_main.json"
    _atomic(machine_path, json.dumps(summary, indent=2, sort_keys=True))

    plain_lines = [
        "# 群体间期事件状态 v0.3 三患者 pilot：白话报告",
        "",
        "## 一句话",
        "",
        "这轮不再问模型能不能只凭最近一次事件猜下一次，而是问：把群体间期事件按真实时间连续读入后，模型留下的状态能否在停止读取未来事件的情况下，预测未来 5、30、120 分钟里会发生多少事件，以及这些事件会招募哪些触点。",
        "",
        "## 这轮真正修正了什么",
        "",
        "- 没有事件发生的有效记录时间正式进入损失；记录中断、发作和发作后排除段不算作‘安静’。",
        "- 每次事件先由旧时刻状态预测，整次事件的波形、频带、参与触点和精确延迟只在评分后用于更新状态。",
        "- contact decoder 只读取旧模型的结构宽度，不读取旧权重；新 grammar 只在患者最早 16% 有效记录上拟合、用随后 4% 选轮次，然后冻结。",
        "- 状态只有 16 维，固定覆盖 5 分钟、30 分钟、2 小时和 6 小时四档；每次事件是有界校正，不再无界累加成事件计数器。",
        "- 状态训练只用随后 50%，下一 10% 选 checkpoint，最后 20% 只做 development 评分。",
        "- 训练不只看下一次事件，还直接要求状态预测未来 5、30、120 分钟的事件数。",
        "- TBPTT 同时限制 1024 次事件和 30 分钟；chunk 边界只断梯度、不清状态。",
        "- 主要评价是固定真实时间锚点后的多事件 open-loop，而不是只看下一次事件。",
        "- 仍有一个不能抹掉的限制：当前触点集合由旧的全记录 refine/packing 选出，所以这是模型层嵌套干净、测量层仍 transductive 的开发性 pilot。",
        "",
        "## 数据与训练是否真的动了",
        "",
    ]
    for subject in manifest["subjects"]:
        row = per_subject[subject]
        plain_lines.append(
            f"- `{subject}`：{row['n_seeds']}/3 seeds；选中 epoch {row['selected_epochs']}；"
            f"验证状态范数中位 {_fmt(row['state_norm_validation'], 3)}；event encoder / state / adapter 相对更新 "
            f"{_fmt(row['encoder_relative_update'], 5)} / {_fmt(row['state_relative_update'], 5)} / {_fmt(row['adapter_relative_update'], 5)}。"
        )
    plain_lines += [
        "",
        "## 核心结果怎么读",
        "",
        "下面所有差值都是‘正确时刻状态的损失 − 对照损失’，所以负数才是正确状态更好。三位患者只是方向分诊，不能写成队列结论。",
        "",
        "| horizon | count vs multiscale | count vs wrong-time | continue vs wrong-time | positive size vs wrong-time | contacts vs wrong-time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        d = directions[horizon]
        cells = []
        for field in (
            "count_correct_minus_multiscale", "count_correct_minus_shifted",
            "continue_correct_minus_shifted", "positive_size_correct_minus_shifted",
            "subset_correct_minus_shifted",
        ):
            x = d[field]
            cells.append(f"{x['n_negative_favourable']}/{x['n_estimable']} ({_fmt(x['patient_median'])})")
        plain_lines.append(f"| {horizon} | " + " | ".join(cells) + " |")
    plain_lines += [
        "",
        "## 当前允许结论",
        "",
        "这张表只能回答三件事：模型是否真正训练、正确时刻的状态是否胜过保留自相关的错时状态、以及它是否胜过一个可解释的多尺度历史基线。只有当多名患者、多个 seed 在较长 horizon 上方向一致，才能把它叫作候选慢预测状态。",
        "",
        "它还不能回答：这个状态能否预测发作、IED 是否反过来塑造状态、或者任何跨患者正式结论。H2b 与 H3 必须读取这轮冻结的轨迹，分别独立检验。",
        "",
        f"机器汇总：`{machine_path}`",
    ]

    technical_lines = [
        "# Group-Event State v0.3 pilot：技术报告",
        "",
        f"- source commit: `{manifest['source_commit']}`",
        f"- completed runs: {len(records)}/9",
        "- partition: 16/4/50/10/20 nested development physical-time split；formal sealed partition 未打开",
        "- grammar: calibration-prefix single K categorical, reported as continue + K|continue + product-form conditional K-subset likelihood",
        "- timing: trapezoidal marked point-process likelihood over valid exposure, including terminal censoring",
        "- state: 16 dimensions; fixed taus = 300/1800/7200/21600 s; bounded tau-dependent event correction",
        "- TBPTT: max 1024 events AND 1800 s; carry+detach, no chunk reset; 300 s segment burn-in",
        "- slow objective: fixed-anchor future-count Poisson NLL at 300/1800/7200 s",
        "- source boundary: legacy learned decoder weights excluded, but legacy full-record contact selection remains upstream-transductive",
        "- open-loop: no future event update; horizons 300/1800/7200 s",
        "",
        "## Per-subject seed-median contrasts",
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
                f"ridge-edge seeds={h['multiscale_ridge_edge_seeds']}/3."
            )
        technical_lines.append("")
    technical_lines += [
        "## Interpretation boundary",
        "",
        "Negative contrasts are favourable. The pilot is not powered for cohort inference. A ridge-edge baseline is retained with a caveat rather than converted into a biological result. Mark comparisons against wrong-time and state-free grammar are valid; a full capacity-matched multiscale mark adapter remains a later comparison and is not silently claimed here.",
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
