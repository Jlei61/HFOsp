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
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3/pilot"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--date", default="2026-09-02")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.result_root / "task_manifest.json").read_text())
    incomplete = [t["id"] for t in manifest["tasks"] if t["status"] != "complete"]
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"pilot queue incomplete: {incomplete[:8]}")
    records = []
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
            if grammar and not grammar.get("outer_train_only", False):
                raise ValueError(f"{subject}: grammar is not outer-TRAIN-only")
            # Every recorded second of the split must enter survival exactly
            # once (within float accumulation tolerance), and no gap may enter.
            for split_name, split_result in (
                ("train", result["history"][0]["train"]),
                ("val", result["history"][0]["validation"]),
                ("test", result["test"]),
            ):
                expected = float(result["recorded_seconds"][split_name])
                observed = float(split_result["observed_seconds"])
                if abs(expected - observed) > 1.0:
                    raise ValueError(
                        f"{subject}/{seed}/{split_name}: survival support {observed} != {expected}"
                    )
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
            h_rows = [r["open_loop"]["horizons"][horizon] for r in rows]
            def contrast(endpoint, a, b):
                return _med([x[endpoint][a] - x[endpoint][b] for x in h_rows])
            h_out[horizon] = {
                "n_seeds": len(h_rows),
                "count_correct_minus_multiscale": contrast(
                    "count_poisson_nll", "correct_state", "multiscale_history"
                ),
                "count_correct_minus_shifted": contrast(
                    "count_poisson_nll", "correct_state", "block_shifted_state"
                ),
                "count_correct_minus_state_free": contrast(
                    "count_poisson_nll", "correct_state", "state_free"
                ),
                "size_correct_minus_shifted": _med([
                    x["mark_nll"]["correct_state"]["size"]
                    - x["mark_nll"]["block_shifted_state"]["size"]
                    for x in h_rows if x["mark_nll"]["block_shifted_state"] is not None
                ]),
                "subset_correct_minus_shifted": _med([
                    x["mark_nll"]["correct_state"]["subset"]
                    - x["mark_nll"]["block_shifted_state"]["subset"]
                    for x in h_rows if x["mark_nll"]["block_shifted_state"] is not None
                ]),
                "size_correct_minus_state_free": _med([
                    x["mark_nll"]["correct_state"]["size"]
                    - x["mark_nll"]["state_free"]["size"] for x in h_rows
                ]),
                "subset_correct_minus_state_free": _med([
                    x["mark_nll"]["correct_state"]["subset"]
                    - x["mark_nll"]["state_free"]["subset"] for x in h_rows
                ]),
                "n_test_anchors": int(h_rows[0]["n_test_anchors"]) if h_rows else 0,
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
            "size_correct_minus_shifted",
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
        "scientific_scope": {
            "partition": "development physical-time TEST, not formal sealed partition",
            "primary_input": "full interictal group-event waveform/multiband/participation/delay after scoring",
            "background_seeg": "not used in this primary pilot",
            "grammar": "outer-TRAIN exact tied-group grammar; legacy learned weights not loaded",
            "state": "20-dimensional fixed-timescale state, 2 min to 12 h",
            "claim_level": "three-patient development pilot only",
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
        "- contact decoder 的结构来自旧模型，但旧模型学过的权重没有进入主臂；新 grammar 只在新的 outer-TRAIN 上训练，然后冻结。",
        "- 状态只有 20 维，覆盖 2 分钟、10 分钟、30 分钟、2 小时和 12 小时五档；每次事件是有界校正，不再无界累加成事件计数器。",
        "- 主要评价是固定真实时间锚点后的多事件 open-loop，而不是只看下一次事件。",
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
        "| horizon | count vs multiscale | count vs wrong-time | size vs wrong-time | contacts vs wrong-time |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        d = directions[horizon]
        cells = []
        for field in (
            "count_correct_minus_multiscale", "count_correct_minus_shifted",
            "size_correct_minus_shifted", "subset_correct_minus_shifted",
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
        "- partition: development physical-time TEST；formal sealed partition 未打开",
        "- grammar: outer-TRAIN exact group-size/STOP + conditional fixed-cardinality subset likelihood",
        "- timing: trapezoidal marked point-process likelihood over valid exposure, including terminal censoring",
        "- state: 20 dimensions; fixed taus = 120/600/1800/7200/43200 s; bounded tau-dependent event correction",
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
                f"size−shift={_fmt(h['size_correct_minus_shifted'])}; "
                f"subset−shift={_fmt(h['subset_correct_minus_shifted'])}; "
                f"anchors={h['n_test_anchors']}, matched={h['n_shift_matched_anchors']}, "
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
