#!/usr/bin/env python3
"""Finalize the corrected observed-support H1/H2 long-horizon exploration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import atomic_json  # noqa: E402

LONG = Path("/data/hfosp_group_event_state_v0_3_5_long_observed_support")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_gain(base: Any, full: Any) -> float | None:
    if not isinstance(base, (int, float)) or not isinstance(full, (int, float)):
        return None
    return float(base - full)


def collect_seed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((LONG / "physical").glob("*/dynamic_rate/*/seed*/card.json")):
        card = read(path)
        scale = path.parents[3].name
        strata = card.get("selection_strata") or {"all": card.get("selection_arms", {})}
        for stratum, arms in strata.items():
            for contrast, base, full in (
                ("dynamic_over_static", "static", "dynamic"),
                ("residual_over_dynamic", "dynamic", "residual"),
                ("correct_time_over_wrong", "block_shift", "residual_paired"),
            ):
                base_nll = arms.get(base, {}).get("nll", [None])[0]
                full_nll = arms.get(full, {}).get("nll", [None])[0]
                rows.append({
                    "family": "H1_rate", "scale": scale, "stratum": stratum,
                    "subject": card["subject"], "seed": int(card["seed"]),
                    "endpoint": "count_nll", "contrast": contrast,
                    "gain": finite_gain(base_nll, full_nll),
                    "n": arms.get(full, {}).get("n", [0])[0],
                })
    for path in sorted((LONG / "physical").glob("*/full_mark/*/decoder_seed*_state_seed*/card.json")):
        card = read(path)
        scale = path.parents[3].name
        selections = {"all": card.get("physical_selection", {})}
        no_seizure = selections["all"].get("no_seizure_crossing")
        if isinstance(no_seizure, dict):
            selections["no_seizure_crossing"] = no_seizure
        for stratum, selection in selections.items():
            for _horizon, arms in selection.get("horizons", {}).items():
                for endpoint, contrast in arms.get("contrasts", {}).items():
                    for name, key in (
                        ("state_over_dynamic_rate", "mark_gain_over_q"),
                        ("state_over_fit_mean", "mark_gain_over_period_mean"),
                        ("correct_time_over_wrong", "correct_time_gain_over_matched_wrong"),
                    ):
                        rows.append({
                            "family": "H1_H2a_physical", "scale": scale,
                            "stratum": stratum, "subject": card["subject"],
                            "seed": int(card["seed"]), "endpoint": endpoint,
                            "contrast": name, "gain": contrast.get(key),
                            "n": arms.get("q_plus_mark", {}).get("n_anchors", 0),
                        })
    for path in sorted((LONG / "event_offset").glob("*/full_mark/*/decoder_seed*_state_seed*/card.json")):
        card = read(path)
        offset = path.parents[3].name
        for _horizon, arms in card.get("selection", {}).get("arms", {}).items():
            for endpoint in ("grammar", "contact_nll", "stop_bce", "next_bce"):
                state = arms.get("rate_plus_mark", {}).get(endpoint, {}).get("mean")
                comparators = {
                    "state_over_dynamic_rate": arms.get("rate_only", {}).get(endpoint, {}).get("mean"),
                    "state_over_fit_mean": arms.get("period_mean_mark", {}).get(endpoint, {}).get("mean"),
                    "correct_time_over_wrong": arms.get("block_shift_mark", {}).get(endpoint, {}).get("mean"),
                }
                for contrast, base in comparators.items():
                    rows.append({
                        "family": "H2a_event_offset", "scale": f"event_{offset}",
                        "stratum": "all", "subject": card["subject"],
                        "seed": int(card["seed"]), "endpoint": endpoint,
                        "contrast": contrast, "gain": finite_gain(base, state),
                        "n": arms.get("rate_plus_mark", {}).get(endpoint, {}).get("n", 0),
                    })
    for path in sorted((LONG / "physical").glob("*/seizure/*/decoder_seed*_state_seed*/card.json")):
        card = read(path)
        scale = path.parents[3].name
        contrasts = card.get("distance_survival", {}).get("arms", {}).get("registered_contrasts", {})
        for name in ("dynamic_rate_increment", "mark_state_without_rate",
                     "mark_state_increment_over_rate", "correct_time_increment_over_matched_wrong"):
            rows.append({
                "family": "H2b_seizure_hazard", "scale": scale, "stratum": "right_censored",
                "subject": card["subject"],
                "seed": int(path.parent.name.split("state_seed", 1)[1]),
                "endpoint": "person_period_log_score", "contrast": name,
                "gain": contrasts.get(name),
                "n": card.get("distance_survival", {}).get("n_person_period_rows", 0),
            })
    return rows


def aggregate(seed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple, list[float]] = {}
    n_values: dict[tuple, list[int]] = {}
    for row in seed_rows:
        key = tuple(row[name] for name in ("family", "scale", "stratum", "subject", "endpoint", "contrast"))
        if isinstance(row.get("gain"), (int, float)):
            grouped.setdefault(key, []).append(float(row["gain"]))
        n_values.setdefault(key, []).append(int(row.get("n") or 0))
    patient_rows = []
    for key in sorted(n_values):
        values = grouped.get(key, [])
        patient_rows.append({
            "family": key[0], "scale": key[1], "stratum": key[2],
            "subject": key[3], "endpoint": key[4], "contrast": key[5],
            "gain": statistics.median(values) if values else None,
            "n_seeds": len(values), "n_median": statistics.median(n_values[key]),
        })
    cohort_groups: dict[tuple, list[float]] = {}
    for row in patient_rows:
        if row["gain"] is None:
            continue
        key = tuple(row[name] for name in ("family", "scale", "stratum", "endpoint", "contrast"))
        cohort_groups.setdefault(key, []).append(float(row["gain"]))
    cohort_rows = [{
        "family": key[0], "scale": key[1], "stratum": key[2], "endpoint": key[3],
        "contrast": key[4], "n_subjects": len(values),
        "n_positive": sum(value > 0 for value in values),
        "median_gain": statistics.median(values),
    } for key, values in sorted(cohort_groups.items())]
    return patient_rows, cohort_rows


def main() -> int:
    done = LONG / "supervisor" / "queue_done.json"
    if not done.is_file():
        raise RuntimeError("corrected long-horizon queue is not complete")
    reports = LONG / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    seed_rows = collect_seed_rows()
    patient_rows, cohort_rows = aggregate(seed_rows)
    write_csv(reports / "seed_rows.csv", seed_rows)
    write_csv(reports / "patient_rows.csv", patient_rows)
    write_csv(reports / "cohort_rows.csv", cohort_rows)
    summary = {
        "format": "group_event_state_v0_3_5_long_observed_support_summary_v3",
        "queue": read(done), "estimability": read(LONG / "estimability_v3.json"),
        "cohort": cohort_rows,
        "claim_boundary": "exploratory development-prefix results; NOT_ESTIMABLE is not biological absence",
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    atomic_json(reports / "summary.json", summary)
    lines = [
        "# v0.3.5 修订后 H1/H2 长尺度探索报告", "",
        "窗口可跨排除或未记录区间，但未观测秒数不当作没有 IED；状态仅可跨不含发作的十分钟以内短缺口继续传递；每个 horizon 有独立的长留出块。正数表示后一个模型优于前一个模型。", "",
        "| 问题 | 尺度 | 分层 | 端点 | 对比 | 同向患者 | 中位改善 |", "|---|---|---|---|---|---:|---:|",
    ]
    for row in cohort_rows:
        lines.append(
            f"| {row['family']} | {row['scale']} | {row['stratum']} | {row['endpoint']} | {row['contrast']} | "
            f"{row['n_positive']}/{row['n_subjects']} | {row['median_gain']:+.6f} |"
        )
    lines += [
        "", "`all` 与 `no_seizure_crossing` 必须并列读：前者是住院真实轨迹的整体预测，后者才允许解释为纯间期慢状态。",
        "这些是 development-prefix 探索，不是正式确认；不能把不可估或零增量写成生物学不存在。", "",
    ]
    (reports / "plain_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
