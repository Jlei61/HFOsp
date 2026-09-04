#!/usr/bin/env python3
"""Aggregate the complete v0.3.5 event-input source attribution experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import OUTPUT_ROOT, atomic_json  # noqa: E402

ATTR = OUTPUT_ROOT / "source_attribution"
VIEWS = ("times_only", "spatial_only", "waveform_only", "multiband_only", "mark_shuffle")
GRAMMAR_METRICS = ("grammar", "next_bce", "stop_bce", "contact_nll")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_cards(view: str):
    for path in sorted((ATTR / view / "full_mark").glob("*/decoder_seed*_state_seed*/card.json")):
        subject, tag = path.parents[1].name, path.parent.name
        full = OUTPUT_ROOT / "full_mark_final" / subject / tag / "card.json"
        if full.is_file():
            yield subject, tag, read(path), read(full)


def grammar_rows() -> list[dict]:
    rows = []
    for view in VIEWS:
        for subject, tag, ablated, full in paired_cards(view):
            for horizon, full_arms in full["selection"]["arms"].items():
                view_arms = ablated["selection"]["arms"][horizon]
                for metric in GRAMMAR_METRICS:
                    full_loss = float(full_arms["rate_plus_mark"][metric]["mean"])
                    view_loss = float(view_arms["rate_plus_mark"][metric]["mean"])
                    rows.append({
                        "view": view, "subject": subject, "tag": tag, "horizon": horizon,
                        "metric": metric, "full_loss": full_loss, "view_loss": view_loss,
                        "full_gain_over_view": view_loss - full_loss,
                        "view_selected_epoch": int(ablated["selected_epoch"]),
                        "full_selected_epoch": int(full["selected_epoch"]),
                    })
    return rows


def functional_rows() -> list[dict]:
    rows = []
    for view in VIEWS:
        for state_path in sorted((ATTR / view / "full_mark").glob("*/decoder_seed*_state_seed*/card.json")):
            subject, tag = state_path.parents[1].name, state_path.parent.name
            vp = ATTR / view / "functional_readouts" / subject / tag / "card.json"
            fp = OUTPUT_ROOT / "functional_readouts_final" / subject / tag / "card.json"
            if not (vp.is_file() and fp.is_file()):
                continue
            vd, fd = read(vp), read(fp)
            for family in ("event_horizons", "physical_horizons"):
                for horizon, endpoints in fd.get(family, {}).items():
                    for endpoint, full_result in endpoints.items():
                        view_result = vd.get(family, {}).get(horizon, {}).get(endpoint)
                        if not view_result:
                            continue
                        full_loss = full_result.get("q_plus_state", {}).get("selection_loss")
                        view_loss = view_result.get("q_plus_state", {}).get("selection_loss")
                        if full_loss is None or view_loss is None:
                            continue
                        rows.append({
                            "view": view, "subject": subject, "tag": tag, "family": family,
                            "horizon": horizon, "endpoint": endpoint,
                            "full_loss": float(full_loss), "view_loss": float(view_loss),
                            "full_gain_over_view": float(view_loss) - float(full_loss),
                        })
    return rows


def patient_summary(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys) + (row["subject"],)
        grouped.setdefault(key, []).append(float(row["full_gain_over_view"]))
    return [
        {**dict(zip(keys + ("subject",), key)), "gain": statistics.median(values), "n_seeds": len(values)}
        for key, values in sorted(grouped.items())
    ]


def cohort_summary(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(float(row["gain"]))
    return [
        {**dict(zip(keys, key)), "n_subjects": len(values),
         "n_full_better": sum(value > 0 for value in values),
         "median_full_gain_over_view": statistics.median(values),
         "patient_values": values}
        for key, values in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    done = ATTR / "supervisor" / "queue_done.json"
    if not done.is_file():
        raise RuntimeError("source attribution queue is not complete")
    queue = read(done)
    grammar = grammar_rows(); functional = functional_rows()
    gp = patient_summary(grammar, ("view", "horizon", "metric"))
    fp = patient_summary(functional, ("view", "family", "horizon", "endpoint"))
    summary = {
        "format": "group_event_state_v0_3_5_source_attribution_summary_v1",
        "queue_status": queue["status"], "queue_failures": queue.get("failed", []),
        "semantics": {
            "positive_full_gain_over_view": "aligned full event content has lower held-out loss than the named input view",
            "times_only": "exact event times and q(t), with no event mark payload",
            "mark_shuffle": "same payload distribution, causally wrong within-segment time alignment",
        },
        "grammar": cohort_summary(gp, ("view", "horizon", "metric")),
        "functional": cohort_summary(fp, ("view", "family", "horizon", "endpoint")),
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    report = ATTR / "reports"
    write_csv(report / "grammar_seed_rows.csv", grammar)
    write_csv(report / "grammar_patient_rows.csv", gp)
    write_csv(report / "functional_seed_rows.csv", functional)
    write_csv(report / "functional_patient_rows.csv", fp)
    atomic_json(report / "summary.json", summary)
    lines = [
        "# v0.3.5 群体事件信息来源消融报告", "",
        "本实验保持事件时刻、q(t)、目标、成熟 contact decoder 和训练配方不变，只改变每次群体事件写入状态的内容。正数表示完整且时刻对齐的事件内容更好。", "",
        "## Contact/STOP grammar", "",
        "| 输入对照 | 未来事件数 | 指标 | 完整内容更好 | 患者中位增益 |", "|---|---:|---|---:|---:|",
    ]
    for row in summary["grammar"]:
        lines.append(f"| {row['view']} | {row['horizon']} | {row['metric']} | {row['n_full_better']}/{row['n_subjects']} | {row['median_full_gain_over_view']:+.6f} |")
    lines += ["", "## 连续事件形态读出", "", "完整逐端点结果见 `functional_patient_rows.csv`；该表不把不同量纲端点强行平均。", "",
              "`times_only` 回答事件内容是否超出发生时刻；`mark_shuffle` 回答同一批内容是否必须放在正确时刻。单一来源臂用于定位空间、波形或多频带信息是否足以承载增量。", ""]
    (report / "plain_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report), "grammar_rows": len(grammar), "functional_rows": len(functional)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
