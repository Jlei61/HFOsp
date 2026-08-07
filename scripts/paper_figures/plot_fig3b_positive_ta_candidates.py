#!/usr/bin/env python3
"""Generate morphology-aware positive-power candidates for Fig. 3B review."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3b_interictal_ictal_shared_field import (
    DIRECT_EARLY_CORR_MIN,
    EARLIEST_PAIR_MIN_NORM,
    _checkpoint_event,
    _checkpoint_rows,
    _extract_clinical_activation,
    _load_record,
    _morphology_metrics,
    _passes_morphology_candidate,
    _score_audit,
    render,
)
from scripts.plot_topic5_interictal_event_envelope_field import load_frozen


OUT_DIR = (
    ROOT
    / "results/paper-ready-figure/fig3b_interictal_ictal_shared_field"
    / "figures/candidates_positive_ta_morphology"
)
SCHEMA_ID = "fig3b_positive_ta_morphology_candidates_v1"


def _passes_candidate_gate(row: dict) -> bool:
    return _passes_morphology_candidate(row, row)


def _write_readme(out_dir: Path, selected: list[dict]) -> Path:
    lines = [
        "# Fig3-B positive-power TA morphology candidates",
        "",
        "本目录用于目视挑选 Fig3-B 右侧发作场，不替代当前 formal candidate。所有入选发作均满足：",
        "15/15 触点 robust-z 为正、TA 为 maxAB winner、`shared_a_signed>0`，并要求 SCL9/ICL11",
        "两个最早端点不落在低值端、rank 0–3 源区高于 rank 11–14 晚期区、power 与 `−rank`",
        f"直接相关不低于 `{DIRECT_EARLY_CORR_MIN:.2f}`。这些阈值只用于构造目视候选，不是统计 gate。",
        "",
    ]
    for order, row in enumerate(selected, start=1):
        stem = row["stem"]
        lines.extend(
            [
                f"### {stem}.png / .pdf",
                "",
                (
                    f"候选 {order}，E1146 seizure {row['seizure_idx']}；"
                    f"shared A={row['shared_a_signed']:.3f}，"
                    f"direct early-rank corr={row['direct_early_rank_correlation']:.3f}，"
                    f"robust-z={row['power_min_robust_z']:.2f}–{row['power_max_robust_z']:.2f}。"
                ),
                "",
                "**关注点**：目视检查上侧 SCL9、下侧 ICL11/ICL9 是否共同形成左侧源区，以及场是否从左侧向右下逐渐减弱。",
                "",
            ]
        )
    lines.extend(
        [
            "### candidate_summary.json / .csv",
            "",
            "保存全部 25 次 complete/exact 发作的真实 robust-z、TA/TB 分数、局部形态指标和逐项 gate。",
            "",
            "**关注点**：最终例图由人工目视决定；不得把 candidate ranking 写成预注册统计或独立验证。",
        ]
    )
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    ds_sid = str(args.subject).replace("/", "_")
    record, frozen_path = _load_record(ds_sid)
    fz = load_frozen(ds_sid)
    rank_a = np.asarray(fz["rank_a"], dtype=float)
    checkpoint_rows = _checkpoint_rows(ds_sid, len(rank_a))
    payload_by_idx = {}
    audited = []

    for checkpoint_row in checkpoint_rows:
        seizure_idx = int(checkpoint_row["seizure_idx"])
        activation, extraction = _extract_clinical_activation(
            ds_sid, seizure_idx, record
        )
        checkpoint, checkpoint_path = _checkpoint_event(ds_sid, seizure_idx)
        score_audit = _score_audit(record, activation, checkpoint)
        errors = [
            float(value["abs_error"])
            for value in (score_audit.get("checkpoint_comparison") or {}).values()
            if "abs_error" in value
        ]
        max_error = max(errors, default=0.0)
        if max_error > 1e-12:
            raise ValueError(
                f"seizure {seizure_idx}: checkpoint score error {max_error:g}"
            )
        row = dict(checkpoint_row)
        row.update(_morphology_metrics(activation, rank_a))
        row["passes_candidate_gate"] = _passes_candidate_gate(row)
        row["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
        row["checkpoint_max_abs_score_error"] = max_error
        audited.append(row)
        payload_by_idx[seizure_idx] = (activation, extraction, score_audit)

    eligible = [row for row in audited if row["passes_candidate_gate"]]
    eligible.sort(
        key=lambda row: (
            -row["direct_early_rank_correlation"],
            -row["early4_minus_late4_normalized"],
            -row["shared_a_signed"],
            row["seizure_idx"],
        )
    )
    selected = eligible[: int(args.max_candidates)]
    if not selected:
        raise ValueError("no candidate passed the morphology-aware positive-power gate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for order, row in enumerate(selected, start=1):
        seizure_idx = int(row["seizure_idx"])
        activation, extraction, score_audit = payload_by_idx[seizure_idx]
        stem = (
            f"candidate_{order:02d}_{ds_sid}_seizure_{seizure_idx:02d}_"
            "interictal_ictal_shared_field"
        )
        row["candidate_order"] = order
        row["stem"] = stem
        display = render(
            ds_sid,
            fz,
            activation,
            args.output_dir / f"{stem}.png",
            args.output_dir / f"{stem}.pdf",
        )
        sidecar = {
            "schema_id": SCHEMA_ID,
            "status": "Fig3-B morphology candidate for visual review",
            "subject": ds_sid,
            "frozen_record": str(frozen_path.relative_to(ROOT)),
            "candidate": row,
            "contact_order": list(fz["names"]),
            "rank_a": rank_a.tolist(),
            "raw_ictal_robust_z_mean": activation.tolist(),
            "ictal_extraction": extraction,
            "score_audit": score_audit,
            "display": display,
        }
        (args.output_dir / f"{stem}_metadata.json").write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "schema_id": SCHEMA_ID,
        "status": "candidate review only; no formal Fig3-B replacement selected",
        "subject": ds_sid,
        "gate": {
            "all_contacts_positive_robust_z": True,
            "shared_a_signed_positive": True,
            "shared_best_template": "A",
            "direct_early_rank_correlation_min": DIRECT_EARLY_CORR_MIN,
            "early4_minus_late4_normalized_min_exclusive": 0.0,
            "earliest2_min_normalized_min": EARLIEST_PAIR_MIN_NORM,
            "earliest_contacts": ["SCL9", "ICL11"],
            "early_region": "TA ranks 0-3",
            "late_region": "TA ranks 11-14",
        },
        "n_complete_exact_candidates": len(audited),
        "n_pass_candidate_gate": len(eligible),
        "selected_seizure_indices": [int(row["seizure_idx"]) for row in selected],
        "selected": selected,
        "all_audited": audited,
    }
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fieldnames = list(audited[0].keys())
    with (args.output_dir / "candidate_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audited)
    readme = _write_readme(args.output_dir, selected)
    print(f"selected seizures: {[row['seizure_idx'] for row in selected]}")
    print(f"wrote {args.output_dir}")
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
