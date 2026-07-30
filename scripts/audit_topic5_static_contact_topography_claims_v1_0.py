#!/usr/bin/env python3
"""Audit manuscript-facing Topic 5/RNN claims after the bounded closeout.

The audit is intentionally lexical first and semantic second.  It does not
rewrite manuscript prose.  It inventories potentially over-strong phrases and
classifies each occurrence as:

* SAFE_BOUNDARY_OR_NEGATION
* DIFFERENT_EMPIRICAL_CONTRACT
* HISTORICAL_MODEL_STAGE
* UNSAFE_CURRENT_CLAIM
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
CSV_OUT = OUT / "claim_consistency_audit.csv"
JSON_OUT = OUT / "CLAIM_CONSISTENCY_AUDIT.json"
REPORT_OUT = (
    ROOT
    / "docs/archive/topic5/"
    "static_contact_topography_claim_consistency_audit_2026-07-28.md"
)

PRIMARY_SOURCE = (
    ROOT / "docs/paper-draft/figure6_static_contact_topography_bounded_result.md"
)
ACTIVE_EXTRAS = [
    ROOT / "docs/paper-draft/README.md",
    ROOT / "docs/main_figure_plan.md",
    ROOT / "docs/paper_overview.md",
    ROOT / "docs/topic5_seizure_subtyping.md",
]
HISTORICAL_FIG6 = {
    ROOT / "docs/paper-draft/figure6_persistent_path_mode_rnn_bounded_negative.md",
    ROOT
    / "docs/paper-draft/figure6_symmetric_axis_propagation_state_bounded_negative.md",
    ROOT / "docs/paper-draft/figure6_competitive_propagation_rnn_bounded_result.md",
    ROOT / "docs/paper-draft/figure6_rnn_axis_static_transfer_v2_4_bounded_negative.md",
}

TERM_PATTERNS = {
    "replay": re.compile(r"\breplay(?:ed|ing)?\b", re.I),
    "shared field": re.compile(r"\bshared[- ]field\b", re.I),
    "shared scaffold recovery": re.compile(
        r"\b(?:recover(?:ed|y|ing)?|identify|identified)\b.{0,45}"
        r"\b(?:shared|pathological|physical)[- ](?:axis|scaffold)\b",
        re.I,
    ),
    "seizure propagation prediction": re.compile(
        r"\bpredict(?:s|ed|ing|ion)?\b.{0,45}\bseizure propagation\b",
        re.I,
    ),
    "latent-state mechanism": re.compile(
        r"\blatent[- ]state (?:transition|mechanism|dynamics)\b", re.I
    ),
    "ordered history drives transfer": re.compile(
        r"\bordered history\b.{0,45}\b(?:drive|drives|transfer)\b", re.I
    ),
}

SAFE_TOKENS = (
    "not ",
    "no ",
    "without ",
    "cannot ",
    "could not ",
    "does not ",
    "did not ",
    "avoid ",
    "forbidden",
    "historical",
    "superseded",
    "bounded",
    "不",
    "未",
    "无",
    "不能",
    "并非",
    "禁",
    "避免",
    "非 ",
)


def manuscript_paths() -> list[Path]:
    paths = sorted((ROOT / "docs/paper-draft").glob("*.md"))
    for path in ACTIVE_EXTRAS:
        if path not in paths:
            paths.append(path)
    return sorted(set(paths))


def historical_header(path: Path, lines: list[str]) -> bool:
    if path == PRIMARY_SOURCE:
        return False
    header = "\n".join(lines[:25]).lower()
    return "historical" in header or "superseded" in header or "历史" in header


def classify(
    path: Path,
    term: str,
    context: str,
    is_historical: bool,
    section_heading: str,
) -> tuple[str, str]:
    context_lower = context.lower()
    if path in HISTORICAL_FIG6 or is_historical:
        return (
            "HISTORICAL_MODEL_STAGE",
            "file is explicitly marked historical/superseded and is not a current claim source",
        )
    if (
        term == "shared field"
        and path.name
        in {
            "topic5_seizure_subtyping.md",
            "paper_overview.md",
            "main_figure_plan.md",
        }
    ):
        return (
            "DIFFERENT_EMPIRICAL_CONTRACT",
            "phrase refers to the pre-existing empirical field analysis, not the RNN closeout",
        )
    if path == PRIMARY_SOURCE and section_heading.lower() in {
        "terminology ledger",
        "not established",
        "claim-evidence map",
    }:
        return (
            "SAFE_BOUNDARY_OR_NEGATION",
            "phrase is listed in a terminology prohibition or explicit not-established section",
        )
    if any(token in context_lower for token in SAFE_TOKENS):
        return (
            "SAFE_BOUNDARY_OR_NEGATION",
            "phrase occurs in an explicit limitation, negation, or forbidden-claim statement",
        )
    return (
        "UNSAFE_CURRENT_CLAIM",
        "potentially over-strong phrase appears in a current manuscript-facing source",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    scanned = manuscript_paths()
    for path in scanned:
        lines = path.read_text(errors="replace").splitlines()
        is_historical = historical_header(path, lines)
        section_heading = ""
        for line_no, line in enumerate(lines, start=1):
            if line.lstrip().startswith("#"):
                section_heading = line.lstrip("#").strip()
            for term, pattern in TERM_PATTERNS.items():
                if not pattern.search(line):
                    continue
                left = max(0, line_no - 2)
                right = min(len(lines), line_no + 1)
                context = " ".join(part.strip() for part in lines[left:right])
                status, reason = classify(
                    path,
                    term,
                    context,
                    is_historical,
                    section_heading,
                )
                rows.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": line_no,
                        "term": term,
                        "text": line.strip(),
                        "classification": status,
                        "reason": reason,
                    }
                )

    with CSV_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "line",
                "term",
                "text",
                "classification",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    unsafe = [
        row for row in rows if row["classification"] == "UNSAFE_CURRENT_CLAIM"
    ]
    summary = {
        "contract": "topic5_static_contact_topography_claim_audit_v1_0",
        "status": "PASS" if not unsafe else "FAIL_UNSAFE_CURRENT_CLAIM",
        "one_sentence_argument": (
            "Patient-specific interictal contact topography corresponds to "
            "early-ictal energy up to polarity, while ordered GRU shows no "
            "detectable independent heldout or cross-state increment."
        ),
        "n_files_scanned": len(scanned),
        "n_occurrences": len(rows),
        "classification_counts": counts,
        "n_unsafe_current_claims": len(unsafe),
        "unsafe_current_claims": unsafe,
        "primary_manuscript_source": str(PRIMARY_SOURCE.relative_to(ROOT)),
        "audit_csv": str(CSV_OUT.relative_to(ROOT)),
    }
    JSON_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    report_lines = [
        "# Topic 5 static contact topography 全文 claim consistency audit",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 扫描 manuscript-facing 文件：{summary['n_files_scanned']}",
        f"- 命中需审阅短语：{summary['n_occurrences']}",
        f"- unsafe current claims：{summary['n_unsafe_current_claims']}",
        "",
        "## 冻结口径",
        "",
        "> Patient-specific interictal contact topography corresponds to "
        "early-ictal energy up to polarity, while ordered GRU shows no "
        "detectable independent heldout or cross-state increment.",
        "",
        "`abs(rho)` 只表示相同或反向的 contact ordering，不能单独写成 positive "
        "replay、shared signed field 或方向性传播。RNN 在当前论文中是 Supplementary "
        "boundary control，不是主文机制。",
        "",
        "## 分类汇总",
        "",
    ]
    for key in sorted(counts):
        report_lines.append(f"- `{key}`：{counts[key]}")
    report_lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- `{CSV_OUT.relative_to(ROOT)}`",
            f"- `{JSON_OUT.relative_to(ROOT)}`",
            "",
            "逐条文本、行号和判定理由见 CSV。历史模型文件仍保留用于 provenance，但其标题头已"
            "明确标为 superseded/historical，不能作为当前 manuscript source。",
            "",
        ]
    )
    if unsafe:
        report_lines.extend(["## 未解决 unsafe claims", ""])
        for row in unsafe:
            report_lines.append(
                f"- `{row['path']}:{row['line']}` — {row['text']}"
            )
        report_lines.append("")
    REPORT_OUT.write_text("\n".join(report_lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
