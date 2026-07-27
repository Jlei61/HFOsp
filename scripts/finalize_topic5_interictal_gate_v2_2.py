#!/usr/bin/env python3
"""Finalize the v2.2 interictal claims and target-value seal."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_interictal_gate_v2_2 import (  # noqa: E402
    evaluate_interictal_target_gate,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded_report(
    *,
    gate: dict[str, Any],
    sequence: dict[str, Any] | None,
) -> str:
    item_lines = "\n".join(
        f"- `{name}`: {status}"
        for name, status in gate["claim_statuses"].items()
    )
    blockers = "\n".join(f"- {value}" for value in gate["blockers"])
    sequence_text = (
        "未完成"
        if sequence is None
        else (
            f"完成；n={sequence.get('n_patients')}，"
            f"median Markov benefit={sequence.get('median_markov_benefit'):.6g}，"
            f"该结果为 nonblocking sensitivity。"
        )
    )
    return (
        "# Topic 5 v2.2 claim-specific closeout\n\n"
        "## 冻结 gate\n\n"
        f"{item_lines}\n\n"
        "## Target 状态\n\n"
        f"- interictal 四项全部通过：{gate['interictal_pass']}\n"
        f"- exact clinical-onset source metadata 就绪："
        f"{gate['source_metadata_ready']}\n"
        f"- early-ictal values 解封："
        f"{gate['early_ictal_values_unlocked']}\n\n"
        f"- early-ictal transfer："
        f"`{gate['early_ictal_transfer_status']}`\n\n"
        "## 阻断原因\n\n"
        f"{blockers if blockers else '- 无'}\n\n"
        "## All-subject sequence sensitivity\n\n"
        f"{sequence_text}\n\n"
        "## 安全科学口径\n\n"
        "Claim 2 是已执行后失败；Claim 3/4 是按预注册 stop rule 锁定未运行，"
        "不能写成失败。每个 claim 独立报告；未通过或未执行的 claim 不由其他 "
        "endpoint 替代。"
        "若 target 保持 sealed，不报告 early-ictal transfer 的数值或方向，也不使用 "
        "SOZ、能量最高触点、A/B source 或患者级 focus 代替逐发作 clinical-onset "
        "source contacts。\n"
    )


def main() -> None:
    analysis = BASE / "formal/analysis"
    claim2_path = analysis / "CLAIM2_STATUS.json"
    claim2 = json.loads(claim2_path.read_text(encoding="utf-8"))
    if claim2.get("status") != "complete":
        raise SystemExit("Claim 2 is not complete")
    claim3 = _read_optional(analysis / "CLAIM3_STATUS.json")
    claim4 = _read_optional(analysis / "CLAIM4_STATUS.json")
    target = json.loads(
        (BASE / "target_audit/TARGET_METADATA_GATE.json").read_text(
            encoding="utf-8"
        )
    )
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise SystemExit("target values were read before final gate")
    gate = evaluate_interictal_target_gate(
        claim2=claim2,
        claim3=claim3,
        claim4=claim4,
        target_metadata=target,
    )
    sequence = _read_optional(analysis / "ALL_SUBJECT_SEQUENCE_STATUS.json")
    if sequence is None or sequence.get("status") != "complete":
        raise SystemExit("all-subject sequence sensitivity is incomplete")
    payload = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "complete",
        **gate,
        "sequence_sensitivity_status": sequence["status"],
        "claim1_predictive_adequacy": (
            (_read_optional(analysis / "CLAIM1_STATUS.json") or {}).get(
                "claim1_sequence_predictability", "NOT_RUN"
            )
        ),
        "target_values_read": False,
    }
    atomic_json(analysis / "INTERICTAL_CLAIM_SUMMARY.json", payload)
    unlock_path = BASE / "formal/EARLY_ICTAL_VALUES_UNLOCKED.json"
    if gate["early_ictal_values_unlocked"]:
        atomic_json(
            unlock_path,
            {
                "status": "UNLOCKED",
                "reason": (
                    "four frozen interictal gates passed and exact per-seizure "
                    "clinical-onset source metadata is ready"
                ),
                "target_values_read": False,
            },
        )
    elif unlock_path.exists():
        raise RuntimeError(
            "stale early-ictal unlock exists while current gate is sealed"
        )
    report = _bounded_report(gate=gate, sequence=sequence)
    (analysis / "CLAIM_SPECIFIC_CLOSEOUT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
