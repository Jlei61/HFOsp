#!/usr/bin/env python3
"""Fail closed while freezing the completed SIG v2.1 evidence stack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "SIG_V2_1_IDENTIFIABILITY_STATE.json"
)
OUTPUT = STATE.with_name("SIG_V2_1_ACCEPTANCE.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE_BOUNDED_SINGLE_GRAPH_DEVELOPMENT":
        raise RuntimeError("v2.1 state is not complete")
    checks = state.get("checks") or {}
    failed = sorted(key for key, value in checks.items() if not bool(value))
    if failed:
        raise RuntimeError(f"v2.1 state has failed checks: {failed}")
    for name, record in (state.get("artifacts") or {}).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise RuntimeError(f"missing v2.1 dependency {name}: {path}")
        if sha256(path) != str(record.get("sha256", "")):
            raise RuntimeError(f"v2.1 dependency hash drift: {name}")
    payload = {
        "contract": "topic5_stable_interaction_identifiability_v2_1_acceptance",
        "status": "ACCEPTED_AND_FROZEN",
        "accepted_object": (
            "single fixed contact-feedback graph under the v2.1 development "
            "contract"
        ),
        "scientific_verdict": state["scientific_verdict"],
        "safe_claim": state["safe_claim"],
        "carry_forward_to_v2_2": [
            "contact feedback can improve within-event conditional generation",
            "single-fixed-graph structure-specific evidence was absent in four calibrated pilot patients",
            "two pilot patients remain unadjudicated",
        ],
        "not_carried_forward": [
            "rank step as long-timescale recurrence",
            "seen-distribution prediction as a structure gate",
            "SNN as an RNN gate",
            "any claim that stable or time-varying pathological structure is absent",
        ],
        "v2_2_relation": (
            "NEW_EVENT_INDEXED_SCIENTIFIC_OBJECT_NOT_A_CAPACITY_EXTENSION_OR_RESCUE"
        ),
        "state_path": str(STATE),
        "state_sha256": sha256(STATE),
        "source_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
