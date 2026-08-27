#!/usr/bin/env python3
"""Write the final fail-closed H2b machine audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.audit import build_machine_audit
from src.topic5_continuous_marked_state_h2b.contract import RESULT_ROOT, atomic_json


def main() -> None:
    output = RESULT_ROOT / "reports/machine_audit.json"
    payload = build_machine_audit(result_root=RESULT_ROOT, repo_root=REPO_ROOT)
    atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "scientific_claim_eligible": payload["scientific_claim_eligible"],
        "r1_7_integration_status": payload["r1_7_integration_status"],
        "failed_checks": payload["failed_checks"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
