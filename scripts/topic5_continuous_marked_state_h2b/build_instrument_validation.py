#!/usr/bin/env python3
"""Bind the three preregistered H2b instrument checks into one audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.contract import (
    RESULT_ROOT, atomic_json, sha256_file, utc_now,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path = RESULT_ROOT) -> dict:
    root = Path(root).resolve()
    risk_audit_path = root / "fits/risk_probe_instrument/risk_probe_machine_audit.json"
    permutation_path = root / "fits/risk_probe_instrument/time_label_permutation.json"
    phenotype_path = (
        root / "fits/phenotype_transfer_instrument/phenotype_transfer_machine_audit.json"
    )
    causality_path = (
        root / "state_cache/instrument/epilepsiae_384/seed_1/causality_perturbation.json"
    )
    risk = _read(risk_audit_path)
    permutation = _read(permutation_path)
    phenotype = _read(phenotype_path)
    causality = _read(causality_path)

    positive = risk.get("positive_synthetic") or {}
    positive_pass = (
        positive.get("status") == "PASS"
        and float(positive.get("state_minus_observation_conditional_log_loss")) < 0
    )
    permutation_pass = (
        int(permutation.get("n_permutations", 0)) >= 100
        and float(permutation.get("null_q025")) < 0.0
        and float(permutation.get("null_q975")) > 0.0
        and abs(float(permutation.get("null_median"))) < 0.15
    )
    causality_pass = (
        causality.get("bitwise_equal_with_nan") is True
        and int(causality.get("n_post_query_observations_perturbed", 0)) > 0
        and causality.get("state_extraction_source_sha256") == sha256_file(
            REPO_ROOT / "src/topic5_continuous_marked_state_h2b/state_extraction.py"
        )
    )
    phenotype_synthetic = phenotype.get("positive_synthetic") or {}
    phenotype_pass = phenotype_synthetic.get("status") == "PASS"
    checks = {
        "positive_synthetic": {
            **positive,
            "status": "PASS" if positive_pass else "FAIL",
        },
        "time_label_permutation": {
            "status": "PASS" if permutation_pass else "FAIL",
            "n_permutations": permutation.get("n_permutations"),
            "null_median": permutation.get("null_median"),
            "null_q025": permutation.get("null_q025"),
            "null_q975": permutation.get("null_q975"),
            "scope": "independent positive synthetic; engineering check only",
        },
        "causality_perturbation": {
            **causality,
            "status": "PASS" if causality_pass else "FAIL",
        },
        "phenotype_positive_synthetic": {
            **phenotype_synthetic,
            "status": "PASS" if phenotype_pass else "FAIL",
        },
    }
    payload = {
        "status": "PASS" if all(
            row["status"] == "PASS" for row in checks.values()
        ) else "FAIL",
        "created_utc": utc_now(),
        **checks,
        "engineering_check_only": True,
        "scientific_h2b_evidence": False,
        "source_artifacts": {
            str(path): sha256_file(path)
            for path in (
                risk_audit_path, permutation_path, phenotype_path, causality_path
            )
        },
        "producer_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    output = root / "reports/instrument_validation.json"
    atomic_json(output, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
