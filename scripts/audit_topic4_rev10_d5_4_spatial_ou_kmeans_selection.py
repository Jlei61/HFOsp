"""Adjudicate the fresh D5.4 KMeans selection networks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev10_d5_3_spatial_ou_kmeans_grid import audit_candidate
from scripts.run_topic4_rev9l_forced_source_worker import _sha256


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_4_spatial_ou_kmeans_selection.json"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def adjudicate(local, permuted, off, *, minimum_purity):
    matrix_ok = bool(local["signed_patient_margin"] > 0)
    support_ok = bool(local["evaluable"])
    purity = (
        None if not support_ok
        else local["balanced_kmeans"]["purity_median"]
    )
    purity_ok = bool(purity is not None and purity > float(minimum_purity))
    if support_ok and purity_ok and matrix_ok:
        status = "REV10D5_4_FRESH_SELECTION_CONFIRMS_KMEANS_CANDIDATE"
    elif not support_ok:
        status = "REV10D5_4_FRESH_SELECTION_SUPPORT_FAIL"
    elif not purity_ok:
        status = "REV10D5_4_FRESH_SELECTION_KMEANS_NOT_REPLICATED"
    else:
        status = "REV10D5_4_FRESH_SELECTION_PATIENT_GEOMETRY_FAIL"
    locality = None
    if local["evaluable"] and permuted["evaluable"]:
        locality = float(
            local["balanced_kmeans"]["purity_median"]
            - permuted["balanced_kmeans"]["purity_median"]
        )
    return {
        "status": status,
        "fresh_local_support_evaluable": support_ok,
        "fresh_local_purity_exceeds_d5_2_anchor": purity_ok,
        "fresh_local_patient_matrix_sign_geometry": matrix_ok,
        "minimum_fresh_purity": float(minimum_purity),
        "local_minus_permuted_balanced_purity": locality,
        "local": local, "permuted": permuted, "off": off,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "selection_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != "REV10D5_4_SPATIAL_OU_KMEANS_SELECTION_FROZEN":
        raise RuntimeError("D5.4 manifest is not frozen")
    if summary.get("status") != "REV10R_RETURNED_ONLY_SELECTION_COMPLETE":
        raise RuntimeError("D5.4 selection aggregate is incomplete")
    candidates = {
        row["candidate_id"]: row for row in manifest["candidate_set"]["candidates"]
    }
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    contract = config["search"]["kmeans_selection"]
    audited = {
        candidate_id: audit_candidate(
            config_path, root, candidate, aggregate[candidate_id], contract,
        )
        for candidate_id, candidate in candidates.items()
    }
    selected = manifest["selection_freeze"]["selected_nonzero_candidate_id"]
    permuted = manifest["selection_freeze"]["matched_permuted_candidate_id"]
    payload = adjudicate(
        audited[selected], audited[permuted], audited["edge_noop"],
        minimum_purity=contract["minimum_fresh_purity"],
    )
    payload.update({
        "selected_local_candidate_id": selected,
        "matched_permuted_candidate_id": permuted,
        "patient_matched_direction_purity_q05": contract[
            "patient_benchmark_q05"
        ],
        "locality_contrast_is_diagnostic_not_gate": True,
        "claim_boundary": (
            "fresh selection networks but development-only; passing authorizes "
            "an untouched confirmation, not a patient or Fig4 success claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
        },
    })
    output = root / "selection_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "local_balanced_purity": payload["local"]["balanced_kmeans"].get(
            "purity_median"
        ),
        "local_signed_patient_margin": payload["local"][
            "signed_patient_margin"
        ],
        "local_minus_permuted_purity": payload[
            "local_minus_permuted_balanced_purity"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
