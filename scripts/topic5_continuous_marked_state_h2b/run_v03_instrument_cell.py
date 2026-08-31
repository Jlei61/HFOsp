#!/usr/bin/env python3
"""Run interictal-only H2b v0.3 instrument diagnostics for one cell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    H2B_V0_3_REVISION,
    assert_safe_output_path,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.state_extraction import (  # noqa: E402
    load_frozen_r16_checkpoint,
)
from src.topic5_continuous_marked_state_h2b.v03_contract import (  # noqa: E402
    assert_frozen_contract_matches,
)
from src.topic5_continuous_marked_state_h2b.v03_instrument import (  # noqa: E402
    scan_interictal_state,
    summarise_instrument_trace,
    trace_npz_payload,
)
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(subject: str, seed: int, *, v02_root: Path, result_root: Path) -> dict:
    contract_path = result_root / "analysis_contract.json"
    assert_frozen_contract_matches(_json(contract_path))
    inventory_path = v02_root / "manifests/r1_7_checkpoint_inventory.json"
    support_path = v02_root / "manifests/support_census.json"
    inventory = _json(inventory_path)
    support = _json(support_path)
    selected = [entry for entry in inventory["entries"]
                if str(entry["subject"]) == str(subject)
                and int(entry["seed"]) == int(seed)]
    if len(selected) != 1:
        raise ValueError(f"inventory identity is not unique: {subject}/seed_{seed}")
    entry = selected[0]
    if not bool(entry.get("checkpoint_available")):
        raise ValueError(f"checkpoint unavailable: {subject}/seed_{seed}")
    support_rows = [row for row in support["patient_rows"]
                    if str(row["subject"]) == str(subject)]
    if len(support_rows) != 1:
        raise ValueError(f"support identity is not unique: {subject}")
    support_row = support_rows[0]
    design_path = Path(support_row["upstream_design_path"]).resolve()
    manifest_path = Path(support_row["upstream_design_manifest_path"]).resolve()
    design_manifest = _json(manifest_path)
    embedding_path = design_path.with_name("explicit_embedding.npy")
    if sha256_file(design_path) != str(design_manifest["design_sha256"]):
        raise ValueError("frozen interictal design SHA256 drift")
    if sha256_file(embedding_path) != str(design_manifest["explicit_embedding_sha256"]):
        raise ValueError("frozen interictal embedding SHA256 drift")
    model, provenance = load_frozen_r16_checkpoint(
        entry["checkpoint_path"],
        expected_sha256=entry["checkpoint_sha256"],
        expected_subject=subject,
        expected_seed=seed,
        require_stable_result=False,
        require_complete_result=True,
        device="cpu",
    )
    checkpoint_result = _json(Path(entry["result_path"]))
    d_state = checkpoint_result["d_state"]
    state_support = d_state["support"]
    joint = d_state["persistent_minus_memoryless"]["joint_nll_per_event"]
    design = load_full_design(design_path)
    embedding = np.load(embedding_path, mmap_mode="r")
    trace = scan_interictal_state(model, design, embedding, device="cpu")
    summary = summarise_instrument_trace(
        model,
        trace,
        state_start=float(state_support["state_start"]),
        state_stop=float(state_support["state_stop"]),
        interictal_persistent_minus_memoryless_joint=float(joint),
    )
    output = result_root / "instrument/by_cell" / subject / f"seed_{seed}"
    trace_path = output / "interictal_d_state_trace.npz"
    _atomic_npz(
        trace_path,
        **trace_npz_payload(
            trace,
            state_start=float(state_support["state_start"]),
            state_stop=float(state_support["state_stop"]),
        ),
    )
    payload = {
        "status": "COMPLETE",
        "revision": "h2b_v0_3_interictal_instrument_cell_v1",
        "h2b_revision": H2B_V0_3_REVISION,
        "created_utc": utc_now(),
        "subject": subject,
        "seed": int(seed),
        "diagnostics": summary,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "source": {
            "checkpoint": provenance,
            "checkpoint_result_sha256": sha256_file(entry["result_path"]),
            "design_path": str(design_path),
            "design_sha256": sha256_file(design_path),
            "design_manifest_path": str(manifest_path),
            "design_manifest_sha256": sha256_file(manifest_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": sha256_file(embedding_path),
            "v0_2_inventory_sha256": sha256_file(inventory_path),
            "v0_2_support_sha256": sha256_file(support_path),
        },
        "data_scope": "interictal TRAIN and D_state only",
        "seizure_risk_outcome_read": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "seizure_gradient_path": False,
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
    }
    atomic_json(output / "instrument_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    args = parser.parse_args()
    result = run(
        args.subject, args.seed,
        v02_root=args.v0_2_root.resolve(), result_root=args.result_root.resolve(),
    )
    print(result["status"], result["subject"], result["seed"],
          result["diagnostics"]["status"])


if __name__ == "__main__":
    main()
