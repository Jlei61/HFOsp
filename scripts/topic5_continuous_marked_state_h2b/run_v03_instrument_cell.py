#!/usr/bin/env python3
"""Run interictal-only H2b v0.3 instrument diagnostics for one cell."""
from __future__ import annotations

import argparse
import csv
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


PRODUCER_SCRIPT = Path(__file__).resolve()
INSTRUMENT_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_instrument.py"
NUISANCE_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_nuisance.py"
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _past_seizure_onsets(path: Path, subject: str) -> np.ndarray:
    """Read only verified onset times; Q6 uses each onset only after it occurred."""
    if not path.is_file():
        return np.empty(0, dtype=np.float64)
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("subject")) != str(subject):
                continue
            if str(row.get("matched", "")).lower() not in {"true", "1"}:
                continue
            if str(row.get("ambiguous", "")).lower() in {"true", "1"}:
                continue
            values.append(float(row["onset_epoch"]))
    return np.unique(np.asarray(values, dtype=np.float64))


def _resolve_interictal_design(
    subject: str, *, v02_root: Path, result_root: Path,
) -> tuple[Path, Path, Path, dict]:
    r1_root = Path(
        "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/"
        "continuous_marked_state/r1"
    )
    candidates = (
        ("v0_3_hash_verified_rebuild", result_root / "upstream_r1_2",
         result_root / "manifests/upstream_rebuild" / f"{subject}.json"),
        ("v0_2_hash_verified_rebuild", v02_root / "upstream_r1_2",
         v02_root / "manifests/upstream_rebuild" / f"{subject}.json"),
        ("frozen_r1_7a_upstream", r1_root / "r1_7a/upstream_r1_2", None),
    )
    frozen_r17b_manifest = (
        r1_root / "r1_7b_cohort_extension/cache" / subject / "manifest.json"
    )
    for route, root, audit_path in candidates:
        design_path = root / "cache" / subject / "full_design.npz"
        manifest_path = root / "cache" / subject / "manifest.json"
        embedding_path = root / "cache" / subject / "explicit_embedding.npy"
        if not all(path.is_file() for path in (
            design_path, manifest_path, embedding_path,
        )):
            continue
        manifest = _json(manifest_path)
        if manifest.get("status") != "COMPLETE":
            continue
        if sha256_file(design_path) != str(manifest.get("design_sha256")):
            raise ValueError(f"{subject}: interictal design SHA256 drift")
        if sha256_file(embedding_path) != str(
            manifest.get("explicit_embedding_sha256")
        ):
            raise ValueError(f"{subject}: interictal embedding SHA256 drift")
        audit = None
        if audit_path is not None:
            if not audit_path.is_file():
                continue
            audit = _json(audit_path)
            if audit.get("status") != "COMPLETE" or not all(
                bool(value) for value in audit.get("checks", {}).values()
            ):
                raise ValueError(f"{subject}: upstream rebuild equivalence failed")
            artifact = audit.get("artifacts", {})
            if str(Path(artifact.get("design", "")).resolve()) != str(
                design_path.resolve()
            ) or artifact.get("design_sha256") != sha256_file(design_path):
                raise ValueError(f"{subject}: rebuild audit/design disagreement")
        if frozen_r17b_manifest.is_file():
            expected = _json(frozen_r17b_manifest)
            if expected.get("design_sha256") != sha256_file(design_path):
                raise ValueError(f"{subject}: design differs from frozen R1.7 source")
        provenance = {
            "route": route,
            "design_manifest": str(manifest_path),
            "design_manifest_sha256": sha256_file(manifest_path),
            "rebuild_audit": str(audit_path) if audit_path is not None else None,
            "rebuild_audit_sha256": (
                sha256_file(audit_path) if audit_path is not None else None
            ),
            "frozen_r1_7b_cache_manifest": (
                str(frozen_r17b_manifest) if frozen_r17b_manifest.is_file() else None
            ),
            "frozen_r1_7b_cache_manifest_sha256": (
                sha256_file(frozen_r17b_manifest)
                if frozen_r17b_manifest.is_file() else None
            ),
        }
        return design_path, manifest_path, embedding_path, provenance
    raise FileNotFoundError(
        f"{subject}: no hash-verified interictal design independent of seizure support"
    )


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


def run(subject: str, seed: int, *, v02_root: Path, result_root: Path,
        n_null_permutations: int = 100) -> dict:
    contract_path = result_root / "analysis_contract.json"
    assert_frozen_contract_matches(_json(contract_path))
    inventory_path = v02_root / "manifests/r1_7_checkpoint_inventory.json"
    inventory = _json(inventory_path)
    selected = [entry for entry in inventory["entries"]
                if str(entry["subject"]) == str(subject)
                and int(entry["seed"]) == int(seed)]
    if len(selected) != 1:
        raise ValueError(f"inventory identity is not unique: {subject}/seed_{seed}")
    entry = selected[0]
    if not bool(entry.get("checkpoint_available")):
        raise ValueError(f"checkpoint unavailable: {subject}/seed_{seed}")
    design_path, manifest_path, embedding_path, design_provenance = (
        _resolve_interictal_design(
            subject, v02_root=v02_root, result_root=result_root,
        )
    )
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
    seizure_crosswalk = v02_root / "manifests/seizure_crosswalk.csv"
    past_onsets = _past_seizure_onsets(seizure_crosswalk, subject)
    timezone_name = (
        "Europe/Berlin" if subject.startswith("epilepsiae_") else "Asia/Shanghai"
    )
    trace = scan_interictal_state(model, design, embedding, device="cpu")
    summary = summarise_instrument_trace(
        model,
        design,
        trace,
        state_start=float(state_support["state_start"]),
        state_stop=float(state_support["state_stop"]),
        interictal_persistent_minus_memoryless_joint=float(joint),
        embedding=embedding,
        rng_seed=int(seed) + int.from_bytes(subject.encode("utf-8"), "little") % 1_000_003,
        timezone_name=timezone_name,
        n_null_permutations=int(n_null_permutations),
        past_seizure_onsets=past_onsets,
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
        "revision": "h2b_v0_3_interictal_instrument_cell_v4",
        "supersedes_revision": "h2b_v0_3_interictal_instrument_cell_v3",
        "h2b_revision": H2B_V0_3_REVISION,
        "created_utc": utc_now(),
        "subject": subject,
        "seed": int(seed),
        "instrument_config": {
            "n_null_permutations": int(n_null_permutations),
        },
        "diagnostics": summary,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "source": {
            "producer_script": str(PRODUCER_SCRIPT),
            "producer_script_sha256": sha256_file(PRODUCER_SCRIPT),
            "instrument_module": str(INSTRUMENT_MODULE),
            "instrument_module_sha256": sha256_file(INSTRUMENT_MODULE),
            "nuisance_module": str(NUISANCE_MODULE),
            "nuisance_module_sha256": sha256_file(NUISANCE_MODULE),
            "checkpoint": provenance,
            "checkpoint_result_sha256": sha256_file(entry["result_path"]),
            "design_path": str(design_path),
            "design_sha256": sha256_file(design_path),
            "design_manifest_path": str(manifest_path),
            "design_manifest_sha256": sha256_file(manifest_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": sha256_file(embedding_path),
            "design_resolution": design_provenance,
            "v0_2_inventory_sha256": sha256_file(inventory_path),
            "past_seizure_crosswalk_path": str(seizure_crosswalk),
            "past_seizure_crosswalk_sha256": (
                sha256_file(seizure_crosswalk) if seizure_crosswalk.is_file() else None
            ),
        },
        "data_scope": (
            "state and prediction targets use interictal TRAIN/D_state only; "
            "verified past seizure onsets enter Q6 only as a causal nuisance"
        ),
        "past_seizure_nuisance_read": bool(len(past_onsets)),
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
    parser.add_argument("--n-null-permutations", type=int, default=100)
    args = parser.parse_args()
    result = run(
        args.subject, args.seed,
        v02_root=args.v0_2_root.resolve(), result_root=args.result_root.resolve(),
        n_null_permutations=int(args.n_null_permutations),
    )
    print(result["status"], result["subject"], result["seed"],
          result["diagnostics"]["status"])


if __name__ == "__main__":
    main()
