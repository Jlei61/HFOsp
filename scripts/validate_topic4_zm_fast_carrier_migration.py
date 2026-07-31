#!/usr/bin/env python3
"""Real-state 500-ms Arm-A exact-continuation gate for Phase D."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import topic4_zm_checkpoint as CK  # noqa: E402
from src import topic4_zm_fast_carrier_contract as C  # noqa: E402
from src import topic4_zm_fast_carrier_state as S  # noqa: E402
from src import topic4_zm_fast_carrier_runtime as RT  # noqa: E402
from scripts import run_topic4_zm_branch_decision as R  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "results/topic4_sef_hfo/zm_fast_carrier_repair/phaseD_input_manifest_v1_4.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/topic4_sef_hfo/zm_fast_carrier_repair/armA_migration_equivalence.json"
)
DURATION_MS = 500.0


def _canonical_sha(payload: dict) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_once(ctx: dict, state: dict) -> tuple[dict, dict, str]:
    slow = R.make_slow(ctx)
    controller = CK.ZMCheckpoint(
        initial_state=state,
        return_final_state=True,
        dump_ext=True,
    )
    result = R.run_segment(
        ctx,
        slow,
        DURATION_MS,
        ckpt=controller,
        fresh_rng=True,
        dump_i_spikes=True,
    )
    if result.get("runaway_early_stop_ms") is not None:
        raise RuntimeError("Arm-A parity continuation truncated as runaway")
    if controller.final_truncated or controller.final_state is None:
        raise RuntimeError("Arm-A parity continuation has no valid final state")
    return (
        S.fingerprint_observables(result),
        S.fingerprint_slow_traces(slow),
        CK.state_hash(controller.final_state),
    )


def _write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite different result: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("refusing production gate without --confirm-run")

    manifest = json.loads(args.input.read_text())
    C.validate_input_manifest(manifest, ROOT)
    source_seed = int(manifest["source"]["seed"])
    if source_seed != 1:
        raise RuntimeError(f"unexpected source seed: {source_seed}")
    ctx = RT.build_source_locked_context(ROOT, manifest, R)
    rows = []
    for row in manifest["source_panel"]:
        row_id = (row["bin_name"], row["fast_phase"])
        source, _ = CK.load_state_npz(
            ROOT / row["path"],
            expected_config_sha=manifest["source"]["canonical_config_sha"],
            expected_engine_sha=row["source_state_manifest"]["engine_sha"],
            expected_dt=manifest["source"]["dt_ms"],
        )
        migrated, transformation = S.load_and_migrate(
            ROOT,
            manifest,
            row_id=row_id,
            contract_already_validated=True,
        )
        old_obs, old_traces, old_final = _run_once(ctx, source)
        new_obs, new_traces, new_final = _run_once(ctx, migrated)
        S.require_exact_continuation(old_obs, new_obs, label=f"{row_id} observables")
        S.require_exact_continuation(old_traces, new_traces, label=f"{row_id} slow traces")
        S.require_exact_continuation(
            {"state_hash": old_final},
            {"state_hash": new_final},
            label=f"{row_id} final state",
        )
        rows.append(
            {
                "bin_name": row_id[0],
                "fast_phase": row_id[1],
                "source_state_hash": transformation["source_state_hash"],
                "migrated_state_hash": transformation["migrated_state_hash"],
                "transformation_sha256": _canonical_sha(transformation),
                "observable_fingerprints": new_obs,
                "slow_trace_fingerprints": new_traces,
                "final_state_hash": new_final,
                "exact_500ms_continuation": True,
            }
        )

    body = {
        "schema": "zm_fast_carrier_armA_migration_equivalence_v1_2026-07-31",
        "input_path": str(args.input.resolve().relative_to(ROOT)),
        "input_file_sha256": C.sha256_file(args.input),
        "input_manifest_sha256": manifest["manifest_sha256"],
        "runtime_git_sha": RT.git_sha(ROOT),
        "duration_ms": DURATION_MS,
        "comparison": "byte_exact_all_observables_slow_traces_and_final_state",
        "excluded_non_scientific_fields": ["wall_s"],
        "all_rows_exact": all(row["exact_500ms_continuation"] for row in rows),
        "rows": rows,
        "production_authorized": False,
    }
    payload = {**body, "manifest_sha256": _canonical_sha(body)}
    _write_once(args.output, payload)
    print(args.output)
    print(payload["manifest_sha256"])


if __name__ == "__main__":
    main()
