#!/usr/bin/env python3
"""Run the CPU-only E384 H2b development-instrument pilot end to end."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load cuda_env's compatible C++ runtime before pandas native extensions.
import torch as _torch  # noqa: F401
import pandas as pd

from src.topic5_continuous_marked_state_h2b.contract import (
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.pilot import (
    E384_SUBJECT,
    build_e384_risk_table,
    prepare_e384_query_inputs,
    state_cache_to_anchor_frame,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable


SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-root", type=Path, default=SOURCE_REPO)
    parser.add_argument("--controls-per-case", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-state-cache", action="store_true",
        help="reuse hash-verified COMPLETE state caches while overwriting downstream probes",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    inputs = prepare_e384_query_inputs(
        source_repo_root=args.source_repo_root,
        result_root=RESULT_ROOT,
    )
    if args.prepare_only:
        print(json.dumps(inputs, ensure_ascii=False, indent=2))
        return

    checkpoint_inventory_path = RESULT_ROOT / "manifests/state_checkpoint_inventory.json"
    checkpoints = json.loads(checkpoint_inventory_path.read_text())
    query_path = Path(inputs["query_path"])
    exclusion_path = Path(inputs["global_exclusion_path"])
    coverage_path = (
        Path(args.source_repo_root).resolve()
        / "results/epi_prssm/continuous_marked_state/r1/r1_2/coverage"
        / f"{E384_SUBJECT}.npz"
    )
    coverage = CoverageTable.load(coverage_path)

    frames = []
    cache_rows = []
    for entry in checkpoints["entries"]:
        seed = int(entry["seed"])
        cache = RESULT_ROOT / "state_cache" / E384_SUBJECT / f"seed_{seed}" / "states.npz"
        if cache.exists() and (not args.overwrite or args.reuse_state_cache):
            manifest = cache.with_suffix(".manifest.json")
            if not manifest.exists():
                raise FileExistsError(f"state cache lacks manifest: {cache}")
        else:
            command = [
                sys.executable,
                "scripts/topic5_continuous_marked_state_h2b/extract_states.py",
                "--subject", E384_SUBJECT,
                "--seed", str(seed),
                "--checkpoint", str(entry["checkpoint_path"]),
                "--checkpoint-sha256", str(entry["checkpoint_sha256_expected"]),
                "--queries", str(query_path),
                "--global-exclusions", str(exclusion_path),
                "--source-repo-root", str(Path(args.source_repo_root).resolve()),
                "--output", str(cache),
                "--device", "cpu",
            ]
            _run_logged(
                command,
                RESULT_ROOT / "logs" / f"extract_{E384_SUBJECT}_seed_{seed}.log",
            )
        frame = state_cache_to_anchor_frame(
            cache_path=cache,
            query_path=query_path,
            coverage=coverage,
            global_exclusion_path=exclusion_path,
            seed=seed,
        )
        feature_path = (
            RESULT_ROOT / "per_subject" / E384_SUBJECT
            / f"seed_{seed}_anchor_features.csv"
        )
        atomic_csv(
            feature_path,
            frame.replace({float("nan"): None}).to_dict(orient="records"),
        )
        frames.append(frame)
        cache_rows.append({
            "subject": E384_SUBJECT,
            "seed": seed,
            "state_cache": str(cache),
            "state_cache_sha256": sha256_file(cache),
            "state_manifest": str(cache.with_suffix(".manifest.json")),
            "state_manifest_sha256": sha256_file(
                cache.with_suffix(".manifest.json")
            ),
            "anchor_features": str(feature_path),
            "anchor_features_sha256": sha256_file(feature_path),
        })

    main_arms = ("B_history", "B_observation", "B_state", "memoryless")
    risk_path = RESULT_ROOT / "risk_sets/e384_risk_sets.csv"
    risk, risk_audit = build_e384_risk_table(
        anchor_frames=frames,
        seizure_path=Path(inputs["seizure_path"]),
        output_path=risk_path,
        controls_per_case=args.controls_per_case,
        arms=main_arms,
        require_wrong_time=False,
    )
    probe_dir = RESULT_ROOT / "fits/e384_instrument"
    command = [
        sys.executable,
        "scripts/topic5_continuous_marked_state_h2b/run_risk_probe.py",
        "--risk-table", str(risk_path),
        "--output-dir", str(probe_dir),
        "--n-permutations", str(args.n_permutations),
        "--arms", *main_arms,
    ]
    if args.overwrite:
        command.append("--overwrite")
    _run_logged(command, RESULT_ROOT / "logs/e384_risk_probe.log")

    wrong_arms = (*main_arms, "wrong_time")
    wrong_path = RESULT_ROOT / "risk_sets/e384_wrong_time_risk_sets.csv"
    wrong_risk, wrong_audit = build_e384_risk_table(
        anchor_frames=frames,
        seizure_path=Path(inputs["seizure_path"]),
        output_path=wrong_path,
        controls_per_case=args.controls_per_case,
        arms=wrong_arms,
        require_wrong_time=True,
    )
    wrong_probe_dir = RESULT_ROOT / "fits/e384_wrong_time_instrument"
    wrong_command = [
        sys.executable,
        "scripts/topic5_continuous_marked_state_h2b/run_risk_probe.py",
        "--risk-table", str(wrong_path),
        "--output-dir", str(wrong_probe_dir),
        "--n-permutations", str(args.n_permutations),
        "--arms", *wrong_arms,
    ]
    if args.overwrite:
        wrong_command.append("--overwrite")
    _run_logged(wrong_command, RESULT_ROOT / "logs/e384_wrong_time_probe.log")

    summary = {
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "scope": "E384 development instrument; not a cohort H2b result",
        "subject": E384_SUBJECT,
        "checkpoint_seeds": sorted(int(row["seed"]) for row in cache_rows),
        "state_caches": cache_rows,
        "risk_table": str(risk_path),
        "risk_table_sha256": sha256_file(risk_path),
        "n_risk_sets": int(risk.risk_set_id.nunique()),
        "n_seizures": int(risk.seizure_id.nunique()),
        "lead_minutes_run": sorted(map(int, risk.lead_minutes.unique())),
        "evaluation_tiers": sorted(risk.evaluation_tier.astype(str).unique()),
        "risk_set_audit": risk_audit,
        "wrong_time_risk_table": str(wrong_path),
        "wrong_time_risk_table_sha256": sha256_file(wrong_path),
        "wrong_time_n_risk_sets": int(wrong_risk.risk_set_id.nunique()),
        "wrong_time_n_seizures": int(wrong_risk.seizure_id.nunique()),
        "wrong_time_risk_set_audit": wrong_audit,
        "probe_audit": str(probe_dir / "risk_probe_machine_audit.json"),
        "probe_audit_sha256": sha256_file(
            probe_dir / "risk_probe_machine_audit.json"
        ),
        "wrong_time_probe_audit": str(
            wrong_probe_dir / "risk_probe_machine_audit.json"
        ),
        "wrong_time_probe_audit_sha256": sha256_file(
            wrong_probe_dir / "risk_probe_machine_audit.json"
        ),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "seizure_loss_updates_state": False,
        "r1_7_used": False,
        "state_cache_reused": bool(args.reuse_state_cache),
        "source_sha256": {
            "scripts/topic5_continuous_marked_state_h2b/run_e384_pilot.py": (
                sha256_file(Path(__file__).resolve())
            ),
            "src/topic5_continuous_marked_state_h2b/pilot.py": sha256_file(
                REPO_ROOT / "src/topic5_continuous_marked_state_h2b/pilot.py"
            ),
        },
    }
    atomic_json(RESULT_ROOT / "reports/e384_pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
