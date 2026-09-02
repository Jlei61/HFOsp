#!/usr/bin/env python3
"""Resource-guarded, resumable v0.5 formal training orchestrator."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
PYTHON = Path(sys.executable)
ARM_INTERNAL = {
    "L0": "L0_LOCAL_ONLY",
    "L1": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2m": "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3": "L3_LOCAL_PLUS_LEARNED_LR",
    "C-suffix": "C_L3_ORDER_SHUFFLED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_snapshot() -> dict:
    contract = json.loads((OUT_ROOT / "RUN_CONTRACT.json").read_text())
    for relative, expected in contract["source_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"source changed after freeze: {relative}")
    if not (OUT_ROOT / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json").exists():
        raise RuntimeError("target physical embargo marker missing")
    if sha256_file(OUT_ROOT / "FORMAL_TRAINING_SCHEDULE.csv") != contract["schedule_sha256"]:
        raise RuntimeError("formal training schedule changed after freeze")
    manifest_path = OUT_ROOT / "INPUT_CACHE_MANIFEST.json"
    if sha256_file(manifest_path) != contract["input_manifest_sha256"]:
        raise RuntimeError("formal input manifest changed after freeze")
    return contract


def validate_formal_inputs(*, include_stage_evidence: bool) -> None:
    """Verify every byte consumed by training against the frozen manifest."""
    manifest = json.loads((OUT_ROOT / "INPUT_CACHE_MANIFEST.json").read_text())
    for record in manifest["cache_records"]:
        fit_root = OUT_ROOT / "cache" / record["fit_id"]
        for name, expected in record["files"].items():
            path = fit_root / name
            if not path.exists():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(expected["size_bytes"]):
                raise RuntimeError(f"formal input size changed: {record['fit_id']}/{name}")
            if sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"formal input hash changed: {record['fit_id']}/{name}")
    if include_stage_evidence:
        for name, expected in manifest["stage_evidence_hashes"].items():
            path = OUT_ROOT / name
            if sha256_file(path) != expected:
                raise RuntimeError(f"stage evidence changed before formal launch: {name}")


def memory_guard(requested: int) -> int:
    info = {}
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            key, value = line.split(":", 1)
            info[key] = int(value.strip().split()[0]) * 1024
    available_gib = info["MemAvailable"] / 2**30
    if available_gib < 32:
        raise RuntimeError(f"system memory headroom too low: {available_gib:.1f} GiB")
    workers = min(int(requested), 12, max(1, int(available_gib // 4)))
    if subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        )
        free_mib = int(output.strip().splitlines()[0])
        if free_mib < 4096:
            raise RuntimeError(f"GPU memory headroom too low: {free_mib} MiB")
        workers = min(workers, max(1, free_mib // 768))
    return workers


def unit_dir(row: dict) -> Path:
    return OUT_ROOT / "formal_units" / row["fit_id"] / ARM_INTERNAL[row["arm"]] / f"seed{row['seed']}"


def run_unit(row: dict, retry: int = 1) -> dict:
    destination = unit_dir(row)
    done = destination / "DONE.json"
    if done.exists():
        return {**row, "status": "SKIPPED_DONE", "returncode": 0}
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(ROOT / "scripts/train_topic5_multiscale_scaffold_unit_v0_5.py"),
        "--fit-id", row["fit_id"], "--arm", row["arm"], "--seed", str(row["seed"]),
    ]
    log = destination / "runner.log"
    for attempt in range(retry + 1):
        with log.open("a") as stream:
            stream.write(f"\nATTEMPT {attempt} COMMAND {' '.join(command)}\n")
            stream.flush()
            environment = dict(os.environ)
            environment.update({
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
            })
            process = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=environment,
            )
        if process.returncode == 0 and done.exists():
            failed_marker = destination / "FAILED.json"
            if failed_marker.exists():
                failed_marker.unlink()
            return {**row, "status": "DONE", "returncode": 0, "attempt": attempt}
        if attempt < retry:
            time.sleep(2)
    error_text = log.read_text(errors="replace")[-8000:]
    failed = {
        **row, "status": "FAILED", "returncode": process.returncode,
        "oom_detected": "out of memory" in error_text.lower(), "log": str(log),
    }
    (destination / "FAILED.json").write_text(json.dumps(failed, indent=2))
    return failed


def run_phase(rows: pd.DataFrame, workers: int, phase: int) -> list[dict]:
    records = rows.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_unit, row): row for row in records}
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if number % 10 == 0 or result["status"] == "FAILED":
                print(json.dumps({
                    "phase": phase, "completed": number, "total": len(records),
                    "failed": sum(item["status"] == "FAILED" for item in results),
                }), flush=True)
    pd.DataFrame(results).to_csv(OUT_ROOT / f"PHASE_{phase}_EXECUTION.csv", index=False)
    failures = [row for row in results if row["status"] == "FAILED"]
    if failures:
        raise RuntimeError(f"phase {phase} has {len(failures)} failed units")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    contract = validate_snapshot()
    validate_formal_inputs(include_stage_evidence=True)
    workers = memory_guard(args.workers)
    schedule = pd.read_csv(OUT_ROOT / "FORMAL_TRAINING_SCHEDULE.csv")
    print(json.dumps({"status": "START", "workers": workers, "units": len(schedule)}), flush=True)
    run_phase(schedule[schedule.phase == 1], workers, 1)
    validate_snapshot()
    validate_formal_inputs(include_stage_evidence=False)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/build_topic5_l2m_graph_controls_v0_5.py")
    ], cwd=ROOT, check=True)
    manifest = pd.read_csv(OUT_ROOT / "L2M_GRAPH_CONTROL_MANIFEST.csv")
    if len(manifest) != 126 or not manifest.all_exact.all():
        raise RuntimeError(f"expected 126 exact L2m controls, found {len(manifest)}")
    run_phase(schedule[schedule.phase == 2], workers, 2)
    validate_snapshot()
    validate_formal_inputs(include_stage_evidence=False)
    done = list((OUT_ROOT / "formal_units").glob("*/*/seed*/DONE.json"))
    if len(done) != int(contract["formal_units"]):
        raise RuntimeError(f"formal DONE count {len(done)} != {contract['formal_units']}")
    payload = {
        "status": "PASS", "formal_units": len(done), "workers": workers,
        "target_values_read": False, "unresolved_oom": 0,
    }
    temporary = OUT_ROOT / "STAGE_E_TRAINING_COMPLETE.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(OUT_ROOT / "STAGE_E_TRAINING_COMPLETE.json")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
