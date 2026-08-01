#!/usr/bin/env python3
"""Manifest-driven, resumable launcher for the Topic 5 sufficiency runs.

Every cell is a directory.  A cell with ``DONE.json`` is complete and skipped;
a directory without ``DONE.json`` is a partial run and blocks resume loudly
instead of being silently overwritten.  The launcher therefore survives a
dropped shell: rerun it with the same manifest and it continues.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_training_sufficiency import plan_cells  # noqa: E402

SCRIPTS = {
    "development": ROOT / "scripts/run_topic5_training_sufficiency_dev_v0_1.py",
    "loso": ROOT / "scripts/run_topic5_training_sufficiency_loso_v0_1.py",
}


def _command(cell: dict, cell_dir: Path) -> list[str]:
    script = SCRIPTS[cell["script"]]
    command = [sys.executable, str(script), "--run-dir", str(cell_dir)]
    for key, value in cell["args"].items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
            continue
        command.extend([flag, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--retry-once", action="store_true")
    args = parser.parse_args()

    manifest_path = (
        args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    )
    manifest = json.loads(manifest_path.read_text())
    root = ROOT / manifest["root"]
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    cells = {entry["cell"]: entry for entry in manifest["cells"]}
    plan = plan_cells(list(cells), root)
    state_path = root / "LAUNCHER_STATE.json"

    def _write_state(status: str, failures: list[str], extra: dict | None = None) -> None:
        payload = {
            "status": status,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "phase": manifest.get("phase"),
            "n_cells": len(cells),
            "n_complete": len(plan_cells(list(cells), root)["complete"]),
            "n_failed": len(failures),
            "failed_cells": failures,
            "workers": int(args.workers),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        payload.update(extra or {})
        state_path.write_text(json.dumps(payload, indent=2) + "\n")

    if plan["blocked"]:
        _write_state("BLOCKED_PARTIAL_CELLS", [], {"blocked_cells": plan["blocked"]})
        raise SystemExit(
            "partial cells block a safe resume; inspect and remove them:\n  "
            + "\n  ".join(plan["blocked"])
        )

    with (root / "launch_manifest.tsv").open("w") as handle:
        handle.write("cell\tstatus\n")
        for cell in cells:
            status = "complete" if cell in plan["complete"] else "pending"
            handle.write(f"{cell}\t{status}\n")

    queue: Queue = Queue()
    for cell in plan["pending"]:
        queue.put(cell)
    failures: list[str] = []
    lock = threading.Lock()
    _write_state("RUNNING", failures, {"n_pending": queue.qsize()})

    def _worker(worker_id: int) -> None:
        while True:
            try:
                cell = queue.get_nowait()
            except Empty:
                return
            cell_dir = root / cell
            cell_dir.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / (cell.replace("/", "__") + ".log")
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES", "0"),
                    "OMP_NUM_THREADS": str(args.cpu_threads),
                    "MKL_NUM_THREADS": str(args.cpu_threads),
                    "PYTHONPATH": str(ROOT),
                }
            )
            attempts = 2 if args.retry_once else 1
            for attempt in range(attempts):
                if cell_dir.exists():
                    # a retry must start from a clean directory
                    for path in sorted(cell_dir.rglob("*"), reverse=True):
                        path.unlink() if path.is_file() else path.rmdir()
                    cell_dir.rmdir()
                with log_path.open("w" if attempt == 0 else "a") as handle:
                    handle.write(f"# attempt {attempt + 1}\n")
                    handle.flush()
                    result = subprocess.run(
                        _command(cells[cell], cell_dir),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        cwd=str(ROOT),
                    )
                if result.returncode == 0 and (cell_dir / "DONE.json").is_file():
                    break
            else:
                pass
            with lock:
                if not (cell_dir / "DONE.json").is_file():
                    failures.append(cell)
                _write_state("RUNNING", failures, {"n_pending": queue.qsize()})

    threads = [
        threading.Thread(target=_worker, args=(index,), daemon=True)
        for index in range(int(args.workers))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    final = plan_cells(list(cells), root)
    status = "COMPLETE" if len(final["complete"]) == len(cells) else "INCOMPLETE"
    _write_state(status, failures, {"n_pending": 0})
    (root / "LAUNCHER_DONE.json").write_text(
        json.dumps(
            {
                "status": status,
                "n_cells": len(cells),
                "n_complete": len(final["complete"]),
                "failed_cells": failures,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": status, "n_complete": len(final["complete"]), "n_cells": len(cells), "failed": failures}))
    if status != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
