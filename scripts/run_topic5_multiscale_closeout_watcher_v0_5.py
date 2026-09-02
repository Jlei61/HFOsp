#!/usr/bin/env python3
"""Detached post-pipeline source-data export and machine closeout audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_FIGURE = ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures"


def write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_related_tests(out: Path) -> None:
    tests = [
        "tests/test_topic5_multiscale_scaffold_cache_v0_5.py",
        "tests/test_topic5_multiscale_effective_scaffold_v0_5.py",
        "tests/test_topic5_lbss_rnn_v0_2.py",
        "tests/test_topic5_lbss_full_tissue_v0_3.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log = out / "PREFINAL_RELATED_PYTEST.log"
    log.write_text(completed.stdout + completed.stderr)
    summary = "\n".join(log.read_text().strip().splitlines()[-4:])
    write(out / "PREFINAL_RELATED_PYTEST_EVIDENCE.json", {
        "contract": "topic5_v0_5_prefinal_related_tests",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "test_files": tests,
        "returncode": int(completed.returncode),
        "summary_tail": summary,
        "log_sha256": sha256_file(log),
        "target_values_read": False,
    })
    if completed.returncode != 0:
        raise RuntimeError(f"related tests failed with rc={completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--upstream-pid", type=int, default=0)
    args = parser.parse_args()
    out, figure = args.out_root.resolve(), args.figure_dir.resolve()
    begin = time.monotonic()
    while not (out / "PIPELINE_COMPLETE.json").exists():
        for failed in ("POSTTRAINING_PIPELINE_FAILED.json", "STAGE_F_TARGET_FREE_FAILED.json"):
            if (out / failed).exists():
                write(out / "CLOSEOUT_WATCHER_FAILED.json", {
                    "status": "UPSTREAM_FAILED", "failed_marker": failed,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                })
                raise SystemExit(1)
        if args.upstream_pid and not Path(f"/proc/{args.upstream_pid}").exists():
            write(out / "CLOSEOUT_WATCHER_FAILED.json", {
                "status": "UPSTREAM_PROCESS_EXITED_WITHOUT_PIPELINE_COMPLETE",
                "upstream_pid": args.upstream_pid,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise SystemExit(3)
        if time.monotonic() - begin > args.timeout_hours * 3600:
            write(out / "CLOSEOUT_WATCHER_FAILED.json", {
                "status": "TIMEOUT", "timeout_hours": args.timeout_hours,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise SystemExit(2)
        time.sleep(max(5, int(args.poll_seconds)))

    try:
        run_related_tests(out)
    except Exception as error:
        write(out / "CLOSEOUT_WATCHER_FAILED.json", {
            "status": "RELATED_TESTS_FAILED", "error": repr(error),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise

    commands = (
        [sys.executable, str(ROOT / "scripts/adjudicate_topic5_multiscale_claims_v0_5.py"),
         "--out-root", str(out)],
        [sys.executable, str(ROOT / "scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_r2.py"),
         "--out-root", str(out), "--figure-dir", str(figure)],
        [sys.executable, str(ROOT / "scripts/export_topic5_figure6_source_data_v0_5.py"),
         "--out-root", str(out), "--figure-dir", str(figure)],
        [sys.executable, str(ROOT / "scripts/finalize_topic5_multiscale_closeout_report_v0_5.py"),
         "--out-root", str(out)],
        [sys.executable, str(ROOT / "scripts/audit_topic5_multiscale_closeout_v0_5.py"),
         "--out-root", str(out), "--figure-dir", str(figure)],
        [sys.executable, str(ROOT / "scripts/sync_topic5_multiscale_closeout_docs_v0_5.py"),
         "--out-root", str(out)],
    )
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            write(out / "CLOSEOUT_WATCHER_FAILED.json", {
                "status": "CLOSEOUT_COMMAND_FAILED", "returncode": result.returncode,
                "command": command, "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise SystemExit(result.returncode)
    write(out / "CLOSEOUT_WATCHER_COMPLETE.json", {
        "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_manifest": str(figure / "FIGURE6_SOURCE_DATA_MANIFEST.json"),
        "closeout_audit": str(out / "CLOSEOUT_AUDIT.json"),
    })


if __name__ == "__main__":
    main()
