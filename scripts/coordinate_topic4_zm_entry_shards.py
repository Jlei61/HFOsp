#!/usr/bin/env python
"""Durable entry-shard coordinator for the live Rev3.1 branch decision.

The coordinator launches no base-grid work.  It waits until the already
running canonical/part workers jointly cover the five locked base levels,
stops the old single-writer entry processes, atomically merges all rows, and
launches only the bracket-required expansion cells.  At most ``--max-snn``
full-network processes are allowed at any time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
for _path in (str(_ROOT), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_topic4_zm_branch_decision as R  # noqa: E402


SESSION = "topic4_zmbd"
SEEDS = (1, 3, 4)
LOG = Path(R.OUT) / "logs" / "entry_shard_coordinator.log"


def _log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[entry-shard-coordinator] {time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    with LOG.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def available_rows(seed):
    root = Path(R.OUT) / "boundaries" / "entry" / f"seed{int(seed)}"
    rows = {}
    canonical = _json(root / "entry_probes.json")
    for row in canonical.get("rows", []):
        if row.get("completed"):
            rows[row["key"]] = row
    for path in sorted((root / "parts").glob("*.json")):
        payload = _json(path)
        row = payload.get("row")
        if payload.get("complete") is True and isinstance(row, dict):
            previous = rows.get(row.get("key"))
            if previous is not None:
                identity = (
                    previous.get("bank_sha"),
                    previous.get("entered_carrier"),
                    previous.get("stationarity_ok"),
                    previous.get("end_reason"),
                )
                candidate = (
                    row.get("bank_sha"),
                    row.get("entered_carrier"),
                    row.get("stationarity_ok"),
                    row.get("end_reason"),
                )
                if identity != candidate:
                    raise RuntimeError(
                        f"seed {seed}: conflicting row {row.get('key')}"
                    )
            rows[row["key"]] = previous or row
    return rows


def expected_base_keys():
    return {
        f"lambda={lam:g}|{R.ENTRY_BASE_REPLICATE}"
        for lam in R.ENTRY_LEVELS
    }


def live_snn_count():
    result = subprocess.run(
        ["pgrep", "-af", r"python scripts/run_topic4_zm_.*\.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("pgrep failed while counting SNN workers")
    return sum(
        any(
            name in line
            for name in (
                "run_topic4_zm_branch_decision.py",
                "run_topic4_zm_entry_cell.py",
                "run_topic4_zm_offset_cell.py",
            )
        )
        for line in result.stdout.splitlines()
    )


def _window_names():
    result = subprocess.run(
        ["tmux", "list-windows", "-t", SESSION, "-F", "#{window_name}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tmux session {SESSION!r} is unavailable")
    return set(result.stdout.splitlines())


def _kill_window(name):
    if name in _window_names():
        subprocess.run(
            ["tmux", "kill-window", "-t", f"{SESSION}:{name}"],
            check=True,
        )
        _log(f"stopped window={name}")


def _merge(seed):
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "merge_topic4_zm_entry_parts.py"),
            "--seed",
            str(seed),
        ],
        cwd=_ROOT,
        check=True,
    )
    return _json(
        Path(R.OUT)
        / "boundaries"
        / "entry"
        / f"seed{seed}"
        / "entry_probes.json"
    )


def _cell_window(seed, lam, replicate):
    tag = f"{float(lam):g}".replace(".", "p")
    rep = replicate.replace("noise_", "").replace("_", "")
    return f"ex{seed}_{tag}_{rep}"


def _launch_cell(seed, lam, replicate):
    name = _cell_window(seed, lam, replicate)
    if name in _window_names():
        return
    log = (
        Path(R.OUT)
        / "logs"
        / f"entry_cell_seed{seed}_{float(lam):g}_{replicate}.log"
    )
    command = (
        f"cd {str(_ROOT)!r} && exec env OMP_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        f"{sys.executable!r} scripts/run_topic4_zm_entry_cell.py "
        f"--seed {int(seed)} --lambda {float(lam):.17g} "
        f"--replicate {replicate!r} --confirm-run >> {str(log)!r} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-window", "-d", "-t", SESSION, "-n", name, command],
        check=True,
    )
    _log(
        f"launched window={name} seed={seed} lambda={float(lam):g} "
        f"replicate={replicate}"
    )


def _restart_finalizer():
    if "finalizer" in _window_names():
        return
    log = Path(R.OUT) / "logs" / "finalizer_console.log"
    command = (
        f"cd {str(_ROOT)!r} && exec bash "
        "scripts/finalize_topic4_zm_branch_when_ready.sh "
        f">> {str(log)!r} 2>&1"
    )
    subprocess.run(
        [
            "tmux",
            "new-window",
            "-d",
            "-t",
            SESSION,
            "-n",
            "finalizer",
            command,
        ],
        check=True,
    )
    _log("restarted wait-only finalizer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=float, default=60.0)
    parser.add_argument("--max-snn", type=int, default=15)
    args = parser.parse_args()
    if args.poll < 10:
        raise SystemExit("--poll must be >=10 s")
    if args.max_snn < 6:
        raise SystemExit("--max-snn must retain the six canonical boundary workers")
    _log(f"started max_snn={args.max_snn} poll={args.poll:g}s")

    base = expected_base_keys()
    while True:
        missing = {
            seed: sorted(base - set(available_rows(seed)))
            for seed in SEEDS
        }
        _log(f"base pending={ {seed: len(keys) for seed, keys in missing.items()} }")
        if not any(missing.values()):
            break
        time.sleep(args.poll)

    # Prevent the old finalizer from declaring a transient worker-liveness P0
    # while canonical entry writers are replaced by part workers.
    _kill_window("finalizer")
    for seed in SEEDS:
        _kill_window(f"seed{seed}_entry")
    time.sleep(2.0)

    manifests = {seed: _merge(seed) for seed in SEEDS}
    while True:
        pending = [
            (seed, cell)
            for seed, manifest in manifests.items()
            for cell in manifest.get("pending_cells", [])
        ]
        if not pending:
            break
        launched = 0
        for seed, cell in pending:
            rows = available_rows(seed)
            key = f"lambda={float(cell['lambda']):g}|{cell['replicate']}"
            if key in rows:
                continue
            if live_snn_count() >= args.max_snn:
                break
            _launch_cell(seed, cell["lambda"], cell["replicate"])
            launched += 1
        _log(
            f"expansion pending={len(pending)} launched={launched} "
            f"live_snn={live_snn_count()}"
        )
        time.sleep(args.poll)
        manifests = {seed: _merge(seed) for seed in SEEDS}

    for seed, manifest in manifests.items():
        if manifest.get("complete") is not True:
            raise RuntimeError(f"seed {seed}: entry merge ended incomplete")
    _restart_finalizer()
    _log("entry shards complete")


if __name__ == "__main__":
    main()
