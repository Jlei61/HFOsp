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
import shlex
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


def early_expansion_cells(rows):
    """Return registered resamples once one seed's base grid is complete."""

    base_rows = [
        {
            "lambda": row["lambda"],
            "entered_carrier": row["entered_carrier"],
        }
        for row in rows.values()
        if row.get("replicate") == R.ENTRY_BASE_REPLICATE
        and row.get("completed")
    ]
    if len(base_rows) != len(R.ENTRY_LEVELS):
        return []
    bracket = R.BD.half_boundary(
        R.BD.jeffreys_probability_curve(
            base_rows, "lambda", "entered_carrier"
        ),
        expected_direction="increasing",
    )
    if bracket.get("status") != "bracketed":
        return []
    pending = []
    for lam in bracket["q_bracket"]:
        for replicate in R.ENTRY_EXPANSION_REPLICATES:
            key = f"lambda={float(lam):g}|{replicate}"
            if key not in rows:
                pending.append(
                    {"lambda": float(lam), "replicate": replicate}
                )
    return pending


def _option_value(tokens, name, *, required=True, default=None):
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"in-flight entry command has bare {name}")
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    if required:
        raise RuntimeError(f"in-flight entry command is missing {name}")
    return default


def _entry_cell_from_command(command):
    """Recover a shard's scientific identity from its live Python argv."""

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        if "run_topic4_zm_entry_cell.py" in command:
            raise RuntimeError(
                "cannot parse a live entry-cell command"
            ) from exc
        return None
    indexes = [
        index
        for index, token in enumerate(tokens)
        if token.endswith("run_topic4_zm_entry_cell.py")
    ]
    if not indexes:
        return None
    script_index = indexes[0]
    if script_index == 0 or not Path(tokens[script_index - 1]).name.startswith(
        "python"
    ):
        return None
    argv = tokens[script_index + 1 :]
    seed = int(_option_value(argv, "--seed"))
    lam = float(_option_value(argv, "--lambda"))
    replicate = _option_value(
        argv,
        "--replicate",
        required=False,
        default=R.ENTRY_BASE_REPLICATE,
    )
    return seed, f"lambda={lam:g}|{replicate}"


def live_entry_cells():
    """Return ``(seed, key)`` for all in-flight isolated entry workers."""

    result = subprocess.run(
        ["ps", "-eo", "args="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ps failed while identifying in-flight entry cells")
    cells = set()
    for command in result.stdout.splitlines():
        parsed = _entry_cell_from_command(command)
        if parsed is not None:
            cells.add(parsed)
    return cells


def missing_covered_by_shards(seed, missing_keys, live_cells):
    return all((int(seed), key) in live_cells for key in missing_keys)


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
        return False
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
    return True


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
        rows_by_seed = {seed: available_rows(seed) for seed in SEEDS}
        missing = {
            seed: sorted(base - set(rows_by_seed[seed]))
            for seed in SEEDS
        }
        live_cells = live_entry_cells()
        windows = _window_names()
        for seed in SEEDS:
            canonical = f"seed{seed}_entry"
            if (
                canonical in windows
                and missing_covered_by_shards(
                    seed, missing[seed], live_cells
                )
            ):
                # The isolated workers now cover every unfinished base cell.
                # Stop the wait-only finalizer first so it cannot mistake the
                # intentional handoff for a vanished required worker.
                _kill_window("finalizer")
                _kill_window(canonical)
                windows.discard(canonical)
                _log(
                    f"handed off seed={seed} canonical entry writer; "
                    f"covered_missing={missing[seed]}"
                )
        # If a handed-off shard dies before writing its atomic part, restore
        # that exact base cell instead of leaving a silent hole.
        live_cells = live_entry_cells()
        for seed in SEEDS:
            if f"seed{seed}_entry" in _window_names():
                continue
            for lam in R.ENTRY_LEVELS:
                key = f"lambda={float(lam):g}|{R.ENTRY_BASE_REPLICATE}"
                if key not in missing[seed] or (seed, key) in live_cells:
                    continue
                if live_snn_count() >= args.max_snn:
                    break
                if _launch_cell(seed, lam, R.ENTRY_BASE_REPLICATE):
                    live_cells.add((seed, key))
        launched = 0
        for seed in SEEDS:
            if missing[seed]:
                continue
            for cell in early_expansion_cells(rows_by_seed[seed]):
                if live_snn_count() >= args.max_snn:
                    break
                if _launch_cell(seed, cell["lambda"], cell["replicate"]):
                    launched += 1
        _log(
            f"base pending="
            f"{ {seed: len(keys) for seed, keys in missing.items()} } "
            f"early_expansion_launched={launched} "
            f"live_snn={live_snn_count()}"
        )
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
    R.write_json_atomic(
        os.path.join(R.OUT, "boundaries", "entry", "entry_shards_complete.json"),
        {
            "complete": True,
            "seeds": list(SEEDS),
            "max_snn": int(args.max_snn),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    _restart_finalizer()
    _log("entry shards complete")


if __name__ == "__main__":
    main()
