#!/usr/bin/env python
"""Durable offset-shard coordinator for the live Rev3.1 branch decision."""
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
LOG = Path(R.OUT) / "logs" / "offset_shard_coordinator.log"


def _log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[offset-shard-coordinator] {time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    with LOG.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _row_identity(row):
    return (
        row.get("bank_sha"),
        row.get("remained_carrier"),
        row.get("low_basin_persisted"),
        row.get("response_valid"),
        row.get("stationarity_ok"),
        row.get("end_reason"),
    )


def available_rows(seed):
    root = Path(R.OUT) / "boundaries" / "offset" / f"seed{int(seed)}"
    rows = {}
    canonical = _json(root / "offset_probes.json")
    for row in canonical.get("rows", []):
        if row.get("completed"):
            rows[row["key"]] = row
    for path in sorted((root / "parts").glob("*.json")):
        payload = _json(path)
        row = payload.get("row")
        if payload.get("complete") is True and isinstance(row, dict):
            previous = rows.get(row.get("key"))
            if previous is not None and _row_identity(previous) != _row_identity(row):
                raise RuntimeError(
                    f"seed {seed}: conflicting row {row.get('key')}"
                )
            rows[row["key"]] = previous or row
    return rows


def base_cells():
    return [
        {
            "family": family,
            "lambda": float(lam),
            "initial_kind": initial_kind,
            "replicate": R.OFFSET_BASE_REPLICATE,
        }
        for family in R.OFFSET_FAMILIES
        for lam in R.OFFSET_LEVELS
        for initial_kind in ("active", "low")
    ]


def cell_key(cell):
    if cell["family"] == "dynamic_ZM":
        return f"dynamic_ZM|late_active|{cell['replicate']}"
    return (
        f"{cell['family']}|lambda={float(cell['lambda']):g}|"
        f"{cell['initial_kind']}|{cell['replicate']}"
    )


def _canonical_next_key(seed):
    """Best-effort skip for the old worker's currently running base cell."""

    path = (
        Path(R.OUT)
        / "boundaries"
        / "offset"
        / f"seed{seed}"
        / "offset_probes.json"
    )
    rows = _json(path).get("rows", [])
    completed = {row.get("key") for row in rows if row.get("completed")}
    for cell in base_cells():
        key = cell_key(cell)
        if key not in completed:
            return key
    return None


def _option_value(tokens, name, *, required=True, default=None):
    """Read either ``--name value`` or ``--name=value`` from argv tokens."""

    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"in-flight offset command has bare {name}")
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    if required:
        raise RuntimeError(f"in-flight offset command is missing {name}")
    return default


def _offset_cell_from_command(command):
    """Recover a shard cell from a live Python command, if applicable.

    Window names are deliberately not trusted here: manually resumed workers
    and future coordinators may use different tmux labels.  The scientific
    identity is encoded by the runner argv itself.
    """

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        if "run_topic4_zm_offset_cell.py" in command:
            raise RuntimeError(
                "cannot parse a live offset-cell command"
            ) from exc
        return None
    script_indexes = [
        index
        for index, token in enumerate(tokens)
        if token.endswith("run_topic4_zm_offset_cell.py")
    ]
    if not script_indexes:
        return None
    # Ignore shell/inspection commands that merely contain the script text.
    script_index = script_indexes[0]
    if script_index == 0 or not Path(tokens[script_index - 1]).name.startswith(
        "python"
    ):
        return None
    argv = tokens[script_index + 1 :]
    seed = int(_option_value(argv, "--seed"))
    family = _option_value(argv, "--family")
    initial_kind = _option_value(
        argv, "--initial-kind", required=False, default="active"
    )
    replicate = _option_value(
        argv,
        "--replicate",
        required=False,
        default=R.OFFSET_BASE_REPLICATE,
    )
    lam_text = _option_value(argv, "--lambda", required=False)
    if family == "dynamic_ZM":
        lam = None
    elif lam_text is None:
        raise RuntimeError(
            "live static offset-cell command is missing --lambda"
        )
    else:
        lam = float(lam_text)
    cell = {
        "family": family,
        "lambda": lam,
        "initial_kind": initial_kind,
        "replicate": replicate,
    }
    # Reuse the contract validator embedded in cell_key for dynamic/static
    # naming rather than maintaining a second identity convention.
    cell_key(cell)
    return seed, cell


def live_offset_cells():
    """Return scientific identities for every in-flight offset shard."""

    result = subprocess.run(
        ["ps", "-eo", "args="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ps failed while identifying in-flight offset cells")
    cells = set()
    for command in result.stdout.splitlines():
        parsed = _offset_cell_from_command(command)
        if parsed is None:
            continue
        seed, cell = parsed
        cells.add((int(seed), cell_key(cell)))
    return cells


def live_snn_count():
    result = subprocess.run(
        ["pgrep", "-af", r"python scripts/run_topic4_zm_.*\.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("pgrep failed while counting SNN workers")
    names = (
        "run_topic4_zm_branch_decision.py",
        "run_topic4_zm_entry_cell.py",
        "run_topic4_zm_offset_cell.py",
    )
    return sum(any(name in line for name in names) for line in result.stdout.splitlines())


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
            str(_SCRIPTS / "merge_topic4_zm_offset_parts.py"),
            "--seed",
            str(seed),
        ],
        cwd=_ROOT,
        check=True,
    )
    return _json(
        Path(R.OUT)
        / "boundaries"
        / "offset"
        / f"seed{seed}"
        / "offset_probes.json"
    )


def _window_name(seed, cell):
    family = {
        "M_alone": "ma",
        "M_SG": "msg",
        "M_Z_recovery": "mzr",
        "dynamic_ZM": "dyn",
    }[cell["family"]]
    if cell["family"] == "dynamic_ZM":
        level = "late"
    else:
        level = f"{float(cell['lambda']):g}".replace(".", "p")
    initial = cell["initial_kind"][0]
    replicate = (
        cell["replicate"]
        .replace("noise_replay", "r")
        .replace("noise_resample_", "s")
    )
    return f"ox{seed}_{family}_{level}_{initial}_{replicate}"


def _launch_cell(seed, cell):
    name = _window_name(seed, cell)
    if name in _window_names():
        return False
    log = Path(R.OUT) / "logs" / f"offset_cell_{name}.log"
    command = [
        sys.executable,
        "scripts/run_topic4_zm_offset_cell.py",
        "--seed",
        str(seed),
        "--family",
        cell["family"],
        "--initial-kind",
        cell["initial_kind"],
        "--replicate",
        cell["replicate"],
        "--confirm-run",
    ]
    if cell.get("lambda") is not None:
        command.extend(["--lambda", f"{float(cell['lambda']):.17g}"])
    shell_command = (
        f"cd {str(_ROOT)!r} && exec env OMP_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        + " ".join(repr(token) for token in command)
        + f" >> {str(log)!r} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-window", "-d", "-t", SESSION, "-n", name, shell_command],
        check=True,
    )
    _log(
        f"launched window={name} seed={seed} key={cell_key(cell)}"
    )
    return True


def _entry_complete():
    sentinel = _json(
        Path(R.OUT)
        / "boundaries"
        / "entry"
        / "entry_shards_complete.json"
    )
    return sentinel.get("complete") is True and all(
        _json(
            Path(R.OUT)
            / "boundaries"
            / "entry"
            / f"seed{seed}"
            / "entry_probes.json"
        ).get("complete")
        is True
        for seed in SEEDS
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
        raise SystemExit("--max-snn must be >=6")
    _log(f"started max_snn={args.max_snn} poll={args.poll:g}s")

    while not _entry_complete():
        _log("waiting for three-seed entry merge")
        time.sleep(args.poll)

    base = base_cells()
    while True:
        rows_by_seed = {seed: available_rows(seed) for seed in SEEDS}
        missing = [
            (seed, cell)
            for seed in SEEDS
            for cell in base
            if cell_key(cell) not in rows_by_seed[seed]
        ]
        if not missing:
            break
        inflight = live_offset_cells() | {
            (seed, key)
            for seed in SEEDS
            if (key := _canonical_next_key(seed)) is not None
        }
        launched = 0
        for seed, cell in missing:
            if (seed, cell_key(cell)) in inflight:
                continue
            if live_snn_count() >= args.max_snn:
                break
            if _launch_cell(seed, cell):
                launched += 1
        _log(
            f"base pending={len(missing)} launched={launched} "
            f"live_snn={live_snn_count()} inflight_skipped={sorted(inflight)}"
        )
        time.sleep(args.poll)

    _kill_window("finalizer")
    for seed in SEEDS:
        _kill_window(f"seed{seed}_offset")
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
            if cell_key(cell) in available_rows(seed):
                continue
            if live_snn_count() >= args.max_snn:
                break
            if _launch_cell(seed, cell):
                launched += 1
        _log(
            f"adaptive pending={len(pending)} launched={launched} "
            f"live_snn={live_snn_count()}"
        )
        time.sleep(args.poll)
        manifests = {seed: _merge(seed) for seed in SEEDS}

    for seed, manifest in manifests.items():
        if manifest.get("complete") is not True:
            raise RuntimeError(f"seed {seed}: offset merge ended incomplete")
    _restart_finalizer()
    _log("offset shards complete")


if __name__ == "__main__":
    main()
