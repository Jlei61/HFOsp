#!/usr/bin/env python3
"""Run the seed-1 12-s fast-inhibition lifecycle batch with bounded resources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
RUNNER = ROOT / "scripts/run_topic4_zm_fast_lifecycle_development.py"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _cfg_id(cfg):
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def build_manifest():
    rows = []
    dep = qmc.LatinHypercube(d=2, seed=20260802).random(16)
    for point in dep:
        rows.append({
            "family": "depression_only_lhs",
            "arm": "i2e",
            "tau_D_ms": round(300 + 550 * float(point[0]), 1),
            "d_star": round(0.55 + 0.30 * float(point[1]), 4),
            "strength_scale": 1.0,
        })
    combined = qmc.LatinHypercube(d=4, seed=20260803).random(16)
    for point in combined:
        rows.append({
            "family": "combined_lhs",
            "arm": "combined",
            "tau_D_ms": round(300 + 550 * float(point[0]), 1),
            "d_star": round(0.55 + 0.30 * float(point[1]), 4),
            "tau_aI_ms": round(60 + 290 * float(point[2]), 1),
            "f_aI": round(0.02 + 0.10 * float(point[3]), 4),
            "strength_scale": 1.0,
        })
    rows.extend([
        dict(family="anchor_burst", arm="i2e", tau_D_ms=300.0, d_star=0.55,
             strength_scale=1.0),
        dict(family="anchor_patch", arm="i2e", tau_D_ms=600.0, d_star=0.75,
             strength_scale=1.0),
        dict(family="anchor_transition", arm="combined", tau_D_ms=600.0, d_star=0.75,
             tau_aI_ms=300.0, f_aI=0.10, strength_scale=0.5),
        dict(family="anchor_weak_patch", arm="combined", tau_D_ms=300.0, d_star=0.55,
             tau_aI_ms=100.0, f_aI=0.10, strength_scale=0.5),
    ])
    for row in rows:
        row.update(
            g_M=1.0, tau_M_ms=500.0, g_Z=1.0,
            T_ms=12000.0, burn_ms=1000.0,
        )
        row["config_id"] = _cfg_id(row)
    return {
        "schema": "topic4_zm_lifecycle_sprint_batch1_v1_2026-08-02",
        "created_at_utc": _now(),
        "seed": 1,
        "paired_noise": True,
        "n_configs": len(rows),
        "rows": rows,
    }


def _mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024 ** 2
    raise RuntimeError("MemAvailable missing")


def _command(row):
    cmd = [
        sys.executable, str(RUNNER), "sprint-cell",
        "--arm", row["arm"],
        "--tau-D-ms", str(row["tau_D_ms"]),
        "--d-star", str(row["d_star"]),
        "--strength-scale", str(row["strength_scale"]),
        "--g-M", str(row["g_M"]),
        "--tau-M-ms", str(row["tau_M_ms"]),
        "--g-Z", str(row["g_Z"]),
        "--T-ms", str(row["T_ms"]),
        "--burn-ms", str(row["burn_ms"]),
        "--confirm-run",
    ]
    if row["arm"] == "combined":
        cmd += ["--tau-aI-ms", str(row["tau_aI_ms"]), "--f-aI", str(row["f_aI"])]
    return cmd


def run_batch(max_workers, min_mem_gb, poll_s):
    manifest_path = OUT / "batch1_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else build_manifest()
    _atomic_json(manifest_path, manifest)
    ledger_path = OUT / "batch1_run_ledger.json"
    ledger = {
        "schema": "topic4_zm_lifecycle_sprint_ledger_v1_2026-08-02",
        "started_at_utc": _now(),
        "max_workers": int(max_workers),
        "min_mem_available_gb": float(min_mem_gb),
        "rows": {row["config_id"]: {"status": "pending", **row} for row in manifest["rows"]},
    }
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text())
        for key, value in previous.get("rows", {}).items():
            if key in ledger["rows"] and value.get("status") == "success":
                ledger["rows"][key] = value
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active = {}
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    while True:
        for key, item in list(active.items()):
            code = item["proc"].poll()
            if code is None:
                continue
            item["stream"].close()
            row = ledger["rows"][key]
            row.update(
                status="success" if code == 0 else "worker_failed",
                returncode=int(code), terminal_time_utc=_now(),
                log_path=str(item["log"].relative_to(ROOT)),
            )
            del active[key]
            _atomic_json(ledger_path, ledger)
        pending = [
            (key, row) for key, row in ledger["rows"].items()
            if row["status"] == "pending"
        ]
        while pending and len(active) < int(max_workers) and _mem_available_gb() >= float(min_mem_gb):
            key, row = pending.pop(0)
            log = logs / f"{key}.log"
            stream = log.open("a")
            cmd = _command(row)
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            row.update(
                status="running", pid=int(proc.pid), start_time_utc=_now(),
                command=cmd, log_path=str(log.relative_to(ROOT)),
            )
            active[key] = {"proc": proc, "stream": stream, "log": log}
            _atomic_json(ledger_path, ledger)
        if not active and not pending:
            break
        ledger["heartbeat_time_utc"] = _now()
        ledger["mem_available_gb"] = _mem_available_gb()
        ledger["n_active"] = len(active)
        _atomic_json(ledger_path, ledger)
        time.sleep(float(poll_s))
    ledger["terminal_time_utc"] = _now()
    ledger["status"] = "complete"
    _atomic_json(ledger_path, ledger)
    return ledger_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--min-mem-gb", type=float, default=90.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    manifest = build_manifest()
    _atomic_json(OUT / "batch1_manifest.json", manifest)
    if args.manifest_only:
        print(OUT / "batch1_manifest.json")
        return
    print(run_batch(args.max_workers, args.min_mem_gb, args.poll_s))


if __name__ == "__main__":
    main()
