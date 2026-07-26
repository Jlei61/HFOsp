#!/usr/bin/env python
"""Phase-0A gate: dynamic-state inventory + canonical-config lock (spec rev3.1 §2.1, plan Task 1).

  python scripts/audit_topic4_zm_dynamic_state.py                 # audit only (no sim, seconds)
  python scripts/audit_topic4_zm_dynamic_state.py --write-lock --seeds 1,3,4

`--write-lock` additionally resolves the per-seed q75 threshold calibration `I_th_EI` by running the
REAL slow-off baseline used by run_zm_snn_native_exit (a short spontaneous run), so the lock records
the number that the science runs will actually use. Calibrations are cached per seed.

Exit status 2 == `blocked_state_inventory` (a mutable engine state nobody classified): Task 2 must
not start.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _p in (_ROOT, _SCRIPTS, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import src.topic4_zm_fork_state as FS  # noqa: E402

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision", "phase0")


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT).decode().strip()
    except Exception:
        return "unknown"


def calibrate_I_th_EI(seed, cache_path):
    """q75 of the settled slow-OFF interictal E-cell inhibitory current (the REAL calibrator)."""
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    key = str(int(seed))
    if key in cache:
        return float(cache[key]["I_th_EI"]), cache[key]
    import run_m4_phaseplane as PP
    import run_zm_snn_native_exit as ZM
    t0 = time.time()
    S = PP.build_substrate(seed=int(seed))
    t_build = time.time() - t0
    t1 = time.time()
    val = ZM._calibrate_I_th_EI(S)
    rec = dict(I_th_EI=float(val), build_s=round(t_build, 1), calib_s=round(time.time() - t1, 1),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2, 2))
    cache[key] = rec
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    return float(val), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--seeds", default="1")
    a = ap.parse_args()

    audit = FS.audit_dynamic_state()
    print(f"[audit] status={audit['status']} rows={audit['n_rows']} "
          f"(simulator={audit['n_simulator_rows']}) scopes={audit['engine_scopes']}")
    if audit["status"] != "ok":
        for p in audit["problems"]:
            print("  !", p)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "blocked_state_inventory.json"), "w") as f:
            json.dump(dict(audit, git_sha=_git_sha()), f, indent=2)
        return 2

    if not a.write_lock:
        print("[audit] ok (dry run; pass --write-lock to write the canonical config lock)")
        return 0

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "state_inventory.json"), "w") as f:
        json.dump(dict(version=FS.INVENTORY_VERSION, git_sha=_git_sha(),
                       audit=audit, rows=FS.build_state_inventory()), f, indent=2)

    cache_path = os.path.join(OUT, "I_th_EI_calibration.json")
    configs = {}
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        val, rec = calibrate_I_th_EI(s, cache_path)
        cfg = FS.build_canonical_config(seed=s, I_th_EI=val)
        configs[str(s)] = dict(config=cfg, config_sha=FS.config_sha(cfg), calibration=rec)
        print(f"[lock] seed={s} I_th_EI={val:.6f} config_sha={FS.config_sha(cfg)[:16]} "
              f"(build {rec.get('build_s')}s, calib {rec.get('calib_s')}s, "
              f"rss {rec.get('peak_rss_gb')}GB)")
    with open(os.path.join(OUT, "canonical_config.json"), "w") as f:
        json.dump(dict(schema=FS.SCHEMA_VERSION, git_sha=_git_sha(), seeds=configs), f, indent=2)
    print(f"[lock] wrote {OUT}/canonical_config.json + state_inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
