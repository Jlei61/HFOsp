#!/usr/bin/env python
"""Lightweight resource monitor for the topic4 M4 SNN-native-exit line (spec §10).

Logs one strict-JSON line per tick to a resource_log.jsonl: timestamp, loadavg,
MemAvailable / MemTotal, swap used, and per-PID RSS (+CPU% if psutil present) for
the PIDs listed in a manifest file (this line's workers only). If MemAvailable drops
below `--mem-stop-frac` of MemTotal, or swap grows more than `--swap-stop-mb` above
the baseline at start, it writes a `protective_stop` record and touches `--stop-file`
(a sentinel the launching agent polls between batches). It NEVER kills processes
itself -- protective action is left to the agent, which only ever stops PIDs in the
manifest (never a global pkill). Dependency-light: /proc parsing + optional psutil.
"""
from __future__ import annotations

import argparse
import json
import os
import time

try:
    import psutil  # optional, only for CPU%
except Exception:  # pragma: no cover
    psutil = None


def _meminfo():
    total = avail = swap_total = swap_free = None
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            v = int(rest.strip().split()[0])  # kB
            if k == "MemTotal":
                total = v
            elif k == "MemAvailable":
                avail = v
            elif k == "SwapTotal":
                swap_total = v
            elif k == "SwapFree":
                swap_free = v
    return total, avail, (swap_total - swap_free if swap_total is not None else None)


def _pid_rss_kb(pid: int):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def _read_manifest(path):
    if not path or not os.path.exists(path):
        return []
    pids = []
    with open(path) as f:
        for tok in f.read().split():
            try:
                pids.append(int(tok))
            except ValueError:
                pass
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="resource_log.jsonl path (append)")
    ap.add_argument("--manifest", default=None, help="file of PIDs (one/line) for THIS line")
    ap.add_argument("--stop-file", default=None, help="sentinel touched on protective stop")
    ap.add_argument("--interval", type=float, default=45.0)
    ap.add_argument("--duration", type=float, default=None, help="seconds; default run until killed")
    ap.add_argument("--mem-stop-frac", type=float, default=0.20)
    ap.add_argument("--swap-stop-mb", type=float, default=1024.0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    total0, _, swap0 = _meminfo()
    procs = {}
    if psutil is not None:
        for pid in _read_manifest(a.manifest):
            try:
                procs[pid] = psutil.Process(pid)
                procs[pid].cpu_percent(None)  # prime
            except Exception:
                pass

    t_start = time.time()
    while True:
        total, avail, swap_used = _meminfo()
        pids = _read_manifest(a.manifest)
        per_pid = {}
        for pid in pids:
            rss = _pid_rss_kb(pid)
            cpu = None
            if psutil is not None:
                p = procs.get(pid)
                if p is None:
                    try:
                        p = psutil.Process(pid)
                        p.cpu_percent(None)
                        procs[pid] = p
                    except Exception:
                        p = None
                if p is not None:
                    try:
                        cpu = p.cpu_percent(None)
                    except Exception:
                        cpu = None
            if rss is not None:
                per_pid[str(pid)] = {"rss_mb": round(rss / 1024, 1), "cpu_pct": cpu}

        mem_frac = (avail / total) if (total and avail) else None
        swap_growth_mb = ((swap_used - swap0) / 1024.0) if (swap_used is not None and swap0 is not None) else None
        protective = bool(
            (mem_frac is not None and mem_frac < a.mem_stop_frac)
            or (swap_growth_mb is not None and swap_growth_mb > a.swap_stop_mb)
        )
        rec = {
            "ts": round(time.time(), 1),
            "loadavg": [round(x, 2) for x in os.getloadavg()],
            "mem_total_gb": round(total / 1024 / 1024, 1) if total else None,
            "mem_avail_gb": round(avail / 1024 / 1024, 1) if avail else None,
            "mem_avail_frac": round(mem_frac, 3) if mem_frac is not None else None,
            "swap_used_mb": round(swap_used / 1024, 1) if swap_used is not None else None,
            "swap_growth_mb": round(swap_growth_mb, 1) if swap_growth_mb is not None else None,
            "n_pids": len(per_pid),
            "per_pid": per_pid,
            "protective_stop": protective,
        }
        with open(a.out, "a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        if protective and a.stop_file:
            with open(a.stop_file, "w") as f:
                f.write(f"protective_stop at {rec['ts']}: mem_frac={mem_frac} swap_growth_mb={swap_growth_mb}\n")

        if a.duration is not None and (time.time() - t_start) >= a.duration:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
