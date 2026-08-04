#!/usr/bin/env python
"""Read completed FCXR-LC3 reconnaissance rows and report the entry measurement.

Answers the bearing question a bare onset time cannot: was the crossing preceded by
enough returning interictal events to call it accumulation, or did the trajectory
ignite before any load had built?  E4 trajectories start cold from rest, so an early
bout is a startup-transient candidate until the count says otherwise.

Rows written before the ledger existed still carry the event list and the bout, which
is enough for the count and the class.  The dose and the per-event slow state are
reported as unavailable with the reason, never silently omitted.

Usage:
    python scripts/analyze_topic4_fcxr_lc3_entry.py [row.json ...]
With no argument it reads every completed row in the registered reconnaissance dir.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_fcxr_lc3_ledger import entry_from_record  # noqa: E402

RECON_DIR = os.path.join(
    ROOT, "results", "topic4_sef_hfo", "fcxr_lc3_dx_spatial_instability",
    "dynamic_reconnaissance")


def _rows(paths):
    if paths:
        return list(paths)
    found = sorted(glob.glob(os.path.join(RECON_DIR, "recon_noise*.json")))
    return [p for p in found
            if ".RUNNING." not in os.path.basename(p)
            and ".DONE." not in os.path.basename(p)
            and ".superseded." not in os.path.basename(p)]


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def report(path):
    with open(path) as f:
        record = json.load(f)
    entry = entry_from_record(record)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  lifecycle          : {(record.get('lifecycle') or {}).get('label')}")
    print(f"  verdict            : {record.get('verdict')}")
    print(f"  record length      : {_fmt(record.get('T_ms'), 0)} ms"
          f"   extended_to_cap={record.get('extended_to_cap')}")
    print(f"  20 s onset search  : onset_seen="
          f"{(record.get('onset_search_20s') or {}).get('onset_seen')}")
    print(f"  summary source     : {entry['source']}")
    print(f"  onset / offset     : {_fmt(entry['onset_ms'], 0)} / "
          f"{_fmt(entry['offset_ms'], 0)} ms")
    print(f"  events before onset: {entry['n_events_before_onset']}"
          f"   of which returning: {entry['n_returning_before_onset']}")
    print(f"  ENTRY CLASS        : {entry['entry_class']}")
    print(f"  first non-returning: {_fmt(entry['first_non_returning_index'])}")
    print(f"  cumulative dose    : Q_af={_fmt(entry['Q_af_to_onset'])}"
          f"  Q_rate={_fmt(entry['Q_rate_to_onset'])}")
    for reason in entry["unavailable"]:
        print(f"    unavailable      : {reason}")

    ledger = record.get("event_ledger")
    if not ledger:
        return
    print(f"\n  event ladder ({ledger['n_events']} events, "
          f"{ledger['n_returning']} returning)")
    print(f"  {'k':>3} {'t_on(ms)':>10} {'dur':>6} {'ret':>4} {'dose_af':>9} "
          f"{'Q_af':>9} {'D_core_A':>9} {'H_core_A':>9} {'X_core_A':>9} {'phase':>11}")
    for ev in ledger["events"]:
        post = ev.get("post") or {}
        d = post.get("D", {}).get("core_A")
        h = post.get("H", {}).get("core_A")
        x = post.get("X", {}).get("core_A")
        print(f"  {ev['index']:>3} {ev['t_on_ms']:>10.1f} {ev['dur_ms']:>6.1f} "
              f"{str(ev['returned']):>4} {ev['dose_af']:>9.4f} {ev['Q_af']:>9.4f} "
              f"{_fmt(d, 4):>9} {_fmt(h, 4):>9} {_fmt(x, 4):>9} {ev['phase']:>11}")
    post_offset = ledger["post_offset"]
    print(f"  post-offset returning events: {post_offset['n_returning']}"
          f"   durations_ms={[round(v, 1) for v in post_offset['durations_ms'][:8]]}")
    print("  frozen interictal reference: 34 returning events, duration 8-19 ms, "
          "participation 0.045-0.071, event rate 0.086-3.15 /s")


def main():
    paths = _rows(sys.argv[1:])
    if not paths:
        print("no completed reconnaissance rows yet")
        return
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()
