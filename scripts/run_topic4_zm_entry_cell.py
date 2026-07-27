#!/usr/bin/env python
"""Run one crash-safe Z-entry boundary cell into an isolated part manifest.

This is an orchestration-only companion to ``run_topic4_zm_branch_decision``.
It deliberately never writes ``entry_probes.json``: many independent workers
may therefore evaluate different lambda/noise cells without sharing a writer.
``merge_topic4_zm_entry_parts.py`` is the sole part-to-canonical merger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _path in (_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_topic4_zm_branch_decision as R  # noqa: E402


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _part_path(seed: int, lam: float, replicate: str) -> str:
    return os.path.join(
        R.OUT,
        "boundaries",
        "entry",
        f"seed{int(seed)}",
        "parts",
        f"lambda_{_slug(lam)}__{replicate}.json",
    )


def _canonical_row(seed: int, key: str):
    path = os.path.join(
        R.OUT, "boundaries", "entry", f"seed{int(seed)}", "entry_probes.json"
    )
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    return next((row for row in payload.get("rows", []) if row.get("key") == key), None)


def run_cell(seed: int, lam: float, replicate: str):
    lam = float(lam)
    if lam not in set(R.ENTRY_LEVELS):
        raise SystemExit(
            f"lambda={lam:g} is not a locked entry level {R.ENTRY_LEVELS}"
        )
    allowed_replicates = {
        R.ENTRY_BASE_REPLICATE,
        *R.ENTRY_EXPANSION_REPLICATES,
    }
    if replicate not in allowed_replicates:
        raise SystemExit(
            f"replicate={replicate!r} is not locked: {sorted(allowed_replicates)}"
        )
    key = f"lambda={lam:g}|{replicate}"
    path = _part_path(seed, lam, replicate)
    if os.path.exists(path):
        old = json.load(open(path))
        if (
            old.get("complete") is True
            and old.get("row", {}).get("key") == key
            and old.get("boundary_version") == R.BD.BOUNDARY_VERSION
        ):
            print(f"[entry-cell] already complete -> {path}", flush=True)
            return old
    existing = _canonical_row(seed, key)
    if (
        existing is not None
        and existing.get("completed") is True
        and existing.get("boundary_version") == R.BD.BOUNDARY_VERSION
    ):
        payload = {
            "complete": True,
            "source": "canonical_row_reuse",
            "boundary_version": R.BD.BOUNDARY_VERSION,
            "row": existing,
        }
        R.write_json_atomic(path, payload)
        print(f"[entry-cell] canonical row already complete -> {path}", flush=True)
        return payload

    ctx = R.build_context(seed, resolution="dt")
    verdict_path = os.path.join(R.OUT, "branch_verdict.json")
    verdict = json.load(open(verdict_path)) if os.path.exists(verdict_path) else {}
    if not R.SR.source_rhythm_authorized(verdict):
        raise SystemExit("entry cell requires the confirmed two-seed source carrier")
    if "carrier_fast_only" not in set(verdict.get("smallest_positive_subsystem") or []):
        raise SystemExit("entry cell is locked to the observed fast-only carrier")

    nE = int(ctx["S"]["NE"])
    anchor_path = os.path.join(R.OUT, "anchors", f"seed{seed}", "anchor.json")
    anchor = json.load(open(anchor_path))
    captured = {
        f"{row['bin_name']}__{row['fast_phase']}": row
        for row in anchor.get("captured_states", [])
    }
    required = (
        "pre_entry__natural",
        "onset_adjacent__natural",
        "bounded_mid__peak",
    )
    missing = [tag for tag in required if tag not in captured]
    if missing:
        raise SystemExit(f"entry cell missing states: {missing}")
    engine_sha = ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"]
    states = {}
    manifests = {}
    for tag in required:
        states[tag], manifests[tag] = R.CK.load_state_npz(
            os.path.join(_ROOT, captured[tag]["path"]),
            expected_config_sha=ctx["cfg_sha"],
            expected_engine_sha=engine_sha,
            expected_dt=ctx["dt"],
        )
    pre = states["pre_entry__natural"]
    onset = states["onset_adjacent__natural"]
    carrier = states["bounded_mid__peak"]
    state = R.BD.interpolate_slow_state(
        pre, carrier, lam, coordinates=("z",), nE=nE
    )
    state["slow.m"] = np.asarray(onset["slow.m"]).copy()
    for slow_key in ("slow.rE_fast", "slow.mu_G", "slow.S_G"):
        state[slow_key] = np.asarray(onset[slow_key]).copy()

    bank = R.NB.build_noise_bank(
        ctx["cfg_sha"],
        seed,
        int(captured["pre_entry__natural"]["t_step"]),
        replicate,
    )
    run = R.run_continuation(
        ctx,
        state,
        "freeze_all",
        bank,
        anchor["locks"],
        T_ms=R.ENTRY_RESPONSE_MS,
    )
    summary = R.summarize_continuation(
        run, anchor["locks"], T_ms=R.ENTRY_RESPONSE_MS
    )
    zE = np.asarray(state["slow.z"], float)[:nE]
    row = {
        "key": key,
        "seed": int(seed),
        "lambda": lam,
        "replicate": replicate,
        "bank_sha": bank["bank_sha"],
        "entered_carrier": bool(summary["survived"] and summary["stationarity_ok"]),
        "completed": True,
        "boundary_version": R.BD.BOUNDARY_VERSION,
        "producer_git_sha": ctx["runtime_git_sha"],
        "z_core_mean": float(zE[ctx["core"]].mean()),
        "z_surround_mean": float(zE[~ctx["core"]].mean()),
        **summary,
    }
    payload = {
        **R.provenance(ctx, phase="entry_boundary_cell"),
        "complete": True,
        "boundary_version": R.BD.BOUNDARY_VERSION,
        "source_state_hashes": {
            tag: manifests[tag]["state_hash"] for tag in required
        },
        "response_ms": R.ENTRY_RESPONSE_MS,
        "row": row,
    }
    R.write_json_atomic(path, payload)
    print(
        f"[entry-cell] seed={seed} lambda={lam:g} rep={replicate} "
        f"entered={row['entered_carrier']} end={summary['end_reason']} "
        f"wall={summary['wall_s']}s -> {path}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(1, 3, 4))
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--replicate", default=R.ENTRY_BASE_REPLICATE)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("refusing N=40000 entry cell without --confirm-run")
    run_cell(args.seed, args.lam, args.replicate)


if __name__ == "__main__":
    main()
