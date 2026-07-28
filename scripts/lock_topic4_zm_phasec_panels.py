#!/usr/bin/env python
"""Lock activity-independent Phase-C neuron panels from canonical geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.run_topic4_zm_branch_decision as R  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
PATH = OUT / "phasec_panels.json"
SEEDS = (1, 3, 4)
ANALYSIS_PER_STRATUM = 512
PAIRWISE_PER_STRATUM = 128


def _ranked(indices, config_sha, seed, label, n):
    scored = [
        (
            hashlib.sha256(
                f"{config_sha}|{seed}|{label}|{int(index)}".encode()
            ).digest(),
            int(index),
        )
        for index in np.asarray(indices, int)
    ]
    scored.sort()
    if len(scored) < n:
        raise RuntimeError(f"{label}: requested {n}, available {len(scored)}")
    return [index for _digest, index in scored[:n]]


def _canonical_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def build_payload():
    rows = {}
    for seed in SEEDS:
        ctx = R.build_context(seed)
        core = np.asarray(ctx["core"], bool)
        core_ids = np.flatnonzero(core)
        surround_ids = np.flatnonzero(~core)
        analysis = (
            _ranked(core_ids, ctx["cfg_sha"], seed, "analysis_core", ANALYSIS_PER_STRATUM)
            + _ranked(
                surround_ids, ctx["cfg_sha"], seed, "analysis_surround",
                ANALYSIS_PER_STRATUM,
            )
        )
        pairwise = (
            _ranked(core_ids, ctx["cfg_sha"], seed, "pairwise_core", PAIRWISE_PER_STRATUM)
            + _ranked(
                surround_ids, ctx["cfg_sha"], seed, "pairwise_surround",
                PAIRWISE_PER_STRATUM,
            )
        )
        row = {
            "seed": seed,
            "config_sha": ctx["cfg_sha"],
            "NE": int(ctx["S"]["NE"]),
            "analysis_panel_E_ids": analysis,
            "analysis_panel_n_core": ANALYSIS_PER_STRATUM,
            "analysis_panel_n_surround": ANALYSIS_PER_STRATUM,
            "pairwise_panel_E_ids": pairwise,
            "pairwise_panel_n_core": PAIRWISE_PER_STRATUM,
            "pairwise_panel_n_surround": PAIRWISE_PER_STRATUM,
            "selection": "sha256(config_sha|seed|panel_stratum|E_local_id)",
            "activity_independent": True,
        }
        row["panel_sha256"] = hashlib.sha256(_canonical_bytes(row)).hexdigest()
        rows[str(seed)] = row
    payload = {
        "schema": "zm_phasec_panels_v1_2026-07-28",
        "seeds": rows,
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload()
    for seed, row in payload["seeds"].items():
        print(
            f"[phasec panels] seed={seed} "
            f"analysis={len(row['analysis_panel_E_ids'])} "
            f"pairwise={len(row['pairwise_panel_E_ids'])} "
            f"sha={row['panel_sha256'][:16]}",
            flush=True,
        )
    if args.check_only:
        if PATH.exists():
            old = json.load(PATH.open())
            if _canonical_bytes(old) != _canonical_bytes(payload):
                raise RuntimeError(
                    "existing Phase-C panel manifest differs from live selection"
                )
        print(json.dumps({
            "status": "validated",
            "manifest_sha256": payload["manifest_sha256"],
            "path_exists": PATH.exists(),
        }, sort_keys=True))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if PATH.exists():
        old = json.load(PATH.open())
        if _canonical_bytes(old) != _canonical_bytes(payload):
            raise RuntimeError("existing Phase-C panel manifest differs; refusing overwrite")
        print(f"[phasec panels] reused {PATH}")
        return
    tmp = PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, PATH)
    print(f"[phasec panels] locked {PATH} sha={payload['manifest_sha256']}", flush=True)


if __name__ == "__main__":
    main()
