#!/usr/bin/env python3
"""Expand four selected fast phenotypes into the seed-1 M response panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_topic4_zm_lifecycle_sprint_batch as B  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
M_COORDS = ((0.0, 500.0),) + tuple(
    (g_m, tau_m) for g_m in (1.0, 3.0, 10.0, 30.0)
    for tau_m in (500.0, 2000.0)
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fast_config(row):
    if "arm" in row and "tau_D_ms" in row:
        source = dict(row)
    else:
        mechanism = row["mechanism"]
        dep = mechanism["i2e_depression"]
        iadapt = mechanism.get("i_adaptation") or {}
        source = {
            "arm": mechanism["arm"],
            "tau_D_ms": dep["tau_D_ms"],
            "d_star": dep["d_star_nominal"],
            "strength_scale": mechanism.get("strength_scale", 1.0),
        }
        if source["arm"] == "combined":
            source.update(
                tau_aI_ms=iadapt["tau_aI_ms"], f_aI=iadapt["f_aI"],
            )
    keep = {
        key: source[key] for key in (
            "arm", "tau_D_ms", "d_star", "strength_scale",
            "tau_aI_ms", "f_aI",
        ) if key in source
    }
    if keep["arm"] not in {"i2e", "combined"}:
        raise ValueError("M panel requires an i2e or combined fast phenotype")
    return keep


def build_manifest(selection, *, T_ms=20000.0):
    selected = list(selection.get("selected", selection.get("rows", [])))
    if len(selected) != 4:
        raise ValueError("M panel requires exactly four selected fast phenotypes")
    prepared = [
        (
            rank,
            _fast_config(source),
            source.get("config_id", source.get("stem", f"selected_{rank}")),
        )
        for rank, source in enumerate(selected)
    ]
    rows = []
    # Coordinate-major ordering makes each 8-worker wave scientifically
    # interpretable across all four phenotypes.  In particular, wave 1 contains
    # every g_M=0 paired control and every native g_M=1,tau_M=500 trajectory,
    # rather than spending eight slots on a single phenotype.
    for g_m, tau_m in M_COORDS:
        for rank, fast, source_id in prepared:
            row = {
                "family": "m_response_panel", "selection_rank": rank,
                "source_fast_id": source_id, **fast,
                "g_M": g_m, "tau_M_ms": tau_m, "g_Z": 1.0,
                "T_ms": float(T_ms), "burn_ms": 1000.0,
            }
            row["config_id"] = B._cfg_id(row)
            rows.append(row)
    return {
        "schema": "topic4_zm_lifecycle_m_response_panel_v1_2026-08-02",
        "created_at_utc": _now(), "seed": 1, "paired_noise": True,
        "selection_source": selection.get("selection_source"),
        "n_selected_fast_phenotypes": len(selected),
        "n_M_coordinates_per_phenotype": len(M_COORDS),
        "n_configs": len(rows), "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection-json", type=Path, required=True)
    ap.add_argument("--T-ms", type=float, default=20000.0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--min-mem-gb", type=float, default=90.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    selection = json.loads(args.selection_json.read_text())
    manifest = build_manifest(selection, T_ms=args.T_ms)
    manifest_path = OUT / "m_panel_manifest.json"
    B._atomic_json(manifest_path, manifest)
    if args.manifest_only:
        print(manifest_path)
        return
    print(B.run_manifest(
        manifest_path, OUT / "m_panel_run_ledger.json",
        args.max_workers, args.min_mem_gb, args.poll_s,
    ))


if __name__ == "__main__":
    main()
