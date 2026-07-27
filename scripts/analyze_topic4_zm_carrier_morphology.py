#!/usr/bin/env python
"""Build the offline native-resolution carrier morphology artifact."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.topic4_zm_carrier_morphology import (  # noqa: E402
    MORPHOLOGY_VERSION,
    characterize_confirmation,
)

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".carrier_morphology.", suffix=".json",
                               dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    rows = []
    for tier in ("dt2", "long"):
        pattern = os.path.join(
            OUT, "confirmations", tier, "seed*", "fork_matrix.json"
        )
        for manifest_path in sorted(glob.glob(pattern)):
            manifest = json.load(open(manifest_path))
            for source_row in manifest.get("rows", []):
                trace_path = os.path.join(
                    os.path.dirname(manifest_path), "traces",
                    f"{source_row['bin_name']}__{source_row['fast_phase']}__"
                    f"{source_row['arm']}__{source_row['replicate']}.npz",
                )
                if not os.path.exists(trace_path):
                    continue
                with np.load(trace_path, allow_pickle=False) as z:
                    required = {"r_all", "lfp", "lfp_fs", "kymo_axial",
                                "burn_in_ms", "bin_ms"}
                    missing = sorted(required - set(z.files))
                    if missing:
                        raise RuntimeError(f"{trace_path}: missing arrays {missing}")
                    metrics = characterize_confirmation(
                        z["r_all"], z["lfp"], float(z["lfp_fs"]),
                        burn_in_ms=float(z["burn_in_ms"]),
                        kymo_axial=z["kymo_axial"], bin_ms=float(z["bin_ms"]),
                    )
                rows.append({
                    "tier": tier,
                    "seed": int(source_row["seed"]),
                    "row_key": source_row["key"],
                    "resolution": source_row.get("resolution"),
                    "dt_ms": source_row.get("dt"),
                    "T_cont_ms": source_row.get("T_cont_ms"),
                    "source_manifest": os.path.relpath(manifest_path, _ROOT),
                    "source_trace": os.path.relpath(trace_path, _ROOT),
                    "source_trace_sha256": _sha256(trace_path),
                    **metrics,
                })
    if not rows:
        raise SystemExit("no completed confirmation traces")
    payload = {
        "morphology_version": MORPHOLOGY_VERSION,
        "analysis_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": sorted(rows, key=lambda x: (x["tier"], x["seed"], x["row_key"])),
    }
    out = os.path.join(OUT, "confirmations", "carrier_morphology.json")
    _write_json_atomic(out, payload)
    print(f"[morphology] rows={len(rows)} -> {os.path.relpath(out, _ROOT)}")


if __name__ == "__main__":
    main()
