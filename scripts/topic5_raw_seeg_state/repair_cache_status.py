#!/usr/bin/env python
"""Validate a raw cache on disk and repair a status file that lies about it.

Why this exists: ``write_status(subject, **summary)`` raised
"got multiple values for argument 'subject'" *after* ``build_subject_cache`` had
already written the zarr, the per-contact scale, ``minute_filled.npy`` and
``cache_index.parquet``. Every artifact was complete and correct; only the
status JSON said FAILED. Rebuilding would have thrown away ~2000 s of work per
subject for a bookkeeping error.

This script does NOT trust the status file and does NOT trust the build. It
re-derives every invariant from the artifacts themselves and only then rewrites
the status, stamping ``repaired_from`` so the repair is auditable. A subject
that fails any invariant is left FAILED and must be rebuilt.

    python scripts/topic5_raw_seeg_state/repair_cache_status.py --subjects all
    python scripts/topic5_raw_seeg_state/repair_cache_status.py --subjects a,b --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.topic5_raw_seeg_state import contract  # noqa: E402


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def validate(subject: str) -> Dict[str, Any]:
    """Re-derive every invariant from the artifacts. Raises on any violation."""
    import pandas as pd

    from src.topic5_raw_seeg_state import raw_cache as RC

    d = contract.cache_dir(subject)
    arr, scale = RC.load_cache(subject)                     # also proves attrs exist
    scale = np.asarray(scale, dtype=np.float64)
    n_min_zarr, rem = divmod(int(arr.shape[0]), contract.MINUTE_SAMPLES)
    if rem:
        raise ValueError(f"zarr length {arr.shape[0]} is not a whole number of minutes")

    con = pd.read_parquet(contract.DATA_DIR / "contact_metadata.parquet")
    n_contacts = int((con.subject == subject).sum())
    if int(arr.shape[1]) != n_contacts:
        raise ValueError(f"zarr has {arr.shape[1]} columns, contact_metadata has {n_contacts}")
    if scale.shape != (n_contacts,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("contact_scale_uv missing, wrong length, or non-positive")

    ci = pd.read_parquet(d / "cache_index.parquet")
    for col in ("minute_index", "minute_start_epoch", "split", "cached", "filled"):
        if col not in ci.columns:
            raise ValueError(f"cache_index.parquet lacks column {col!r}")
    if int(ci.minute_index.max()) >= n_min_zarr:
        raise ValueError("cache_index references a minute beyond the zarr")

    cached = ci[ci.cached.astype(bool)]
    if cached.empty:
        raise ValueError("no cached minute")
    # the sealed bound is the one thing that must never be even slightly wrong
    dev_end = contract.dev_end_epoch(subject)
    last_end = float(cached.minute_start_epoch.max()) + 60.0
    if last_end > dev_end + 1e-6:
        raise ValueError(f"a cached minute ends at {last_end:.3f} >= dev_end {dev_end:.3f}")
    contract.assert_not_sealed(subject, cached.minute_start_epoch.to_numpy())

    # spot-check that the cached minutes actually carry signal
    idx = np.asarray(sorted(cached.minute_index.astype(int)))
    probe = idx[np.linspace(0, idx.size - 1, min(5, idx.size)).round().astype(int)]
    empty = []
    for mi in probe:
        a = int(mi) * contract.MINUTE_SAMPLES
        if not np.abs(np.asarray(arr[a:a + contract.MINUTE_SAMPLES, :])).max():
            empty.append(int(mi))
    if empty:
        raise ValueError(f"cached minutes {empty} are all zero")

    on_disk = dir_bytes(d)
    raw_bytes = int(arr.shape[0]) * int(arr.shape[1]) * 2
    return {
        "state": "DONE",
        "n_minutes_zarr": n_min_zarr,
        "n_minutes_cached": int(cached.shape[0]),
        "n_contacts": n_contacts,
        "bytes_uncompressed": raw_bytes,
        "bytes_on_disk": on_disk,
        "compression_ratio": round(raw_bytes / max(on_disk, 1), 3),
        "cached_hours": round(float(cached.shape[0]) / 60.0, 3),
        "last_cached_minute_end_epoch": last_end,
        "dev_end_epoch": dev_end,
        "validated_probe_minutes": [int(x) for x in probe],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", required=True, help="'all' or a comma list")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    subjects = (contract.cohort_subjects() if args.subjects == "all"
                else [s.strip() for s in args.subjects.split(",") if s.strip()])

    repaired: List[str] = []
    already: List[str] = []
    broken: List[str] = []
    for subject in subjects:
        st_path = contract.cache_dir(subject) / "BUILD_STATUS.json"
        if not st_path.exists():
            print(f"  {subject:24s} NO_STATUS (never built)")
            continue
        old = json.loads(st_path.read_text())
        if old.get("state") == "DONE" and old.get("contract_version") == contract.CONTRACT_VERSION:
            already.append(subject)
            print(f"  {subject:24s} already DONE")
            continue
        try:
            payload = validate(subject)
        except Exception as exc:  # noqa: BLE001
            broken.append(subject)
            print(f"  {subject:24s} STILL BROKEN :: {type(exc).__name__}: {exc}")
            continue
        payload.update({
            "subject": subject,
            "contract_version": contract.CONTRACT_VERSION,
            "code_revision": contract.code_revision(),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "repaired_from": {"state": old.get("state"), "error": old.get("error")},
            "repair_note": ("status rewritten by repair_cache_status.py after every "
                            "artifact invariant was re-derived from disk; the build "
                            "itself was not re-run"),
        })
        print(f"  {subject:24s} REPAIRED  cached={payload['n_minutes_cached']} min, "
              f"{payload['bytes_on_disk']/1e9:.2f} GB on disk, "
              f"ratio {payload['compression_ratio']}x"
              + ("  [dry-run]" if args.dry_run else ""))
        if not args.dry_run:
            contract.atomic_write_json(st_path, payload)
        repaired.append(subject)

    print(f"\nrepaired={len(repaired)} already_done={len(already)} still_broken={len(broken)}")
    if broken:
        print("must be rebuilt: " + ", ".join(broken), file=sys.stderr)
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
