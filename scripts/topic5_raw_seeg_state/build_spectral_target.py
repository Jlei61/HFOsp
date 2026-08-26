#!/usr/bin/env python
"""Build the minute spectral target + train stats + artifact mask.  Worker B.

Runs strictly after ``build_raw_cache.py`` for the same subject, in this order:
targets (and the broadband / saturation side arrays) -> train-only statistics ->
artifact mask -> hand the mask back to Worker A's
``data_contract.refine_minute_index_with_artifacts`` so the window index knows
which contact-minutes survived.  If that refinement hook does not exist yet the
mask is still written and a TODO is logged instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import contract  # noqa: E402


def status_path(subject: str) -> Path:
    return contract.cache_dir(subject) / "TARGET_STATUS.json"


def write_status(subject: str, /, **fields) -> None:
    """``subject`` is positional-only on purpose.

    The per-subject summaries these builders return carry their own
    ``subject`` key, so a plain ``write_status(subject, **summary)`` raised
    "got multiple values for argument 'subject'" -- and it raised AFTER the
    zarr, the scale and the cache index had all been written, so a 2000 s
    build ended up marked FAILED with every artifact on disk and correct.
    Positional-only lets the duplicate key land harmlessly in ``fields``.
    """
    payload = {"subject": subject, "contract_version": contract.CONTRACT_VERSION,
               "code_revision": contract.code_revision(),
               "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    payload.update(fields)
    contract.atomic_write_json(status_path(subject), payload)


def is_done(subject: str) -> bool:
    p = status_path(subject)
    if not p.exists():
        return False
    try:
        st = json.loads(p.read_text())
    except Exception:
        return False
    return st.get("state") == "DONE" and st.get("contract_version") == contract.CONTRACT_VERSION


def _refine_window_index(subject: str, data_dir: Path, log) -> dict:
    """Feed the artifact mask back into Worker A's window index.

    Two polarity/shape conversions happen here and both are load-bearing:

    * ``spectral_target.artifact_mask`` returns **True = artifact** (the R0.1
      brief's convention).  ``data_contract.refine_minute_index_with_artifacts``
      documents its input as **True = survives**.  We pass ``~mask``.
    * A's hook wants only the ``contact_valid`` columns, in ``channel_index``
      order, because that subset is the denominator of the valid-contact
      fraction.  Our cache carries every contact_metadata row, so we subset.

    The refined rows are written per subject to ``<cache>/<subject>/
    window_index_refined.parquet``.  The parent process merges them into
    ``data/window_index.parquet`` once, single-threaded -- several worker
    processes must never write that shared file.
    """
    try:
        from src.topic5_raw_seeg_state import data_contract
    except ImportError:
        log(f"TODO {subject}: src/topic5_raw_seeg_state/data_contract.py does not exist "
            "yet; artifact mask written but window_index NOT refined")
        return {"refined": False, "reason": "data_contract module missing"}
    fn = getattr(data_contract, "refine_minute_index_with_artifacts", None)
    if fn is None:
        log(f"TODO {subject}: data_contract.refine_minute_index_with_artifacts is not "
            "defined yet; artifact mask written but window_index NOT refined")
        return {"refined": False, "reason": "refine_minute_index_with_artifacts missing"}

    import numpy as np
    import pandas as pd
    import zarr

    from src.topic5_raw_seeg_state import spectral_target as st

    contacts = pd.read_parquet(data_dir / "contact_metadata.parquet")
    contacts = contacts[contacts["subject"] == subject].sort_values("channel_index")
    keep = contacts["contact_valid"].to_numpy().astype(bool)
    if not keep.any():
        return {"refined": False, "reason": "no contact_valid channels"}

    wi = pd.read_parquet(data_dir / "window_index.parquet")
    wi = wi[wi["subject"] == subject].sort_values("minute_index").reset_index(drop=True)
    mask = np.asarray(
        zarr.open_array(str(st.artifact_mask_path(subject)), mode="r")[:], dtype=bool
    )
    if mask.shape != (len(wi), len(contacts)):
        raise ValueError(
            f"{subject}: artifact mask {mask.shape} does not match window_index "
            f"({len(wi)} minutes) x contact_metadata ({len(contacts)} contacts)"
        )
    refined = fn(wi, ~mask[:, keep])
    out_path = contract.cache_dir(subject) / "window_index_refined.parquet"
    refined.to_parquet(out_path, index=False)
    log(f"{subject}: refined window index -> {out_path} "
        f"(usable {int(refined['minute_usable'].sum())}/{len(refined)}, "
        f"ctx_ok {int(refined['ctx_ok'].sum())})")
    return {
        "refined": True,
        "path": str(out_path),
        "n_minutes": int(len(refined)),
        "n_minute_usable": int(refined["minute_usable"].sum()),
        "n_ctx_ok": int(refined["ctx_ok"].sum()),
        "n_horizon_ok": {f"h{h}": int(refined[f"h{h}_ok"].sum())
                         for h in contract.HORIZONS_MIN},
    }


def merge_refined_window_index(subjects, data_dir: Path, log) -> dict:
    """Fold every ``window_index_refined.parquet`` into the shared window index.

    Runs once, in the parent process, after the pool has drained.  The
    pre-artifact frame is kept as ``window_index_preartifact.parquet`` the first
    time so Worker A's original output is never lost.
    """
    import pandas as pd

    target = data_dir / "window_index.parquet"
    backup = data_dir / "window_index_preartifact.parquet"
    parts = {s: contract.cache_dir(s) / "window_index_refined.parquet" for s in subjects}
    parts = {s: p for s, p in parts.items() if p.exists()}
    if not parts:
        return {"merged": 0}
    base = pd.read_parquet(target)
    if not backup.exists():
        base.to_parquet(backup, index=False)
    keep = base[~base["subject"].isin(parts)]
    merged = pd.concat([keep] + [pd.read_parquet(p) for p in parts.values()],
                       ignore_index=True)
    merged = merged.sort_values(["subject", "minute_index"]).reset_index(drop=True)
    merged = merged[list(contract.WINDOW_INDEX_COLUMNS)]
    tmp = target.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(target)
    log(f"merged artifact-refined window index for {len(parts)} subjects -> {target}")
    return {"merged": len(parts), "subjects": sorted(parts), "backup": str(backup)}


def _worker(subject: str, data_dir: str) -> dict:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    from src.topic5_raw_seeg_state import spectral_target as st

    t0 = time.time()
    say = lambda m: print(m, flush=True)  # noqa: E731
    write_status(subject, state="RUNNING", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        tgt = st.build_subject_targets(subject, log=say)
        stats = st.compute_train_stats(subject, log=say)
        mask = st.artifact_mask(subject, log=say)
        # Second standardisation pass: the artifact tail carries most of the
        # variance, so target_mean/target_std must be re-estimated on the clean
        # train contact-minutes. See refine_train_stats_with_artifacts.
        stats = st.refine_train_stats_with_artifacts(subject, log=say)
        refine = _refine_window_index(subject, Path(data_dir), say)
        summary = {
            "state": "DONE",
            "target": tgt,
            "artifact": mask,
            "n_train_minutes": stats["n_train_minutes"],
            "standardisation_audit": stats.get("standardisation_audit"),
            "n_contacts": stats["n_contacts"],
            "window_index_refinement": refine,
            "wall_seconds": time.time() - t0,
        }
        write_status(subject, **summary)
        return {"subject": subject, **summary}
    except Exception as exc:
        tb = traceback.format_exc()
        write_status(subject, state="FAILED", error=str(exc), traceback=tb,
                     wall_seconds=time.time() - t0)
        return {"subject": subject, "state": "FAILED", "error": str(exc), "traceback": tb}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--jobs", type=int, default=6, help="processes (keep <= 10)")
    ap.add_argument("--data-dir", default=str(contract.DATA_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.subjects.strip() == "all":
        subjects = contract.cohort_subjects()
    else:
        subjects = [s for s in args.subjects.replace(",", " ").split() if s]
    known = set(contract.cohort_subjects())
    unknown = [s for s in subjects if s not in known]
    if unknown:
        raise SystemExit(f"unknown subjects (not in the frozen split): {unknown}")

    missing = [s for s in subjects if not contract.raw_cache_path(s).exists()]
    if missing:
        raise SystemExit(f"raw cache missing for {missing}; run build_raw_cache.py first")

    todo = subjects if args.force else [s for s in subjects if not is_done(s)]
    print(f"subjects={len(subjects)} todo={len(todo)}", flush=True)
    if not todo:
        return 0

    jobs = max(1, min(int(args.jobs), 10, len(todo)))
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_worker, s, args.data_dir): s for s in todo}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            extra = (f"artifact_rate={r['artifact']['artifact_rate_cached']:.4f}"
                     if r["state"] == "DONE" else r.get("error", ""))
            print(f"[{r['state']}] {r['subject']} {extra}", flush=True)

    failed = [r["subject"] for r in results if r["state"] != "DONE"]
    done = [r["subject"] for r in results if r["state"] == "DONE"]
    merge = merge_refined_window_index(done, Path(args.data_dir),
                                       lambda m: print(m, flush=True))
    contract.atomic_write_json(
        contract.LOG_DIR / "build_spectral_target_last_run.json",
        {"n_requested": len(subjects), "n_done": len(done),
         "failed": failed, "window_index_merge": merge, "results": results},
    )
    print(f"done={len(results) - len(failed)} failed={len(failed)} {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
