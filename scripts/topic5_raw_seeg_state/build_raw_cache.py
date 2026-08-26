#!/usr/bin/env python
"""Build the decimated int16 raw cache, one process per subject.  Worker B.

    LD_LIBRARY_PATH=$CONDA_LIB $PY scripts/topic5_raw_seeg_state/build_raw_cache.py \
        --subjects yuquan_huanghanwen --jobs 1

Resumability: each subject gets ``<cache>/<subject>/BUILD_STATUS.json`` written
atomically with state RUNNING / DONE / FAILED.  A subject already DONE under the
current ``contract.CONTRACT_VERSION`` is skipped unless ``--force``.
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
    return contract.cache_dir(subject) / "BUILD_STATUS.json"


def read_status(subject: str) -> dict:
    p = status_path(subject)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


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
    st = read_status(subject)
    return (st.get("state") == "DONE"
            and st.get("contract_version") == contract.CONTRACT_VERSION)


def _worker(subject: str, data_dir: str, cache_cap: bool, chunk_minutes: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import pandas as pd

    from src.topic5_raw_seeg_state import raw_cache as rc

    t0 = time.time()
    write_status(subject, state="RUNNING", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        d = Path(data_dir)
        manifest = pd.read_parquet(d / "dataset_manifest.parquet")
        contacts = pd.read_parquet(d / "contact_metadata.parquet")
        window = pd.read_parquet(d / "window_index.parquet")
        split = contract.load_subject_splits()[subject]
        summary = rc.build_subject_cache(
            subject, manifest, contacts, window,
            out_path=contract.raw_cache_path(subject),
            train_end_epoch=split.train_end_epoch,
            dev_end_epoch=split.dev_end_epoch,
            cache_cap=cache_cap,
            chunk_minutes=chunk_minutes,
            log=lambda m: print(m, flush=True),
        )
        write_status(subject, state="DONE", **summary)
        return {"subject": subject, "state": "DONE", **summary}
    except Exception as exc:
        tb = traceback.format_exc()
        write_status(subject, state="FAILED", error=str(exc), traceback=tb,
                     wall_seconds=time.time() - t0)
        return {"subject": subject, "state": "FAILED", "error": str(exc), "traceback": tb}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default="all",
                    help="'all' or a comma/space separated subject list")
    ap.add_argument("--jobs", type=int, default=6, help="processes (keep <= 10)")
    ap.add_argument("--data-dir", default=str(contract.DATA_DIR))
    ap.add_argument("--chunk-minutes", type=int, default=5)
    ap.add_argument("--no-cache-cap", action="store_true",
                    help="cache every covered dev minute instead of the 36h/12h caps")
    ap.add_argument("--force", action="store_true", help="rebuild subjects already DONE")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.subjects.strip() == "all":
        subjects = contract.cohort_subjects()
    else:
        subjects = [s for s in args.subjects.replace(",", " ").split() if s]
    known = set(contract.cohort_subjects())
    unknown = [s for s in subjects if s not in known]
    if unknown:
        raise SystemExit(f"unknown subjects (not in the frozen split): {unknown}")

    todo = subjects if args.force else [s for s in subjects if not is_done(s)]
    skipped = sorted(set(subjects) - set(todo))
    print(f"subjects={len(subjects)} todo={len(todo)} skipped_done={len(skipped)}", flush=True)
    if args.dry_run:
        print(json.dumps({"todo": todo, "skipped": skipped}, indent=2))
        return 0
    if not todo:
        return 0

    jobs = max(1, min(int(args.jobs), 10, len(todo)))
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_worker, s, args.data_dir, not args.no_cache_cap,
                            args.chunk_minutes): s for s in todo}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = r["state"]
            extra = (f"{r.get('bytes_on_disk', 0)/1e9:.2f} GB "
                     f"{r.get('wall_seconds', 0):.0f}s" if tag == "DONE" else r.get("error", ""))
            print(f"[{tag}] {r['subject']} {extra}", flush=True)

    failed = [r["subject"] for r in results if r["state"] != "DONE"]
    contract.atomic_write_json(
        contract.LOG_DIR / "build_raw_cache_last_run.json",
        {"n_requested": len(subjects), "n_done": len(results) - len(failed),
         "failed": failed, "results": results},
    )
    print(f"done={len(results) - len(failed)} failed={len(failed)} {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
