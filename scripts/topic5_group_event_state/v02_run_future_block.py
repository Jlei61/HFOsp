#!/usr/bin/env python3
"""Future-block evaluation for one or more patients (Agent A, A1/A2).

Runs the shared anchor grid, the baseline and -- when a producer's frozen state
is supplied -- the nested increment and its block-shift null.  With no producer
this is a complete ``B_multiscale`` run, which is exactly what can be done while
the GPUs are held by another queue.

Usage:

    python scripts/topic5_group_event_state/v02_run_future_block.py \
        --subjects epilepsiae_916 epilepsiae_253 --workers 8

    # later, once producers exist
    ... --state-dir /data/hfosp_group_event_state_v0_2/agent_a/states/P_slow_seed1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from src.topic5_group_event_state.v02.evaluate import (  # noqa: E402
    EvaluationConfig,
    evaluate_subject,
)
from src.topic5_group_event_state.v02.readout import ReadoutConfig  # noqa: E402
from src.topic5_group_event_state.v02.registry import (  # noqa: E402
    atomic_write_json,
    source_commit,
)
from src.topic5_group_event_state.v02.runtime import (  # noqa: E402
    ResourceLease,
    already_done,
    config_fingerprint,
    save_result,
    write_status,
)
from src.topic5_group_event_state.v02.subject import (  # noqa: E402
    SubjectTimelineConfig,
    load_subject_timeline,
    timeline_summary,
    trainability,
)

DEFAULT_OUT = Path("/data/hfosp_group_event_state_v0_2/agent_a/future_block")
SHARED_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/shared"
)

# Pre-registered cohorts (contract clause file, section "预注册的队列").
A1_SMOKE = ("epilepsiae_916", "epilepsiae_253", "epilepsiae_1073")
A2_MIDSTAGE = (
    "epilepsiae_1073", "epilepsiae_1077", "epilepsiae_1096", "epilepsiae_1125",
    "epilepsiae_1146", "epilepsiae_253", "epilepsiae_384", "epilepsiae_548",
)


def _load_states(state_dirs: list[Path], subject: str, n_anchors: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for d in state_dirs:
        path = Path(d) / f"{subject}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            values = np.asarray(z["state"], dtype=np.float64)
            if "t_anchor" in z and values.shape[0] != n_anchors:
                raise ValueError(
                    f"{path}: {values.shape[0]} state rows for {n_anchors} anchors"
                )
        out[Path(d).name] = values
    return out


def _run_one(args: tuple) -> dict:
    subject, out_root, state_dirs, cfg_hash, eval_kwargs = args
    started = time.time()
    result_path = Path(out_root) / "per_subject" / f"{subject}.json"
    if already_done(result_path, cfg_hash):
        return {"subject": subject, "status": "skipped_done"}
    try:
        tl = load_subject_timeline(subject, config=SubjectTimelineConfig())
        states = _load_states([Path(d) for d in state_dirs], subject, tl.grid.n_anchors)
        result = evaluate_subject(
            tl,
            states,
            config=EvaluationConfig(
                readout=ReadoutConfig(max_iter=eval_kwargs["max_iter"]),
                run_mlp_baseline=eval_kwargs["run_mlp"],
                mlp_hidden=eval_kwargs["mlp_hidden"],
                shift_extra_steps=tuple(eval_kwargs["shift_extra_steps"]),
            ),
        )
        result["timeline"] = timeline_summary(tl)
        result["trainability"] = trainability(tl)
        result["seconds"] = round(time.time() - started, 1)
        save_result(result_path, result, cfg_hash)
        return {
            "subject": subject,
            "status": "ok",
            "seconds": result["seconds"],
            "n_anchors": tl.grid.n_anchors,
        }
    except Exception as exc:
        payload = {
            "subject": subject,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "seconds": round(time.time() - started, 1),
        }
        atomic_write_json(Path(out_root) / "failures" / f"{subject}.json", payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--cohort", choices=("a1", "a2", "all"), default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--state-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--shift-extra-steps", type=int, nargs="+", default=[1, 4, 12])
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    if args.subjects:
        subjects = list(args.subjects)
    elif args.cohort == "a1":
        subjects = list(A1_SMOKE)
    elif args.cohort == "a2":
        subjects = list(A2_MIDSTAGE)
    else:
        subjects = sorted(
            p.name for p in Path("/data/hfosp_group_event_state_v0_1/dataset").iterdir()
            if (p / "index.json").exists()
        )

    out_root = Path(args.out_root) / args.tag
    (out_root / "per_subject").mkdir(parents=True, exist_ok=True)
    (out_root / "failures").mkdir(parents=True, exist_ok=True)

    eval_kwargs = {
        "max_iter": args.max_iter,
        "run_mlp": not args.no_mlp,
        "mlp_hidden": args.mlp_hidden,
        "shift_extra_steps": list(args.shift_extra_steps),
    }
    commit = source_commit(ROOT)
    cfg_hash = config_fingerprint(
        SubjectTimelineConfig().as_dict(), eval_kwargs,
        sorted(Path(d).name for d in args.state_dir), commit,
    )

    lease = ResourceLease(SHARED_ROOT / "resource_leases" / f"agent_a_{args.tag}.json",
                          "agent_a")
    lease.acquire(gpus=[], slots=args.workers, task="future_block",
                  out_root=str(out_root), config_hash=cfg_hash)
    status_path = out_root / "STATUS.json"
    write_status(status_path, state="running", n_subjects=len(subjects),
                 done=0, pid=os.getpid(), config_hash=cfg_hash, commit=commit)

    payload = [
        (s, str(out_root), [str(d) for d in args.state_dir], cfg_hash, eval_kwargs)
        for s in subjects
    ]
    results = []
    started = time.time()
    try:
        with mp.get_context("spawn").Pool(processes=max(1, args.workers)) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_one, payload), start=1):
                results.append(res)
                print(f"[{i}/{len(subjects)}] {res['subject']}: {res['status']} "
                      f"{res.get('seconds', '')}", flush=True)
                write_status(status_path, state="running", n_subjects=len(subjects),
                             done=i, pid=os.getpid(), config_hash=cfg_hash,
                             commit=commit, elapsed=round(time.time() - started, 1))
                lease.beat(gpus=[], slots=args.workers, task="future_block",
                           done=i, n_subjects=len(subjects))
    finally:
        lease.release()

    manifest = {
        "tag": args.tag,
        "commit": commit,
        "config_hash": cfg_hash,
        "timeline_config": SubjectTimelineConfig().as_dict(),
        "eval_config": eval_kwargs,
        "state_dirs": [str(d) for d in args.state_dir],
        "subjects": sorted(subjects),
        "results": sorted(results, key=lambda r: r["subject"]),
        "n_ok": sum(1 for r in results if r["status"] in ("ok", "skipped_done")),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    atomic_write_json(out_root / "manifest.json", manifest)
    write_status(status_path, state="finished", n_subjects=len(subjects),
                 done=len(results), n_failed=manifest["n_failed"],
                 config_hash=cfg_hash, commit=commit)
    print(json.dumps({k: manifest[k] for k in
                      ("n_ok", "n_failed", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
