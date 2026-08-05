"""Milestone H launcher: run the frozen cohort, seed-major, resumable.

Priority order is every patient at seed 1 before any patient reaches seed 2, so
that an interrupted run still yields complete cohort coverage at one seed rather
than a few patients at three.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

ARMS = (
    "STATIC_CONTACT",
    "ORDINARY_GRU",
    "CONTACT_GRAPH_RNN",
    "LATENT_FIXED_LOCAL_RNN",
    "LATENT_LEARNED_SPATIAL_RNN",
)


def unit_dir(subject: str, arm: str, seed: int) -> Path:
    return OUT_ROOT / "per_subject" / subject / arm / f"seed{seed}"


def worker(work: "queue.Queue", config: Path, log_lock: threading.Lock,
           log_path: Path) -> None:
    while True:
        try:
            subject, arm, seed = work.get_nowait()
        except queue.Empty:
            return
        out = unit_dir(subject, arm, seed)
        if (out / "DONE.json").exists():
            work.task_done()
            continue
        started = time.time()
        result = subprocess.run(
            [PYTHON, str(ROOT / "scripts/train_topic5_slp_unit.py"),
             "--subject", subject, "--arm", arm, "--seed", str(seed),
             "--config", str(config), "--out", str(out)],
            capture_output=True, text=True,
        )
        with log_lock:
            with log_path.open("a") as handle:
                handle.write(
                    f"{subject}\t{arm}\tseed{seed}\t{time.time() - started:.0f}s\t"
                    f"rc={result.returncode}\t{result.stdout.strip()[-160:]}\n"
                )
        print(f"[{time.strftime('%H:%M:%S')}] {subject} {arm} seed{seed} "
              f"({time.time() - started:.0f}s) rc={result.returncode}", flush=True)
        work.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    manifest = json.loads((OUT_ROOT / "INPUT_MANIFEST.json").read_text())
    subjects = args.subjects or manifest["frozen_cohort"]["primary"]

    work: "queue.Queue" = queue.Queue()
    planned = []
    for seed in args.seeds:
        for subject in subjects:
            for arm in args.arms:
                work.put((subject, arm, seed))
                planned.append({"subject": subject, "arm": arm, "seed": seed})

    (OUT_ROOT / "EXPERIMENT_MATRIX.csv").write_text(
        "subject,arm,seed\n" + "\n".join(
            f"{p['subject']},{p['arm']},{p['seed']}" for p in planned
        ) + "\n"
    )
    log_path = OUT_ROOT / "cohort_run.log"
    print(f"planned {len(planned)} units, {args.workers} workers", flush=True)

    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(work, args.config, lock, log_path),
                         daemon=True)
        for _ in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    done = sum(1 for p in planned
               if (unit_dir(p["subject"], p["arm"], p["seed"]) / "DONE.json").exists())
    failed = sum(1 for p in planned
                 if (unit_dir(p["subject"], p["arm"], p["seed"]) / "FAILED.json").exists())
    print(f"complete: {done}/{len(planned)}   failed: {failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
