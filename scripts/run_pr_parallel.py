#!/usr/bin/env python3
"""Generic parallel launcher for Topic 0 rerun PR steps.

Splits the 40-subject cohort into N groups and launches N parallel
subprocesses of ``scripts/run_interictal_propagation.py`` (each with
disjoint ``--subjects`` list). Each subprocess writes its per-subject JSON
to a unique path so per_subject writes are race-free; the shared cohort
summary writes (e.g. ``pr5a_novel_template_gate.json``) race and the
last-finisher wins. After all subprocesses succeed, run the per-PR
consolidator (separate script) to rebuild the cohort summary from all
per-subject JSONs.

Usage::

    python scripts/run_pr_parallel.py --flag pr5-gate --n-parallel 10
    python scripts/run_pr_parallel.py --flag pr5-recruitment --n-parallel 8
    python scripts/run_pr_parallel.py --flag pr4a-followup --n-parallel 10

Requires Topic 0 §3.1 phantom-rank fix (the launcher hard-codes
``--masked-features`` since it's for the broad re-derivation rerun).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling", "liyouran",
    "songzishuo", "zhangbichen", "zhaochenxi", "zhaojinrui", "zhourongxuan",
    "zhangjiaqi", "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin", "gaolan", "wangyiyang",
]
EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]


def _split_round_robin(items: List[str], n: int) -> List[List[str]]:
    """Round-robin split — balances heavy subjects across groups."""
    groups: List[List[str]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        groups[i % n].append(item)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--flag",
        required=True,
        help="The --pr* flag to pass to run_interictal_propagation.py "
        "(e.g. 'pr5-gate', 'pr5-recruitment', 'pr4a-followup')",
    )
    ap.add_argument("--n-parallel", type=int, default=10)
    ap.add_argument(
        "--log-prefix",
        default=None,
        help="Log file prefix; default = step5_<flag>_masked",
    )
    ap.add_argument(
        "--subjects-only",
        nargs="+",
        default=None,
        help="Optional explicit subject list; overrides cohort default.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without launching.",
    )
    args = ap.parse_args()

    if args.subjects_only:
        all_subjects = list(args.subjects_only)
    else:
        all_subjects = YUQUAN_SUBJECTS + EPILEPSIAE_SUBJECTS

    n = min(args.n_parallel, len(all_subjects))
    groups = _split_round_robin(all_subjects, n)
    log_prefix = args.log_prefix or f"step5_{args.flag.replace('-', '_')}_masked"
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Launching {n} parallel processes for --{args.flag} ...")
    print(f"Subjects per group: {[len(g) for g in groups]}")
    for i, g in enumerate(groups):
        print(f"  group {i}: {' '.join(g)}")

    if args.dry_run:
        print("\n(dry-run; no processes launched)")
        return 0

    procs = []
    t0 = time.perf_counter()
    for i, group in enumerate(groups):
        log_path = log_dir / f"{log_prefix}_g{i:02d}.log"
        cmd = [
            sys.executable,
            "scripts/run_interictal_propagation.py",
            f"--{args.flag}",
            "--masked-features",
            "--subjects",
            *group,
        ]
        print(f"\n[group {i}] -> {log_path}")
        print(f"  {' '.join(cmd)}")
        with open(log_path, "w") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT))
        procs.append((i, p, log_path))

    print(f"\nAll {n} groups launched. Waiting...")
    fail_count = 0
    for i, p, log_path in procs:
        rc = p.wait()
        elapsed = time.perf_counter() - t0
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  group {i}: rc={rc} {status}  ({elapsed:.0f}s elapsed)")
        if rc != 0:
            fail_count += 1
            print(f"    log tail:")
            tail = subprocess.check_output(["tail", "-20", str(log_path)])
            for line in tail.decode().splitlines():
                print(f"      {line}")

    total = time.perf_counter() - t0
    print(f"\nDone. n_groups={n}, n_failed={fail_count}, total_wall={total:.0f}s")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
