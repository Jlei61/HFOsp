"""H2: does the latent field predict at contacts it never trained on?

This is the one question in the design that no earlier model in this line could
ask.  A contact-node model carries one parameter row per contact, so a contact
absent from training has no parameters at all.  Putting the state in tissue and
making contacts pure observation ports is what makes the question askable.

Both compared arms run without a per-contact bias (spec 7.1) -- with it the test
is undefined at a contact that never appeared in the loss.  Both arms hold out
the same contacts, chosen from the patient and seed alone, so neither is scored
on an easier subset.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"

ARMS = ("CONTACT_GRAPH_RNN", "LATENT_LEARNED_SPATIAL_RNN")
MODES = ("weak", "strong")


def cell_dir(subject: str, arm: str, mode: str, seed: int) -> Path:
    return OUT / "leave_contact_out" / subject / arm / mode / f"seed{seed}"


def worker(work: "queue.Queue", config: Path, fraction: float) -> None:
    while True:
        try:
            subject, arm, mode, seed = work.get_nowait()
        except queue.Empty:
            return
        out = cell_dir(subject, arm, mode, seed)
        if not (out / "DONE.json").exists():
            subprocess.run(
                [PYTHON, str(ROOT / "scripts/train_topic5_slp_unit.py"),
                 "--subject", subject, "--arm", arm, "--seed", str(seed),
                 "--config", str(config), "--out", str(out),
                 "--holdout-contacts", f"auto:{fraction}", "--holdout-mode", mode],
                capture_output=True, text=True,
            )
        print(f"[lco] {subject} {arm} {mode} seed{seed} "
              f"{'ok' if (out / 'DONE.json').exists() else 'FAILED'}", flush=True)
        work.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1])
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads((OUT / "INPUT_MANIFEST.json").read_text())
    subjects = args.subjects or manifest["frozen_cohort"]["primary"]

    work: "queue.Queue" = queue.Queue()
    for seed in args.seeds:
        for subject in subjects:
            for mode in MODES:
                for arm in ARMS:
                    work.put((subject, arm, mode, seed))

    threads = [threading.Thread(target=worker, args=(work, args.config, args.fraction),
                                daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = []
    for seed in args.seeds:
        for subject in subjects:
            for mode in MODES:
                for arm in ARMS:
                    done = cell_dir(subject, arm, mode, seed) / "DONE.json"
                    if not done.exists():
                        continue
                    payload = json.loads(done.read_text())
                    rows.append({
                        "subject": subject, "arm": arm, "mode": mode, "seed": seed,
                        "n_holdout_contacts": payload.get("n_holdout_contacts"),
                        "heldout_next_bce": payload.get("heldout_contact_next_bce"),
                        "heldout_contact_nll": payload.get("heldout_contact_contact_nll"),
                        "heldout_top1": payload.get("heldout_contact_top1"),
                        "retained_next_bce": payload.get("retained_contact_next_bce"),
                        "retained_contact_nll": payload.get("retained_contact_contact_nll"),
                        "converged": payload.get("converged"),
                    })
    if not rows:
        raise SystemExit("no leave-contact-out cell completed")

    with (OUT / "leave_contact_out_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"contract": "topic5_slp_leave_contact_out_v0_1",
               "holdout_fraction": args.fraction,
               "n_cells": len(rows), "comparisons": {}}
    for mode in MODES:
        by_arm = {}
        for arm in ARMS:
            per_subject = {}
            for row in rows:
                if row["arm"] == arm and row["mode"] == mode and row["heldout_next_bce"]:
                    per_subject.setdefault(row["subject"], []).append(row["heldout_next_bce"])
            by_arm[arm] = {s: float(np.median(v)) for s, v in per_subject.items()}
        common = sorted(set(by_arm[ARMS[0]]) & set(by_arm[ARMS[1]]))
        if len(common) < 3:
            summary["comparisons"][mode] = {"status": "INSUFFICIENT", "n": len(common)}
            continue
        # positive delta means the latent field wins at unseen contacts
        delta = np.array([by_arm["CONTACT_GRAPH_RNN"][s]
                          - by_arm["LATENT_LEARNED_SPATIAL_RNN"][s] for s in common])
        rng = np.random.default_rng(20260806)
        boot = [float(np.median(rng.choice(delta, len(delta), replace=True)))
                for _ in range(4000)]
        summary["comparisons"][mode] = {
            "status": "COMPLETE", "n": len(common), "subjects": common,
            "median_delta": float(np.median(delta)),
            "bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
            "n_positive": int((delta > 0).sum()),
            "wilcoxon_two_sided_p": float(stats.wilcoxon(delta).pvalue),
            "per_patient_delta": {s: float(d) for s, d in zip(common, delta)},
            "positive_means": "latent field better at contacts it never trained on",
        }
        entry = summary["comparisons"][mode]
        print(f"\n{mode:7s} n={entry['n']:2d} median={entry['median_delta']:+.4f} "
              f"pos={entry['n_positive']}/{entry['n']} p={entry['wilcoxon_two_sided_p']:.3g}")

    (OUT / "leave_contact_out_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
