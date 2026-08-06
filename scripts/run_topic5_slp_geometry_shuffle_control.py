"""Does the wiring economy need the real geometry, or would any distances do?

The learned arm is trained twice per patient: once with the true node positions
feeding the connection-cost term, once with those positions permuted.  Everything
else -- the observation operator, the real contact coordinates, the events, the
seed -- is identical, so the only thing that changes is which pairs the cost
calls "far apart".

If the two predict equally well, then "long connections are expensive" is a
decoration rather than a working constraint, and no result in this run may be
attributed to the spatial prior.
"""
from __future__ import annotations

import argparse
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
ARM = "LATENT_LEARNED_SPATIAL_RNN"


def worker(work: "queue.Queue", config: Path) -> None:
    while True:
        try:
            subject, shuffled = work.get_nowait()
        except queue.Empty:
            return
        out = OUT / "geometry_shuffle" / subject / ("shuffled" if shuffled else "real")
        if not (out / "DONE.json").exists():
            command = [
                PYTHON, str(ROOT / "scripts/train_topic5_slp_unit.py"),
                "--subject", subject, "--arm", ARM, "--seed", "1",
                "--config", str(config), "--out", str(out),
            ]
            if shuffled:
                command.append("--shuffle-wiring-geometry")
            subprocess.run(command, capture_output=True, text=True)
        print(f"[shuffle] {subject} {'shuffled' if shuffled else 'real':9s} "
              f"{'ok' if (out / 'DONE.json').exists() else 'FAILED'}", flush=True)
        work.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads((OUT / "INPUT_MANIFEST.json").read_text())
    subjects = args.subjects or manifest["frozen_cohort"]["primary"]

    work: "queue.Queue" = queue.Queue()
    for subject in subjects:
        for shuffled in (False, True):
            work.put((subject, shuffled))
    threads = [threading.Thread(target=worker, args=(work, args.config), daemon=True)
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    pairs = []
    for subject in subjects:
        cells = {}
        for name in ("real", "shuffled"):
            path = OUT / "geometry_shuffle" / subject / name / "DONE.json"
            if path.exists():
                cells[name] = json.loads(path.read_text())
        if len(cells) == 2:
            pairs.append({
                "subject": subject,
                "real_test_next_bce": cells["real"]["test_next_bce"],
                "shuffled_test_next_bce": cells["shuffled"]["test_next_bce"],
                "delta": cells["shuffled"]["test_next_bce"] - cells["real"]["test_next_bce"],
                "real_mean_edge_length": cells["real"].get("mean_edge_length"),
                "shuffled_mean_edge_length": cells["shuffled"].get("mean_edge_length"),
            })

    if len(pairs) < 3:
        raise SystemExit(f"only {len(pairs)} paired cells completed")

    delta = np.array([p["delta"] for p in pairs])
    rng = np.random.default_rng(20260806)
    boot = [float(np.median(rng.choice(delta, len(delta), replace=True)))
            for _ in range(4000)]
    real_len = np.array([p["real_mean_edge_length"] for p in pairs
                         if p["real_mean_edge_length"] is not None])
    shuffled_len = np.array([p["shuffled_mean_edge_length"] for p in pairs
                             if p["shuffled_mean_edge_length"] is not None])

    verdict = {
        "contract": "topic5_slp_geometry_shuffle_control_v0_1",
        "n_patients": len(pairs),
        "positive_means": "real geometry predicts better than permuted geometry",
        "median_delta": float(np.median(delta)),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        "n_positive": int((delta > 0).sum()),
        "wilcoxon_two_sided_p": float(stats.wilcoxon(delta).pvalue),
        "mean_edge_length_real": float(np.median(real_len)) if len(real_len) else None,
        "mean_edge_length_shuffled": float(np.median(shuffled_len)) if len(shuffled_len) else None,
        "reading": (
            "the spatial prior is doing work: connections learned against the real "
            "geometry predict better"
            if np.median(delta) > 0 and stats.wilcoxon(delta).pvalue < 0.05 else
            "the spatial prior is not doing measurable work here: a graph learned "
            "against permuted distances predicts as well, so no result in this run "
            "may be attributed to short connections being cheaper"
        ),
        "patients": pairs,
    }
    (OUT / "geometry_shuffle_control.json").write_text(json.dumps(verdict, indent=1))
    print(f"\nn={verdict['n_patients']} median={verdict['median_delta']:+.4f} "
          f"pos={verdict['n_positive']}/{verdict['n_patients']} "
          f"p={verdict['wilcoxon_two_sided_p']:.3g}")
    print(f"edge length real {verdict['mean_edge_length_real']} vs "
          f"shuffled {verdict['mean_edge_length_shuffled']}")
    print(verdict["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
