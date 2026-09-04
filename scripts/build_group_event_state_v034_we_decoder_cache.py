#!/usr/bin/env python3
"""Re-split the v0.5 wiring-economy RNN caches by *recorded time* for the v0.3.4 frozen decoder.

The v0.5 line split events by count (56/12/12 % of events); for E253 that training
window reaches past our 80 % recorded-time boundary and for E548 its validation
window lies inside our STATE_SELECTION.  Here the same cache arrays are kept
byte-identical except ``split``, which becomes

    0 (train)      recorded time  <  60 %
    1 (validation) 60 % <= t < 65 %
    2 (test)       65 % <= t < 70 %
   -1 (unused)     t >= 70 %      (STATE_SELECTION, DEVELOPMENT, sealed never touched)

Boundaries come from the locked human-input manifests (20/60/70/80 % epochs) and
the 65 % point is recomputed with the same recorded-time rule; the recomputed 60 %
and 70 % points must match the manifest to 1e-6 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.topic5_group_event_state.v03.partition import recorded_epoch_at_fraction

SOURCE = Path("/data/hfosp_rnn_external_results/results/topic5_multiscale_effective_scaffold_v0_5")
HUMAN = Path("/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs")
OUT = Path("/data/hfosp_group_event_state_v0_3_4/we_decoder")
FITS = {
    "epilepsiae_253": "epilepsiae_253__own_a",
    "epilepsiae_1096": "epilepsiae_1096__own_a",
    "epilepsiae_384": "epilepsiae_384__shared",
    "epilepsiae_1125": "epilepsiae_1125__own_a",
    "epilepsiae_1146": "epilepsiae_1146__shared",
    "epilepsiae_548": "epilepsiae_548__shared",
    "epilepsiae_583": "epilepsiae_583__shared",
    "epilepsiae_922": "epilepsiae_922__own_a",
}
COPY = ("plane.npz", "events_raw.npz", "train_only_modes.npz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boundaries(subject: str) -> dict[str, float]:
    manifest = json.loads((HUMAN / subject / "manifest_v3.json").read_text())
    npz = Path(manifest["input_path"])
    with np.load(npz, allow_pickle=False) as z:
        bounds = np.asarray(z["target_segment_bounds"], dtype=np.float64)
    intervals = [SimpleNamespace(start_epoch=float(a), stop_epoch=float(b), duration_seconds=float(b - a)) for a, b in bounds]
    rec = {f"{int(100 * q)}pct": float(recorded_epoch_at_fraction(intervals, q)) for q in (0.60, 0.65, 0.70)}
    pb = manifest["report"]["phase_boundaries_epoch"]
    for key in ("60pct", "70pct"):
        if abs(rec[key] - float(pb[key])) > 1e-6:
            raise ValueError(f"{subject}: recomputed {key} {rec[key]} != manifest {pb[key]}")
    return {**rec, "manifest": str(HUMAN / subject / "manifest_v3.json"), "manifest_sha256": sha256(HUMAN / subject / "manifest_v3.json"),
            "human_input_sha256": sha256(npz)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    (out / "cache").mkdir(parents=True, exist_ok=True)
    records = {}
    for subject, fit in FITS.items():
        src = SOURCE / "cache" / fit
        dst = out / "cache" / fit
        dst.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            shutil.copyfile(src / name, dst / name)
        b = boundaries(subject)
        with np.load(src / "events.npz", allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        t = arrays["event_abs_time"].astype(np.float64)
        # Chronological 80/10/10 by event count among covered events strictly
        # before the 70 % recorded-time boundary; nothing at or after 70 % is
        # ever used.  (A pure recorded-time 60/65/70 rule leaves E1146 with no
        # covered validation events at all, so the count rule is the uniform one.)
        split = np.full(t.shape, -1, dtype=np.int8)
        covered = np.flatnonzero(t < b["70pct"])
        covered = covered[np.argsort(t[covered], kind="stable")]
        n = covered.size
        n_train = int(np.floor(0.8 * n)); n_val = int(np.floor(0.1 * n))
        split[covered[:n_train]] = 0
        split[covered[n_train:n_train + n_val]] = 1
        split[covered[n_train + n_val:]] = 2
        b["train_stop_epoch"] = float(t[covered[n_train - 1]]); b["validation_stop_epoch"] = float(t[covered[n_train + n_val - 1]])
        b["test_stop_epoch"] = float(t[covered[-1]])
        old = arrays["split"].copy()
        arrays["split"] = split
        np.savez_compressed(dst / "events.npz", **arrays)
        prov = json.loads((src / "provenance.json").read_text())
        prov["v034_recorded_time_split"] = {
            "rule": "chronological 80/10/10 by event count among covered events with recorded time < 70pct; >=70pct unused",
            **b,
            "n_train": int((split == 0).sum()), "n_validation": int((split == 1).sum()),
            "n_test": int((split == 2).sum()), "n_unused": int((split == -1).sum()),
            "v0_5_original_split_counts": {int(v): int(c) for v, c in zip(*np.unique(old, return_counts=True))},
            "source_cache": str(src),
        }
        prov["n_train"], prov["n_validation"], prov["n_test"] = prov["v034_recorded_time_split"]["n_train"], prov["v034_recorded_time_split"]["n_validation"], prov["v034_recorded_time_split"]["n_test"]
        (dst / "provenance.json").write_text(json.dumps(prov, indent=2))
        records[fit] = {"subject": subject, **prov["v034_recorded_time_split"],
                        "files": {n: sha256(dst / n) for n in (*COPY, "events.npz", "provenance.json")},
                        "source_files": {n: sha256(src / n) for n in (*COPY, "events.npz", "provenance.json")}}
        print(subject, fit, {k: records[fit][k] for k in ("n_train", "n_validation", "n_test", "n_unused")}, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True).stdout.strip()
    (out / "INPUT_CACHE_MANIFEST.json").write_text(json.dumps({
        "contract": "topic5_we_decoder_v034_recorded_time_split_input_manifest",
        "fits": len(records), "required_files_per_fit": [*COPY, "events.npz", "provenance.json"],
        "split_rule": "chronological 80/10/10 by event count among covered events before the 70pct recorded-time boundary; >=70pct unused (-1)",
        "cache_records": records, "target_values_read": False, "development_targets_read": False,
        "source_root": str(SOURCE), "producer": str(Path(__file__).resolve()), "producer_sha256": sha256(Path(__file__).resolve()),
    }, indent=2))
    (out / "RUN_CONTRACT.json").write_text(json.dumps({
        "contract": "topic5_we_decoder_v034_recorded_time_split", "git_commit": commit,
        "trainer": "scripts/train_topic5_lbss_unit_v0_2.py (copied from codex/topic5-lbss-rnn-v0-1)",
        "arm": "L3_LOCAL_PLUS_LEARNED_LR", "seeds": [0, 1, 2], "defaults": "unchanged v0.5 DEFAULTS",
        "development_targets_read": False, "sealed_partition_opened": False, "seizure_outcomes_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
