#!/usr/bin/env python3
"""Build recorded-time-split mature-decoder caches for v0.3.5 expansion.

This is the same byte-preserving cache conversion used by the original eight
patients.  It only replaces the old event-count split with a chronological
80/10/10 split among source-decoder events strictly before the registered 70%
recorded-time boundary.  The 70--80% state-selection period stays unused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v03.partition import recorded_epoch_at_fraction  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import align_events  # noqa: E402

SOURCE = Path("/data/hfosp_rnn_external_results/results/topic5_multiscale_effective_scaffold_v0_5")
HUMAN = Path("/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs")
DATASET = Path("/data/hfosp_group_event_state_v0_1/dataset")
OUT = Path("/data/hfosp_group_event_state_v0_3_4/we_decoder")
FITS = {
    "epilepsiae_1077": "epilepsiae_1077__own_a",
    "epilepsiae_958": "epilepsiae_958__shared",
    "yuquan_chengshuai": "yuquan_chengshuai__shared",
    "yuquan_pengzihang": "yuquan_pengzihang__shared",
    "yuquan_xuxinyi": "yuquan_xuxinyi__own_a",
    "yuquan_zhangbichen": "yuquan_zhangbichen__own_a",
    "yuquan_zhangjiaqi": "yuquan_zhangjiaqi__shared",
    "yuquan_zhangkexuan": "yuquan_zhangkexuan__own_a",
}
COPY = ("plane.npz", "events_raw.npz", "train_only_modes.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def boundaries(subject: str) -> dict[str, float | str]:
    manifest_path = HUMAN / subject / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = Path(manifest["input_path"])
    with np.load(artifact, allow_pickle=False) as stored:
        raw = np.asarray(stored["target_segment_bounds"], dtype=np.float64)
    intervals = [
        SimpleNamespace(start_epoch=float(a), stop_epoch=float(b), duration_seconds=float(b - a))
        for a, b in raw
    ]
    result = {
        f"{int(100 * fraction)}pct": float(recorded_epoch_at_fraction(intervals, fraction))
        for fraction in (0.60, 0.65, 0.70, 0.80)
    }
    registered = manifest["report"]["phase_boundaries_epoch"]
    for key in ("60pct", "70pct", "80pct"):
        if abs(float(result[key]) - float(registered[key])) > 1e-6:
            raise ValueError(f"{subject}: recomputed {key} differs from immutable manifest")
    return {
        **result, "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "human_input_sha256": sha256(artifact),
    }


def dataset_event_times(subject: str) -> np.ndarray:
    root = DATASET / subject
    with np.load(root / "scalars.npz", allow_pickle=False) as stored:
        order = np.asarray(stored["interictal_index"], dtype=np.int64)
        return np.asarray(stored["t_abs"], dtype=np.float64)[order]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    (args.out / "cache").mkdir(parents=True, exist_ok=True)
    records = {}
    for subject, fit in FITS.items():
        source = SOURCE / "cache" / fit
        target = args.out / "cache" / fit
        if not (HUMAN / subject / "manifest_v3.json").is_file():
            raise FileNotFoundError(f"missing immutable human prefix for {subject}")
        target.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            shutil.copyfile(source / name, target / name)
        b = boundaries(subject)
        with np.load(source / "events.npz", allow_pickle=False) as stored:
            arrays = {key: np.asarray(stored[key]) for key in stored.files}
        time = np.asarray(arrays["event_abs_time"], dtype=np.float64)
        split = np.full(time.shape, -1, dtype=np.int8)
        covered = np.flatnonzero(time < float(b["70pct"]))
        covered = covered[np.argsort(time[covered], kind="stable")]
        if covered.size < 30:
            raise ValueError(f"{subject}: too few mature-decoder events before 70pct")
        n_train = int(np.floor(0.8 * covered.size))
        n_validation = int(np.floor(0.1 * covered.size))
        split[covered[:n_train]] = 0
        split[covered[n_train:n_train + n_validation]] = 1
        split[covered[n_train + n_validation:]] = 2
        arrays["split"] = split
        with (target / "events.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)

        provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
        ours = dataset_event_times(subject)
        alignment = align_events(ours, time)
        selection = (ours >= float(b["70pct"])) & (ours < float(b["80pct"]))
        n_selection_events = int(selection.sum())
        n_selection_aligned = int(np.sum(alignment[selection] >= 0))
        provenance["v035_recorded_time_split"] = {
            "rule": "chronological 80/10/10 by event count among mature-decoder events before 70pct; >=70pct unused",
            **b,
            "n_train": int(np.sum(split == 0)),
            "n_validation": int(np.sum(split == 1)),
            "n_test": int(np.sum(split == 2)),
            "n_unused": int(np.sum(split == -1)),
            "n_group_events_in_state_selection": n_selection_events,
            "n_group_events_aligned_to_decoder_in_state_selection": n_selection_aligned,
            "state_selection_decoder_coverage": (
                n_selection_aligned / n_selection_events if n_selection_events else 0.0
            ),
            "source_cache": str(source),
        }
        provenance["n_train"] = int(np.sum(split == 0))
        provenance["n_validation"] = int(np.sum(split == 1))
        provenance["n_test"] = int(np.sum(split == 2))
        (target / "provenance.json").write_text(json.dumps(provenance, indent=2))
        records[fit] = {"subject": subject, **provenance["v035_recorded_time_split"]}
        print(subject, fit, n_selection_aligned, "/", n_selection_events, flush=True)

    manifest = args.out / "V035_EXPANSION_INPUT_CACHE_MANIFEST.json"
    manifest.write_text(json.dumps({
        "format": "group_event_state_v0_3_5_decoder_expansion_cache_manifest_v1",
        "records": records, "target_values_read": False,
        "development_targets_read": False, "sealed_partition_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
