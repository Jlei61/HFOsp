"""Per-patient grid, observation kernel and events for v0.2.

Events, plane coordinates and the chronological split are taken unchanged from
the v0.1 cache -- they were built and checked there, and rebuilding them would
add a second definition of the same thing.  What is new is the lattice: v0.1
placed irregular latent nodes, but finite-difference transport needs a regular
grid with a defined along-axis and across-axis direction.

The plane is the one v0.1 fitted on the whole recording, so this inherits its
status: retrospective, test-informed geometry.  A train-only axis would need the
upstream propagation-endpoint machinery re-run per split, which is not done
here; the status is recorded rather than quietly assumed away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_propagation_operator import build_grid  # noqa: E402
from src.topic5_virtual_seeg_operator import build_observation_operator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"
V1 = (ROOT.parent / "topic5-slp-rnn"
      / "results/topic5_spatial_latent_propagation_rnn_v0_1")

GEOMETRY_STATUS = "RETROSPECTIVE_TEST_INFORMED_GEOMETRY"
TRAIN_ONLY_AXIS = "NOT_ACHIEVED"
TRAIN_ONLY_AXIS_REASON = (
    "the propagation axis is derived from source/sink endpoints of the rank "
    "events, so a train-only axis needs the upstream endpoint estimation re-run "
    "on the training split. That was not wired up here. Every geometry-dependent "
    "result in v0.2 therefore inherits v0.1's retrospective status and no "
    "leave-contact-out number may be read as prospective."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells-per-side", type=int, default=40)
    args = parser.parse_args()

    cohort = json.loads((V1 / "INPUT_MANIFEST.json").read_text())["frozen_cohort"]
    subjects = cohort["primary"]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)

    records = []
    for subject in subjects:
        src = V1 / "cache" / subject
        plane = np.load(src / "plane_coordinates.npz", allow_pickle=True)
        events = np.load(src / "events.npz")
        contacts = plane["xy_mm"]
        sigma = float(plane["sigma_mm"][0])

        centres, shape, mask = build_grid(
            contacts, sigma, max_cells_per_side=args.max_cells_per_side
        )
        H = build_observation_operator(contacts, centres, sigma)
        seen = (H > 0).sum(axis=1)
        if seen.min() < 3:
            raise SystemExit(
                f"{subject}: a contact reads only {seen.min()} grid cells; the "
                "readout has collapsed onto a per-contact parameter"
            )

        destination = OUT / "cache" / subject
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "grid.npz", centres=centres,
                            shape=np.array(shape), mask=mask, sigma_mm=np.array([sigma]))
        np.savez_compressed(destination / "seeg_operator.npz", H=H)
        np.savez_compressed(destination / "events.npz",
                            group_ids=events["group_ids"], split=events["split"])

        split = events["split"]
        records.append({
            "subject": subject,
            "n_contacts": int(len(contacts)),
            "grid_shape": list(shape),
            "n_cells_total": int(shape[0] * shape[1]),
            "n_cells_in_domain": int(mask.sum()),
            "sigma_mm": sigma,
            "min_cells_seen_per_contact": int(seen.min()),
            "n_train": int((split == 0).sum()),
            "n_validation": int((split == 1).sum()),
            "n_test": int((split == 2).sum()),
            "events_sha256": sha256(src / "events.npz"),
            "plane_sha256": sha256(src / "plane_coordinates.npz"),
        })
        print(f"  {subject:24s} {records[-1]['n_contacts']:3d} contacts  "
              f"grid {shape}  {records[-1]['n_cells_in_domain']:4d} in domain  "
              f"min cells/contact {seen.min()}")

    summary = {
        "contract": "topic5_spo_cache_v0_2",
        "reused_from_v0_1": ["group_ids", "chronological split", "plane_coordinates",
                             "sigma rule"],
        "new_in_v0_2": ["regular finite-difference grid", "observation operator on that grid"],
        "geometry_status": GEOMETRY_STATUS,
        "train_only_axis": TRAIN_ONLY_AXIS,
        "train_only_axis_reason": TRAIN_ONLY_AXIS_REASON,
        "n_subjects": len(records),
        "patients": records,
    }
    (OUT / "cache" / "CACHE_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    (OUT / "INPUT_MANIFEST.json").write_text(json.dumps({
        "contract": "topic5_spo_input_manifest_v0_2",
        "frozen_cohort": cohort,
        "geometry_status": GEOMETRY_STATUS,
        "train_only_axis": TRAIN_ONLY_AXIS,
        "train_only_axis_reason": TRAIN_ONLY_AXIS_REASON,
        "upstream": str(V1.relative_to(ROOT.parent)),
    }, indent=1))
    print(f"\ncached {len(records)} patients; geometry status {GEOMETRY_STATUS}; "
          f"train-only axis {TRAIN_ONLY_AXIS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
