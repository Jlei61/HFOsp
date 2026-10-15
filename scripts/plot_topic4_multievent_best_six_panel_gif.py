#!/usr/bin/env python3
"""Render the frozen v2.1 winner in the earlier six-panel review format.

This is a plotting-only chronological preview.  It does not alter the frozen
candidate, event detector, labels, scores, or nomination.
"""
from pathlib import Path
import json
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_xy_raw_propagation_video import PATIENT, META
from scripts.review_topic4_same_network_events import movie, sha


SEARCH = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1"
RUN_STEM = "v2_1_pop1_de_b_002_topo_6101_dyn_7101"
WORKER = SEARCH / "execution/confirmation_24s/workers" / f"{RUN_STEM}.json"
OBSERVATION = SEARCH / "execution/confirmation_24s/repaired_observation" / f"{RUN_STEM}.json"
CANDIDATES = SEARCH / "execution/confirmation_24s/candidate_manifest.json"
EVALUATOR = ROOT / "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl"
OUTPUT = SEARCH / "figures" / f"six_panel_{RUN_STEM}"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    worker = read(WORKER)
    observation = read(OBSERVATION)
    arrays_path = Path(worker["arrays"]["path"])
    observation_arrays_path = Path(observation["observation_arrays_path"])
    if sha(arrays_path) != worker["arrays"]["sha256"]:
        raise RuntimeError("frozen worker arrays changed")
    if sha(observation_arrays_path) != observation["observation_arrays_sha256"]:
        raise RuntimeError("frozen repaired-observation arrays changed")

    candidates = read(CANDIDATES)["candidates"]
    candidate = next(row for row in candidates if row["candidate_id"] == worker["candidate_id"])
    with np.load(arrays_path) as source:
        z = {key: source[key] for key in (
            "contact_names", "contact_xy_mm", "sheet_activity_counts",
            "sheet_activity_frame_ms", "contact_envelope",
            "contact_envelope_dt_ms", "topology_seed",
        )}
    with np.load(observation_arrays_path) as source:
        centroid_ms = np.asarray(source["centroid_ms"], float)
        windows_ms = np.asarray(source["windows_ms"], float)
        primary = np.asarray(source["primary_event_indices"], int)
    with np.load(PATIENT) as source:
        patient = {key: source[key] for key in source.files}
    with EVALUATOR.open("rb") as handle:
        evaluator = pickle.load(handle)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    adapted_worker = {
        "candidate_id": worker["candidate_id"],
        "seed": f"topo {worker['topology_seed']}, dyn {worker['dynamics_seed']}",
        "observation": {
            "centroid_ms": centroid_ms.tolist(),
            "windows_ms": windows_ms.tolist(),
            "primary_event_indices": primary.tolist(),
        },
    }
    result = movie(
        adapted_worker, z, patient, evaluator, candidate, OUTPUT,
        max_events=6,
        selection_note=(
            "Frozen G4 winner; predeclared lowest confirmation topology/dynamics "
            "pair; first six detected windows in chronological order, including "
            "non-primary windows; no visual-likeness selection."
        ),
    )
    result.update({
        "display_scope": "first six of all detected windows",
        "worker_path": str(WORKER),
        "worker_sha256": sha(WORKER),
        "observation_path": str(OBSERVATION),
        "observation_sha256": sha(OBSERVATION),
        "candidate_manifest_path": str(CANDIDATES),
        "candidate_manifest_sha256": sha(CANDIDATES),
        "patient_cache_path": str(PATIENT),
        "patient_cache_sha256": sha(PATIENT),
        "patient_movie_metadata_path": str(META),
        "patient_movie_metadata_sha256": sha(META),
    })
    (OUTPUT / "six_panel_preview_metadata.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(result["path"])


if __name__ == "__main__":
    main()
