#!/usr/bin/env python3
"""Build anatomy-only, calibration-prefix caches for the v0.3.7 contact decoder.

The mature wiring-economy decoder architecture is retained, but the old
propagation-template plane is not.  That plane was estimated from the complete
interictal record and is therefore unsuitable for the nested v0.3.7 primary
analysis.  Here the patient plane is a deterministic PCA projection of the
implant coordinates, and every event used for decoder fitting or selection is
strictly earlier than the registered 20-percent state-FIT boundary.

No development, seizure-outcome, or sealed target is opened by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json, sha256_file  # noqa: E402
from src.topic5_lbss_rnn_v0_2 import build_pool_contract, strong_component_audit  # noqa: E402


DATASET_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
)
INPUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs")
OUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/decoder_strict")
DEFAULT_SUBJECTS = (
    "epilepsiae_253",
    "epilepsiae_958",
    "epilepsiae_1077",
    "epilepsiae_1125",
)
NODE_SEED = 20260904
MIN_CONTACTS = 6


def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _densify(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups, dtype=np.int16)
    out = np.full_like(values, -1)
    for row_index, row in enumerate(values):
        present = np.unique(row[row >= 0])
        mapping = {int(old): int(new) for new, old in enumerate(present)}
        for contact_index, old in enumerate(row):
            if old >= 0:
                out[row_index, contact_index] = mapping[int(old)]
    return out


def _anatomical_plane(coords_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project implant coordinates without consulting any event value."""

    coords = np.asarray(coords_xyz, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("anatomical PCA requires finite [contact, 3] coordinates")
    centre = coords.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(coords - centre, full_matrices=False)
    if vh.shape[0] < 2 or singular[1] <= max(1e-8, singular[0] * 1e-6):
        raise ValueError("implant geometry cannot support a two-dimensional anatomy-only plane")
    axes = vh[:2].copy()
    # SVD axis signs are arbitrary.  Resolve them from the coordinate loadings,
    # never from propagation labels or event statistics.
    for axis in range(2):
        pivot = int(np.argmax(np.abs(axes[axis])))
        if axes[axis, pivot] < 0:
            axes[axis] *= -1.0
    xy = (coords - centre) @ axes.T
    return xy.astype(np.float32), centre.reshape(3).astype(np.float32), axes.astype(np.float32)


def _anatomy_tissue_layout(xy: np.ndarray, *, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Build a local observation plane from anatomy alone.

    Contacts are retained as guaranteed support nodes.  Additional background
    nodes are selected by deterministic farthest-point sampling from a padded
    grid.  No event rank, template, or patient outcome enters this function.
    """

    contacts = np.asarray(xy, dtype=np.float64)
    pairwise = np.linalg.norm(contacts[:, None] - contacts[None], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    pitch = float(np.median(pairwise.min(axis=1)))
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("implant geometry has no positive contact pitch")
    sigma = max(0.5, 0.5 * pitch)
    margin = max(3.0 * sigma, pitch)
    step = max(0.5, min(pitch, sigma))
    lower = contacts.min(axis=0) - margin
    upper = contacts.max(axis=0) + margin
    gx = np.arange(lower[0], upper[0] + step, step)
    gy = np.arange(lower[1], upper[1] + step, step)
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    target_nodes = min(192, max(64, 6 * len(contacts)))
    nodes = [row.copy() for row in contacts]
    if len(grid) + len(nodes) < target_nodes:
        raise ValueError("anatomy grid is too small for the requested tissue plane")
    rng = np.random.default_rng(int(seed))
    distance = np.linalg.norm(grid[:, None] - np.asarray(nodes)[None], axis=-1).min(axis=1)
    distance += rng.uniform(0.0, 1e-12, size=distance.shape)
    for _ in range(target_nodes - len(nodes)):
        index = int(np.argmax(distance))
        candidate = grid[index]
        if np.min(np.linalg.norm(np.asarray(nodes) - candidate, axis=1)) > 1e-7:
            nodes.append(candidate.copy())
        distance = np.minimum(distance, np.linalg.norm(grid - candidate, axis=1))
        distance[index] = -np.inf
    node_array = np.asarray(nodes, dtype=np.float64)
    d_contact = np.linalg.norm(contacts[:, None] - node_array[None], axis=-1)
    observation = np.exp(-(d_contact ** 2) / (2.0 * sigma ** 2))
    observation[d_contact > 3.0 * sigma] = 0.0
    row_sum = observation.sum(axis=1, keepdims=True)
    if not np.all(row_sum > 0):
        raise RuntimeError("an anatomy contact has no local tissue support")
    observation /= row_sum
    return sigma, node_array.astype(np.float32), observation.astype(np.float32)


def _split_prefix(n_events: int) -> np.ndarray:
    if int(n_events) < 30:
        raise ValueError("strict calibration prefix has fewer than 30 eligible events")
    n_train = int(np.floor(0.8 * n_events))
    n_validation = int(np.floor(0.1 * n_events))
    split = np.full(n_events, 2, dtype=np.int8)
    split[:n_train] = 0
    split[n_train:n_train + n_validation] = 1
    if not all(np.any(split == value) for value in (0, 1, 2)):
        raise ValueError("strict decoder split contains an empty partition")
    return split


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    return result.stdout.strip()


def build_subject(
    subject: str,
    out_root: Path,
    *,
    dataset_root: Path = DATASET_ROOT,
    input_root: Path = INPUT_ROOT,
) -> dict:
    dataset_path = dataset_root / f"{subject}.npz"
    dataset_meta_path = dataset_root / f"{subject}.json"
    input_manifest_path = input_root / subject / "manifest_v3.json"
    for path in (dataset_path, dataset_meta_path, input_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _read(input_manifest_path)
    boundary = float(manifest["report"]["phase_boundaries_epoch"]["20pct"])

    with np.load(dataset_path, allow_pickle=False) as stored:
        names_all = np.asarray(stored["contact_names"]).astype(str)
        coords_all = np.asarray(stored["contact_coords"], dtype=np.float64)
        finite_contact = np.isfinite(coords_all).all(axis=1)
        columns = np.flatnonzero(finite_contact)
        if columns.size < MIN_CONTACTS:
            raise ValueError(f"{subject}: only {columns.size} contacts have static coordinates")
        names = names_all[columns]
        coords = coords_all[columns]
        times_all = np.asarray(stored["event_abs_time"], dtype=np.float64)
        source_index_all = np.asarray(stored["event_source_index"], dtype=np.int64)
        raw_groups = np.asarray(stored["event_group_ids"], dtype=np.int16)[:, columns]
        raw_lag = np.asarray(stored["event_lag_raw"], dtype=np.float32)[:, columns]

    prefix = np.flatnonzero(np.isfinite(times_all) & (times_all < boundary))
    prefix = prefix[np.argsort(times_all[prefix], kind="stable")]
    ranks = _densify(raw_groups[prefix])
    participants = np.sum(ranks >= 0, axis=1)
    groups = np.asarray(
        [len(np.unique(row[row >= 0])) for row in ranks], dtype=np.int16,
    )
    eligible = (participants >= 3) & (groups >= 2)
    prefix = prefix[eligible]
    ranks = ranks[eligible]
    lag = raw_lag[prefix]
    groups = groups[eligible]
    times = times_all[prefix]
    source_index = source_index_all[prefix]
    split = _split_prefix(len(prefix))
    if not float(times.max()) < boundary:
        raise AssertionError("strict decoder cache crossed the state-FIT boundary")

    xy, centre, axes = _anatomical_plane(coords)
    sigma, nodes, observation = _anatomy_tissue_layout(xy, seed=NODE_SEED)
    distance = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1).astype(np.float32)
    pools = build_pool_contract(distance)
    graph = strong_component_audit(pools.local_mask, supported=np.abs(observation).sum(axis=0) > 0)
    if not (
        graph["all_nodes_one_strong_component"]
        and float(graph["contact_supported_pairwise_reachability"]) == 1.0
        and int(graph["minimum_in_degree"]) >= 1
        and int(graph["minimum_out_degree"]) >= 1
    ):
        raise RuntimeError(f"{subject}: anatomy-only tissue graph failed connectivity audit")

    fit_id = f"{subject}__anatomy"
    target = out_root / "cache" / fit_id
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "plane.npz",
        contacts_xy_mm=xy,
        contacts_xyz_mm=coords.astype(np.float32),
        anatomical_pca_center=centre,
        anatomical_pca_axes=axes,
        nodes_xy_mm=nodes,
        H=observation,
        D_mm=distance,
        sigma_mm=np.asarray([sigma], dtype=np.float32),
        scale_mm=np.asarray([1.0], dtype=np.float32),
        latent_domain_version=np.asarray(["ANATOMY_ONLY_PCA_FULL_TISSUE_V0_3_7"]),
    )
    np.savez_compressed(
        target / "events_raw.npz",
        ranks=ranks,
        base_split=split,
        event_group_count=groups,
        event_lag_raw=lag,
        event_abs_time=times,
        event_source_index=source_index,
        event_dataset_index=prefix.astype(np.int64),
        contact_names=names,
    )
    # Mode fields are retained only for API compatibility.  The mature L3
    # decoder loss never consumes them, so neutral sentinels avoid importing a
    # full-record cluster assignment into the strict prefix.
    neutral_mode = np.full(len(ranks), -1, dtype=np.int8)
    np.savez_compressed(
        target / "events.npz",
        ranks=ranks,
        split=split,
        mode=neutral_mode,
        full_train_mode=neutral_mode,
        prefix_posterior=np.full((len(ranks), 1), 1.0, dtype=np.float32),
        prefix_entropy=np.zeros(len(ranks), dtype=np.float32),
        event_abs_time=times,
        event_source_index=source_index,
        event_dataset_index=prefix.astype(np.int64),
    )
    provenance = {
        "format": "group_event_state_v0_3_7_strict_decoder_cache_v1",
        "fit_id": fit_id,
        "subject": subject,
        "scope": "anatomy_only",
        "n_contacts": int(len(names)),
        "n_joint_contacts": int(len(names)),
        "joint_contacts": names.tolist(),
        "contact_vocabulary_source": "static implant montage names with finite coordinates",
        "contact_vocabulary_event_selected": False,
        "geometry_source": "static implant 3D coordinates projected by event-blind PCA",
        "geometry_uses_event_values": False,
        "n_nodes": int(len(nodes)),
        "n_events": int(len(ranks)),
        "n_train": int(np.sum(split == 0)),
        "n_validation": int(np.sum(split == 1)),
        "n_test": int(np.sum(split == 2)),
        "decoder_max_used_time": float(times.max()),
        "state_fit_start_20pct": boundary,
        "strictly_pre_state_fit": bool(float(times.max()) < boundary),
        "mode_fields_consumed_by_decoder_loss": False,
        "event_values_after_20pct_read_into_output": False,
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_metadata_sha256": sha256_file(dataset_meta_path),
        "state_input_manifest_sha256": sha256_file(input_manifest_path),
        "contact_names_sha256": _digest_array(np.asarray(names, dtype="U")),
        "contact_coordinates_sha256": _digest_array(coords.astype(np.float32)),
        "plane_sha256": sha256_file(target / "plane.npz"),
        "events_sha256": sha256_file(target / "events.npz"),
        "events_raw_sha256": sha256_file(target / "events_raw.npz"),
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
        "target_values_read": False,
    }
    atomic_json(target / "provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    rows = [
        build_subject(
            str(subject), out_root,
            dataset_root=args.dataset_root.resolve(), input_root=args.input_root.resolve(),
        )
        for subject in args.subjects
    ]
    atomic_json(out_root / "INPUT_CACHE_MANIFEST.json", {
        "format": "group_event_state_v0_3_7_strict_decoder_input_manifest_v1",
        "split_contract": (
            "chronological 80/10/10 within eligible group events strictly earlier than "
            "the registered 20pct state-FIT boundary"
        ),
        "geometry_contract": "event-blind anatomy-only PCA of implant coordinates",
        "fits": {row["fit_id"]: row for row in rows},
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    atomic_json(out_root / "RUN_CONTRACT.json", {
        "format": "group_event_state_v0_3_7_strict_decoder_run_contract_v1",
        "git_commit_at_cache_build": _git_head(),
        "trainer": "scripts/train_topic5_lbss_unit_v0_2.py",
        "arm": "L3_LOCAL_PLUS_LEARNED_LR",
        "seeds": [0, 1, 2],
        "patient_local_fit_max_time": "strictly before state FIT 20pct boundary",
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    print(json.dumps({
        "out_root": str(out_root),
        "subjects": [row["subject"] for row in rows],
        "events": {row["subject"]: row["n_events"] for row in rows},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
