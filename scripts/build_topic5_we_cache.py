"""WE-SLP-RNN v0.3 cache builder: geometry, events, and the A/B label join.

One cache per *fit*, not per patient.  A patient whose two interictal templates
share a propagation axis gets a single fit on the shared plane and keeps every
event; a patient whose templates do not share an axis gets two fits, each on its
own template's plane and each trained only on that template's events.  Forcing
the second group onto one plane would express one of the two modes in a frame
that its own geometry would have flipped.

The A/B labels never enter training.  They are attached here so the *evaluation*
can stratify by mode, and the join is the single most dangerous step in the
version: ``adaptive_cluster.labels`` is indexed over the subset of events with at
least three participating channels, while ``event_source_index`` is indexed over
all events.  Indexing one with the other runs out of range on three of the 21
patients and silently mislabels the rest, so the mapping is reconstructed and
checked rather than assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.topic5_shared_propagation_field import (  # noqa: E402
    load_subject_rank_events,
    sha256_file,
)
from src.topic5_virtual_seeg_operator import (  # noqa: E402
    kernel_sigma_mm,
    resolve_node_count,
)

DATASET_DIR = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
FIELD_DIR = ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
PROP_DIR = ROOT / "results/interictal_propagation_masked/per_subject"
EPILEPSIAE_LAGPAT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
YUQUAN_LAGPAT = Path("/mnt/yuquan_data/yuquan_24h_edf")
OUT_ROOT = ROOT / "results/topic5_wiring_economy_slp_rnn_v0_3"

# Frozen cohort: the narrow-tree intersection from SLP-RNN v0.1's INPUT_MANIFEST.
COHORT: List[str] = [
    "epilepsiae_1084", "epilepsiae_1146", "epilepsiae_1150", "epilepsiae_253",
    "epilepsiae_384", "epilepsiae_442", "epilepsiae_548", "epilepsiae_590",
    "epilepsiae_620", "epilepsiae_922", "epilepsiae_958",
    "yuquan_chengshuai", "yuquan_huanghanwen", "yuquan_litengsheng",
    "yuquan_liyouran", "yuquan_pengzihang", "yuquan_songzishuo",
    "yuquan_xuxinyi", "yuquan_zhangbichen", "yuquan_zhangkexuan",
    "yuquan_zhaochenxi",
]

NODE_SEED = 20260808
MIN_RANKS_PER_EVENT = 2
MIN_PARTICIPATING = 3  # the rule adaptive_cluster's valid-event subset was built with
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_LABEL_COVERAGE = 0.98


def densify_ranks(group_ids: np.ndarray) -> np.ndarray:
    """Re-number ranks to 0..T-1 per event after contacts have been dropped.

    Dropping a contact can empty a whole rank set.  Leaving the gap would make
    the model predict an empty set at that step, which is not a rank set of the
    patient's event -- it is an artefact of the montage intersection.
    """
    out = np.full(group_ids.shape, -1, dtype=np.int16)
    for e, row in enumerate(group_ids):
        present = np.unique(row[row >= 0])
        remap = {int(old): new for new, old in enumerate(present)}
        for c, value in enumerate(row):
            if value >= 0:
                out[e, c] = remap[int(value)]
    return out


def lagpat_dir(subject: str) -> Path:
    dataset, name = subject.split("_", 1)
    if dataset == "yuquan":
        return YUQUAN_LAGPAT / name
    legacy = EPILEPSIAE_LAGPAT / name / "all_recs"
    return legacy if legacy.exists() else EPILEPSIAE_LAGPAT / name


def event_mode_labels(subject: str, source_index: np.ndarray) -> Dict[str, Any]:
    """Attach A/B cluster labels to the sealed dataset's events.

    ``adaptive_cluster.labels`` covers only events with >= 3 participating
    channels, in their original order; the sealed dataset indexes all events.
    The bridge is the valid-event index vector, which is not stored anywhere and
    has to be recomputed from the raw participation booleans under the same
    rule the producer used.
    """
    payload = json.loads((PROP_DIR / f"{subject}.json").read_text())
    adaptive = payload["adaptive_cluster"]
    labels = np.asarray(adaptive["labels"], dtype=np.int64)

    loaded = load_subject_propagation_events(lagpat_dir(subject))
    bools = loaded["bools"]
    n_all = int(bools.shape[1])
    valid_idx = _valid_event_indices(bools, min_participating=MIN_PARTICIPATING)

    if len(valid_idx) != len(labels):
        raise RuntimeError(
            f"{subject}: valid-event reconstruction gives {len(valid_idx)} events "
            f"but adaptive_cluster.labels has {len(labels)}; the label join is "
            "not certified and must not be guessed"
        )
    if len(np.unique(source_index)) != len(source_index):
        raise RuntimeError(f"{subject}: event_source_index has duplicates")
    if int(source_index.max()) >= n_all:
        raise RuntimeError(
            f"{subject}: event_source_index reaches {int(source_index.max())} but the "
            f"raw record only has {n_all} events"
        )

    label_full = np.full(n_all, -1, dtype=np.int8)
    label_full[valid_idx] = labels.astype(np.int8)
    mode = label_full[source_index]
    coverage = float(np.mean(mode >= 0))

    # Which cluster is template A?  The cluster templates and the frozen
    # gradient-field ranks are two views of the same thing, so the assignment is
    # whichever pairing correlates best -- never the raw cluster id, which is
    # only a k-means output ordering.
    field = json.loads((FIELD_DIR / f"{subject}.json").read_text())["interictal_field"]
    order = [str(c) for c in field["contact_order"]]
    names = [str(c) for c in payload["channel_names"]]
    take = [names.index(c) for c in order]
    rank_a = np.asarray(field["rank_a"], float)
    rank_b = np.asarray(field["rank_b"], float)
    templates = {
        int(c["cluster_id"]): np.asarray(c["template_rank"], float)[take]
        for c in adaptive["clusters"]
    }
    ids = sorted(templates)
    if len(ids) != 2:
        raise RuntimeError(f"{subject}: expected two clusters, found {len(ids)}")
    direct = (spearmanr(templates[ids[0]], rank_a).statistic
              + spearmanr(templates[ids[1]], rank_b).statistic)
    swapped = (spearmanr(templates[ids[0]], rank_b).statistic
               + spearmanr(templates[ids[1]], rank_a).statistic)
    mode_to_template = ({str(ids[0]): "a", str(ids[1]): "b"} if direct >= swapped
                        else {str(ids[0]): "b", str(ids[1]): "a"})

    return {
        "mode": mode,
        "label_coverage": coverage,
        "mode_to_template": mode_to_template,
        "template_match_direct": float(direct),
        "template_match_swapped": float(swapped),
        "n_raw_events": n_all,
        "n_valid_events": int(len(valid_idx)),
    }


def plane_scopes(subject: str) -> Dict[str, Dict[str, Any]]:
    """Return the planes this patient contributes, keyed by fit scope."""
    field = json.loads((FIELD_DIR / f"{subject}.json").read_text())["interictal_field"]
    planes = field["planes"]
    shared = planes.get("shared")
    if shared is not None and shared.get("status") == "ok":
        return {"shared": shared}
    for key in ("own_a", "own_b"):
        if planes.get(key, {}).get("status") != "ok":
            raise RuntimeError(f"{subject}: plane {key} is not solved and there is no shared plane")
    return {"own_a": planes["own_a"], "own_b": planes["own_b"]}


def build_fit(subject: str, scope: str, plane: Dict[str, Any], record, mode: np.ndarray,
              label_info: Dict[str, Any], out_root: Path,
              sigma_override: float | None = None) -> Dict[str, Any]:
    field = json.loads((FIELD_DIR / f"{subject}.json").read_text())["interictal_field"]
    order = [str(c) for c in field["contact_order"]]
    points = np.asarray(plane["points"], float)
    scale_mm = float(plane["scale_mm"])
    by_name = {name: points[i] * scale_mm for i, name in enumerate(order)}

    event_names = [str(n) for n in record.contact_names]
    joint = [n for n in event_names if n in by_name]
    if len(joint) < 8:
        raise RuntimeError(f"{subject}/{scope}: only {len(joint)} joint contacts")
    columns = np.array([event_names.index(n) for n in joint], int)
    contacts_xy = np.stack([by_name[n] for n in joint])

    ranks = densify_ranks(record.group_ids[:, columns])
    n_ranks = np.array([len(np.unique(r[r >= 0])) for r in ranks], int)
    keep = n_ranks >= MIN_RANKS_PER_EVENT

    if scope != "shared":
        wanted = [int(k) for k, v in label_info["mode_to_template"].items()
                  if v == scope.split("_", 1)[1]]
        keep = keep & np.isin(mode, wanted)

    train, validation, test = record.development_split(VALIDATION_FRACTION, TEST_FRACTION)
    split = np.full(len(ranks), -1, np.int8)
    split[train] = 0
    split[validation] = 1
    split[test] = 2
    split[~keep] = -1

    sigma = float(sigma_override) if sigma_override else kernel_sigma_mm(contacts_xy)
    n_nodes, nodes_xy, H, nominal = resolve_node_count(contacts_xy, sigma, seed=NODE_SEED)
    node_distance = np.linalg.norm(nodes_xy[:, None, :] - nodes_xy[None, :, :], axis=-1)

    fit_id = f"{subject}__{scope}"
    out_dir = out_root / "cache" / fit_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "plane.npz",
        contacts_xy_mm=contacts_xy.astype(np.float32),
        nodes_xy_mm=nodes_xy.astype(np.float32),
        H=H.astype(np.float32),
        D_mm=node_distance.astype(np.float32),
        sigma_mm=np.array([sigma], np.float32),
        scale_mm=np.array([scale_mm], np.float32),
    )
    np.savez_compressed(
        out_dir / "events.npz",
        ranks=ranks.astype(np.int16),
        split=split,
        mode=mode.astype(np.int8),
    )
    provenance = {
        "fit_id": fit_id,
        "subject": subject,
        "scope": scope,
        "contacts": joint,
        "n_contacts": len(joint),
        "n_nodes": int(n_nodes),
        "nominal_n_nodes": int(nominal),
        "sigma_mm": float(sigma),
        "support_radius_mm": float(3.0 * sigma),
        "scale_mm": scale_mm,
        "node_seed": NODE_SEED,
        "min_ranks_per_event": MIN_RANKS_PER_EVENT,
        "n_events_kept": int(keep.sum()),
        "n_train": int((split == 0).sum()),
        "n_validation": int((split == 1).sum()),
        "n_test": int((split == 2).sum()),
        "mode_counts": {str(k): int((mode[keep] == k).sum()) for k in (-1, 0, 1)},
        "label_coverage": label_info["label_coverage"],
        "mode_to_template": label_info["mode_to_template"],
        "template_match_direct": label_info["template_match_direct"],
        "template_match_swapped": label_info["template_match_swapped"],
        "dataset_sha256": record.input_sha256,
        "field_sha256": sha256_file(FIELD_DIR / f"{subject}.json"),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--sigma-mm", type=float, default=None,
                        help="fix the read-out kernel width across the cohort instead of "
                             "deriving it per patient from contact spacing")
    args = parser.parse_args()

    subjects = args.subjects or COHORT
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    low_coverage: List[str] = []

    for subject in subjects:
        record = load_subject_rank_events(DATASET_DIR, subject)
        label_info = event_mode_labels(subject, record.event_source_index)
        if label_info["label_coverage"] < MIN_LABEL_COVERAGE:
            low_coverage.append(subject)
        for scope, plane in plane_scopes(subject).items():
            row = build_fit(subject, scope, plane, record, label_info["mode"],
                            label_info, args.out_root, args.sigma_mm)
            rows.append(row)
            print(f"{row['fit_id']:34s} C={row['n_contacts']:3d} M={row['n_nodes']:3d} "
                  f"sigma={row['sigma_mm']:4.1f}mm kept={row['n_events_kept']:6d} "
                  f"train={row['n_train']:6d} cover={row['label_coverage']:.4f}")

    manifest = {
        "contract": "topic5_wiring_economy_slp_rnn_v0_3_cache",
        "cohort": subjects,
        "n_patients": len(subjects),
        "n_fits": len(rows),
        "shared_fits": [r["fit_id"] for r in rows if r["scope"] == "shared"],
        "split_fits": [r["fit_id"] for r in rows if r["scope"] != "shared"],
        "sigma_override_mm": args.sigma_mm,
        "min_label_coverage": MIN_LABEL_COVERAGE,
        "low_coverage_subjects": low_coverage,
        "fits": rows,
    }
    path = args.out_root / "INPUT_MANIFEST.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.rename(path)
    print(f"\n{len(rows)} fits from {len(subjects)} patients "
          f"({len(manifest['shared_fits'])} shared, {len(manifest['split_fits'])} split)")
    if low_coverage:
        print(f"LOW A/B LABEL COVERAGE: {low_coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
