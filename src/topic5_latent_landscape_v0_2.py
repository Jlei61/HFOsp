"""Frozen-input contracts for Topic 5.2 latent propagation landscape v0.2.

This module deliberately has no early-ictal or SNN field reader.  It resolves
the complete Topic 5.1 checkpoint matrix, hashes the full decoder state, and
provides small deterministic helpers shared by the later streaming and
perturbation stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract
from src.topic5_rnn_motif_v0_4 import RolloutSizeHead, state_features


PUBLIC_TO_INTERNAL_ARM = {
    "L0": "L0_LOCAL_ONLY",
    "L1": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2m": "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3": "L3_LOCAL_PLUS_LEARNED_LR",
    "C-suffix": "C_L3_ORDER_SHUFFLED",
}
PUBLIC_ARMS = tuple(PUBLIC_TO_INTERNAL_ARM)
SEEDS = (0, 1, 2)
REUSED_PUBLIC_ARMS = frozenset({"L0", "L1", "L3"})


def classify_future_field_axis(
    scope: str, mode0_label: str, mode1_label: str
) -> dict[str, object]:
    """Freeze the scientifically admissible mode-axis tier and sign.

    Only a shared fit whose two train modes map bijectively to A and B has a
    canonical within-model A/B axis.  An own fit remains useful for a generic
    within-fit mode contrast, but its two modes must not be relabelled A/B.
    """
    scope = str(scope)
    labels = (str(mode0_label), str(mode1_label))
    if scope == "shared":
        if set(labels) != {"A", "B"}:
            return {
                "tier": "FIELD_AXIS_NOT_IDENTIFIABLE",
                "reason": "SHARED_MAPPING_NOT_BIJECTIVE_AB",
                "positive_mode": None,
                "negative_mode": None,
                "positive_label": None,
                "negative_label": None,
                "canonical_ab": False,
            }
        positive_mode = labels.index("A")
        negative_mode = labels.index("B")
        return {
            "tier": "CANONICAL_AB_SHARED",
            "reason": "SAME_FIT_TRAIN_MODES_BIJECTIVE_TO_AB",
            "positive_mode": positive_mode,
            "negative_mode": negative_mode,
            "positive_label": "A",
            "negative_label": "B",
            "canonical_ab": True,
        }
    if scope in {"own_a", "own_b"}:
        expected = "A" if scope == "own_a" else "B"
        if labels != (expected, expected):
            return {
                "tier": "FIELD_AXIS_NOT_IDENTIFIABLE",
                "reason": "OWN_SCOPE_MAPPING_MISMATCH",
                "positive_mode": None,
                "negative_mode": None,
                "positive_label": None,
                "negative_label": None,
                "canonical_ab": False,
            }
        return {
            "tier": "WITHIN_FIT_MODE_ONLY",
            "reason": "TWO_TRAIN_MODES_WITHIN_ONE_GEOMETRY_SCOPE",
            "positive_mode": 1,
            "negative_mode": 0,
            "positive_label": "mode1",
            "negative_label": "mode0",
            "canonical_ab": False,
        }
    return {
        "tier": "FIELD_AXIS_NOT_IDENTIFIABLE",
        "reason": "UNKNOWN_FIT_SCOPE",
        "positive_mode": None,
        "negative_mode": None,
        "positive_label": None,
        "negative_label": None,
        "canonical_ab": False,
    }


def rank_matrix_to_event_fields(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert contact-rank rows to parent-compatible full/start-removed fields.

    Non-participating contacts are zero.  The observed first rank is NaN in the
    start-removed field so it cannot directly carry the future-field contrast.
    """
    ranks = np.asarray(ranks)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must be event x contact, got {ranks.shape}")
    full = np.zeros(ranks.shape, dtype=np.float64)
    recurrence = np.zeros(ranks.shape, dtype=np.float64)
    for index, row in enumerate(ranks):
        participating = row >= 0
        if not np.any(participating):
            continue
        n_ranks = int(np.max(row[participating])) + 1
        full[index, participating] = 1.0 - row[participating] / max(n_ranks - 1, 1)
        seed = row == 0
        recurrence[index, seed] = np.nan
        post = row >= 1
        recurrence[index, post] = 1.0 - (row[post] - 1) / max(n_ranks - 2, 1)
    return full, recurrence


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Return a content hash without deserialising the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays including name, dtype, shape, and contiguous bytes."""
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def enumerate_cell_keys(fit_ids: Iterable[str]) -> list[tuple[str, str, int]]:
    return [
        (str(fit_id), arm, seed)
        for fit_id in fit_ids
        for arm in PUBLIC_ARMS
        for seed in SEEDS
    ]


def stable_event_hash(
    patient: str,
    split: int,
    event_source_index: int,
    event_dataset_index: int,
) -> str:
    """Hash an event identity without using its label, field, or response."""
    return hashlib.sha256(
        f"{patient}\0{int(split)}\0{int(event_source_index)}\0{int(event_dataset_index)}".encode(
            "utf-8"
        )
    ).hexdigest()


def response_blind_event_sample(
    *,
    patient: str,
    split: np.ndarray,
    event_source_index: np.ndarray,
    event_dataset_index: np.ndarray,
    phase_defined: np.ndarray,
    caps: Mapping[int, int],
) -> pd.DataFrame:
    """Select the lowest identity hashes within each split.

    The selector deliberately has no mode, field-coordinate, hidden-state, or
    outcome argument.  That makes response-dependent sampling impossible at
    this API boundary.
    """
    split = np.asarray(split)
    source = np.asarray(event_source_index)
    dataset = np.asarray(event_dataset_index)
    phase_defined = np.asarray(phase_defined, dtype=bool)
    if not (split.shape == source.shape == dataset.shape == phase_defined.shape):
        raise ValueError("event sampling arrays must have identical shapes")
    rows: list[dict[str, object]] = []
    for split_id, cap in sorted((int(k), int(v)) for k, v in caps.items()):
        eligible = np.flatnonzero((split == split_id) & phase_defined)
        hashes = [
            stable_event_hash(patient, split_id, int(source[i]), int(dataset[i]))
            for i in eligible
        ]
        ordered = sorted(zip(hashes, eligible.tolist()), key=lambda item: (item[0], item[1]))
        selected = ordered[: min(cap, len(ordered))]
        denominator = len(ordered)
        for rank, (digest, index) in enumerate(selected):
            rows.append({
                "event_array_index": int(index),
                "split": split_id,
                "event_source_index": int(source[index]),
                "event_dataset_index": int(dataset[index]),
                "identity_sha256": digest,
                "hash_rank_within_split": int(rank),
                "eligible_events_in_split": int(denominator),
                "selected_events_in_split": int(len(selected)),
                "inclusion_fraction": float(len(selected) / denominator) if denominator else 0.0,
            })
    return pd.DataFrame(rows)


def resolve_unit_dir(
    parent_root: Path,
    old_root: Path,
    fit_id: str,
    public_arm: str,
    seed: int,
    reused_fits: set[str],
) -> tuple[Path, str]:
    if public_arm not in PUBLIC_TO_INTERNAL_ARM:
        raise ValueError(f"unknown public arm {public_arm!r}")
    internal = PUBLIC_TO_INTERNAL_ARM[public_arm]
    if fit_id in reused_fits and public_arm in REUSED_PUBLIC_ARMS:
        return old_root / "per_fit" / fit_id / internal / f"seed{seed}", "V0_3_EXACT_REUSE"
    return parent_root / "formal_units" / fit_id / internal / f"seed{seed}", "V0_5_FORMAL_UNIT"


@dataclass(frozen=True)
class CheckpointCell:
    patient: str
    fit_id: str
    geometry_view: str
    public_arm: str
    internal_arm: str
    seed: int
    checkpoint_source: str
    unit_dir: str
    metrics_path: str
    weights_path: str
    size_decoder_path: str
    graph_path: str
    checkpoint_sha256: str
    size_decoder_sha256: str
    graph_sha256: str
    metrics_sha256: str
    model_config_sha256: str
    split_sha256: str
    event_identity_sha256: str
    H_sha256: str
    node_mask_sha256: str
    contact_order_sha256: str
    decoder_sha256: str
    n_nodes: int
    n_contacts: int
    n_events_axis_train: int
    n_events_axis_validation: int
    n_events_test: int
    target_values_read: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _required_unit_files(unit_dir: Path) -> dict[str, Path]:
    return {
        "metrics": unit_dir / "metrics.json",
        "weights": unit_dir / "weights.pt",
        "size_decoder": unit_dir / "rollout_size_head.pt",
        "graph": unit_dir / "graph.npz",
        "done": unit_dir / "DONE.json",
    }


def _contact_order(provenance: Mapping[str, object], events_raw: np.lib.npyio.NpzFile) -> list[str]:
    if "joint_contacts" in provenance:
        return [str(value) for value in provenance["joint_contacts"]]
    if "contact_names" in events_raw.files:
        return [str(value) for value in events_raw["contact_names"].tolist()]
    raise KeyError("neither joint_contacts nor events_raw contact_names is available")


def resolve_checkpoint_cells(parent_root: Path, old_root: Path) -> list[CheckpointCell]:
    """Resolve and validate the complete 42 x 5 x 3 analysis matrix."""
    parent_root = Path(parent_root).resolve()
    old_root = Path(old_root).resolve()
    census = pd.read_csv(parent_root / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(parent_root / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    if len(census) != 42 or census["fit_id"].astype(str).nunique() != 42:
        raise RuntimeError("frozen parent census must contain 42 unique fits")
    reused_fits = set(
        reuse.loc[
            reuse["checkpoint_reuse_eligible"].map(parse_bool), "fit_id"
        ].astype(str)
    )
    if len(reused_fits) != 11:
        raise RuntimeError(f"expected 11 exact-reuse fits, found {len(reused_fits)}")

    cells: list[CheckpointCell] = []
    fit_lookup = census.set_index(census["fit_id"].astype(str), drop=False)
    for fit_id, public_arm, seed in enumerate_cell_keys(census["fit_id"].astype(str)):
        fit = fit_lookup.loc[fit_id]
        unit_dir, source = resolve_unit_dir(
            parent_root, old_root, fit_id, public_arm, seed, reused_fits
        )
        paths = _required_unit_files(unit_dir)
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{fit_id}/{public_arm}/seed{seed} missing {missing}: {unit_dir}")
        metrics = json.loads(paths["metrics"].read_text())
        done = json.loads(paths["done"].read_text())
        if metrics.get("fit_id") != fit_id:
            raise RuntimeError(f"metrics fit mismatch: {paths['metrics']}")
        if metrics.get("arm") != PUBLIC_TO_INTERNAL_ARM[public_arm]:
            raise RuntimeError(f"metrics arm mismatch: {paths['metrics']}")
        if int(metrics.get("seed", -1)) != seed:
            raise RuntimeError(f"metrics seed mismatch: {paths['metrics']}")
        if metrics.get("target_values_read") is not False or done.get("target_values_read") is not False:
            raise RuntimeError(f"target marker is not false: {unit_dir}")
        if not bool(done.get("ok")):
            raise RuntimeError(f"unit is not complete: {unit_dir}")

        cache = parent_root / "cache" / fit_id
        plane_path = cache / "plane.npz"
        events_path = cache / "events.npz"
        events_raw_path = cache / "events_raw.npz"
        provenance_path = cache / "provenance.json"
        for path in (plane_path, events_path, events_raw_path, provenance_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        provenance = json.loads(provenance_path.read_text())
        with (
            np.load(plane_path, allow_pickle=False) as plane,
            np.load(events_path, allow_pickle=False) as events,
            np.load(events_raw_path, allow_pickle=False) as events_raw,
            np.load(paths["graph"], allow_pickle=False) as graph,
        ):
            split = np.asarray(events["split"])
            keep = split >= 0
            event_identity = {
                "event_source_index": np.asarray(events["event_source_index"])[keep],
                "event_dataset_index": np.asarray(events["event_dataset_index"])[keep],
                "event_abs_time": np.asarray(events["event_abs_time"])[keep],
            }
            contact_order = _contact_order(provenance, events_raw)
            n_contacts = int(provenance.get("n_joint_contacts", provenance.get("n_contacts", -1)))
            n_nodes = int(provenance["n_nodes"])
            if np.asarray(plane["H"]).shape != (n_contacts, n_nodes):
                raise RuntimeError(f"H shape mismatch for {fit_id}")
            if np.asarray(graph["mask"]).shape != (n_nodes, n_nodes):
                raise RuntimeError(f"node mask shape mismatch for {unit_dir}")
            if len(contact_order) != n_contacts:
                raise RuntimeError(f"contact order mismatch for {fit_id}")
            split_hash = arrays_sha256({
                "split": split,
                "event_source_index": np.asarray(events["event_source_index"]),
                "event_dataset_index": np.asarray(events["event_dataset_index"]),
            })
            event_identity_hash = arrays_sha256(event_identity)
            h_hash = arrays_sha256({"H": np.asarray(plane["H"])})
            mask_hash = arrays_sha256({"node_mask": np.asarray(graph["mask"], np.uint8)})
        contact_hash = canonical_json_sha256(contact_order)
        weight_hash = sha256_file(paths["weights"])
        size_hash = sha256_file(paths["size_decoder"])
        decoder_hash = canonical_json_sha256({
            "model_weights_sha256": weight_hash,
            "size_decoder_sha256": size_hash,
            "stop_head_location": "model_weights.stop_head",
            "decoder_contract": "topic5_rnn_motif_v0_4",
        })
        cells.append(CheckpointCell(
            patient=str(fit["subject"]),
            fit_id=fit_id,
            geometry_view=str(fit["scope"]),
            public_arm=public_arm,
            internal_arm=PUBLIC_TO_INTERNAL_ARM[public_arm],
            seed=seed,
            checkpoint_source=source,
            unit_dir=str(unit_dir),
            metrics_path=str(paths["metrics"]),
            weights_path=str(paths["weights"]),
            size_decoder_path=str(paths["size_decoder"]),
            graph_path=str(paths["graph"]),
            checkpoint_sha256=weight_hash,
            size_decoder_sha256=size_hash,
            graph_sha256=sha256_file(paths["graph"]),
            metrics_sha256=sha256_file(paths["metrics"]),
            model_config_sha256=canonical_json_sha256(metrics["config"]),
            split_sha256=split_hash,
            event_identity_sha256=event_identity_hash,
            H_sha256=h_hash,
            node_mask_sha256=mask_hash,
            contact_order_sha256=contact_hash,
            decoder_sha256=decoder_hash,
            n_nodes=n_nodes,
            n_contacts=n_contacts,
            n_events_axis_train=int(np.count_nonzero(split == 0)),
            n_events_axis_validation=int(np.count_nonzero(split == 1)),
            n_events_test=int(np.count_nonzero(split == 2)),
            target_values_read=False,
        ))
    audit_checkpoint_cells(cells)
    return cells


def audit_checkpoint_cells(cells: Iterable[CheckpointCell]) -> dict[str, object]:
    cells = list(cells)
    keys = {(row.fit_id, row.public_arm, row.seed) for row in cells}
    if len(cells) != 630 or len(keys) != 630:
        raise RuntimeError(f"checkpoint matrix is not 630 unique cells: {len(cells)}/{len(keys)}")
    source_counts = pd.Series([row.checkpoint_source for row in cells]).value_counts().to_dict()
    if source_counts != {"V0_5_FORMAL_UNIT": 531, "V0_3_EXACT_REUSE": 99}:
        raise RuntimeError(f"checkpoint source counts drifted: {source_counts}")
    frame = pd.DataFrame(row.to_dict() for row in cells)
    fit_errors: list[str] = []
    for fit_id, group in frame.groupby("fit_id", sort=False):
        for column in ("split_sha256", "event_identity_sha256", "H_sha256", "contact_order_sha256"):
            if group[column].nunique() != 1:
                fit_errors.append(f"{fit_id}:{column}")
        if group["n_nodes"].nunique() != 1 or group["n_contacts"].nunique() != 1:
            fit_errors.append(f"{fit_id}:shape")
    if fit_errors:
        raise RuntimeError(f"within-fit frozen-input drift: {fit_errors[:10]}")
    if bool(frame["target_values_read"].any()):
        raise RuntimeError("checkpoint manifest contains a target-read cell")
    return {
        "resolved_cells": len(cells),
        "unique_cells": len(keys),
        "patients": int(frame["patient"].nunique()),
        "fits": int(frame["fit_id"].nunique()),
        "source_counts": {key: int(value) for key, value in source_counts.items()},
        "within_fit_input_consistency": True,
        "target_values_read": False,
    }


def estimate_resource_budget(parent_root: Path, cells: Iterable[CheckpointCell]) -> dict[str, object]:
    """Estimate the rejected full archive and the two-pass storage envelope."""
    parent_root = Path(parent_root)
    frame = pd.DataFrame(row.to_dict() for row in cells)
    rows: list[dict[str, int | str]] = []
    for fit_id, group in frame.groupby("fit_id", sort=False):
        cache = parent_root / "cache" / fit_id
        with np.load(cache / "events.npz", allow_pickle=False) as events:
            ranks = np.asarray(events["ranks"])
            split = np.asarray(events["split"])
        keep = split >= 0
        lengths = np.where(
            np.any(ranks[keep] >= 0, axis=1),
            np.max(np.where(ranks[keep] >= 0, ranks[keep], -1), axis=1) + 1,
            0,
        )
        test_lengths = lengths[split[keep] == 2]
        n_nodes = int(group["n_nodes"].iloc[0])
        n_contacts = int(group["n_contacts"].iloc[0])
        unit_count = len(group)
        steps = int(lengths.sum())
        test_events = int(np.count_nonzero(split[keep] == 2))
        full_bytes_per_step = 4 * (n_nodes + 2 * n_contacts) + 32
        pass1_bytes_per_step = 4 * 16 + 24
        selected_states_per_unit = int(np.minimum(test_lengths, 3).sum())
        selected_bytes_per_state = 4 * (n_nodes + 3 * n_contacts) + 256
        rows.append({
            "fit_id": str(fit_id),
            "unit_count": unit_count,
            "event_steps": steps,
            "test_events": test_events,
            "selected_states_per_unit": selected_states_per_unit,
            "full_archive_bytes": steps * unit_count * full_bytes_per_step,
            "pass1_projection_bytes": steps * unit_count * pass1_bytes_per_step,
            "pass2_selected_q_bytes": selected_states_per_unit * unit_count * selected_bytes_per_state,
        })
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("event_steps", "test_events", "full_archive_bytes",
                    "pass1_projection_bytes", "pass2_selected_q_bytes")
    }
    return {
        "contract": "topic5_latent_landscape_resource_preflight_v0_2",
        "analysis_cells": int(len(frame)),
        "fits": int(frame["fit_id"].nunique()),
        "estimate_notes": {
            "full_archive": "hidden + pre/post contact arrays + bookkeeping for every state",
            "pass1_projection": "upper bound if every 16D projection were persisted; streaming summaries are smaller",
            "pass2_selected_q": "three frozen phases per heldout event when event length permits",
            "rollout_and_patch_outputs": "must be measured by sentinel before cohort execution",
        },
        "totals": totals,
        "per_fit": rows,
    }


@dataclass
class DecoderState:
    """The executable closed-loop state; hidden state alone is insufficient."""

    h: torch.Tensor
    recruited: torch.Tensor
    rank_index: int

    def clone(self) -> "DecoderState":
        return DecoderState(
            h=self.h.detach().clone(),
            recruited=self.recruited.detach().clone(),
            rank_index=int(self.rank_index),
        )


def parameter_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def load_frozen_cell(
    parent_root: Path,
    manifest_row: Mapping[str, object],
    device: torch.device,
) -> tuple[LBSSModel, RolloutSizeHead, dict[str, object], dict[str, np.ndarray]]:
    """Instantiate one exact frozen model and its complete decoder."""
    parent_root = Path(parent_root)
    metrics_path = Path(str(manifest_row["metrics_path"]))
    graph_path = Path(str(manifest_row["graph_path"]))
    metrics = json.loads(metrics_path.read_text())
    cache = parent_root / "cache" / str(manifest_row["fit_id"])
    provenance = json.loads((cache / "provenance.json").read_text())
    with np.load(cache / "plane.npz", allow_pickle=False) as handle:
        plane = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(graph_path, allow_pickle=False) as handle:
        graph = {name: np.asarray(handle[name]) for name in handle.files}
    cfg = metrics["config"]
    pools = build_pool_contract(
        plane["D_mm"], cfg["density"], cfg["added_fraction"],
        cfg.get("r_local_multiplier", 2.0),
    )
    fixed = graph["added_mask"] if metrics["arm"] == "L2M_MACRO_MATCHED_RANDOM_LR" else None
    n_contacts = int(provenance.get("n_joint_contacts", provenance.get("n_contacts", -1)))
    model = LBSSModel(LBSSConfig(
        arm=str(metrics["arm"]),
        n_contacts=n_contacts,
        n_nodes=int(provenance["n_nodes"]),
        observation_operator=plane["H"],
        node_distance_mm=plane["D_mm"],
        local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added,
        seed=int(metrics["seed"]),
        state_dim=int(cfg["state_dim"]),
        fixed_added_mask=fixed,
    )).to(device)
    model.load_state_dict(torch.load(
        Path(str(manifest_row["weights_path"])), map_location=device, weights_only=True
    ))
    model.freeze_mask()
    model.eval()
    decoder = RolloutSizeHead(n_contacts).to(device)
    decoder.load_state_dict(torch.load(
        Path(str(manifest_row["size_decoder_path"])), map_location=device, weights_only=True
    ))
    decoder.eval()
    return model, decoder, metrics, plane


@torch.no_grad()
def manual_teacher_forced_trace(
    model: LBSSModel,
    x: torch.Tensor,
    recruited: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Replay hidden, pre-mask logits, and STOP one step at a time."""
    if x.ndim != 3 or recruited.shape != x.shape:
        raise ValueError("x and recruited must be aligned (batch, step, contact) tensors")
    h = torch.zeros(x.shape[0], model.n_nodes * model.state_dim, device=x.device)
    hidden: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    stop: list[torch.Tensor] = []
    for step in range(x.shape[1]):
        h = model._step(h, x[:, step])
        hidden.append(h.detach().clone())
        logits.append(model._readout(h))
        features = state_features(model, h, step, recruited[:, step].mean(-1))
        stop.append(model._stop(h, features[:, 2], recruited[:, step].mean(-1)))
    return {
        "hidden": torch.stack(hidden, dim=1),
        "pre_mask_logits": torch.stack(logits, dim=1),
        "stop_logits": torch.stack(stop, dim=1),
    }


@torch.no_grad()
def decoder_snapshot(
    model: LBSSModel,
    size_decoder: RolloutSizeHead,
    state: DecoderState,
    *,
    force_continue: bool = False,
) -> dict[str, object]:
    """Evaluate STOP, cardinality, repeat mask, and deterministic contact choice."""
    if state.h.shape[0] != 1 or state.recruited.shape != (1, model.n_contacts):
        raise ValueError("decoder_snapshot currently requires one aligned state")
    fraction = state.recruited.mean(-1)
    features = state_features(model, state.h, state.rank_index, fraction)
    pre_mask_logits = model._readout(state.h)
    stop_logit = model._stop(state.h, features[:, 2], fraction)
    stop_probability = torch.sigmoid(stop_logit)
    size_logits = size_decoder(features)
    available = state.recruited <= 0
    should_stop = bool(stop_probability.item() >= 0.5 or not bool(available.any()))
    picked: list[int] = []
    if force_continue or not should_stop:
        eligible = np.flatnonzero(available[0].detach().cpu().numpy())
        if eligible.size:
            count = min(int(size_logits.argmax(-1).item()) + 1, int(eligible.size))
            score = pre_mask_logits[0].detach().cpu().numpy()
            order = np.lexsort((eligible, -score[eligible]))
            picked = eligible[order[:count]].astype(int).tolist()
    return {
        "features": features.detach().clone(),
        "pre_mask_logits": pre_mask_logits.detach().clone(),
        "stop_logit": stop_logit.detach().clone(),
        "stop_probability": stop_probability.detach().clone(),
        "size_logits": size_logits.detach().clone(),
        "available": available.detach().clone(),
        "should_stop": should_stop,
        "picked": picked,
    }


def select_replay_event_indices(ranks: np.ndarray, split: np.ndarray, per_split: int = 10) -> np.ndarray:
    """Select deterministic length-quantile events from each frozen split."""
    ranks = np.asarray(ranks)
    split = np.asarray(split)
    if ranks.shape[0] != split.shape[0]:
        raise ValueError("ranks and split must align")
    lengths = np.where(
        np.any(ranks >= 0, axis=1),
        np.max(np.where(ranks >= 0, ranks, -1), axis=1) + 1,
        0,
    )
    chosen: list[int] = []
    for label in (0, 1, 2):
        candidates = np.flatnonzero(split == label)
        if candidates.size == 0:
            raise RuntimeError(f"split {label} contains no events")
        order = candidates[np.lexsort((candidates, lengths[candidates]))]
        take = min(int(per_split), int(order.size))
        positions = np.linspace(0, order.size - 1, take).round().astype(int)
        chosen.extend(order[positions].tolist())
    return np.asarray(chosen, dtype=int)
