"""Frame-cache loader for the Topic 5.2 dynamical motif RNN.

Kept apart from the model module so the model code can be statically proven
free of template / mode / seizure identifiers.  Mode labels are loaded here
because scoring needs them; they never enter a model input.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.topic5_lbss_rnn_v0_2 import build_pool_contract

FRAMES = ("GEOMETRY_ONLY_PCA2", "PARENT_FROZEN_FRAME", "SYNTHETIC")


@dataclass(frozen=True)
class FrameUnit:
    frame: str
    unit_id: str
    subject: str
    contact_names: list[str]
    shafts: list[str]
    contacts_xy_mm: np.ndarray
    nodes_xy_mm: np.ndarray
    H: np.ndarray
    D_mm: np.ndarray
    local_mask: np.ndarray
    r_local_mm: float
    sigma_mm: float
    ranks: np.ndarray
    split: np.ndarray
    mode_label: np.ndarray
    mode_posterior: np.ndarray
    event_abs_time: np.ndarray
    event_lag_raw: np.ndarray
    provenance: dict

    @property
    def n_contacts(self) -> int:
        return len(self.contact_names)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes_xy_mm.shape[0])

    def indices(self, split_value: int) -> np.ndarray:
        return np.flatnonzero(self.split == split_value)

    def observed_indices(self) -> np.ndarray:
        """Events the trainer may see: train, calibration and development test."""
        return np.flatnonzero(self.split >= 0)


def load_frame_unit(root: Path, frame: str, unit_id: str) -> FrameUnit:
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}")
    if frame in ("GEOMETRY_ONLY_PCA2", "SYNTHETIC"):
        return _load_geometry_unit(root, unit_id, frame)
    return _load_parent_unit(root, unit_id)


def _load_geometry_unit(root: Path, unit_id: str, frame: str = "GEOMETRY_ONLY_PCA2") -> FrameUnit:
    directory = root / "frame_cache" / frame / unit_id
    plane = np.load(directory / "plane.npz", allow_pickle=False)
    events = np.load(directory / "events.npz", allow_pickle=True)
    provenance = json.loads((directory / "provenance.json").read_text())
    return FrameUnit(
        frame=frame,
        unit_id=unit_id,
        subject=provenance["subject"],
        contact_names=[str(v) for v in events["contact_names"]],
        shafts=[str(v) for v in events["shafts"]],
        contacts_xy_mm=np.asarray(plane["contacts_xy_mm"], dtype=np.float32),
        nodes_xy_mm=np.asarray(plane["nodes_xy_mm"], dtype=np.float32),
        H=np.asarray(plane["H"], dtype=np.float32),
        D_mm=np.asarray(plane["D_mm"], dtype=np.float32),
        local_mask=np.asarray(plane["local_mask"], dtype=np.uint8),
        r_local_mm=float(provenance["r_local_mm"]),
        sigma_mm=float(provenance["sigma_mm"]),
        ranks=np.asarray(events["ranks"]),
        split=np.asarray(events["split"]),
        mode_label=np.asarray(events["prefix_mode"]),
        mode_posterior=np.asarray(events["prefix_posterior"], dtype=np.float32),
        event_abs_time=np.asarray(events["event_abs_time"]),
        event_lag_raw=np.asarray(events["event_lag_raw"], dtype=np.float32),
        provenance=provenance,
    )


def _load_parent_unit(root: Path, unit_id: str) -> FrameUnit:
    parent = root.parent / "topic5_multiscale_effective_scaffold_v0_5" / "cache" / unit_id
    plane = np.load(parent / "plane.npz", allow_pickle=False)
    events = np.load(parent / "events.npz", allow_pickle=True)
    raw = np.load(parent / "events_raw.npz", allow_pickle=True)
    provenance = json.loads((parent / "provenance.json").read_text())
    pools = build_pool_contract(np.asarray(plane["D_mm"], dtype=float))
    field_shafts = _parent_shafts(provenance["subject"], [str(v) for v in raw["contact_names"]])
    return FrameUnit(
        frame="PARENT_FROZEN_FRAME",
        unit_id=unit_id,
        subject=provenance["subject"],
        contact_names=[str(v) for v in raw["contact_names"]],
        shafts=field_shafts,
        contacts_xy_mm=np.asarray(plane["contacts_xy_mm"], dtype=np.float32),
        nodes_xy_mm=np.asarray(plane["nodes_xy_mm"], dtype=np.float32),
        H=np.asarray(plane["H"], dtype=np.float32),
        D_mm=np.asarray(plane["D_mm"], dtype=np.float32),
        local_mask=np.asarray(pools.local_mask, dtype=np.uint8),
        r_local_mm=float(pools.r_local_mm),
        sigma_mm=float(plane["sigma_mm"][0]),
        ranks=np.asarray(raw["ranks"]),
        split=np.asarray(events["split"]),
        mode_label=np.asarray(events["mode"]),
        mode_posterior=np.asarray(events["prefix_posterior"], dtype=np.float32),
        event_abs_time=np.asarray(raw["event_abs_time"]),
        event_lag_raw=np.asarray(raw["event_lag_raw"], dtype=np.float32),
        provenance=provenance,
    )


def _parent_shafts(subject: str, contact_names: list[str]) -> list[str]:
    field_root = Path("/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/"
                      "template_gradient_fields/per_subject")
    record = json.loads((field_root / f"{subject}.json").read_text())
    lookup = dict(zip([str(v) for v in record["names"]], [str(v) for v in record["shafts"]]))
    return [lookup[name] for name in contact_names]


def layout_axes_in_frame(unit: FrameUnit) -> dict[str, dict]:
    """Contact-cloud PCA1 and dominant-shaft axis expressed as in-frame angles."""
    if unit.frame == "GEOMETRY_ONLY_PCA2":
        return unit.provenance["layout_axes_in_frame"]
    field_root = Path("/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/"
                      "template_gradient_fields/per_subject")
    record = json.loads((field_root / f"{unit.subject}.json").read_text())
    names = [str(v) for v in record["names"]]
    coords = np.asarray(record["coords"], dtype=float)
    columns = [names.index(name) for name in unit.contact_names]
    points = coords[columns]
    plane = np.load(
        Path("/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/results/"
             "topic5_multiscale_effective_scaffold_v0_5/cache") / unit.unit_id / "plane.npz",
        allow_pickle=False,
    )
    del plane
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pca1 = vt[0]
    counts: dict[str, int] = {}
    for shaft in unit.shafts:
        counts[shaft] = counts.get(shaft, 0) + 1
    dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    members = [index for index, shaft in enumerate(unit.shafts) if shaft == dominant]
    shaft_axis = np.full(3, np.nan)
    if len(members) >= 2:
        block = points[members] - points[members].mean(axis=0)
        shaft_axis = np.linalg.svd(block, full_matrices=False)[2][0]
    # Recover the frame basis from the contact projection: solve for u, w.
    basis, *_ = np.linalg.lstsq(centred, unit.contacts_xy_mm - unit.contacts_xy_mm.mean(axis=0),
                                rcond=None)
    u, w = basis[:, 0], basis[:, 1]
    u = u / max(np.linalg.norm(u), 1e-12)
    w = w / max(np.linalg.norm(w), 1e-12)
    out = {}
    for name, vector in (("contact_cloud_pca1", pca1), ("dominant_shaft", shaft_axis)):
        if not np.all(np.isfinite(vector)):
            out[name] = {"estimable": False, "theta_rad": None, "in_plane_norm": None}
            continue
        planar = np.array([vector @ u, vector @ w])
        norm = float(np.linalg.norm(planar))
        out[name] = {
            "estimable": bool(norm > 0.2),
            "theta_rad": float(np.arctan2(planar[1], planar[0]) % np.pi) if norm > 1e-9 else None,
            "in_plane_norm": norm,
        }
    return out
