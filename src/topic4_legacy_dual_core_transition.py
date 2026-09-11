"""Rebuild the frozen rev11-NLC substrate for the Z/M ictal-transition round.

The producer of record for that substrate is
``scripts/run_topic4_rev10_r_edge_flow_worker.py``. That script is NEVER modified
here -- changing it would make the rev11-NLC round non-reproducible in place --
so this module mirrors its construction sequence instead. The sequence is
order-sensitive: ``net["rng"]`` is re-seeded at a specific point and the field
query must run before the edge mapper, so reordering silently changes numbers.
Gate A (``scripts/audit_topic4_zm_ictal_transition.py --gate parity``) is what
proves the mirror is exact.
"""
from __future__ import annotations

import hashlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), os.path.join(str(ROOT), "src", "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PATHWAYS = ("E_to_E", "E_to_I")


@contextmanager
def _artifact_working_directory(artifact_root):
    if artifact_root is None:
        yield
        return
    previous = os.getcwd()
    os.chdir(artifact_root)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class Substrate:
    """Everything a run needs, plus the provenance that proves it is frozen."""

    params: Any
    net: dict
    n_e: int
    n_i: int
    positions_e: np.ndarray
    positions_i: np.ndarray
    h_e: np.ndarray
    h_i: np.ndarray
    vtheta: np.ndarray
    delta_vtheta: np.ndarray
    montage: Any
    contact_names: list
    contact_xy: np.ndarray
    shaft_ids: np.ndarray
    valid_contacts: np.ndarray
    edge_audit: dict
    edge_coefficients: np.ndarray
    ee_out_gain: np.ndarray
    etoi_out_gain: np.ndarray
    axis_unit: np.ndarray
    axis_source_xy: np.ndarray
    axis_sink_xy: np.ndarray
    detector_threshold: float
    engine: dict
    stage: dict
    network_cache: dict
    field_transform: Any = None
    extras: dict = field(default_factory=dict)


def load_round_config(path):
    import json
    return json.loads(Path(path).read_text())


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_path(relative_path, artifact_root=None):
    local = ROOT / relative_path
    if local.exists() or artifact_root is None:
        return local
    return Path(artifact_root) / relative_path


def verify_frozen_inputs(config, *, artifact_root=None):
    """Hash every declared input. Raises rather than warning -- a drifted input
    means the substrate is not the frozen one and every number downstream is
    about a different model."""
    records = {}
    for key, record in config["inputs"].items():
        path = _input_path(record["path"], artifact_root)
        if not path.exists():
            raise RuntimeError(f"input missing: {record['path']}")
        digest = _sha256_file(path)
        records[key] = {"path": record["path"], "expected": record["sha256"],
                        "observed": digest, "match": digest == record["sha256"]}
        if digest != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    return {"all_match": True, "records": records}


def _load_json_input(record, *, artifact_root=None):
    import json
    return json.loads(_input_path(record["path"], artifact_root).read_text())


def _outgoing_by_pathway(matrices, n_e, pathway):
    """Outgoing weight per E source. The mapper conserves the INCOMING budget
    per target by contract, so the incoming totals cannot show its effect; only
    the outgoing side varies."""
    total = np.zeros(n_e, float)
    for matrix in matrices:
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        cols = np.asarray(coo.col, np.int64)
        mask = rows < n_e if pathway == "E_to_E" else rows >= n_e
        if not np.any(mask):
            continue
        total += np.bincount(cols[mask],
                             weights=np.asarray(coo.data[mask], float),
                             minlength=n_e)
    return total


def _gain(pre, post):
    out = np.full(pre.shape, np.nan, float)
    good = pre > 0.0
    out[good] = post[good] / pre[good]
    return out


def _placement_with_artifact_root(stage, artifact_root):
    """Mirror the frozen placement while resolving its result inputs explicitly."""
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci,
    )
    root = "." if artifact_root is None else str(artifact_root)
    montage, _, _, _ = gradient_shared_template_foci(stage["subject"], 3, root=root)
    _, source_names, sink_names = template_source_foci(
        stage["subject"], "narrow", 3, root=root,
    )
    registered = register_to_sheet(
        montage, source_names, sink_names,
        L=stage["engine"]["L"], target_inter_core_mm=None,
    )
    axis = registered["sink_centroid"] - registered["source_centroid"]
    registered["axis_unit_vec"] = axis / np.linalg.norm(axis)
    return registered


def dose_local_connectivity_coefficients(coefficients, *, ee_dose=1.0,
                                         etoi_dose=1.0):
    """Scale the two learned pathway rows without changing their direction."""
    values = np.asarray(coefficients, float)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("local connectivity coefficients must have shape (2, n)")
    doses = np.asarray([ee_dose, etoi_dose], float)
    if not np.all(np.isfinite(doses)) or np.any(doses < 0.0):
        raise ValueError("pathway doses must be finite and non-negative")
    return values.copy() * doses[:, None]


def _cache_record(cache_hit, cache_source):
    """Always carry the pickle hash. ``_load_network`` only reports
    ``cache_sha256`` on the cache-HIT path, so a freshly built network would
    otherwise have no verifiable identity -- which is exactly the case the
    parity gate needs to check."""
    record = {"hit": bool(cache_hit), **(cache_source or {})}
    path = record.get("frozen_cache_path")
    if path and "cache_sha256" not in record and os.path.exists(path):
        record["cache_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return record


def build_substrate(config, candidate_id, seed, *, cache_dir, field_transform=None,
                    ee_dose=1.0, etoi_dose=1.0,
                    node_candidate_override=None, node_depth_shrinkage=1.0,
                    node_gain=1.0, node_dispersion_candidate_override=None,
                    ee_ellipse_angle_deg=45.0,
                    ee_ellipse_aspect_ratio=2.0,
                    artifact_root=None):
    """Reconstruct one frozen arm on one network seed.

    ``field_transform`` is a square-symmetry element name; when given, the node
    field is queried at inverse-transformed positions and the two directed flow
    coefficients are rotated by the same matrix, which keeps the field-and-flow
    RULE a rigid image of the original. It does NOT make the substrate an
    isometric copy -- the realized graph, its patient-derived anisotropy and the
    contacts stay fixed. See src/topic4_zm_d4.py.
    """
    import json

    from params import Params
    from scripts.run_topic4_rev10_sa_spectral_field_worker import _candidate_node
    from scripts.run_topic4_core_field_stage3_fit import _load_cmrun
    from scripts.run_topic4_rev9_node_kick_canary import _load_network
    from src.sef_hfo_observation import VirtualMontage
    from src.topic4_continuous_field import continuous_field_h_with_queries
    from src.topic4_core_field_rev9 import (
        reconstruct_node_from_dual_fields, reconstruct_node_from_h,
    )
    from src.topic4_core_field_runner import _placement
    from src.topic4_graph_edge_flow import array_sha256
    from src.topic4_local_connectivity import continuous_local_e_source_flow
    from src.topic4_manual_dual_core import (
        budget_matched_dual_core_h, dual_core_query_h,
    )
    from src.topic4_rev20_dual_core_mechanism import (
        fixed_topology_ee_ellipse_redistribution,
    )

    verify_frozen_inputs(config, artifact_root=artifact_root)
    inputs = config["inputs"]
    manifest = _load_json_input(
        inputs["frozen_substrate_manifest"], artifact_root=artifact_root,
    )
    if manifest["status"] != "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_LIBRARY_FROZEN":
        raise RuntimeError("frozen substrate manifest status changed")
    matches = [row for row in manifest["candidate_set"]["candidates"]
               if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"candidate {candidate_id!r} is outside the frozen library")
    candidate = matches[0]

    base = _load_json_input(inputs["rev9_base_config"], artifact_root=artifact_root)
    stage = _load_json_input(inputs["stage_config"], artifact_root=artifact_root)
    contract = _load_json_input(inputs["contact_contract"], artifact_root=artifact_root)
    anchor_config = _load_json_input(
        inputs["node_anchor_config"], artifact_root=artifact_root,
    )
    detector_audit = _load_json_input(
        inputs["common_detector_audit"], artifact_root=artifact_root,
    )
    detector = float(config["engine_detector"]["population_active_fraction_threshold"])
    if detector != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("common detector changed")

    engine = stage["engine"]
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1

    # ---- 1-3: params, network, E positions (order copied from the producer) ----
    params = Params(g=engine["g"], L=engine["L"], density=engine["density"],
                    T=float(config["simulation"]["duration_ms"]), dt=engine["dt"],
                    nu_ext_ratio=cmrun.DRIVE, seed=int(seed))
    reg = (
        _placement(stage) if artifact_root is None
        else _placement_with_artifact_root(stage, artifact_root)
    )
    with _artifact_working_directory(artifact_root):
        net, n_e, n_i, cache_hit, cache_source = _load_network(
            params, stage, reg, int(seed), base, str(cache_dir))
    positions = np.asarray(net["pos"][:n_e], float)
    positions_i = np.asarray(net["pos"][n_e:], float)

    # ---- 4-5: node field and coefficient hash ----
    frozen_node_candidate = candidate["node_field"]
    node_candidate = (
        frozen_node_candidate if node_candidate_override is None
        else dict(node_candidate_override)
    )
    field_type = node_candidate.get("field_type")
    manual_field_audit = None
    if field_type == "spline_continuous":
        node = _candidate_node(node_candidate, positions, n_total=n_e + n_i,
                               stage=stage, config=anchor_config)
    elif field_type == "manual_dual_core_budget_matched":
        if field_transform is not None:
            raise RuntimeError("manual dual-core field does not support field transforms")
        h_manual, manual_field_audit = budget_matched_dual_core_h(
            positions, np.asarray(node_candidate["centers_mm"], float),
            target_count=int(node_candidate["target_count"]),
        )
        node = reconstruct_node_from_h(
            h_manual, n_total=n_e + n_i,
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"],
        )
    else:
        raise RuntimeError(f"unsupported Node field type: {field_type!r}")
    depth_shrinkage = float(node_depth_shrinkage)
    gain = float(node_gain)
    if depth_shrinkage != 1.0 or gain != 1.0:
        node = reconstruct_node_from_h(
            node["h"], n_total=n_e + n_i,
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"], depth_shrinkage=depth_shrinkage,
            node_gain=gain,
        )
    if node_dispersion_candidate_override is not None:
        if depth_shrinkage != 1.0 or gain != 1.0:
            raise RuntimeError("dual Node fields cannot also change depth or scalar gain")
        dispersion_candidate = dict(node_dispersion_candidate_override)
        if dispersion_candidate.get("field_type") != "spline_continuous":
            raise RuntimeError("Node dispersion field must be a continuous spline")
        dispersion_node = _candidate_node(
            dispersion_candidate, positions, n_total=n_e + n_i,
            stage=stage, config=anchor_config,
        )
        node = reconstruct_node_from_dual_fields(
            node["h"], dispersion_node["h"], n_total=n_e + n_i,
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"],
        )
    expected_mass = (
        float(node_candidate["target_count"])
        if field_type == "manual_dual_core_budget_matched"
        else float(stage["N_core_manual"])
    )
    if not np.isclose(node["h"].sum(), expected_mass, atol=1e-8):
        raise RuntimeError("Node anchor field budget changed")
    coefficients = np.asarray(candidate["coefficients"], float)
    if array_sha256(coefficients) != candidate["coefficients_sha256"]:
        raise RuntimeError("edge coefficient hash changed")

    # ---- 6: the producer re-seeds here, before the edge mapper ----
    net["rng"] = np.random.default_rng(int(seed))

    # ---- 7: E/I field query, optionally through the spatial transform ----
    query_e, query_i = positions, positions_i
    coefficients_eff = coefficients
    if field_transform is not None:
        from src.topic4_zm_d4 import (inverse_query_positions,
                                      transform_flow_coefficients)
        query_e = inverse_query_positions(positions, field_transform, L=engine["L"])
        query_i = inverse_query_positions(positions_i, field_transform, L=engine["L"])
        coefficients_eff = transform_flow_coefficients(coefficients, field_transform)
    coefficients_eff = dose_local_connectivity_coefficients(
        coefficients_eff, ee_dose=ee_dose, etoi_dose=etoi_dose)
    if field_type == "spline_continuous":
        h_e, h_i, field_query_audit = continuous_field_h_with_queries(
            node_candidate["coefficients"], query_e, query_i,
            n_basis=node_candidate["n_basis"], degree=node_candidate["degree"],
            target_count=stage["N_core_manual"], L=engine["L"])
    else:
        h_e = np.asarray(node["h"], float)
        h_i = dual_core_query_h(
            query_i, np.asarray(node_candidate["centers_mm"], float),
            distance_cutoff_mm=manual_field_audit["distance_cutoff_mm"],
        )
        field_query_audit = {
            **manual_field_audit,
            "query_I_selected_count": int(np.sum(h_i)),
        }
    if field_transform is None:
        if not np.array_equal(h_e, node["h"]):
            raise RuntimeError("E/I field query changed the frozen E-node field")
        vtheta, delta_vtheta = node["vtheta"], node["delta_vtheta"]
    else:
        transformed = reconstruct_node_from_h(
            h_e, n_total=n_e + n_i, quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"], depth_shrinkage=depth_shrinkage,
            node_gain=gain)
        vtheta, delta_vtheta = transformed["vtheta"], transformed["delta_vtheta"]

    # ---- 8: fixed-topology EE geometry, then learned local mapper ----
    local = config["local_connectivity_basis"]
    net, ellipse_audit = fixed_topology_ee_ellipse_redistribution(
        net, np.asarray(net["pos"], float),
        length_scale=float(local["E_to_E_length_scale_mm"]),
        angle_deg=float(ee_ellipse_angle_deg),
        aspect_ratio=float(ee_ellipse_aspect_ratio),
    )
    pre_bins = list(net["ampa_by_delay"])
    pre_ee = _outgoing_by_pathway(pre_bins, n_e, "E_to_E")
    pre_etoi = _outgoing_by_pathway(pre_bins, n_e, "E_to_I")
    mapped_net, edge_audit = continuous_local_e_source_flow(
        net, np.asarray(net["pos"], float), np.concatenate([h_e, h_i]),
        coefficients_eff,
        l_ee=float(local["E_to_E_length_scale_mm"]),
        l_e_to_i=float(local["E_to_I_length_scale_mm"]),
        raw_logit_clip=candidate.get("raw_logit_clip"))
    post_ee = _outgoing_by_pathway(mapped_net["ampa_by_delay"], n_e, "E_to_E")
    post_etoi = _outgoing_by_pathway(mapped_net["ampa_by_delay"], n_e, "E_to_I")
    if not np.isclose(pre_ee.sum(), _outgoing_by_pathway(pre_bins, n_e, "E_to_E").sum()):
        raise RuntimeError("pre-mapping bins were mutated by the mapper")

    # ---- 9: frozen contact montage ----
    contacts = contract["contacts"]
    contact_names = [row["contact_name"] for row in contacts]
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    shaft_ids = np.asarray([row["shaft_id"] for row in contacts], dtype="U8")
    montage = VirtualMontage(contact_xy, contact_names,
                             provenance="rev10_r_observation_only_contact_contract")
    valid = cmrun.valid_mask(montage, positions, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("all frozen contacts must be locally readable")

    axis = np.asarray(reg["sink_centroid"], float) - np.asarray(reg["source_centroid"], float)
    return Substrate(
        params=params, net=mapped_net, n_e=int(n_e), n_i=int(n_i),
        positions_e=positions, positions_i=positions_i,
        h_e=np.asarray(h_e, float), h_i=np.asarray(h_i, float),
        vtheta=np.asarray(vtheta, float),
        delta_vtheta=np.asarray(delta_vtheta, float),
        montage=montage, contact_names=contact_names, contact_xy=contact_xy,
        shaft_ids=shaft_ids, valid_contacts=np.asarray(valid, bool),
        edge_audit=edge_audit, edge_coefficients=coefficients_eff,
        ee_out_gain=_gain(pre_ee, post_ee), etoi_out_gain=_gain(pre_etoi, post_etoi),
        axis_unit=axis / np.linalg.norm(axis),
        axis_source_xy=np.asarray(reg["source_centroid"], float),
        axis_sink_xy=np.asarray(reg["sink_centroid"], float),
        detector_threshold=detector, engine=engine, stage=stage,
        network_cache=_cache_record(cache_hit, cache_source),
        field_transform=field_transform,
        extras={"field_query_audit": field_query_audit, "cmrun": cmrun,
                "placement": reg, "candidate": candidate,
                "node_candidate": node_candidate,
                "node_candidate_override": node_candidate_override is not None,
                "node_dispersion_candidate_override": (
                    node_dispersion_candidate_override is not None
                ),
                "node_mapping_audit": node["mapping_audit"],
                "manual_field_audit": manual_field_audit,
                "ellipse_audit": ellipse_audit,
                "frozen_node_field_sha256": frozen_node_candidate["field_sha256"],
                "pathway_dose": {"E_to_E": float(ee_dose),
                                   "E_to_I": float(etoi_dose)},
                "contact_contract": contract, "manifest": manifest},
    )


def make_slow(substrate, zm_cfg, *, trace_weights_E=None):
    """Z/M slow protocol, or None when the arm runs with slow state off."""
    from src.topic4_zm_slow_vars import ZMTracedSlowVars as MZSlowVars
    from src.snn_engine.mz_slow_vars import MZSlowVarsConfig
    if zm_cfg.get("mode", "off") == "off":
        return None
    passive = bool(zm_cfg.get("passive", False))
    use_z = bool(zm_cfg.get("use_z"))
    use_m = bool(zm_cfg.get("use_m"))
    if not (use_z or use_m):
        raise RuntimeError("an active slow-variable arm must enable Z or M")
    slow = MZSlowVars(
        substrate.n_e + substrate.n_i, substrate.params.V_th,
        MZSlowVarsConfig(use_z=use_z, use_m=use_m,
                         I_th_EI=float(zm_cfg["I_th_EI"]),
                         tau_z=float(zm_cfg["tau_z"]),
                         tau_adp=float(zm_cfg["tau_adp"]),
                         eta_m=float(zm_cfg["eta_m"]),
                         trace_stride_steps=int(zm_cfg["trace_stride_steps"])),
        NE=substrate.n_e,
        core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
        trace_weights_E=trace_weights_E)
    if passive:
        slow.enable_passive_mode()
    return slow


def make_external_drive(substrate, ou_cfg, seed):
    from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive
    if ou_cfg.get("mode", "off") == "off":
        return None
    return SpatialOUDrive(
        substrate.positions_e, float(substrate.engine["L"]),
        float(substrate.engine["dt"]),
        SpatialOUConfig(mode=ou_cfg["mode"],
                        sigma_rate_per_ms=float(ou_cfg["sigma_rate_per_ms"]),
                        tau_ms=float(ou_cfg["tau_ms"]),
                        ell_mm=float(ou_cfg["ell_mm"]),
                        update_interval_ms=float(ou_cfg["update_interval_ms"]),
                        grid_spacing_mm=float(ou_cfg["grid_spacing_mm"]),
                        seed=int(seed) + int(ou_cfg["seed_offset"])))
