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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), os.path.join(str(ROOT), "src", "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PATHWAYS = ("E_to_E", "E_to_I")


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


def verify_frozen_inputs(config):
    """Hash every declared input. Raises rather than warning -- a drifted input
    means the substrate is not the frozen one and every number downstream is
    about a different model."""
    records = {}
    for key, record in config["inputs"].items():
        path = ROOT / record["path"]
        if not path.exists():
            raise RuntimeError(f"input missing: {record['path']}")
        digest = _sha256_file(path)
        records[key] = {"path": record["path"], "expected": record["sha256"],
                        "observed": digest, "match": digest == record["sha256"]}
        if digest != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    return {"all_match": True, "records": records}


def _load_json_input(record):
    import json
    return json.loads((ROOT / record["path"]).read_text())


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


def build_substrate(config, candidate_id, seed, *, cache_dir, field_transform=None):
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
    from src.topic4_core_field_runner import _placement
    from src.topic4_graph_edge_flow import array_sha256
    from src.topic4_local_connectivity import continuous_local_e_source_flow

    verify_frozen_inputs(config)
    inputs = config["inputs"]
    manifest = _load_json_input(inputs["frozen_substrate_manifest"])
    if manifest["status"] != "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_LIBRARY_FROZEN":
        raise RuntimeError("frozen substrate manifest status changed")
    matches = [row for row in manifest["candidate_set"]["candidates"]
               if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"candidate {candidate_id!r} is outside the frozen library")
    candidate = matches[0]

    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    contract = _load_json_input(inputs["contact_contract"])
    anchor_config = _load_json_input(inputs["node_anchor_config"])
    detector_audit = _load_json_input(inputs["common_detector_audit"])
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
    reg = _placement(stage)
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, reg, int(seed), base, str(cache_dir))
    positions = np.asarray(net["pos"][:n_e], float)
    positions_i = np.asarray(net["pos"][n_e:], float)

    # ---- 4-5: node field and coefficient hash ----
    node_candidate = candidate["node_field"]
    node = _candidate_node(node_candidate, positions, n_total=n_e + n_i,
                           stage=stage, config=anchor_config)
    if not np.isclose(node["h"].sum(), float(stage["N_core_manual"]), atol=1e-8):
        raise RuntimeError("Node anchor field budget changed")
    coefficients = np.asarray(candidate["coefficients"], float)
    if array_sha256(coefficients) != candidate["coefficients_sha256"]:
        raise RuntimeError("edge coefficient hash changed")

    # ---- 6: the producer re-seeds here, before the edge mapper ----
    net["rng"] = np.random.default_rng(int(seed))

    # ---- 7: E/I field query, optionally through the spatial transform ----
    if node_candidate["field_type"] != "spline_continuous":
        raise RuntimeError("rev11-NLC requires the frozen continuous spline Node field")
    query_e, query_i = positions, positions_i
    coefficients_eff = coefficients
    if field_transform is not None:
        from src.topic4_zm_d4 import (inverse_query_positions,
                                      transform_flow_coefficients)
        query_e = inverse_query_positions(positions, field_transform, L=engine["L"])
        query_i = inverse_query_positions(positions_i, field_transform, L=engine["L"])
        coefficients_eff = transform_flow_coefficients(coefficients, field_transform)
    h_e, h_i, field_query_audit = continuous_field_h_with_queries(
        node_candidate["coefficients"], query_e, query_i,
        n_basis=node_candidate["n_basis"], degree=node_candidate["degree"],
        target_count=stage["N_core_manual"], L=engine["L"])
    if field_transform is None:
        if not np.array_equal(h_e, node["h"]):
            raise RuntimeError("E/I field query changed the frozen E-node field")
        vtheta, delta_vtheta = node["vtheta"], node["delta_vtheta"]
    else:
        from scripts.run_topic4_rev10_sa_spectral_field_worker import (
            reconstruct_node_from_h)
        transformed = reconstruct_node_from_h(
            h_e, n_total=n_e + n_i, quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"])
        vtheta, delta_vtheta = transformed["vtheta"], transformed["delta_vtheta"]

    # ---- 8: local connectivity mapper (pre-mapping bins captured first) ----
    pre_bins = list(net["ampa_by_delay"])
    pre_ee = _outgoing_by_pathway(pre_bins, n_e, "E_to_E")
    pre_etoi = _outgoing_by_pathway(pre_bins, n_e, "E_to_I")
    local = config["local_connectivity_basis"]
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
                "contact_contract": contract, "manifest": manifest},
    )


def make_slow(substrate, zm_cfg, *, trace_weights_E=None):
    """Z/M slow protocol, or None when the arm runs with slow state off."""
    from src.topic4_zm_slow_vars import ZMTracedSlowVars as MZSlowVars
    from src.snn_engine.mz_slow_vars import MZSlowVarsConfig
    if zm_cfg.get("mode", "off") == "off":
        return None
    if not (zm_cfg.get("use_z") and zm_cfg.get("use_m")):
        raise RuntimeError("the active Z/M arm must use Z and M together")
    return MZSlowVars(
        substrate.n_e + substrate.n_i, substrate.params.V_th,
        MZSlowVarsConfig(use_z=True, use_m=True,
                         I_th_EI=float(zm_cfg["I_th_EI"]),
                         tau_z=float(zm_cfg["tau_z"]),
                         tau_adp=float(zm_cfg["tau_adp"]),
                         eta_m=float(zm_cfg["eta_m"]),
                         trace_stride_steps=int(zm_cfg["trace_stride_steps"])),
        NE=substrate.n_e,
        core_mask_E=np.asarray(substrate.h_e >= 0.5, bool),
        trace_weights_E=trace_weights_E)


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
