"""Run paired forced-source/sham capacity assays for one arm and network seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_rev9_node_kick_canary import (  # noqa: E402
    _candidate,
    _load_network,
)
from scripts.run_topic4_rev9_factorial_worker import _event_histogram  # noqa: E402
from src.topic4_component_pair_edge import (  # noqa: E402
    component_background_membership,
    component_pair_normalized_ee,
)
from src.topic4_core_connectivity import field_normalized_ee_pair  # noqa: E402
from src.topic4_core_field_profile import normalized_rank_curve  # noqa: E402
from src.topic4_core_field_rev9 import (  # noqa: E402
    assign_frozen_modes,
    component_contributions,
    reconstruct_frozen_node,
)
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import (  # noqa: E402
    exclude_injected_packet_frame,
    paired_excess_geometry,
    select_source_indices,
    select_triggered_event,
)
from src.topic4_rev9_factorial import arm_contract  # noqa: E402


DEFAULT_CONFIG = "config/topic4_rev9l_forced_source.json"
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _runtime_provenance(expected_commit=None):
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths = sorted(paths)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT, text=True).strip()
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_hashes = None
    modules_match_expected = None
    if expected_commit is not None:
        expected_commit = subprocess.check_output(
            ["git", "rev-parse", str(expected_commit)], cwd=ROOT,
            text=True).strip()
        expected_hashes = {}
        for path in paths:
            content = subprocess.check_output(
                ["git", "show", f"{expected_commit}:{path}"], cwd=ROOT)
            expected_hashes[path] = hashlib.sha256(content).hexdigest()
        modules_match_expected = all(
            expected_hashes[path] == _sha256(ROOT / path) for path in paths)
    return {
        "git_commit": current_commit,
        "expected_git_commit": expected_commit,
        "runtime_modules_match_expected_commit": modules_match_expected,
        "runtime_modules_dirty": bool(dirty),
        "runtime_module_sha256": {path: _sha256(ROOT / path) for path in paths},
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "systemd_unit": os.environ.get("REV9L_SYSTEMD_UNIT"),
    }


def _load_json_input(record):
    if _sha256(record["path"]) != record["sha256"]:
        raise RuntimeError(f"input hash changed: {record['path']}")
    return json.loads(Path(record["path"]).read_text())


def _load_reference(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in (
            "grid", "center", "components", "score_center", "score_scale",
            "reference_z", "directions",
        )}


def _event_diagnostics(spikes, cmrun, detect_events, *, trigger_ms, latency_ms):
    active, bin_width = cmrun.active_fraction(spikes, cmrun.DT, cmrun.BIN_MS)
    baseline_start = int(cmrun.BASELINE_MS[0] / bin_width)
    baseline_stop = min(len(active), int(cmrun.BASELINE_MS[1] / bin_width))
    floor = (float(np.min(active, initial=0.0)) if baseline_stop <= baseline_start
             else float(np.percentile(active[baseline_start:baseline_stop], 95)))
    peak = float(np.max(active, initial=0.0))
    threshold = floor + cmrun.CAL_FRAC * (peak - floor)
    events = detect_events(active, bin_width, event_on_frac=threshold)
    triggered = select_triggered_event(
        events, trigger_ms=trigger_ms, max_latency_ms=latency_ms)
    serialized = [{
        "t_on": float(event["t_on"]),
        "t_off": float(event["t_off"]),
        "returned": bool(event.get("returned", False)),
        "peak_ext": float(event.get("peak_ext", np.nan)),
    } for event in events]
    return {
        "active_fraction": np.asarray(active, np.float32),
        "bin_width_ms": float(bin_width),
        "floor": floor,
        "peak": peak,
        "threshold": float(threshold),
        "events": serialized,
        "triggered_event": None if triggered is None else {
            "t_on": float(triggered["t_on"]),
            "t_off": float(triggered["t_off"]),
            "returned": bool(triggered.get("returned", False)),
            "duration_ms": float(triggered["t_off"] - triggered["t_on"]),
        },
    }


def _source_records(config, requested):
    lookup = {source["id"]: source for source in config["packet"]["sources"]}
    unknown = sorted(set(requested) - set(lookup))
    if unknown:
        raise ValueError(f"unknown forced source(s): {unknown}")
    return [lookup[source] for source in requested]


def _pad_rows(rows, *, fill=np.nan, dtype=float):
    width = max((len(row) for row in rows), default=0)
    output = np.full((len(rows), width), fill, dtype=dtype)
    for index, row in enumerate(rows):
        output[index, :len(row)] = row
    return output


def _pad_envelopes(rows, n_contacts):
    width = max((row.shape[1] for row in rows), default=0)
    output = np.full((len(rows), int(n_contacts), width), np.nan, np.float32)
    for index, row in enumerate(rows):
        output[index, :, :row.shape[1]] = row
    return output


def resolve_dynamics_seed(network_seed, dynamics_seed=None):
    """Keep legacy parity while allowing independent noise repeats."""
    value = int(network_seed if dynamics_seed is None else dynamics_seed)
    if value < 0 or value >= 2 ** 63:
        raise ValueError("dynamics seed must lie in [0, 2**63)")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arm", required=True,
                        choices=("Null", "Node", "Edge", "Node+Edge"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dynamics-seed", type=int)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--packet-fractions", nargs="+", type=float)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    parser.add_argument("--expected-commit")
    parser.add_argument("--component-pair-gamma", nargs=6, type=float)
    parser.add_argument("--candidate-id")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    allowed_roles = {
        "forced_initiation_propagation_capacity_development_audit",
        "component_pair_edge_forced_oracle_development",
    }
    if config["scientific_role"] not in allowed_roles:
        raise RuntimeError("L1 scientific role changed")
    is_component_pair = (
        config["scientific_role"] == "component_pair_edge_forced_oracle_development")
    if is_component_pair and (args.component_pair_gamma is None or not args.candidate_id):
        parser.error("component-pair oracle requires gamma and candidate-id")
    if not is_component_pair and (args.component_pair_gamma is not None
                                  or args.candidate_id is not None):
        parser.error("component-pair arguments are not valid for the L1 role")
    if is_component_pair and args.arm != "Edge":
        parser.error("component-pair L2 isolates the Edge-only arm")
    allowed_seeds = set(sum(config["network_seeds"].values(), []))
    if args.seed not in allowed_seeds:
        parser.error("--seed is outside the rev9-L frozen seed sets")
    dynamics_seed = resolve_dynamics_seed(args.seed, args.dynamics_seed)
    sources = _source_records(
        config, args.sources or config["packet"]["formal_sources"])
    fractions = np.asarray(
        args.packet_fractions or config["packet"]["canary_fractions_of_E"], float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or np.any((fractions <= 0.0) | (fractions >= 1.0))):
        parser.error("packet fractions must be finite and lie in (0, 1)")

    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    selection = _load_json_input(inputs["selection"])
    frozen_summary = _load_json_input(inputs["frozen_readouts_json"])
    for key in ("rev9l_l0_config", "frozen_readouts_npz"):
        if _sha256(inputs[key]["path"]) != inputs[key]["sha256"]:
            raise RuntimeError(f"input hash changed: {inputs[key]['path']}")
    candidate = _candidate(base)
    if candidate["theta_sha256"] != selection["selected_theta_sha256"]:
        raise RuntimeError("rev9 base candidate differs from frozen selection")

    output_root = Path(config["output_root"])
    slug = args.arm.lower().replace("+", "_")
    output_json = Path(args.out_json or output_root / "workers" /
                       f"{slug}_seed{args.seed}.json")
    output_npz = Path(args.out_npz or output_root / "workers" /
                      f"{slug}_seed{args.seed}.npz")
    cache_dir = str(Path(args.cache_dir or
                         "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"))
    output_json.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    engine = stage["engine"]
    simulation = config["simulation"]
    reg = _placement(stage)
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]).detect_events
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
    ).snn_event_envelope
    execution_provenance = _runtime_provenance(args.expected_commit)
    if execution_provenance["runtime_modules_dirty"]:
        raise RuntimeError("runtime modules are dirty")
    if (args.expected_commit is not None
            and not execution_provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("runtime modules differ from the launcher commit")

    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=simulation["duration_ms"], dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed))
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, reg, int(args.seed), base, cache_dir)
    positions = np.asarray(net["pos"][:n_e], float)
    node = reconstruct_frozen_node(
        candidate["theta"], positions, n_total=n_e + n_i,
        target_count=stage["N_core_manual"],
        quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"],
        v_base=engine["v_base"], K=candidate["K"], L=engine["L"])
    contributions = component_contributions(
        candidate["theta"], positions, K=candidate["K"], L=engine["L"])
    membership = component_background_membership(node["h"], contributions)
    switches = arm_contract(args.arm)
    edge_diagnostics = None
    edge_family = "none"
    if switches["edge"]:
        if is_component_pair:
            net, edge_diagnostics = component_pair_normalized_ee(
                net, node["h"], membership, args.component_pair_gamma,
                alpha=float(config["alpha_star"]))
            edge_family = "component_pair_residual_target_normalized"
        else:
            net, edge_diagnostics = field_normalized_ee_pair(
                net, node["h"], config["alpha_star"], beta=0.0,
                active_vth_shift=node["delta_vtheta"])
            edge_family = "scalar_field_assortative"
    vtheta = (node["vtheta"] if switches["node"] else
              np.full(n_e + n_i, float(engine["v_base"])))

    with np.load(inputs["frozen_readouts_npz"]["path"], allow_pickle=False) as frozen:
        classifier = {
            "embedding_centroids": np.asarray(
                frozen["classifier_embedding_centroids"], float),
            "ood_distance_thresholds": np.asarray(
                frozen["classifier_ood_thresholds"], float),
        }
    reference_path = frozen_summary["inputs"]["reference"]["path"]
    if _sha256(reference_path) != frozen_summary["inputs"]["reference"]["sha256"]:
        raise RuntimeError("frozen profile reference hash changed")
    reference = _load_reference(reference_path)
    axial = __import__(
        "scripts.run_topic4_core_field_stage3_profile_round1",
        fromlist=["axial_map"]).axial_map()
    grid = np.asarray(reference["grid"], float)
    contact_names = sorted(axial, key=axial.get)
    montage = reg["montage_sheet"]
    valid_contacts = cmrun.valid_mask(
        montage, positions, engine["L"], params.Rr)

    net["rng"] = np.random.default_rng(dynamics_seed)
    sham = simulate_kick(
        params, net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=vtheta,
        early_stop_runaway=simulation["early_stop_runaway"])
    sham_spikes = np.asarray(sham["E_spk_bool"], bool)
    sham_envelope, envelope_dt, _ = snn_event_envelope(
        sham_spikes, positions, montage, engine["dt"])
    sham_events = _event_diagnostics(
        sham_spikes, cmrun, detect_events,
        trigger_ms=simulation["forced_spike_ms"],
        latency_ms=simulation["trigger_max_latency_ms"])

    run_rows = []
    rank_curves, rank_rows, packet_masks = [], [], []
    inclusive_rank_curves, inclusive_rank_rows = [], []
    positive_counts, signed_counts, excess_envelopes = [], [], []
    inclusive_excess_envelopes = []
    active_forced = []
    assigned_distances, assigned_labels, assigned_ood = [], [], []
    for source in sources:
        for fraction in fractions:
            packet_n = max(1, int(round(float(fraction) * n_e)))
            indices = select_source_indices(
                positions, source, n_cells=packet_n,
                component_contribution=contributions)
            packet_e = np.zeros(n_e, bool)
            packet_e[indices] = True
            packet_all = np.zeros(n_e + n_i, bool)
            packet_all[:n_e] = packet_e
            source_center = positions[indices].mean(axis=0)

            net["rng"] = np.random.default_rng(dynamics_seed)
            forced = simulate_kick(
                params, net, KICK_BOOST=0.0, t_kick=1e9,
                V_th_per_neuron=vtheta, forced_spike_mask=packet_all,
                forced_spike_ms=simulation["forced_spike_ms"],
                early_stop_runaway=simulation["early_stop_runaway"])
            forced_spikes = np.asarray(forced["E_spk_bool"], bool)
            trigger_step = int(round(simulation["forced_spike_ms"] / engine["dt"]))
            pretrigger_identical = bool(np.array_equal(
                forced_spikes[:trigger_step], sham_spikes[:trigger_step]))
            forced_envelope, forced_envelope_dt, _ = snn_event_envelope(
                forced_spikes, positions, montage, engine["dt"])
            if forced_envelope_dt != envelope_dt:
                raise RuntimeError("forced/sham envelope frame dt changed")
            inclusive_n_frame = min(
                forced_envelope.shape[1], sham_envelope.shape[1])
            inclusive_excess_envelope = np.clip(
                forced_envelope[:, :inclusive_n_frame]
                - sham_envelope[:, :inclusive_n_frame],
                0.0, None)

            exclude_packet_frame = bool(config["readout"].get(
                "exclude_injected_frame_from_primary", False))
            if exclude_packet_frame:
                response_spikes = exclude_injected_packet_frame(
                    forced_spikes, sham_spikes, packet_e,
                    trigger_step=trigger_step)
                response_envelope, response_dt, _ = snn_event_envelope(
                    response_spikes, positions, montage, engine["dt"])
                if response_dt != envelope_dt:
                    raise RuntimeError("packet-excluded envelope frame dt changed")
            else:
                response_envelope = forced_envelope
            n_frame = min(response_envelope.shape[1], sham_envelope.shape[1])
            excess_envelope = np.clip(
                response_envelope[:, :n_frame] - sham_envelope[:, :n_frame],
                0.0, None)

            primary_window = config["readout"]["primary_window_ms"]
            response_available = n_frame * envelope_dt >= float(primary_window[1])
            readout = (cmrun.read_event(
                excess_envelope, envelope_dt, montage, valid_contacts,
                tuple(primary_window), reg["axis_unit_vec"],
                k_dir=int(engine["k_dir"]),
                part_min=2 * int(engine["k_dir"]) + 1)
                if response_available else {"n_part": 0, "ranks": None})
            rank_dict = readout.get("ranks") or {}
            rank_row = np.asarray([
                np.nan if rank_dict.get(name) is None else float(rank_dict[name])
                for name in contact_names], float)
            curve = normalized_rank_curve(rank_dict, axial, grid=grid)
            curve_row = (np.full(len(grid), np.nan) if curve is None
                         else np.asarray(curve, float))
            if curve is None:
                assigned = {"labels": np.asarray([-1]),
                            "distance_matrix": np.full((1, 2), np.nan),
                            "ood": np.asarray([True])}
            else:
                assigned = assign_frozen_modes(
                    np.asarray(curve)[None, :], classifier, reference)

            inclusive_response_available = (
                inclusive_n_frame * envelope_dt >= float(primary_window[1]))
            inclusive_readout = (cmrun.read_event(
                inclusive_excess_envelope, envelope_dt, montage, valid_contacts,
                tuple(primary_window), reg["axis_unit_vec"],
                k_dir=int(engine["k_dir"]),
                part_min=2 * int(engine["k_dir"]) + 1)
                if inclusive_response_available else {"n_part": 0, "ranks": None})
            inclusive_rank_dict = inclusive_readout.get("ranks") or {}
            inclusive_rank_row = np.asarray([
                np.nan if inclusive_rank_dict.get(name) is None
                else float(inclusive_rank_dict[name])
                for name in contact_names], float)
            inclusive_curve = normalized_rank_curve(
                inclusive_rank_dict, axial, grid=grid)
            inclusive_curve_row = (
                np.full(len(grid), np.nan) if inclusive_curve is None
                else np.asarray(inclusive_curve, float))

            geometry_end = min(
                float(simulation["paired_response_end_ms"]),
                len(forced_spikes) * engine["dt"],
                len(sham_spikes) * engine["dt"])
            geometry = paired_excess_geometry(
                forced_spikes, sham_spikes, positions, packet_e,
                dt_ms=engine["dt"], start_ms=simulation["forced_spike_ms"],
                end_ms=geometry_end, source_center=source_center)
            forced_events = _event_diagnostics(
                forced_spikes, cmrun, detect_events,
                trigger_ms=simulation["forced_spike_ms"],
                latency_ms=simulation["trigger_max_latency_ms"])
            respike_start = trigger_step + 1
            respike_stop = min(
                len(forced_spikes), len(sham_spikes),
                int(round(geometry_end / engine["dt"])))
            source_respike_signed = (
                forced_spikes[respike_start:respike_stop, packet_e].sum()
                - sham_spikes[respike_start:respike_stop, packet_e].sum())
            source_radius = np.linalg.norm(
                positions[indices] - source_center[None, :], axis=1)
            selected_contribution = (
                contributions[indices, int(source["component_1based"]) - 1]
                if source["kind"] == "component" else None)
            row = {
                "source_id": source["id"],
                "source_kind": source["kind"],
                "source_config_xy_mm": source["xy_mm"],
                "source_actual_center_xy_mm": source_center.tolist(),
                "packet_fraction_of_E": float(fraction),
                "packet_n_E": int(packet_n),
                "packet_indices_sha256": hashlib.sha256(
                    np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                "packet_radius_p50_mm": float(np.quantile(source_radius, 0.5)),
                "packet_radius_p95_mm": float(np.quantile(source_radius, 0.95)),
                "component_contribution_min": (
                    None if selected_contribution is None
                    else float(np.min(selected_contribution))),
                "component_contribution_max": (
                    None if selected_contribution is None
                    else float(np.max(selected_contribution))),
                "pretrigger_spikes_bit_identical": pretrigger_identical,
                "forced_spike_requested_count": int(
                    forced["forced_spike_requested_count"]),
                "forced_spike_collision_count": int(
                    forced["forced_spike_collision_count"]),
                "paired_excess_readout": {
                    "window_ms": primary_window,
                    "injected_source_frame_excluded": exclude_packet_frame,
                    "response_available": bool(response_available),
                    "n_part": int(readout.get("n_part", 0)),
                    "curve_usable": bool(curve is not None),
                    "assigned_mode": (None if curve is None else
                                      int(assigned["labels"][0])),
                    "assigned_distance_to_A_B": (
                        [None, None] if curve is None else
                        assigned["distance_matrix"][0].astype(float).tolist()),
                    "ood": bool(assigned["ood"][0]),
                },
                "inclusive_packet_frame_sensitivity": {
                    "response_available": bool(inclusive_response_available),
                    "n_part": int(inclusive_readout.get("n_part", 0)),
                    "curve_usable": bool(inclusive_curve is not None),
                },
                "paired_geometry": {
                    key: value for key, value in geometry.items()
                    if key not in {"signed_spike_count_per_E",
                                   "positive_spike_count_per_E"}
                },
                "source_respike_signed_mass_after_injected_frame": float(
                    source_respike_signed),
                "forced_triggered_event": forced_events["triggered_event"],
                "sham_triggered_event": sham_events["triggered_event"],
                "forced_n_events": len(forced_events["events"]),
                "sham_n_events": len(sham_events["events"]),
                "runaway_early_stop_ms": forced["runaway_early_stop_ms"],
                "simulated_until_ms": float(len(forced_spikes) * engine["dt"]),
                "wall_seconds": float(forced["wall_s"]),
            }
            run_rows.append(row)
            rank_curves.append(curve_row)
            rank_rows.append(rank_row)
            inclusive_rank_curves.append(inclusive_curve_row)
            inclusive_rank_rows.append(inclusive_rank_row)
            packet_masks.append(packet_e)
            positive_counts.append(geometry["positive_spike_count_per_E"])
            signed_counts.append(geometry["signed_spike_count_per_E"])
            excess_envelopes.append(excess_envelope.astype(np.float32))
            inclusive_excess_envelopes.append(
                inclusive_excess_envelope.astype(np.float32))
            active_forced.append(forced_events["active_fraction"])
            assigned_distances.append(assigned["distance_matrix"][0])
            assigned_labels.append(int(assigned["labels"][0]))
            assigned_ood.append(bool(assigned["ood"][0]))
            print(json.dumps({
                "progress": "forced_pair_complete", "arm": args.arm,
                "seed": args.seed, "source": source["id"],
                "packet_fraction": float(fraction),
                "curve_usable": bool(curve is not None),
                "n_part": int(readout.get("n_part", 0)),
                "downstream_positive_spike_mass": geometry[
                    "downstream_positive_spike_mass"],
                "runaway_early_stop_ms": forced["runaway_early_stop_ms"],
                "wall_seconds": float(forced["wall_s"]),
            }), flush=True)

    _atomic_npz(
        output_npz,
        source_ids=np.asarray([row["source_id"] for row in run_rows], dtype="U32"),
        packet_fraction_of_E=np.asarray(
            [row["packet_fraction_of_E"] for row in run_rows], float),
        packet_n_E=np.asarray([row["packet_n_E"] for row in run_rows], np.int64),
        packet_masks_E=np.asarray(packet_masks, bool),
        rank_curves=np.asarray(rank_curves, np.float32),
        contact_ranks=np.asarray(rank_rows, np.float32),
        inclusive_packet_frame_rank_curves=np.asarray(
            inclusive_rank_curves, np.float32),
        inclusive_packet_frame_contact_ranks=np.asarray(
            inclusive_rank_rows, np.float32),
        contact_names=np.asarray(contact_names, dtype="U32"),
        paired_positive_spike_count_E=np.asarray(positive_counts, np.float32),
        paired_signed_spike_count_E=np.asarray(signed_counts, np.float32),
        excess_contact_envelope=_pad_envelopes(
            excess_envelopes, len(contact_names)),
        inclusive_packet_frame_excess_contact_envelope=_pad_envelopes(
            inclusive_excess_envelopes, len(contact_names)),
        envelope_dt_ms=np.asarray(envelope_dt, float),
        forced_active_fraction=_pad_rows(active_forced, dtype=np.float32),
        sham_active_fraction=np.asarray(sham_events["active_fraction"], np.float32),
        active_fraction_bin_ms=np.asarray(sham_events["bin_width_ms"], float),
        assigned_distance_to_A_B=np.asarray(assigned_distances, np.float32),
        assigned_mode=np.asarray(assigned_labels, np.int8),
        assigned_ood=np.asarray(assigned_ood, bool),
        positions_E=np.asarray(positions, np.float32),
        h=np.asarray(node["h"], np.float32),
        delta_vtheta=np.asarray(
            node["delta_vtheta"] if switches["node"] else np.zeros(n_e),
            np.float32),
    )
    payload = {
        "status": "REV9L_FORCED_SOURCE_WORKER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "arm": args.arm,
        "edge_family": edge_family,
        "candidate_id": args.candidate_id,
        "component_pair_gamma": args.component_pair_gamma,
        "switches": switches,
        "seed": int(args.seed),
        "network_seed": int(args.seed),
        "dynamics_seed": dynamics_seed,
        "sources": [source["id"] for source in sources],
        "packet_fractions_of_E": fractions.tolist(),
        "simulation": simulation,
        "sham": {
            "n_events": len(sham_events["events"]),
            "triggered_event": sham_events["triggered_event"],
            "runaway_early_stop_ms": sham["runaway_early_stop_ms"],
            "simulated_until_ms": float(len(sham_spikes) * engine["dt"]),
            "wall_seconds": float(sham["wall_s"]),
        },
        "runs": run_rows,
        "network": {
            "cache_hit": bool(cache_hit),
            "cache_source": cache_source,
            "n_E": int(n_e), "n_I": int(n_i),
            "node_hashes": node["hashes"],
            "edge_diagnostics": edge_diagnostics,
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "inputs": {
            key: {"path": record["path"], "sha256": record["sha256"]}
            for key, record in inputs.items()
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "elapsed_seconds": float(time.time() - started),
        "provenance": execution_provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"], "arm": args.arm,
        "network_seed": args.seed, "dynamics_seed": dynamics_seed,
        "n_pairs": len(run_rows),
        "n_curve_usable": sum(
            row["paired_excess_readout"]["curve_usable"] for row in run_rows),
        "n_runaway": sum(row["runaway_early_stop_ms"] is not None for row in run_rows),
        "elapsed_seconds": payload["elapsed_seconds"],
        "arrays_sha256": payload["arrays"]["sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
