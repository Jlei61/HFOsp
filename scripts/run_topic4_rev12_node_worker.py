#!/usr/bin/env python3
"""Run one rev12-ND continuous Node field on one frozen SNN network."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz, _runtime_provenance,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    annotate_population_excursions_with_lineages,
    assign_detector_fragments_to_directed_lineages,
    assign_detector_fragments_to_cascades,
    bin_neuron_spikes,
    binned_contact_envelope,
    cascade_event_windows,
    directed_lineage_onset_maps,
    directed_spatiotemporal_lineages,
    event_source_onset_maps,
    excitatory_psp_tail_support_ms,
    lineage_restricted_contact_readout,
    lineage_restricted_neuron_contact_readout,
    local_ee_delay_quantile_ms,
    merge_detected_event_fragments,
    population_excursion_episodes,
    neuron_contact_sampling_weights,
    root_coactivity_event_windows,
    sheet_bin_indices,
    sheet_contact_sampling_weights,
    sheet_activity_movie,
    spatiotemporal_cascade_labels,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, make_external_drive,
)
from kick_probe import simulate_kick  # noqa: E402
from src.sef_hfo_events import RETURN_FRAC, detect_events  # noqa: E402
from src.sef_hfo_snn_adapter import snn_event_envelope  # noqa: E402


ALLOWED_SCIENTIFIC_ROLES = {
    "development_only_node_dualmode_refit",
    "development_only_event_identity_canary",
    "development_only_engine_derived_event_identity_canary",
}


def _validate_scientific_role(role: str) -> None:
    if str(role) not in ALLOWED_SCIENTIFIC_ROLES:
        raise RuntimeError("rev12-ND scientific role changed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else root / relative


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _config_at_commit(config_path: Path, commit: str) -> str | None:
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{config_path.relative_to(ROOT)}"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, ValueError):
        return None
    return hashlib.sha256(content).hexdigest()


def _event_peak_active_fraction(event: dict, active: np.ndarray,
                                active_dt_ms: float) -> float:
    """Read the peak population activity inside the frozen event window."""
    values = np.asarray(active, float)
    active_dt_ms = float(active_dt_ms)
    if values.ndim != 1 or active_dt_ms <= 0.0:
        raise ValueError("active fraction must be one-dimensional with positive dt")
    start = max(0, int(np.floor(float(event["t_on"]) / active_dt_ms)))
    stop = min(len(values), int(np.ceil(float(event["t_off"]) / active_dt_ms)))
    if stop <= start:
        raise ValueError("event window contains no population-activity sample")
    return float(np.max(values[start:stop]))


def _directed_parent_contract(event_unit: dict, *, params, net: dict,
                              engine: dict, frame_ms: float,
                              local_delay_ms: float | None = None) -> dict:
    """Resolve root memory from frozen model timescales, never contact data."""
    name = str(event_unit["name"])
    frame_ms = float(frame_ms)
    if frame_ms <= 0.0:
        raise ValueError("lineage frame duration must be positive")
    neighborhood = int(event_unit.get("forward_parent_neighborhood_bins", 1))
    if neighborhood < 0:
        raise ValueError("lineage parent neighborhood cannot be negative")
    if name in {
            "persistent_directed_spatiotemporal_lineage",
            "causal_population_excursion",
            "persistent_root_coactivity_episode"}:
        method = str(event_unit.get(
            "causal_memory_method", "global_fast_state_decay",
        ))
        if method == "global_fast_state_decay":
            multiple = float(event_unit["fast_state_decay_multiples"])
            if multiple <= 0.0:
                raise ValueError("fast-state decay multiple must be positive")
            fast_tau_ms = max(float(params.tau_m_E), float(params.tau_d_GABA))
            delay_ms = int(net["max_delay_steps"]) * float(engine["dt"])
            response_support_ms = multiple * fast_tau_ms
            sensitivity_parameter = "fast_state_decay_multiples"
            sensitivity_value = multiple
        elif method == "local_ee_psp_tail":
            multiple = None
            fast_tau_ms = None
            tail_fraction = float(event_unit["psp_tail_fraction"])
            delay_quantile = float(event_unit["local_ee_delay_quantile"])
            delay_ms = (
                float(local_delay_ms) if local_delay_ms is not None
                else local_ee_delay_quantile_ms(
                    net["ampa_by_delay"], np.asarray(net["pos"][:net["NE"]], float),
                    dt_ms=float(engine["dt"]),
                    bin_mm=float(event_unit["movie_bin_mm"]),
                    neighborhood_bins=neighborhood,
                    quantile=delay_quantile,
                )
            )
            response_support_ms = excitatory_psp_tail_support_ms(
                tau_r_ms=float(params.tau_r_AMPA),
                tau_d_ms=float(params.tau_d_AMPA),
                tau_m_ms=float(params.tau_m_E), dt_ms=float(engine["dt"]),
                tail_fraction=tail_fraction,
            )
            sensitivity_parameter = "psp_tail_fraction"
            sensitivity_value = tail_fraction
        else:
            raise ValueError(f"unknown causal-memory method: {method}")
        memory_ms = response_support_ms + delay_ms
        gap_frames = max(1, int(np.ceil(memory_ms / frame_ms)))
    elif name == "directed_spatiotemporal_lineage":
        method = "adjacent_movie_frame"
        multiple = None
        fast_tau_ms = None
        delay_ms = None
        response_support_ms = None
        memory_ms = None
        sensitivity_parameter = None
        sensitivity_value = None
        gap_frames = int(event_unit.get("forward_parent_frame_gap", 1))
    else:
        raise ValueError(f"unsupported directed event unit: {name}")
    if gap_frames <= 0:
        raise ValueError("lineage parent gap must be positive")
    return {
        "fast_state_decay_multiples": multiple,
        "fast_state_tau_ms": fast_tau_ms,
        "causal_memory_method": method,
        "sensitivity_parameter": sensitivity_parameter,
        "sensitivity_value": sensitivity_value,
        "response_support_ms": response_support_ms,
        "local_or_global_delay_support_ms": delay_ms,
        "maximum_delay_ms": delay_ms,
        "causal_memory_ms": memory_ms,
        "forward_parent_frame_gap": gap_frames,
        "forward_parent_neighborhood_bins": neighborhood,
    }


def _event_contact_readout(*, events: list[dict], envelope: np.ndarray,
                           envelope_dt_ms: float, montage, valid_contacts: np.ndarray,
                           positions_e: np.ndarray, movie: dict | None,
                           lineage_labels: np.ndarray | None,
                           spikes: np.ndarray | None = None,
                           spike_dt_ms: float | None = None,
                           readout: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the configured contact readout after event roots are frozen."""
    source = str(readout.get(
        "source", "full_contact_envelope_within_event_window",
    ))
    n_contacts = len(montage.names)
    audit = {"source": source}
    if source == "lineage_restricted_neuron_activity":
        if (movie is None or lineage_labels is None or spikes is None
                or spike_dt_ms is None):
            raise RuntimeError("exact lineage readout requires spikes and a directed movie")
        neuron_bins, _ = sheet_bin_indices(
            np.asarray(positions_e, float),
            bin_mm=float(movie["bin_mm"]), sheet_mm=float(movie["sheet_mm"]),
        )
        weights = neuron_contact_sampling_weights(
            np.asarray(positions_e, float), np.asarray(montage.contacts, float),
            kernel_width_mm=float(readout["kernel_width_mm"]),
        )
        binned_spikes = bin_neuron_spikes(
            np.asarray(spikes, bool), dt_ms=float(spike_dt_ms),
            frame_ms=float(movie["frame_ms"]),
        )
        restricted = lineage_restricted_neuron_contact_readout(
            binned_spikes, neuron_bins, np.asarray(lineage_labels), events, weights,
            frame_ms=float(movie["frame_ms"]),
            smooth_ms=float(readout["smooth_ms"]),
            participation_margin_fraction=float(
                readout["participation_margin_fraction"]
            ),
            timing_fraction=float(readout["timing_fraction"]),
        )
        audit.update({
            "kernel_width_mm": float(readout["kernel_width_mm"]),
            "smooth_ms": float(readout["smooth_ms"]),
            "spatial_sampler": "exact_normalized_per_neuron_gaussian",
            "root_assignment": "movie_lineage_label_at_each_neuron_bin_and_frame",
            "parity_status": "EXACT_SHARED_PER_NEURON_KERNEL",
        })
        return restricted["onsets"], restricted["ranks"], audit

    if source == "lineage_restricted_sheet_activity":
        if movie is None or lineage_labels is None:
            raise RuntimeError("lineage-restricted readout requires a directed movie")
        neuron_bins, sheet_size = sheet_bin_indices(
            np.asarray(positions_e, float),
            bin_mm=float(movie["bin_mm"]), sheet_mm=float(movie["sheet_mm"]),
        )
        population = np.bincount(
            neuron_bins, minlength=sheet_size * sheet_size,
        ).reshape(sheet_size, sheet_size)
        weights = sheet_contact_sampling_weights(
            np.asarray(montage.contacts, float), population,
            bin_mm=float(movie["bin_mm"]),
            kernel_width_mm=float(readout["kernel_width_mm"]),
        )
        proxy = binned_contact_envelope(
            np.asarray(movie["activity_counts"]), weights,
            frame_ms=float(movie["frame_ms"]),
            smooth_ms=float(readout["smooth_ms"]),
        )
        frozen = np.asarray(envelope, float)
        if proxy.shape != frozen.shape:
            raise RuntimeError(
                "binned and frozen contact envelopes have different shapes"
            )
        correlations = np.asarray([
            np.corrcoef(proxy[index], frozen[index])[0, 1]
            for index in range(n_contacts)
        ], float)
        minimum = float(readout["minimum_full_trace_pearson"])
        if (not np.all(np.isfinite(correlations))
                or float(np.min(correlations)) < minimum):
            raise RuntimeError("binned contact sampler fails full-trace parity")
        restricted = lineage_restricted_contact_readout(
            np.asarray(movie["activity_counts"]), np.asarray(lineage_labels),
            events, weights, frame_ms=float(movie["frame_ms"]),
            smooth_ms=float(readout["smooth_ms"]),
            participation_margin_fraction=float(
                readout["participation_margin_fraction"]
            ),
            timing_fraction=float(readout["timing_fraction"]),
        )
        audit.update({
            "kernel_width_mm": float(readout["kernel_width_mm"]),
            "smooth_ms": float(readout["smooth_ms"]),
            "minimum_full_trace_pearson": minimum,
            "full_trace_pearson_per_contact": correlations,
            "full_trace_pearson_median": float(np.median(correlations)),
            "full_trace_pearson_minimum": float(np.min(correlations)),
            "parity_status": "PASS",
        })
        return restricted["onsets"], restricted["ranks"], audit

    if source not in {
        "full_contact_envelope_within_event_window",
        "full_contact_envelope_within_lineage_window",
    }:
        raise RuntimeError(f"unknown contact readout source: {source}")
    onset_rows, rank_rows = [], []
    for event in events:
        onset, rank = _contact_onsets(
            envelope, envelope_dt_ms, montage, valid_contacts,
            (event["t_on"], event["t_off"]),
            readout["participation_margin_fraction"],
            readout["timing_fraction"],
        )
        onset_rows.append(onset)
        rank_rows.append(rank)
    return (
        np.asarray(onset_rows, float).reshape((-1, n_contacts)),
        np.asarray(rank_rows, float).reshape((-1, n_contacts)),
        audit,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    _validate_scientific_role(config["scientific_role"])
    active_seeds = {
        int(seed) for key in (
            "canary_network_seeds", "fit_network_seeds",
            "selection_network_seeds", "confirmation_network_seeds",
        ) for seed in config["search"].get(key, [])
    }
    if args.seed not in active_seeds:
        parser.error("seed is outside every frozen rev12-ND pool")

    artifact_root = args.artifact_root.resolve()
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev12-ND candidate manifest is stale")
    matches = [
        row for row in manifest["candidates"]
        if row["candidate_id"] == args.candidate_id
    ]
    if len(matches) != 1:
        parser.error("candidate is outside the frozen rev12-ND manifest")
    candidate = matches[0]

    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(expected_commit)
    provenance.update({
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "config_sha256_at_expected_commit": _config_at_commit(
            config_path, expected_commit,
        ),
        "systemd_unit": os.environ.get("REV12ND_SYSTEMD_UNIT"),
    })
    if (provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]
            or provenance["config_sha256"] != provenance["config_sha256_at_expected_commit"]):
        raise RuntimeError("rev12-ND runtime modules or config are not frozen")

    output_root = artifact_root / config["output_root"]
    stem = f"{args.candidate_id}_seed_{args.seed}"
    out_json = args.out_json or output_root / "workers" / f"{stem}.json"
    out_npz = args.out_npz or output_root / "workers" / f"{stem}.npz"
    cache_dir = artifact_root / config["network_cache"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    transition_path = _resolve(
        artifact_root, config["inputs"]["transition_config"]["path"],
    )
    transition = load_round_config(transition_path)
    substrate = build_substrate(
        transition, "node_baseline", args.seed, cache_dir=str(cache_dir),
        ee_dose=0.0, etoi_dose=0.0,
        node_candidate_override=candidate["node_field"],
        artifact_root=artifact_root,
    )
    simulation = config["search"]["simulation"]
    substrate.params.T = float(simulation["duration_ms"])
    substrate.net["rng"] = np.random.default_rng(args.seed)

    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
        external_e_rate_drive=make_external_drive(
            substrate, transition["spatial_ou"], args.seed,
        ),
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(
        spikes, float(substrate.engine["dt"]), cmrun.BIN_MS,
    )
    detector_fragments = detect_events(
        active, active_dt, event_on_frac=substrate.detector_threshold,
    )
    event_unit = config.get("event_unit")
    movie = None
    movie_arrays = {}
    cascade_arrays = {}
    compound_fragments = []
    directed_source = None
    lineage_labels = None
    lineage_sensitivity = []
    directed_assignment_by_fragment = {}
    if event_unit is None:
        detected = detector_fragments
        event_unit_runtime = {"name": "detector_fragment"}
    elif event_unit["name"] == "settled_episode":
        detected = merge_detected_event_fragments(
            detector_fragments,
            maximum_gap_ms=float(event_unit["episode_merge_gap_ms"]),
            sample_dt_ms=float(active_dt),
        )
        event_unit_runtime = {
            "name": "settled_episode",
            "episode_merge_gap_ms": float(event_unit["episode_merge_gap_ms"]),
        }
    elif event_unit["name"] == "population_excursion":
        low_fraction = float(event_unit["low_threshold_fraction"])
        if not np.isclose(low_fraction, float(RETURN_FRAC)):
            raise RuntimeError("population-excursion low fraction drifted from detector")
        fast_tau_ms = max(
            float(substrate.params.tau_m_E),
            float(substrate.params.tau_d_GABA),
        )
        delay_ms = (
            int(substrate.net["max_delay_steps"])
            * float(substrate.engine["dt"])
        )
        reset_ms = (
            float(event_unit["fast_state_decay_multiples"]) * fast_tau_ms
            + delay_ms
        )
        pre_roll_ms = float(event_unit.get("pre_roll_ms", fast_tau_ms))
        detected = population_excursion_episodes(
            detector_fragments, active, sample_dt_ms=float(active_dt),
            event_on_threshold=float(substrate.detector_threshold),
            low_threshold_fraction=low_fraction, reset_ms=reset_ms,
            pre_roll_ms=pre_roll_ms,
        )
        event_unit_runtime = {
            "name": "population_excursion",
            "event_on_threshold": float(substrate.detector_threshold),
            "low_threshold_fraction": low_fraction,
            "low_threshold": low_fraction * float(substrate.detector_threshold),
            "fast_tau_ms": fast_tau_ms,
            "fast_state_decay_multiples": float(
                event_unit["fast_state_decay_multiples"]
            ),
            "maximum_delay_ms": delay_ms,
            "reset_ms": reset_ms,
            "pre_roll_ms": pre_roll_ms,
            "contact_geometry_used_for_boundary": False,
        }
    elif event_unit["name"] == "spatiotemporal_cascade":
        movie_config = config["source_topology"]["full_sheet_movie"]
        if not bool(movie_config.get("enabled", False)):
            raise RuntimeError("cascade events require the whole-run sheet movie")
        movie = sheet_activity_movie(
            spikes, substrate.positions_e,
            dt_ms=float(substrate.engine["dt"]),
            frame_ms=float(movie_config["frame_ms"]),
            bin_mm=float(config["source_topology"]["bin_mm"]),
            sheet_mm=float(substrate.engine["L"]),
        )
        cascade = spatiotemporal_cascade_labels(
            movie["activity_counts"],
            minimum_active_neurons=int(event_unit["minimum_active_neurons"]),
        )
        assignments = assign_detector_fragments_to_cascades(
            movie["activity_counts"], cascade["labels"], detector_fragments,
            frame_ms=float(movie["frame_ms"]),
            minimum_dominance=float(event_unit["minimum_dominance"]),
        )
        detected, compound_fragments = cascade_event_windows(
            cascade["components"], assignments, detector_fragments,
            frame_ms=float(movie["frame_ms"]),
            total_ms=len(active) * float(active_dt),
        )
        event_unit_runtime = {
            "name": "spatiotemporal_cascade",
            "event_on_threshold": float(substrate.detector_threshold),
            "minimum_active_neurons": int(event_unit["minimum_active_neurons"]),
            "minimum_dominance": float(event_unit["minimum_dominance"]),
            "movie_frame_ms": float(movie["frame_ms"]),
            "movie_bin_mm": float(movie["bin_mm"]),
            "contact_geometry_used_for_boundary": False,
            "n_spatiotemporal_components": int(len(cascade["components"])),
            "n_compound_detector_fragments": int(len(compound_fragments)),
            "compound_detector_fragment_fraction": float(
                len(compound_fragments) / max(1, len(detector_fragments))
            ),
            "compound_fragments": compound_fragments,
        }
        cascade_arrays = {
            "detector_fragment_dominant_cascade_id": np.asarray([
                -1 if row["dominant_cascade_id"] is None
                else int(row["dominant_cascade_id"])
                for row in assignments
            ], np.int32),
            "detector_fragment_dominance": np.asarray([
                row["dominant_activity_fraction"] for row in assignments
            ], np.float32),
            "detector_fragment_compound": np.asarray([
                row["compound"] for row in assignments
            ], bool),
        }
    elif event_unit["name"] in {
            "directed_spatiotemporal_lineage",
            "persistent_directed_spatiotemporal_lineage",
            "causal_population_excursion",
            "persistent_root_coactivity_episode"}:
        movie_config = config["source_topology"]["full_sheet_movie"]
        if not bool(movie_config.get("enabled", False)):
            raise RuntimeError("directed lineage events require the whole-run sheet movie")
        movie = sheet_activity_movie(
            spikes, substrate.positions_e,
            dt_ms=float(substrate.engine["dt"]),
            frame_ms=float(movie_config["frame_ms"]),
            bin_mm=float(config["source_topology"]["bin_mm"]),
            sheet_mm=float(substrate.engine["L"]),
        )
        parent_contract = _directed_parent_contract(
            event_unit, params=substrate.params, net=substrate.net,
            engine=substrate.engine, frame_ms=float(movie["frame_ms"]),
        )
        parent_gap_frames = int(parent_contract["forward_parent_frame_gap"])
        parent_neighborhood_bins = int(
            parent_contract["forward_parent_neighborhood_bins"]
        )
        lineage = directed_spatiotemporal_lineages(
            movie["activity_counts"],
            minimum_active_neurons=int(event_unit["minimum_active_neurons"]),
            maximum_parent_gap_frames=parent_gap_frames,
            parent_neighborhood_bins=parent_neighborhood_bins,
        )
        assignments = assign_detector_fragments_to_directed_lineages(
            movie["activity_counts"], lineage["labels"], detector_fragments,
            frame_ms=float(movie["frame_ms"]),
            minimum_dominance=float(event_unit["minimum_dominance"]),
        )
        directed_assignment_by_fragment = {
            int(row["detector_fragment_index"]): row for row in assignments
        }
        if event_unit["name"] == "causal_population_excursion":
            low_fraction = float(event_unit["low_threshold_fraction"])
            if not np.isclose(low_fraction, float(RETURN_FRAC)):
                raise RuntimeError("causal episode low fraction drifted from detector")
            episodes = population_excursion_episodes(
                detector_fragments, active, sample_dt_ms=float(active_dt),
                event_on_threshold=float(substrate.detector_threshold),
                low_threshold_fraction=low_fraction,
                reset_ms=float(parent_contract["causal_memory_ms"]),
                pre_roll_ms=float(event_unit.get(
                    "pre_roll_ms", parent_contract["fast_state_tau_ms"],
                )),
            )
            detected = annotate_population_excursions_with_lineages(
                episodes, movie["activity_counts"], lineage["labels"],
                frame_ms=float(movie["frame_ms"]),
            )
            compound_fragments = []
        elif event_unit["name"] == "persistent_root_coactivity_episode":
            detected = root_coactivity_event_windows(
                lineage["components"], assignments, detector_fragments,
                lineage["labels"], frame_ms=float(movie["frame_ms"]),
                total_ms=len(active) * float(active_dt),
            )
            compound_fragments = []
        else:
            detected, compound_fragments = cascade_event_windows(
                lineage["components"], assignments, detector_fragments,
                frame_ms=float(movie["frame_ms"]),
                total_ms=len(active) * float(active_dt),
            )
        directed_source = directed_lineage_onset_maps(
            lineage["labels"], detected, frame_ms=float(movie["frame_ms"]),
        )
        active_local = (
            movie["activity_counts"] >= int(event_unit["minimum_active_neurons"])
        )
        active_mass = float(np.sum(movie["activity_counts"][active_local]))
        collision_mass = float(np.sum(
            movie["activity_counts"][lineage["collision_mask"]]
        ))
        event_unit_runtime = {
            "name": event_unit["name"],
            "event_on_threshold": float(substrate.detector_threshold),
            "minimum_active_neurons": int(event_unit["minimum_active_neurons"]),
            "minimum_dominance": float(event_unit["minimum_dominance"]),
            "movie_frame_ms": float(movie["frame_ms"]),
            "movie_bin_mm": float(movie["bin_mm"]),
            **parent_contract,
            "contact_geometry_used_for_boundary": False,
            "n_directed_roots": int(len(lineage["components"])),
            "n_compound_detector_fragments": int(np.sum([
                assignment["compound"] for assignment in assignments
            ])),
            "compound_detector_fragment_fraction": float(
                np.mean([assignment["compound"] for assignment in assignments])
                if assignments else 0.0
            ),
            "collision_activity_mass_fraction": float(
                collision_mass / active_mass if active_mass > 0.0 else 0.0
            ),
            "causal_rule": lineage["causal_rule"],
            "compound_fragments": compound_fragments,
            "multi_root_event_fraction": float(np.mean([
                len(event.get("lineage_ids", [])) > 1
                for event in detected
            ])) if detected else 0.0,
            "all_detector_fragments_represented": bool(
                {
                    int(fragment) for event in detected
                    for fragment in event["detector_fragment_indices"]
                }.union({
                    int(row["detector_fragment_index"])
                    for row in compound_fragments
                }) == set(range(len(detector_fragments)))
            ),
            "detector_fragment_membership_excess": int(
                sum(len(event["detector_fragment_indices"]) for event in detected)
                + len(compound_fragments) - len(detector_fragments)
            ),
        }
        if event_unit["name"] in {
                "persistent_directed_spatiotemporal_lineage",
                "causal_population_excursion",
                "persistent_root_coactivity_episode"}:
            memory_method = str(event_unit.get(
                "causal_memory_method", "global_fast_state_decay",
            ))
            if memory_method == "local_ee_psp_tail":
                sensitivity_parameter = "psp_tail_fraction"
                primary_value = float(event_unit["psp_tail_fraction"])
                sensitivity_values = [
                    float(value) for value in event_unit.get(
                        "sensitivity_psp_tail_fractions", [primary_value],
                    )
                ]
            else:
                sensitivity_parameter = "fast_state_decay_multiples"
                primary_value = float(event_unit["fast_state_decay_multiples"])
                sensitivity_values = [
                    float(value) for value in event_unit.get(
                        "sensitivity_fast_state_decay_multiples", [primary_value],
                    )
                ]
            if (len(set(sensitivity_values)) != len(sensitivity_values)
                    or primary_value not in sensitivity_values):
                raise RuntimeError("persistent-lineage sensitivity grid is invalid")
            for sensitivity_value in sensitivity_values:
                if np.isclose(sensitivity_value, primary_value):
                    variant_lineage = lineage
                    variant_assignments = assignments
                    variant_events = detected
                    variant_compounds = compound_fragments
                    variant_contract = parent_contract
                else:
                    variant_unit = {
                        **event_unit, sensitivity_parameter: sensitivity_value,
                    }
                    variant_contract = _directed_parent_contract(
                        variant_unit, params=substrate.params, net=substrate.net,
                        engine=substrate.engine, frame_ms=float(movie["frame_ms"]),
                        local_delay_ms=parent_contract.get(
                            "local_or_global_delay_support_ms"
                        ) if memory_method == "local_ee_psp_tail" else None,
                    )
                    variant_lineage = directed_spatiotemporal_lineages(
                        movie["activity_counts"],
                        minimum_active_neurons=int(event_unit["minimum_active_neurons"]),
                        maximum_parent_gap_frames=int(
                            variant_contract["forward_parent_frame_gap"]
                        ),
                        parent_neighborhood_bins=int(
                            variant_contract["forward_parent_neighborhood_bins"]
                        ),
                    )
                    variant_assignments = assign_detector_fragments_to_directed_lineages(
                        movie["activity_counts"], variant_lineage["labels"],
                        detector_fragments, frame_ms=float(movie["frame_ms"]),
                        minimum_dominance=float(event_unit["minimum_dominance"]),
                    )
                    if event_unit["name"] == "causal_population_excursion":
                        variant_episodes = population_excursion_episodes(
                            detector_fragments, active,
                            sample_dt_ms=float(active_dt),
                            event_on_threshold=float(substrate.detector_threshold),
                            low_threshold_fraction=float(
                                event_unit["low_threshold_fraction"]
                            ),
                            reset_ms=float(variant_contract["causal_memory_ms"]),
                            pre_roll_ms=float(event_unit.get(
                                "pre_roll_ms", variant_contract["fast_state_tau_ms"],
                            )),
                        )
                        variant_events = annotate_population_excursions_with_lineages(
                            variant_episodes, movie["activity_counts"],
                            variant_lineage["labels"],
                            frame_ms=float(movie["frame_ms"]),
                        )
                        variant_compounds = []
                    elif event_unit["name"] == "persistent_root_coactivity_episode":
                        variant_events = root_coactivity_event_windows(
                            variant_lineage["components"], variant_assignments,
                            detector_fragments, variant_lineage["labels"],
                            frame_ms=float(movie["frame_ms"]),
                            total_ms=len(active) * float(active_dt),
                        )
                        variant_compounds = []
                    else:
                        variant_events, variant_compounds = cascade_event_windows(
                            variant_lineage["components"], variant_assignments,
                            detector_fragments, frame_ms=float(movie["frame_ms"]),
                            total_ms=len(active) * float(active_dt),
                        )
                lineage_sensitivity.append({
                    "parameter": sensitivity_parameter,
                    "value": sensitivity_value,
                    "contract": variant_contract,
                    "lineage": variant_lineage,
                    "assignments": variant_assignments,
                    "events": variant_events,
                    "compounds": variant_compounds,
                })
        lineage_labels = lineage["labels"]
        label_dtype = np.int16 if np.max(lineage_labels, initial=0) <= 32767 else np.int32
        cascade_arrays = {
            "directed_lineage_labels": lineage_labels.astype(label_dtype),
            "directed_lineage_collision_mask": lineage["collision_mask"],
            "detector_fragment_dominant_lineage_id": np.asarray([
                -1 if row["dominant_lineage_id"] is None
                else int(row["dominant_lineage_id"])
                for row in assignments
            ], np.int32),
            "detector_fragment_dominance": np.asarray([
                row["dominant_activity_fraction"] for row in assignments
            ], np.float32),
            "detector_fragment_collision_fraction": np.asarray([
                row["collision_activity_fraction"] for row in assignments
            ], np.float32),
            "detector_fragment_compound": np.asarray([
                row["compound"] for row in assignments
            ], bool),
        }
    else:
        raise RuntimeError(f"unknown event unit: {event_unit['name']}")
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, substrate.positions_e, substrate.montage,
        float(substrate.engine["dt"]),
    )
    readout = config["search"]["contact_readout"]
    onsets, ranks, contact_readout_audit = _event_contact_readout(
        events=detected, envelope=envelope, envelope_dt_ms=envelope_dt,
        montage=substrate.montage, valid_contacts=substrate.valid_contacts,
        positions_e=substrate.positions_e, movie=movie,
        lineage_labels=lineage_labels, spikes=spikes,
        spike_dt_ms=float(substrate.engine["dt"]), readout=readout,
    )
    if lineage_sensitivity:
        n_variants = len(lineage_sensitivity)
        maximum_events = max(len(row["events"]) for row in lineage_sensitivity)
        n_contacts = len(substrate.contact_names)
        sensitivity_onsets = np.full(
            (n_variants, maximum_events, n_contacts), np.nan, np.float32,
        )
        sensitivity_ranks = np.full_like(sensitivity_onsets, np.nan)
        sensitivity_returned = np.zeros((n_variants, maximum_events), bool)
        sensitivity_event_counts = np.zeros(n_variants, np.int16)
        sensitivity_fragment_partition = np.full(
            (n_variants, len(detector_fragments)), -32768, np.int32,
        )
        sensitivity_rows = []
        for variant_index, variant in enumerate(lineage_sensitivity):
            variant_events = variant["events"]
            if np.isclose(
                    variant["value"], float(parent_contract["sensitivity_value"])):
                variant_onsets, variant_ranks = onsets, ranks
            else:
                variant_onsets, variant_ranks, _ = _event_contact_readout(
                    events=variant_events, envelope=envelope,
                    envelope_dt_ms=envelope_dt, montage=substrate.montage,
                    valid_contacts=substrate.valid_contacts,
                    positions_e=substrate.positions_e, movie=movie,
                    lineage_labels=variant["lineage"]["labels"], spikes=spikes,
                    spike_dt_ms=float(substrate.engine["dt"]), readout=readout,
                )
            count = len(variant_events)
            sensitivity_event_counts[variant_index] = count
            sensitivity_onsets[variant_index, :count] = variant_onsets
            sensitivity_ranks[variant_index, :count] = variant_ranks
            sensitivity_returned[variant_index, :count] = np.asarray([
                event["returned"] for event in variant_events
            ], bool)
            for event_index, event in enumerate(variant_events):
                for fragment in event["detector_fragment_indices"]:
                    fragment = int(fragment)
                    previous = sensitivity_fragment_partition[
                        variant_index, fragment
                    ]
                    sensitivity_fragment_partition[variant_index, fragment] = (
                        event_index if previous == -32768 else -(fragment + 1)
                    )
            for fragment in range(len(detector_fragments)):
                if sensitivity_fragment_partition[variant_index, fragment] == -32768:
                    sensitivity_fragment_partition[variant_index, fragment] = -(fragment + 1)
            components = variant["lineage"]["components"]
            sensitivity_rows.append({
                "sensitivity_parameter": variant["parameter"],
                "sensitivity_value": variant["value"],
                **variant["contract"],
                "n_directed_roots": len(components),
                "n_events": count,
                "n_compound_fragments": len(variant["compounds"]),
                "compound_fragment_fraction": (
                    float(np.mean([
                        assignment["compound"]
                        for assignment in variant["assignments"]
                    ])) if variant["assignments"] else 0.0
                ),
                "n_roots_resumed_after_gap": int(np.sum([
                    component["resumed_after_gap_count"] > 0
                    for component in components
                ])),
                "maximum_observed_parent_gap_frames": int(max([
                    component["maximum_parent_gap_frames_observed"]
                    for component in components
                ], default=0)),
                "multi_root_event_fraction": float(np.mean([
                    len(event.get("lineage_ids", [])) > 1
                    for event in variant_events
                ])) if variant_events else 0.0,
            })
        event_unit_runtime["memory_sensitivity"] = sensitivity_rows
        cascade_arrays.update({
            "lineage_sensitivity_values": np.asarray([
                row["value"] for row in lineage_sensitivity
            ], np.float32),
            "lineage_sensitivity_event_counts": sensitivity_event_counts,
            "lineage_sensitivity_onsets": sensitivity_onsets,
            "lineage_sensitivity_ranks": sensitivity_ranks,
            "lineage_sensitivity_returned": sensitivity_returned,
            "lineage_sensitivity_fragment_partition": sensitivity_fragment_partition,
        })
        if parent_contract["sensitivity_parameter"] == "fast_state_decay_multiples":
            cascade_arrays["lineage_sensitivity_multiples"] = np.asarray([
                row["value"] for row in lineage_sensitivity
            ], np.float32)
    event_rows = []
    for index, event in enumerate(detected):
        onset = onsets[index]
        event_row = {
            "event_index": int(index), "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "trigger_t_on_ms": float(event.get("trigger_t_on", event["t_on"])),
            "trigger_t_off_ms": float(event.get("trigger_t_off", event["t_off"])),
            "reset_start_ms": event.get("reset_start_ms"),
            "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": _event_peak_active_fraction(
                event, active, active_dt,
            ),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.isfinite(onset).sum()),
            "n_detector_fragments": int(len(event.get(
                "detector_fragment_indices", [int(index)],
            ))),
            "detector_fragment_indices": event.get(
                "detector_fragment_indices", [int(index)],
            ),
            "cascade_id": event.get("cascade_id"),
        }
        if event_unit is not None and event_unit["name"] in {
                "directed_spatiotemporal_lineage",
                "persistent_directed_spatiotemporal_lineage",
                "causal_population_excursion",
                "persistent_root_coactivity_episode"}:
            fragment_rows = [
                directed_assignment_by_fragment[int(fragment)]
                for fragment in event_row["detector_fragment_indices"]
            ]
            event_row.update({
                "lineage_id": event.get("cascade_id"),
                "lineage_ids": event.get("lineage_ids", [event.get("cascade_id")]),
                "root_count": int(event.get("root_count", 1)),
                "minimum_fragment_dominance": float(min(
                    row["dominant_activity_fraction"] for row in fragment_rows
                )),
                "maximum_fragment_collision_fraction": float(max(
                    row["collision_activity_fraction"] for row in fragment_rows
                )),
            })
        event_rows.append(event_row)
    returned = np.asarray([row["returned"] for row in event_rows], bool)
    event_t_on = np.asarray([row["t_on_ms"] for row in event_rows], float)
    event_trigger_t_on = np.asarray([
        row["trigger_t_on_ms"] for row in event_rows
    ], float)
    if directed_source is None:
        source = event_source_onset_maps(
            spikes, substrate.positions_e, event_trigger_t_on, returned,
            dt_ms=float(substrate.engine["dt"]),
            sheet_mm=float(substrate.engine["L"]),
            bin_mm=float(config["source_topology"]["bin_mm"]),
        )
    else:
        source = {
            **directed_source,
            "activity_counts": np.zeros(
                (len(detected), 0, *movie["activity_counts"].shape[1:]),
                dtype=np.uint16,
            ),
            "relative_times_ms": np.asarray([], float),
            "bin_mm": float(movie["bin_mm"]),
            "sheet_mm": float(movie["sheet_mm"]),
        }
    movie_config = config["source_topology"].get("full_sheet_movie", {})
    if movie is None and bool(movie_config.get("enabled", False)):
        movie = sheet_activity_movie(
            spikes, substrate.positions_e,
            dt_ms=float(substrate.engine["dt"]),
            frame_ms=float(movie_config["frame_ms"]),
            bin_mm=float(config["source_topology"]["bin_mm"]),
            sheet_mm=float(substrate.engine["L"]),
        )
    if movie is not None:
        movie_arrays = {
            "sheet_activity_counts": movie["activity_counts"],
            "sheet_activity_frame_ms": np.asarray(movie["frame_ms"], float),
        }
    del spikes

    _atomic_npz(
        out_npz,
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        shaft_ids=np.asarray(substrate.shaft_ids, dtype="U8"),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float64),
        onsets=onsets.astype(np.float32), ranks=ranks.astype(np.float32),
        event_t_on_ms=event_t_on.astype(np.float32),
        event_trigger_t_on_ms=event_trigger_t_on.astype(np.float32),
        event_t_off_ms=np.asarray([row["t_off_ms"] for row in event_rows], np.float32),
        event_returned=returned,
        event_fragment_count=np.asarray([
            row["n_detector_fragments"] for row in event_rows
        ], np.int16),
        event_directed_root_id=np.asarray([
            -1 if row.get("lineage_id") is None else int(row["lineage_id"])
            for row in event_rows
        ], np.int32),
        event_root_count=np.asarray([
            int(row.get("root_count", 1)) for row in event_rows
        ], np.int32),
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        source_onset_maps_ms=source["onset_maps_ms"],
        source_activity_counts=source["activity_counts"],
        source_activity_relative_ms=source["relative_times_ms"],
        source_onset_evaluable=source["evaluable"],
        source_bin_mm=np.asarray(source["bin_mm"], float),
        source_sheet_mm=np.asarray(source["sheet_mm"], float),
        positions_E=np.asarray(substrate.positions_e, np.float32),
        h=np.asarray(substrate.h_e, np.float32),
        delta_vtheta=np.asarray(substrate.delta_vtheta, np.float32),
        edge_coefficients=np.asarray(substrate.edge_coefficients, np.float64),
        **movie_arrays,
        **cascade_arrays,
    )
    payload = {
        "status": "REV12ND_NODE_WORKER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "candidate_id": args.candidate_id,
        "field_sha256": candidate["node_field"]["field_sha256"],
        "seed": int(args.seed),
        "simulation": {
            "duration_ms": float(simulation["duration_ms"]),
            "runaway_early_stop_ms": result.get("runaway_early_stop_ms"),
            "wall_seconds": float(time.time() - started),
        },
        "events": event_rows,
        "event_unit": {
            "raw_detector_fragment_count": int(len(detector_fragments)),
            "episode_count": int(len(detected)),
            **event_unit_runtime,
            "rationale": (
                "rev12 event grouping uses latent population activity only; virtual "
                "contacts are read out after episode boundaries are frozen"
            ),
        },
        "contact_readout": contact_readout_audit,
        "source_topology": {
            "n_evaluable_returned_events": int(np.sum(source["evaluable"] & returned)),
            "bin_mm": source["bin_mm"],
            "window_ms": [-20.0, 80.0],
            "baseline_window_ms": [-120.0, -20.0],
            "source_map": (
                "event_triggered_persistent_recruitment"
                if directed_source is None
                else "directed_lineage_first_arrival"
            ),
            "full_sheet_movie": {
                "stored": bool(movie_arrays),
                "frame_ms": (
                    None if not movie_arrays
                    else float(movie_config["frame_ms"])
                ),
            },
        },
        "mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": bool(np.allclose(substrate.edge_coefficients, 0.0)),
        },
        "arrays": {"path": str(out_npz), "sha256": _sha256(out_npz)},
        "provenance": provenance,
    }
    atomic_write_json(_json_safe(payload), str(out_json))
    print(json.dumps({
        "status": payload["status"], "candidate": args.candidate_id,
        "seed": args.seed, "n_returned": int(np.sum(returned)),
        "n_source_maps": payload["source_topology"]["n_evaluable_returned_events"],
        "output": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
