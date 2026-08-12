"""Run one observation-invariant field candidate on one spontaneous SNN seed."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.freeze_topic4_rev10_sa_spectral_field_candidates import (  # noqa: E402,F401
    build_manifest,
)
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_rev9_node_kick_canary import _load_network  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz,
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.sef_hfo_observation import VirtualMontage, extract_lagpat  # noqa: E402
from src.topic4_core_field_rev9 import (  # noqa: E402
    reconstruct_frozen_node,
    reconstruct_node_from_h,
)
from src.topic4_core_field_runner import _placement, atomic_write_json  # noqa: E402
from src.topic4_continuous_field import continuous_field_h  # noqa: E402
from src.topic4_spectral_field import spectral_field_h  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field.json"


def _contact_onsets(envelope, envelope_dt, montage, valid, window,
                    margin_fraction, timing_fraction):
    selected = np.asarray(envelope, float)[np.asarray(valid, bool)]
    valid_names = np.asarray(montage.names)[np.asarray(valid, bool)]
    output_onsets = np.full(len(montage.names), np.nan)
    output_ranks = np.full(len(montage.names), np.nan)
    if not len(selected):
        return output_onsets, output_ranks
    floor = float(selected.min())
    margin = float(margin_fraction) * (float(selected.max()) - floor)
    artifact = extract_lagpat(
        selected, float(envelope_dt), [tuple(window)], floor, margin,
        timing_frac=float(timing_fraction), tie_tol=float(envelope_dt),
    )
    lookup = {str(name): index for index, name in enumerate(montage.names)}
    for local_index, name in enumerate(valid_names):
        target = lookup[str(name)]
        output_onsets[target] = artifact.lag_raw[local_index, 0]
        output_ranks[target] = artifact.ranks[local_index, 0]
    return output_onsets, output_ranks


def _candidate_node(candidate, positions, *, n_total, stage, config):
    engine = stage["engine"]
    if candidate["field_type"] == "gaussian_k3_benchmark":
        return reconstruct_frozen_node(
            candidate["theta"], positions, n_total=n_total,
            target_count=stage["N_core_manual"],
            quantile_seed=stage["quantile_seed"],
            core_mean=engine["core_mean"], core_std=engine["core_std"],
            v_base=engine["v_base"], K=3, L=engine["L"],
        )
    if candidate["field_type"] != "spectral_continuous":
        if candidate["field_type"] != "spline_continuous":
            raise ValueError(
                f"unknown continuous worker field type: {candidate['field_type']}"
            )
        h, _ = continuous_field_h(
            candidate["coefficients"], positions,
            n_basis=candidate["n_basis"], degree=candidate["degree"],
            target_count=stage["N_core_manual"], L=engine["L"],
        )
    else:
        h, _ = spectral_field_h(
            candidate["coefficients"], positions,
            max_harmonic=config["field"]["max_harmonic"],
            target_count=stage["N_core_manual"], L=engine["L"],
        )
    return reconstruct_node_from_h(
        h, n_total=n_total, quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"],
        v_base=engine["v_base"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    allowed_roles = {
        "development_only_observation_invariant_continuous_node_field_search",
        "development_only_observation_invariant_uniform_allocation_refinement",
        "development_only_stable_spline_random_field_screen",
        "development_only_v3_to_stable_spline_bridge",
        "development_only_stable_spline_adaptive_interpolation",
        "development_only_stable_spline_selection_confirmation",
        "development_only_stable_spline_final_confirmation",
        "development_only_mode_conditioned_boundary_refinement",
    }
    if config["scientific_role"] not in allowed_roles:
        raise RuntimeError("spectral search role changed")
    allowed_seeds = {
        int(value) for value in (
            config["search"]["network_seeds"]
            + config["search"]["selection_network_seeds"]
            + config["search"].get("final_network_seeds", [])
        )
    }
    if args.seed not in allowed_seeds:
        parser.error("--seed is outside the frozen development seed sets")
    if config["search"]["edge"] != "off" or config["search"]["beta"] != "closed":
        raise RuntimeError("spectral search must remain Node-only")

    manifest_path = Path(config["output_root"]) / "candidate_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("candidate manifest must be frozen before workers start")
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("candidate manifest uses another config")
    matches = [
        row for row in manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == args.candidate_id
    ]
    if len(matches) != 1:
        parser.error("--candidate-id is outside the frozen spectral library")
    candidate = matches[0]

    provenance = _runtime_provenance(args.expected_commit)
    provenance["systemd_unit"] = os.environ.get("REV10SA_SYSTEMD_UNIT")
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if provenance["runtime_modules_dirty"] or config_dirty:
        raise RuntimeError("spectral worker runtime modules or config are dirty")
    if (args.expected_commit is not None
            and not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("spectral worker modules differ from launcher commit")

    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    contract = _load_json_input(inputs["contact_contract"])
    detector_audit = _load_json_input(inputs["common_detector_audit"])
    detector = float(config["search"]["detector"][
        "population_active_fraction_threshold"
    ])
    if detector != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("common detector changed")

    output_root = ROOT / config["output_root"]
    stem = f"{candidate['candidate_id']}_seed_{args.seed}"
    output_json = Path(args.out_json or output_root / "workers" / f"{stem}.json")
    output_npz = Path(args.out_npz or output_root / "workers" / f"{stem}.npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(args.cache_dir or ROOT /
                         "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"))

    started = time.time()
    engine = stage["engine"]
    simulation = config["search"]["simulation"]
    reg = _placement(stage)
    cmrun = _load_cmrun()
    cmrun.DT = float(engine["dt"])
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    detect_events = __import__(
        "src.sef_hfo_events", fromlist=["detect_events"]
    ).detect_events
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
    ).snn_event_envelope
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=float(simulation["duration_ms"]), dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=int(args.seed),
    )
    net, n_e, n_i, cache_hit, cache_source = _load_network(
        params, stage, reg, int(args.seed), base, cache_dir,
    )
    positions = np.asarray(net["pos"][:n_e], float)
    node = _candidate_node(
        candidate, positions, n_total=n_e + n_i, stage=stage, config=config,
    )
    if not np.isclose(node["h"].sum(), float(stage["N_core_manual"]), atol=1e-8):
        raise RuntimeError("spectral field budget projection failed")

    contacts = contract["contacts"]
    contact_names = [row["contact_name"] for row in contacts]
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    shaft_ids = np.asarray([row["shaft_id"] for row in contacts], dtype="U8")
    montage = VirtualMontage(
        contact_xy, contact_names,
        provenance="rev10_sa_observation_only_contact_contract",
    )
    valid = cmrun.valid_mask(montage, positions, engine["L"], params.Rr)
    if not np.all(valid):
        raise RuntimeError("all frozen contacts must be locally readable")

    net["rng"] = np.random.default_rng(int(args.seed))
    result = simulate_kick(
        params, net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=node["vtheta"],
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = cmrun.active_fraction(spikes, engine["dt"], cmrun.BIN_MS)
    detected = detect_events(active, active_dt, event_on_frac=detector)
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, positions, montage, engine["dt"],
    )
    readout = config["search"]["contact_readout"]
    onset_rows, rank_rows, event_rows = [], [], []
    for event_index, event in enumerate(detected):
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, valid,
            (event["t_on"], event["t_off"]),
            readout["participation_margin_fraction"],
            readout["timing_fraction"],
        )
        onset_rows.append(onset)
        rank_rows.append(rank)
        event_rows.append({
            "event_index": int(event_index),
            "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": float(event["peak_ext"]),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.isfinite(onset).sum()),
            "ICL_recruited_fraction": float(np.isfinite(
                onset[shaft_ids == "ICL"]
            ).mean()),
            "SCL_recruited_fraction": float(np.isfinite(
                onset[shaft_ids == "SCL"]
            ).mean()),
        })
    onsets = np.asarray(onset_rows, dtype=float).reshape((-1, len(contact_names)))
    ranks = np.asarray(rank_rows, dtype=float).reshape((-1, len(contact_names)))
    _atomic_npz(
        output_npz,
        contact_names=np.asarray(contact_names, dtype="U16"),
        shaft_ids=shaft_ids,
        contact_xy_mm=contact_xy,
        onsets=onsets.astype(np.float32),
        ranks=ranks.astype(np.float32),
        event_t_on_ms=np.asarray([row["t_on_ms"] for row in event_rows], np.float32),
        event_t_off_ms=np.asarray([row["t_off_ms"] for row in event_rows], np.float32),
        event_returned=np.asarray([row["returned"] for row in event_rows], bool),
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        positions_E=np.asarray(positions, np.float32),
        h=np.asarray(node["h"], np.float32),
        delta_vtheta=np.asarray(node["delta_vtheta"], np.float32),
    )
    payload = {
        "status": "REV10SA_SPECTRAL_FIELD_WORKER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "candidate": candidate,
        "seed": int(args.seed),
        "events": event_rows,
        "run": {
            "n_common_detector_events": int(len(detected)),
            "n_returned_events": int(sum(row["returned"] for row in event_rows)),
            "runaway_early_stop_ms": result["runaway_early_stop_ms"],
            "peak_active_fraction": float(np.max(active, initial=0.0)),
            "fraction_time_above_common_detector": float(np.mean(active > detector)),
        },
        "field": {
            "sum_h": float(node["h"].sum()),
            "max_h": float(node["h"].max(initial=0.0)),
            "n_h_ge_0p5": int(np.sum(node["h"] >= 0.5)),
            "n_h_ge_0p9": int(np.sum(node["h"] >= 0.9)),
            "node_hashes": node["hashes"],
        },
        "network": {
            "n_E": int(n_e), "n_I": int(n_i),
            "cache_hit": bool(cache_hit), "cache_source": cache_source,
        },
        "simulation": {
            **simulation,
            "common_detector_threshold": detector,
            "wall_seconds": float(time.time() - started),
        },
        "arrays": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }
    atomic_write_json(payload, output_json)
    print(json.dumps({
        "status": payload["status"], "candidate": args.candidate_id,
        "seed": args.seed, "n_events": len(detected),
        "runaway": result["runaway_early_stop_ms"],
        "elapsed_seconds": payload["simulation"]["wall_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
