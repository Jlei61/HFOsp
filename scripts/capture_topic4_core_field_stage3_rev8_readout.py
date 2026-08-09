"""Capture the frozen rev8 candidate's representative virtual-SEEG run.

This producer reruns exactly one final-confirmation network and verifies every
usable rank curve against the hashed pooled event artifact before saving LFP and
per-neuron onset fields for the Fig. 4 renderers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _load_cmrun  # noqa: E402
from scripts.run_topic4_core_field_stage3_joint_confirm import _atomic_npz  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_core_field_stage3_rev8_confirm import (  # noqa: E402
    CONFIRM_OUT,
    PROFILES_OUT,
)
from src.topic4_core_field import (  # noqa: E402
    build_vth,
    core_thresholds,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_core_field_profile import normalized_rank_curve  # noqa: E402
from src.topic4_core_field_runner import (  # noqa: E402
    _placement,
    atomic_write_json,
    get_network,
    provenance,
)
from src.topic4_core_field_stage3 import params_to_h, spatial_diagnostics  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
OUT_DIR = f"{ROOT}/joint_confirmation_rev8"
READOUT_OUT = f"{OUT_DIR}/representative_readout.json"
FIGDATA_OUT = f"{OUT_DIR}/representative_figdata.npz"


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default=CONFIRM_OUT)
    parser.add_argument("--profiles", default=PROFILES_OUT)
    parser.add_argument("--readout-out", default=READOUT_OUT)
    parser.add_argument("--figdata-out", default=FIGDATA_OUT)
    args = parser.parse_args()

    confirmation = json.load(open(args.confirmation))
    if confirmation["event_profiles"]["sha256"] != _sha256(args.profiles):
        raise RuntimeError("confirmation/event-profile hash mismatch")
    candidate = confirmation["candidates"][0]
    seed = int(confirmation["representative_run"]["seed"])
    representative_indices = confirmation["representative_run"][
        "local_event_index_by_mode"]
    arrays = np.load(args.profiles)
    pooled_seed_ids = np.asarray(arrays["model_seed_ids"], int)
    pooled_local_indices = np.asarray(arrays["model_local_event_indices"], int)
    pooled_labels = np.asarray(arrays["model_labels"], int)
    pooled_curves = np.asarray(arrays["model_curves"], float)
    pooled_map = {
        (int(network_seed), int(local_index)): (int(label), curve)
        for network_seed, local_index, label, curve in zip(
            pooled_seed_ids, pooled_local_indices, pooled_labels, pooled_curves)
    }

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    engine = cfg["engine"]
    cmrun = _load_cmrun()
    cmrun.KDIR = int(engine["k_dir"])
    cmrun.PART_MIN = 2 * int(engine["k_dir"]) + 1
    reg = _placement(cfg)
    params_cls = __import__("params").Params
    simulate_kick = __import__("kick_probe").simulate_kick
    lfp_recorder_cls = __import__("lfp").LFPRecorder
    detect_events = __import__("src.sef_hfo_events", fromlist=["detect_events"]).detect_events
    snn_event_envelope = __import__(
        "src.sef_hfo_snn_adapter", fromlist=["snn_event_envelope"]
    ).snn_event_envelope

    p = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=cfg["duration_ms"], dt=engine["dt"],
        nu_ext_ratio=cmrun.DRIVE, seed=seed,
    )
    cache = os.path.join(STAGE2, "network_cache")
    net, n_e, n_i, cache_hit = get_network(
        p, reg["theta_deg"], engine["AR"], cache)
    pos_e = net["pos"][:n_e]
    theta = np.asarray(candidate["theta"], float)
    h = params_to_h(theta, pos_e, int(candidate["K"]),
                    float(engine["L"]), float(cfg["N_core_manual"]))
    depth = signed_depth(core_thresholds(
        sample_core_quantiles(n_e, cfg["quantile_seed"]),
        engine["core_mean"], engine["core_std"]), engine["v_base"])
    vth = build_vth(
        h, depth, n_total=n_e + n_i, n_E=n_e, v_base=engine["v_base"])

    montage = reg["montage_sheet"]
    valid = cmrun.valid_mask(montage, pos_e, engine["L"], p.Rr)
    recorder = lfp_recorder_cls(
        p, net["pos"], net["labels"], sites=montage.contacts)
    net["rng"] = np.random.default_rng(seed)
    result = simulate_kick(
        p, net, KICK_BOOST=0.0, kick_center=list(reg["center"]),
        r_kick=engine["core_r"], t_kick=1e9,
        V_th_per_neuron=vth, lfp_recorder=recorder,
    )
    spikes = result["E_spk_bool"]
    active_fraction, bin_width = cmrun.active_fraction(
        spikes, engine["dt"], cmrun.BIN_MS)
    baseline_start = int(cmrun.BASELINE_MS[0] / bin_width)
    baseline_stop = int(cmrun.BASELINE_MS[1] / bin_width)
    floor = float(np.percentile(
        active_fraction[baseline_start:baseline_stop], 95))
    bar = floor + cmrun.CAL_FRAC * (float(active_fraction.max()) - floor)
    detected = detect_events(active_fraction, bin_width, event_on_frac=bar)
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, pos_e, montage, engine["dt"])
    axial = axial_map()
    grid = np.asarray(arrays["grid"], float)

    records = []
    captured_curves = {}
    for local_index, event in enumerate(detected):
        readout = cmrun.read_event(
            envelope, envelope_dt, montage, valid,
            (event["t_on"], event["t_off"]), reg["axis_unit_vec"],
            k_dir=int(engine["k_dir"]),
            part_min=2 * int(engine["k_dir"]) + 1,
        )
        curve = normalized_rank_curve(readout["ranks"], axial, grid=grid)
        mode = None
        if curve is not None:
            key = (seed, local_index)
            if key not in pooled_map:
                raise RuntimeError(f"capture produced an unconfirmed usable event {key}")
            mode, expected = pooled_map[key]
            if not np.allclose(curve, expected, atol=2e-7, rtol=0.0):
                raise RuntimeError(f"capture curve drift for event {key}")
            captured_curves[key] = curve
        records.append(dict(
            local_event_index=int(local_index),
            t_on=float(event["t_on"]), t_off=float(event["t_off"]),
            returned=bool(event.get("returned", False)),
            n_part=int(readout["n_part"]),
            ranks=readout["ranks"], mode=mode,
        ))
    expected_keys = {
        key for key in pooled_map if key[0] == seed
    }
    if set(captured_curves) != expected_keys:
        raise RuntimeError("capture did not reproduce the complete confirmed seed event set")

    representatives = {}
    for mode in (0, 1):
        local_index = representative_indices[str(mode)]
        if local_index is None:
            representatives[f"mode_{mode}"] = np.array(None, dtype=object)
            continue
        record = records[int(local_index)]
        if record["mode"] != mode:
            raise RuntimeError("representative event mode changed on capture")
        onset = cmrun.per_neuron_onset(
            spikes, record["t_on"], record["t_off"], engine["dt"])
        representatives[f"mode_{mode}"] = np.array(dict(
            meta=record, onset=np.asarray(onset, np.float32)), dtype=object)

    diagnostics = spatial_diagnostics(
        h, pos_e, reg["center"], reg["axis_unit_vec"])
    _atomic_npz(
        args.figdata_out,
        reg=np.array(dict(
            center=np.asarray(reg["center"], float),
            source_centroid=np.asarray(reg["source_centroid"], float),
            sink_centroid=np.asarray(reg["sink_centroid"], float),
            axis_unit=np.asarray(reg["axis_unit_vec"], float),
            theta_deg=float(reg["theta_deg"]),
            L=float(engine["L"]),
        ), dtype=object),
        contacts=np.asarray(montage.contacts, np.float32),
        names=np.asarray(montage.names, dtype="U32"),
        valid=np.asarray(valid, bool),
        lfp_trace=np.asarray(result["lfp_trace"], np.float32),
        times=np.asarray(result["times"], np.float32),
        active_fraction=np.asarray(active_fraction, np.float32),
        bin_width=np.asarray(bin_width, float),
        posE=np.asarray(pos_e, np.float32),
        h=np.asarray(h, np.float32),
        vth=np.asarray(vth[:n_e], np.float32),
        theta=np.asarray(theta, float),
        **representatives,
    )
    readout_payload = dict(
        status="REV8_REPRESENTATIVE_READOUT_CAPTURED",
        subject=cfg["subject"], seed=seed,
        candidate_id=candidate["candidate_id"],
        theta_sha256=candidate["theta_sha256"], K=int(candidate["K"]),
        events=records,
        representative_local_event_index_by_mode=representative_indices,
        n_detected=int(len(detected)), n_usable=int(len(captured_curves)),
        mode_counts={
            str(mode): int(sum(record["mode"] == mode for record in records))
            for mode in (0, 1)
        },
        field_diagnostics=dict(
            r_bar=float(diagnostics["r_bar"]),
            s_bar=float(diagnostics["s_bar"]),
            c_axis_2mm=float(diagnostics["c_axis"][2.0]),
        ),
        input_confirmation=dict(
            path=args.confirmation, sha256=_sha256(args.confirmation)),
        input_profiles=dict(path=args.profiles, sha256=_sha256(args.profiles)),
        figdata=dict(path=args.figdata_out, sha256=_sha256(args.figdata_out)),
        network_cache_hit=bool(cache_hit),
        exact_curve_reproduction=True,
        provenance=provenance(),
    )
    atomic_write_json(readout_payload, args.readout_out)
    print(json.dumps({
        "status": readout_payload["status"], "seed": seed,
        "n_usable": readout_payload["n_usable"],
        "mode_counts": readout_payload["mode_counts"],
        "figdata_sha256": readout_payload["figdata"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
