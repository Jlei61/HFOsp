#!/usr/bin/env python3
"""I0 qualification on the real frozen substrate (short synthetic runs only).

Checks (plan §5): default None == explicit B0; observer on/off byte parity;
initial voltage enters the engine and differs on exactly K cells; static arrays
unchanged across arms; identical external-input digests across the three arms
of one dynamics seed; different seeds differ; the B0 path reproduces the
mainline worker's historical trajectory prefix; resource estimates.
None of these runs enters the scientific sample.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src import topic4_initial_state_runtime as rt  # noqa: E402
from src.topic4_initial_state import RunObserver, array_sha256, compare_input_digests  # noqa: E402
from src.topic4_zm_ictal_transition import make_external_drive  # noqa: E402
from scripts.run_topic4_initial_state_worker import load_frozen, run_unit  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402


def _sim(substrate, transition, seed, duration_ms, **kwargs):
    substrate.params.T = float(duration_ms)
    substrate.net["rng"] = np.random.default_rng(int(seed))
    drive = make_external_drive(substrate, transition["spatial_ou"], int(seed))
    return simulate_kick(substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
                         V_th_per_neuron=substrate.vtheta, slow=None, early_stop_runaway=True,
                         external_e_rate_drive=drive, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    parser.add_argument("--short-ms", type=float, default=1500.0)
    parser.add_argument("--unit-ms", type=float, default=2000.0)
    args = parser.parse_args()
    design = rt.load_design(args.design)
    out = rt.output_root(design)
    frozen_path = out / "frozen_manifest.json"
    frozen, membership = load_frozen(design, frozen_path)
    qual_dir = out / "qualification"
    qual_dir.mkdir(parents=True, exist_ok=True)
    stage = design["stages"]["screen"]
    seeds = stage["dynamics_seeds"]
    t0 = time.time()
    substrate, candidate, transition, execution, parameter_audit, network = rt.build_frozen_substrate(
        design, stage["topology_seed"], seeds[0], frozen_manifest=frozen)
    build_seconds = time.time() - t0
    evaluator = rt.load_evaluator(design)
    objective = rt.load_objective(design)
    contract = rt.load_observation_contract(design)
    identity_before = rt.static_identity(substrate, parameter_audit)
    n_e, n_total = substrate.n_e, substrate.n_e + substrate.n_i
    checks = {}
    dt = float(substrate.engine["dt"])

    # ---- 1. default None == explicit B0 (E1) ----
    b0 = np.load(frozen["arms"]["B0"]["file"])
    t1 = time.time()
    plain = _sim(substrate, transition, seeds[0], args.short_ms)
    short_wall = time.time() - t1
    explicit = _sim(substrate, transition, seeds[0], args.short_ms, initial_voltage=b0)
    checks["default_none_equals_explicit_B0"] = bool(
        np.array_equal(plain["rate_E"], explicit["rate_E"])
        and np.array_equal(plain["E_spk_bool"], explicit["E_spk_bool"])
        and np.array_equal(plain["initial_V"], explicit["initial_V"]))
    del explicit

    # ---- 2. observer on/off byte parity (E7) ----
    groups = {"coreA": membership["S_A"], "coreB": membership["S_B"]}
    steps = int(round(args.short_ms / dt))
    observer = RunObserver(dt_ms=dt, n_e=n_e, n_total=n_total, groups=groups, n_steps=steps)
    observed = _sim(substrate, transition, seeds[0], args.short_ms, initial_voltage=b0, step_observer=observer)
    checks["observer_does_not_change_trajectory"] = bool(
        np.array_equal(plain["rate_E"], observed["rate_E"])
        and np.array_equal(plain["E_spk_bool"], observed["E_spk_bool"]))
    digest_b0_short = observer.finish()["segments"]
    del observed

    # ---- 3. mainline parity: historical 2511 prefix (B0 path == mainline worker) ----
    hist_path = Path(design["sources"]["baseline_worker_arrays"]["path"])
    checks["historical_arrays_hash_ok"] = rt.sha(hist_path) == design["sources"]["baseline_worker_arrays"]["sha256"]
    hist = np.load(hist_path)
    legacy = _sim(substrate, transition, 2511, args.short_ms)
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(np.asarray(legacy["E_spk_bool"], bool), dt, cmrun.BIN_MS)
    n_active = len(active)
    hist_active = np.asarray(hist["active_fraction"], np.float32)[:n_active]
    from src.sef_hfo_snn_adapter import snn_event_envelope
    env, env_dt, _ = snn_event_envelope(np.asarray(legacy["E_spk_bool"], bool), substrate.positions_e,
                                        substrate.montage, dt)
    n_interior = int(env.shape[1] - 20)          # exclude the 5 ms Gaussian tail at the crop edge
    hist_env = np.asarray(hist["contact_envelope"], np.float32)[:, :n_interior]
    checks["mainline_prefix_active_fraction_identical"] = bool(
        np.array_equal(np.asarray(active, np.float32), hist_active))
    checks["mainline_prefix_contact_envelope_identical"] = bool(
        np.array_equal(np.asarray(env[:, :n_interior], np.float32), hist_env))
    checks["static_identity_matches_historical_worker"] = bool(all(
        identity_before[k] == rt.read(hist_path.with_suffix(".json"))["static_array_identity"][k]
        for k in ("positions_E_sha256", "h_sha256", "delta_vtheta_sha256", "vtheta_sha256",
                  "ampa_topology_sha256", "ampa_values_sha256", "gaba_topology_sha256",
                  "gaba_values_sha256")))
    del legacy, hist

    # ---- 4. three formal-style short units on one seed + one other seed ----
    units = {}
    for arm in ("B0", "B1", "B2"):
        stem = f"qual_{stage['topology_seed']}_dyn_{seeds[0]}_{arm}"
        units[arm] = run_unit(
            design=design, frozen=frozen, membership=membership, substrate=substrate,
            candidate=candidate, transition=transition, parameter_audit=parameter_audit,
            network=network, evaluator=evaluator, objective=objective, contract=contract,
            stage="screen", topology_seed=stage["topology_seed"], dynamics_seed=seeds[0], arm=arm,
            duration_ms=args.unit_ms, out_json=qual_dir / f"{stem}.json",
            out_npz=qual_dir / f"{stem}.npz", qualification_tag="I0_short_unit")
    other = run_unit(
        design=design, frozen=frozen, membership=membership, substrate=substrate,
        candidate=candidate, transition=transition, parameter_audit=parameter_audit,
        network=network, evaluator=evaluator, objective=objective, contract=contract,
        stage="screen", topology_seed=stage["topology_seed"], dynamics_seed=seeds[1], arm="B0",
        duration_ms=args.unit_ms, out_json=qual_dir / f"qual_{stage['topology_seed']}_dyn_{seeds[1]}_B0.json",
        out_npz=qual_dir / f"qual_{stage['topology_seed']}_dyn_{seeds[1]}_B0.npz",
        qualification_tag="I0_short_unit_other_seed")
    identity_after = rt.static_identity(substrate, parameter_audit)
    checks["static_arrays_unchanged_after_all_runs"] = identity_before == identity_after
    checks["static_identity_equal_across_arms"] = all(
        units[a]["static_array_identity"] == units["B0"]["static_array_identity"] for a in ("B1", "B2"))
    # initial voltage differs on exactly K cells by the increment, everything else equal
    v = {arm: np.load(qual_dir / f"qual_{stage['topology_seed']}_dyn_{seeds[0]}_{arm}.npz")["initial_V"]
         for arm in ("B0", "B1", "B2")}
    k = int(len(membership["S_A"]))
    inc = float(design["initialization"]["increment_mV"])
    d1, d2 = v["B1"] - v["B0"], v["B2"] - v["B0"]
    checks["B1_differs_from_B0_on_exactly_K_core_A_cells"] = bool(
        np.count_nonzero(d1) == k and np.all(d1[membership["S_A"]] == inc)
        and np.all(np.delete(d1, membership["S_A"]) == 0))
    checks["B2_differs_from_B0_on_exactly_K_core_B_cells"] = bool(
        np.count_nonzero(d2) == k and np.all(d2[membership["S_B"]] == inc)
        and np.all(np.delete(d2, membership["S_B"]) == 0))
    checks["initial_voltage_below_threshold_margin"] = all(
        units[a]["initial_voltage"]["threshold_margin"]["pass"] for a in units)
    checks["increment_matched_B1_B2"] = (
        units["B1"]["initial_voltage"]["total_increment_mV"] == units["B2"]["initial_voltage"]["total_increment_mV"]
        and units["B1"]["initial_voltage"]["histogram_counts"] == units["B2"]["initial_voltage"]["histogram_counts"])
    # external input replay across arms (D2) and difference across seeds (D3)
    seg = {a: units[a]["external_input_digest"]["segments"] for a in units}
    cmp_01 = compare_input_digests(seg["B0"], seg["B1"])
    cmp_02 = compare_input_digests(seg["B0"], seg["B2"])
    cmp_other = compare_input_digests(seg["B0"], other["external_input_digest"]["segments"])
    checks["external_inputs_identical_B0_B1_all_segments"] = bool(
        cmp_01["all_complete_segments_equal"] and not cmp_01["prefix_only"])
    checks["external_inputs_identical_B0_B2_all_segments"] = bool(
        cmp_02["all_complete_segments_equal"] and not cmp_02["prefix_only"])
    checks["external_inputs_differ_across_dynamics_seeds"] = not cmp_other["all_complete_segments_equal"]
    ext_sums = {a: np.load(qual_dir / f"qual_{stage['topology_seed']}_dyn_{seeds[0]}_{a}.npz")["ext_segment_sums"]
                for a in units}
    checks["exact_poisson_segment_sums_identical_across_arms"] = bool(
        np.array_equal(ext_sums["B0"], ext_sums["B1"]) and np.array_equal(ext_sums["B0"], ext_sums["B2"]))
    checks["short_unit_digest_prefix_matches_direct_engine_run"] = bool(
        compare_input_digests(digest_b0_short, seg["B0"])["all_complete_segments_equal"])
    # the perturbation is effective: trajectories diverge, and we report when
    rate = {a: np.load(qual_dir / f"qual_{stage['topology_seed']}_dyn_{seeds[0]}_{a}.npz")["rate_E_hz_per_step"]
            for a in units}
    def first_divergence_ms(a, b):
        diff = np.flatnonzero(rate[a] != rate[b])
        return None if diff.size == 0 else float(diff[0] * dt)
    divergence = {"B1_vs_B0_ms": first_divergence_ms("B1", "B0"),
                  "B2_vs_B0_ms": first_divergence_ms("B2", "B0"),
                  "B1_vs_B2_ms": first_divergence_ms("B1", "B2")}
    checks["initial_state_changes_the_trajectory"] = all(v is not None for v in divergence.values())
    # observation from 0 s with 500 ms burn-in inside the frozen observer
    checks["raw_acquisition_starts_at_zero"] = bool(
        units["B0"]["simulation"]["actual_duration_ms"] == args.unit_ms
        and units["B0"]["observation"]["burnin_ms"] == 500.0)
    checks["no_runaway_in_short_units"] = all(u["physical_status"].startswith("COMPLETE") for u in units.values())

    # ---- resource estimate ----
    unit_wall = float(np.mean([units[a]["simulation"]["wall_seconds"] for a in units]))
    sim_wall = float(np.mean([units[a]["simulation"]["simulation_wall_seconds"] for a in units]))
    formal_ms = float(design["simulation"]["duration_ms"])
    npz_bytes = int(np.mean([(qual_dir / f"qual_{stage['topology_seed']}_dyn_{seeds[0]}_{a}.npz").stat().st_size
                             for a in units]))
    resources = {
        "substrate_build_seconds": build_seconds,
        "short_run_ms": args.short_ms, "short_run_wall_seconds": short_wall,
        "unit_ms": args.unit_ms, "unit_wall_seconds_mean": unit_wall,
        "unit_simulation_wall_seconds_mean": sim_wall,
        "projected_formal_simulation_seconds": sim_wall * formal_ms / args.unit_ms,
        "projected_formal_unit_seconds": sim_wall * formal_ms / args.unit_ms + (unit_wall - sim_wall) * formal_ms / args.unit_ms + build_seconds,
        "projected_screen_worker_hours_36_units": 36 * (sim_wall * formal_ms / args.unit_ms + (unit_wall - sim_wall) * formal_ms / args.unit_ms + build_seconds) / 3600,
        "peak_rss_gib_this_process": rt.peak_rss_gib(),
        "short_unit_npz_bytes": npz_bytes,
        "projected_formal_npz_bytes": int(npz_bytes * formal_ms / args.unit_ms),
        "note": "projection is linear in duration; not a wall-clock guarantee",
    }
    passed = all(checks.values())
    report = {
        "status": "I0_QUALIFICATION_PASS" if passed else "I0_QUALIFICATION_FAIL",
        "created_unix": time.time(), "checks": checks, "divergence_ms": divergence,
        "digest_comparisons": {"B0_B1": cmp_01, "B0_B2": cmp_02, "B0_other_seed": cmp_other},
        "resources": resources, "frozen_manifest_sha256": rt.sha(frozen_path),
        "scientific_sample": False, "n_short_units": 4,
        "unit_outputs": [str(p) for p in sorted(qual_dir.glob("qual_*.json"))],
        "membership_K": k, "increment_mV": inc,
        "rng_note": ("initial voltage and observer consume no simulation RNG: the "
                     "None-vs-explicit and observer on/off checks are byte-identical "
                     "and the per-segment digests are equal across arms"),
    }
    rt.write(out / "qualification.json", report)
    print({"status": report["status"], "failed": [k for k, v in checks.items() if not v],
           "projected_formal_unit_minutes": resources["projected_formal_unit_seconds"] / 60,
           "peak_rss_gib": resources["peak_rss_gib_this_process"]})


if __name__ == "__main__":
    main()
