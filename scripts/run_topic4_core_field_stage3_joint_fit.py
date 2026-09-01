"""Fit the free-centre field to the frozen rev6 joint rank-curve distance.

This is a new optimization namespace. It does not resume or reinterpret the
first-round one-dimensional-TV checkpoint. Physical field parameters are decoded
from standardized latent coordinates, and candidates below the usable-event gate
retain a graded feasibility key instead of collapsing into one dead-zone score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _evaluate  # noqa: E402
from scripts.run_topic4_core_field_stage3_fit import _load_cmrun  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.topic4_core_field_cmaes import CMAES  # noqa: E402
from src.topic4_core_field_profile import (MIN_PARTICIPANTS,  # noqa: E402
                                           OBJECTIVE_N_EVENTS,
                                           fixed_count_sliced_distance,
                                           rank_curve_table)
from src.topic4_core_field_runner import (_placement, atomic_write_json,  # noqa: E402
                                          canonical_checksum, get_network,
                                          provenance)
from src.topic4_core_field_stage3 import latent_to_theta, n_free  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
REFERENCE_PATH = f"{ROOT}/joint_observable/rank_curve_reference.npz"
OUT = f"{ROOT}/joint_fit"
SEED_POOL = tuple(range(601, 681))
MIN_PROFILE_EVENTS = OBJECTIVE_N_EVENTS
SIGMA0 = 0.65
DEFAULT_POPSIZE = 16
OBJECTIVE_ID = "rev6_joint_rank_curve_fixed_n20_v2"
INIT_ID = "axis_free_space_filling_v1"


def load_reference(path=REFERENCE_PATH):
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in (
        "center", "components", "score_center", "score_scale",
        "reference_z", "directions")}


def candidate_fitness(distance, n_usable, participant_credit,
                      min_events=MIN_PROFILE_EVENTS):
    """Lexicographic key, larger is better, with a non-flat dead zone."""
    n_usable = int(n_usable)
    participant_credit = float(participant_credit)
    if n_usable >= int(min_events) and np.isfinite(distance):
        return (1.0, -float(distance), float(n_usable))
    return (0.0, float(n_usable), participant_credit)


def score_candidate(results, axial, reference, min_events=MIN_PROFILE_EVENTS):
    good = [row for row in results if "error" not in row]
    events = [event for row in good for event in row.get("events", [])]
    curves = rank_curve_table(events, axial)
    distance = (fixed_count_sliced_distance(curves, reference, min_events)
                if len(curves) >= int(min_events) else float("nan"))
    participant_credit = float(sum(row.get("participant_credit", 0.0) for row in good))
    n_detected = int(sum(row.get("n_detected", len(row.get("events", []))) for row in good))
    max_n_part = int(max((row.get("max_n_part", 0) for row in good), default=0))
    row = dict(
        distance=(None if not np.isfinite(distance) else float(distance)),
        n_usable=int(len(curves)),
        n_objective=int(min_events) if len(curves) >= int(min_events) else 0,
        n_detected=n_detected,
        participant_credit=participant_credit,
        max_n_part=max_n_part,
        n_networks=int(len(good)),
        n_failed=int(len(results) - len(good)),
    )
    return candidate_fitness(distance, len(curves), participant_credit, min_events), row


def _initial_latent(K, restart):
    """Axis-free, space-filling restart without patient-position information.

    The first pilot drew every centre close to the sheet midpoint, where most
    candidates produced no usable event. Components now start on a rotated
    regular polygon. The rotation and radius vary deterministically by restart;
    neither the frozen patient axis nor any previous fit enters the initializer.
    """
    K, restart = int(K), int(restart)
    rng = np.random.default_rng(31000 + 100 * K + restart)
    z = np.zeros(n_free(K), float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    rotation = (restart * golden_angle + rng.uniform(0.0, 2.0 * np.pi)) % (2.0 * np.pi)
    radius_fractions = (0.58, 0.70, 0.80)
    radius_fraction = radius_fractions[restart % len(radius_fractions)]
    latent_radius = 1.5 * np.arctanh(radius_fraction)

    # sigma ~= 1.1 mm after the logistic decoder: compact enough to ignite,
    # but isotropic so the initializer does not impose a propagation direction.
    sigma_unit = ((np.log(1.1) - np.log(0.4)) /
                  (np.log(6.0) - np.log(0.4)))
    sigma_latent = np.log(sigma_unit / (1.0 - sigma_unit))
    for k in range(K):
        b = 5 * k
        angle = rotation + 2.0 * np.pi * k / K
        z[b:b + 2] = latent_radius * np.array([np.cos(angle), np.sin(angle)])
        z[b + 2:b + 4] = sigma_latent + rng.normal(0.0, 0.08, size=2)
        z[b + 4] = rng.normal(0.0, 0.20)
    if K > 1:
        z[5 * K:] = 0.0
    return z


def _reference_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _numeric_contract_sha256(prov=None):
    """Hash the exact modules that can change candidate numbers."""
    prov = provenance() if prov is None else prov
    return canonical_checksum(prov["module_sha256"], drop=())


def _run_contract(args, reference_sha, config_checksum, numeric_contract_sha):
    return dict(
        objective_id=OBJECTIVE_ID,
        initializer_id=INIT_ID,
        K=int(args.K),
        restart=int(args.restart),
        popsize=int(args.popsize),
        seeds_per_candidate=int(args.seeds_per_candidate),
        min_events=int(args.min_events),
        reference_sha256=str(reference_sha),
        config_checksum=str(config_checksum),
        numeric_contract_sha256=str(numeric_contract_sha),
    )


def _resume_mismatches(checkpoint, expected_contract):
    actual = checkpoint.get("run_contract")
    if not isinstance(actual, dict):
        return ["run_contract missing (legacy or incompatible checkpoint)"]
    return [f"{key}: checkpoint={actual.get(key)!r}, current={value!r}"
            for key, value in expected_contract.items() if actual.get(key) != value]


def _unique_seed_cache_jobs(seeds, cfg, cache_dir):
    """One cache build per distinct network, independent of population size."""
    return [(int(seed), cfg, cache_dir) for seed in dict.fromkeys(seeds)]


def _precache_network(job):
    seed, cfg, cache_dir = job
    try:
        engine_path = os.path.join("src", "snn_engine")
        if engine_path not in sys.path:
            sys.path.insert(0, engine_path)
        from params import Params

        cmrun = _load_cmrun()
        e = cfg["engine"]
        reg = _placement(cfg)
        p = Params(g=e["g"], L=e["L"], density=e["density"],
                   T=cfg["duration_ms"], dt=e["dt"],
                   nu_ext_ratio=cmrun.DRIVE, seed=int(seed))
        _, _, _, hit = get_network(p, reg["theta_deg"], e["AR"], cache_dir)
        return dict(seed=int(seed), cache_hit=bool(hit))
    except Exception as exc:  # noqa: BLE001
        return dict(seed=int(seed), error=repr(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--restart", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seeds-per-candidate", type=int, default=2)
    ap.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    ap.add_argument("--max-gens", type=int, default=16)
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--min-events", type=int, choices=(OBJECTIVE_N_EVENTS,),
                    default=MIN_PROFILE_EVENTS)
    ap.add_argument("--pilot-stop-dead-fraction", type=float, default=0.50)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    axial = axial_map()
    reference = load_reference()
    reference_sha = _reference_sha256(REFERENCE_PATH)
    prov = provenance()
    numeric_contract_sha = _numeric_contract_sha256(prov)
    run_contract = _run_contract(args, reference_sha, cfg["checksum"],
                                 numeric_contract_sha)
    tag = f"K{args.K}_r{args.restart}"
    ck_path = os.path.join(args.out, f"checkpoint_{tag}.json")
    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(ck_path):
        checkpoint = json.load(open(ck_path))
        mismatches = _resume_mismatches(checkpoint, run_contract)
        if mismatches:
            raise SystemExit("checkpoint contract mismatch; refusing to resume:\n- "
                             + "\n- ".join(mismatches))
        es = CMAES.from_state(checkpoint["optimizer"])
        history = checkpoint["history"]
        generation_summary = checkpoint["generation_summary"]
        print(f"[{tag}] resuming at generation {es.generation}", flush=True)
    else:
        es = CMAES(_initial_latent(args.K, args.restart), SIGMA0,
                   seed=32000 + 100 * args.K + args.restart,
                   popsize=args.popsize)
        history, generation_summary = [], []
        print(f"[{tag}] fresh {INIT_ID} start, dim={n_free(args.K)}, "
              f"popsize={args.popsize}", flush=True)

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    cache = os.path.join(STAGE2, "network_cache")
    started = time.time()
    stop_reason = None
    while es.generation < args.max_gens and (time.time() - started) / 3600.0 < args.hours:
        latent = es.ask()
        physical = [latent_to_theta(z, args.K, float(cfg["engine"]["L"])) for z in latent]
        rng = np.random.default_rng(33000 + 100 * args.K + es.generation)
        seeds = [int(x) for x in rng.choice(
            SEED_POOL, size=args.seeds_per_candidate, replace=False)]
        cache_jobs = _unique_seed_cache_jobs(seeds, cfg, cache)
        with Pool(min(args.workers, len(cache_jobs)), maxtasksperchild=1) as pool:
            cache_rows = pool.map(_precache_network, cache_jobs)
        cache_errors = [row for row in cache_rows if "error" in row]
        if cache_errors:
            raise RuntimeError(f"network cache prewarm failed: {cache_errors}")
        jobs = [(theta, seed, cfg, cache, args.K, MIN_PARTICIPANTS)
                for theta in physical for seed in seeds]
        with Pool(args.workers, maxtasksperchild=1) as pool:
            raw = pool.map(_evaluate, jobs)

        keys, rows = [], []
        for i, (z, theta) in enumerate(zip(latent, physical)):
            chunk = raw[i * len(seeds):(i + 1) * len(seeds)]
            key, row = score_candidate(chunk, axial, reference, args.min_events)
            row.update(latent=[float(x) for x in z],
                       theta=[float(x) for x in theta], seeds=seeds,
                       generation=int(es.generation))
            keys.append(key)
            rows.append(row)
        es.tell(latent, keys)
        history.extend(rows)

        feasible = [row for row in rows if row["distance"] is not None]
        best = min(feasible, key=lambda row: row["distance"]) if feasible else max(
            rows, key=lambda row: (row["n_usable"], row["participant_credit"]))
        summary = dict(
            generation=int(es.generation),
            seeds=seeds,
            network_cache_hits=int(sum(row["cache_hit"] for row in cache_rows)),
            network_cache_builds=int(sum(not row["cache_hit"] for row in cache_rows)),
            sigma=float(es.sigma),
            feasible_fraction=float(len(feasible) / len(rows)),
            zero_usable_fraction=float(np.mean([row["n_usable"] == 0 for row in rows])),
            median_usable=float(np.median([row["n_usable"] for row in rows])),
            best_distance=(None if not feasible else float(best["distance"])),
            best_n_usable=int(best["n_usable"]),
            best_participant_credit=float(best["participant_credit"]),
        )
        generation_summary.append(summary)
        print(f"[{tag}] gen {es.generation:2d} sigma={es.sigma:.3f} "
              f"feasible={summary['feasible_fraction']:.0%} "
              f"zero={summary['zero_usable_fraction']:.0%} "
              f"median-events={summary['median_usable']:.1f} "
              f"best={summary['best_distance']}", flush=True)

        atomic_write_json(dict(
            objective=OBJECTIVE_ID, initializer=INIT_ID,
            run_contract=run_contract,
            K=int(args.K), restart=int(args.restart), popsize=int(args.popsize),
            seeds_per_candidate=int(args.seeds_per_candidate),
            min_events=int(args.min_events), reference_path=REFERENCE_PATH,
            reference_sha256=reference_sha, config_checksum=cfg["checksum"],
            optimizer=es.get_state(), history=history,
            generation_summary=generation_summary, stop_reason=stop_reason,
            provenance=prov), ck_path)

        if len(generation_summary) >= 3:
            recent = generation_summary[-3:]
            if all(row["zero_usable_fraction"] > args.pilot_stop_dead_fraction
                   for row in recent):
                stop_reason = ("three consecutive generations exceeded the frozen "
                               "zero-usable dead-zone threshold")
                print(f"[{tag}] STOP: {stop_reason}", flush=True)
                break

    if stop_reason is None and es.generation >= args.max_gens:
        stop_reason = "max_generations"
    elif stop_reason is None:
        stop_reason = "wall_clock_limit"
    checkpoint = json.load(open(ck_path))
    checkpoint["stop_reason"] = stop_reason
    atomic_write_json(checkpoint, ck_path)
    print(f"[{tag}] finished: {stop_reason}; checkpoint {ck_path}")


if __name__ == "__main__":
    main()
