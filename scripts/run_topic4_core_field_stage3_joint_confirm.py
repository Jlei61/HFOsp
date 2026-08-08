"""Re-evaluate preselected rev6 field candidates on unseen network seeds.

Confirmation measures transfer; it does not choose a winner. The training
global best, final-generation best, and final CMA mean are frozen before any
confirmation simulation. Every distance uses exactly 20 model events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from scripts.calibrate_topic4_core_field_stage3_joint_observable import (  # noqa: E402
    DISTANCE_BOOTSTRAP_SEED,
    HELD_OUT_FRAC,
    N_DISTANCE_BOOTSTRAP,
    SPLIT_SEED,
    _matched_distance,
    _model_curves,
    _patient_curves,
    _prototype_diagnostic,
)
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _evaluate  # noqa: E402
from scripts.run_topic4_core_field_stage3_joint_fit import (  # noqa: E402
    MIN_PROFILE_EVENTS,
    REFERENCE_PATH,
    _precache_network,
    _unique_seed_cache_jobs,
    load_reference,
    score_candidate,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    PROFILE_REFERENCE_N,
    fixed_count_indices,
    rank_curve_table,
    sliced_embedding_distance,
    split_by_block,
    transform_rank_curves,
)
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_core_field_stage3 import latent_to_theta  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
CHECKPOINT = f"{ROOT}/joint_fit_clean_pilot_rev6/checkpoint_K3_r0.json"
CALIBRATION = f"{ROOT}/joint_observable/calibration_summary.json"
OUT = f"{ROOT}/joint_confirmation_pilot_rev6.json"
CONFIRM_SEED_POOL = tuple(range(501, 560))
HELDOUT_REFERENCE_SEED = 20260815


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def select_candidates(checkpoint, sheet_length):
    """Freeze candidate roles before confirmation and merge exact duplicates."""
    feasible = [row for row in checkpoint["history"]
                if row.get("distance") is not None]
    if not feasible:
        raise ValueError("checkpoint contains no feasible candidate")
    global_best = min(feasible, key=lambda row: row["distance"])
    final_generation = max(int(row["generation"]) for row in checkpoint["history"])
    final_feasible = [row for row in feasible
                      if int(row["generation"]) == final_generation]
    if not final_feasible:
        raise ValueError("final generation contains no feasible candidate")
    final_best = min(final_feasible, key=lambda row: row["distance"])
    mean_theta = latent_to_theta(
        checkpoint["optimizer"]["mean"], int(checkpoint["K"]), sheet_length)

    raw = [
        ("training_global_best", np.asarray(global_best["theta"], float), global_best),
        ("final_generation_best", np.asarray(final_best["theta"], float), final_best),
        ("final_optimizer_mean", np.asarray(mean_theta, float), None),
    ]
    selected = []
    for role, theta, source in raw:
        duplicate = next((row for row in selected
                          if np.array_equal(theta, np.asarray(row["theta"]))), None)
        if duplicate is not None:
            duplicate["roles"].append(role)
            continue
        selected.append(dict(
            candidate_id=f"candidate_{len(selected)}",
            roles=[role],
            theta=theta.tolist(),
            theta_sha256=hashlib.sha256(theta.astype("<f8").tobytes()).hexdigest(),
            training=(None if source is None else dict(
                generation=int(source["generation"]),
                distance=float(source["distance"]),
                n_usable=int(source["n_usable"]),
                seeds=[int(seed) for seed in source["seeds"]],
            )),
        ))
    return selected


def confirmation_seeds(checkpoint, n_confirm):
    fit_seeds = {int(seed) for row in checkpoint["history"] for seed in row["seeds"]}
    selected = [seed for seed in CONFIRM_SEED_POOL if seed not in fit_seeds][
        :int(n_confirm)]
    if len(selected) != int(n_confirm) or set(selected) & fit_seeds:
        raise ValueError("could not construct an independent confirmation seed pool")
    return selected, sorted(fit_seeds)


def evaluation_errors(raw, candidates, seeds):
    """Attach candidate and network identity to every caught worker error."""
    rows = []
    for index, row in enumerate(raw):
        if "error" not in row:
            continue
        candidate_index, seed_index = divmod(index, len(seeds))
        rows.append(dict(
            candidate_id=candidates[candidate_index]["candidate_id"],
            roles=candidates[candidate_index]["roles"],
            seed=int(seeds[seed_index]),
            error=str(row["error"]),
        ))
    return rows


def _distance_to_target(curves, reference, target_z, n_events=MIN_PROFILE_EVENTS):
    index = fixed_count_indices(len(curves), n_events)
    if index is None:
        return None
    z = transform_rank_curves(np.asarray(curves)[index], reference)
    value = sliced_embedding_distance(z, target_z, reference["directions"])
    return None if not np.isfinite(value) else float(value)


def _bootstrap_to_target(curves, reference, target_z, seed,
                         n_events=MIN_PROFILE_EVENTS,
                         n_bootstrap=N_DISTANCE_BOOTSTRAP):
    curves = np.asarray(curves, float)
    if len(curves) < int(n_events):
        return None
    rng = np.random.default_rng(int(seed))
    values = np.asarray([
        sliced_embedding_distance(
            transform_rank_curves(
                curves[rng.choice(len(curves), size=int(n_events), replace=False)],
                reference),
            target_z,
            reference["directions"],
        )
        for _ in range(int(n_bootstrap))
    ])
    return dict(n_events=int(n_events), n_bootstrap=int(n_bootstrap), seed=int(seed),
                median=float(np.median(values)),
                p05=float(np.quantile(values, 0.05)),
                p95=float(np.quantile(values, 0.95)))


def _posthoc_diagnostic(curves, reference):
    row = _prototype_diagnostic(curves, reference)
    row.pop("prototypes", None)
    return row


def _control_distributions(calibration, patient_heldout, axial, grid, reference):
    curves = dict(patient_heldout=patient_heldout)
    for key in ("hand_placed_two_cores", "stage2_filament"):
        curves[key] = _model_curves(
            calibration["inputs"]["model_paths"][key], axial, grid)
    out = {}
    for index, (key, values) in enumerate(curves.items()):
        row = _matched_distance(
            values, reference, MIN_PROFILE_EVENTS,
            DISTANCE_BOOTSTRAP_SEED + 20 + index)
        row.pop("values")
        out[key] = row
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--reference", default=REFERENCE_PATH)
    parser.add_argument("--calibration", default=CALIBRATION)
    parser.add_argument("--n-confirm", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    checkpoint = json.load(open(args.checkpoint))
    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    reference = load_reference(args.reference)
    reference_file = np.load(args.reference)
    grid = np.asarray(reference_file["grid"], float)
    axial = axial_map()
    calibration = json.load(open(args.calibration))
    if _sha256(args.reference) != checkpoint["run_contract"]["reference_sha256"]:
        raise SystemExit("checkpoint and confirmation reference hashes differ")
    if cfg["checksum"] != checkpoint["run_contract"]["config_checksum"]:
        raise SystemExit("checkpoint and confirmation config checksums differ")

    candidates = select_candidates(checkpoint, float(cfg["engine"]["L"]))
    seeds, fit_seeds = confirmation_seeds(checkpoint, args.n_confirm)
    patient, blocks = _patient_curves(axial, grid)
    train_index, heldout_index = split_by_block(blocks, HELD_OUT_FRAC, SPLIT_SEED)
    patient_heldout = patient[heldout_index]
    heldout_z_all = transform_rank_curves(patient_heldout, reference)
    rng = np.random.default_rng(HELDOUT_REFERENCE_SEED)
    heldout_take = min(PROFILE_REFERENCE_N, len(heldout_z_all))
    heldout_z = heldout_z_all[
        rng.choice(len(heldout_z_all), size=heldout_take, replace=False)]
    controls = _control_distributions(
        calibration, patient_heldout, axial, grid, reference)

    cache = os.path.join(STAGE2, "network_cache")
    cache_jobs = _unique_seed_cache_jobs(seeds, cfg, cache)
    with Pool(min(args.workers, len(cache_jobs)), maxtasksperchild=1) as pool:
        cache_rows = pool.map(_precache_network, cache_jobs)
    errors = [row for row in cache_rows if "error" in row]
    if errors:
        raise RuntimeError(f"confirmation network cache prewarm failed: {errors}")

    jobs = [(candidate["theta"], seed, cfg, cache, int(checkpoint["K"]), 6)
            for candidate in candidates for seed in seeds]
    with Pool(args.workers, maxtasksperchild=1) as pool:
        raw = pool.map(_evaluate, jobs)
    simulation_errors = evaluation_errors(raw, candidates, seeds)
    if simulation_errors:
        atomic_write_json(dict(
            status="FAIL_CLOSED_SIMULATION_ERRORS",
            checkpoint=dict(path=args.checkpoint, sha256=_sha256(args.checkpoint)),
            confirm_network_seeds=seeds,
            errors=simulation_errors,
            provenance=provenance(),
        ), args.out)
        raise RuntimeError(
            f"confirmation failed closed with {len(simulation_errors)} simulation errors; "
            f"see {args.out}")

    result_rows = []
    for candidate_index, candidate in enumerate(candidates):
        chunk = raw[candidate_index * len(seeds):(candidate_index + 1) * len(seeds)]
        good = [row for row in chunk if "error" not in row]
        events = [event for row in good for event in row.get("events", [])]
        curves = rank_curve_table(events, axial, grid=grid)
        _, score = score_candidate(chunk, axial, reference, MIN_PROFILE_EVENTS)
        train_bootstrap = (None if len(curves) < MIN_PROFILE_EVENTS else
                           _matched_distance(
                               curves, reference, MIN_PROFILE_EVENTS,
                               DISTANCE_BOOTSTRAP_SEED + 100 + candidate_index))
        if train_bootstrap is not None:
            train_bootstrap.pop("values")
        loo = []
        for held_seed, held_row in zip(seeds, chunk):
            keep_events = [event for seed, row in zip(seeds, chunk)
                           if seed != held_seed and "error" not in row
                           for event in row.get("events", [])]
            keep_curves = rank_curve_table(keep_events, axial, grid=grid)
            loo.append(dict(
                held_seed=int(held_seed),
                n_events=int(len(keep_curves)),
                distance_patient_train=_distance_to_target(
                    keep_curves, reference, reference["reference_z"]),
            ))
        result_rows.append(dict(
            **candidate,
            confirm=dict(
                deterministic_distance_patient_train=score["distance"],
                bootstrap_distance_patient_train=train_bootstrap,
                deterministic_distance_patient_heldout=_distance_to_target(
                    curves, reference, heldout_z),
                bootstrap_distance_patient_heldout=_bootstrap_to_target(
                    curves, reference, heldout_z,
                    DISTANCE_BOOTSTRAP_SEED + 200 + candidate_index),
                n_usable=int(len(curves)),
                n_detected=int(score["n_detected"]),
                n_failed_networks=int(score["n_failed"]),
                event_count_by_seed={str(seed): int(len(row.get("events", [])))
                                     for seed, row in zip(seeds, chunk)},
                leave_one_network_out=loo,
                posthoc_prototypes=_posthoc_diagnostic(curves, reference),
            ),
        ))

    output = dict(
        status="UNSEEN_NETWORK_CONFIRMATION_MEASUREMENT_COMPLETE",
        scientific_role=("candidate transfer screen; candidates were frozen before confirmation, "
                         "but this is not K/restart identifiability or lifecycle acceptance"),
        checkpoint=dict(path=args.checkpoint, sha256=_sha256(args.checkpoint),
                        run_contract=checkpoint["run_contract"]),
        reference=dict(path=args.reference, sha256=_sha256(args.reference)),
        patient_split=dict(unit="recording block", frac=HELD_OUT_FRAC,
                           seed=SPLIT_SEED, n_train=int(len(train_index)),
                           n_heldout=int(len(heldout_index)),
                           heldout_reference_n=int(heldout_take),
                           heldout_reference_seed=HELDOUT_REFERENCE_SEED),
        fit_network_seeds=fit_seeds,
        confirm_network_seeds=seeds,
        network_cache=dict(hits=int(sum(row.get("cache_hit", False)
                                        for row in cache_rows)),
                           builds=int(sum(not row.get("cache_hit", False)
                                          for row in cache_rows))),
        objective_event_count=MIN_PROFILE_EVENTS,
        patient_floor_train=calibration["optimization_patient_floor"],
        optimization_controls_n20=controls,
        candidates=result_rows,
        limitations=[
            "single K=3 restart and three optimization generations",
            "six confirmation networks do not establish K or field identifiability",
            "event bootstrap does not model network-seed clustering; leave-one-network-out is reported separately",
            "post-hoc prototype opposition is a falsification diagnostic and was not optimized",
        ],
        provenance=provenance(),
    )
    atomic_write_json(output, args.out)
    print(f"wrote {args.out}")
    for row in result_rows:
        confirm = row["confirm"]
        print(f"{row['candidate_id']} {','.join(row['roles'])}: "
              f"n={confirm['n_usable']} train={confirm['deterministic_distance_patient_train']} "
              f"heldout={confirm['deterministic_distance_patient_heldout']} "
              f"prototype_r={confirm['posthoc_prototypes'].get('prototype_correlation')}")


if __name__ == "__main__":
    main()
