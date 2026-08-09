"""Optimize a free field against rev6 distance plus rev8 patient-mode structure.

Only patient training recording blocks enter this objective.  Patient held-out
blocks and the final unseen-network seed pool are not read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _evaluate, _load_cmrun  # noqa: E402
from scripts.run_topic4_core_field_stage3_joint_fit import (  # noqa: E402
    INIT_ID,
    REFERENCE_PATH,
    _initial_latent,
    _numeric_contract_sha256,
    _precache_network,
    _reference_sha256,
    _resume_mismatches,
    _unique_seed_cache_jobs,
    load_reference,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.topic4_core_field_cmaes import CMAES  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    MIN_PARTICIPANTS,
    MODE_LOSS_WEIGHT,
    MODE_MIN_CLUSTER_EVENTS,
    MODE_OBJECTIVE_N_EVENTS,
    OBJECTIVE_N_EVENTS,
    fixed_count_kmeans_mode_loss,
    fixed_count_sliced_distance,
    rank_curve_table,
)
from src.topic4_core_field_runner import (  # noqa: E402
    atomic_write_json,
    canonical_checksum,
    provenance,
)
from src.topic4_core_field_stage3 import latent_to_theta, n_free  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
TARGET_PATH = (
    f"{ROOT}/joint_kmeans_training_target_rev8/"
    "patient_train_kmeans_target_rev8.npz"
)
OUT = f"{ROOT}/joint_fit_kmeans_rev8"
OBJECTIVE_ID = "rev8_curve_plus_patient_train_kmeans_v1"
REV8_INIT_ID = f"{INIT_ID}_rev8"
REV81_OBJECTIVE_ID = "rev8_1_curve_plus_patient_train_kmeans_weight2_v1"
REV81_INIT_ID = "rev8_1_training_elite_mode_warm_start_v1"
REV81_MODE_LOSS_WEIGHT = 2.0
TRAIN_SEED_POOL = tuple(range(701, 761))
SELECTION_SEED_POOL = tuple(range(761, 767))
FINAL_CONFIRM_SEED_POOL = tuple(range(801, 807))
DEFAULT_POPSIZE = 12
SIGMA0 = 0.55
# KMeans/BLAS executes in the parent between generations.  Forking after that
# can inherit a locked native thread-pool state, so every worker generation must
# start from a fresh interpreter.
WORKER_CONTEXT = get_context("spawn")


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_training_target(path=TARGET_PATH):
    data = np.load(path)
    required = (
        "patient_train_mode_prototypes",
        "patient_train_mode_counts",
        "patient_train_target_similarity_matrix",
        "grid",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(f"rev8 training target is missing {missing}")
    return {key: np.asarray(data[key]) for key in required}


def rev8_candidate_fitness(
        distance, mode, n_usable, participant_credit,
        mode_loss_weight=MODE_LOSS_WEIGHT, mode_sign_tier=False):
    """Lexicographic feasibility followed by the frozen scalar objective."""
    n_usable = int(n_usable)
    participant_credit = float(participant_credit)
    if n_usable < OBJECTIVE_N_EVENTS or not np.isfinite(distance):
        return (0.0, float(n_usable), participant_credit)
    if n_usable < MODE_OBJECTIVE_N_EVENTS or mode.get("status") != "ok":
        return (1.0, float(n_usable), -float(distance), participant_credit)

    mode_loss = float(mode["mode_matrix_loss"])
    min_cluster = int(mode["min_cluster_count"])
    if not bool(mode["support_eligible"]):
        return (2.0, float(min_cluster), -mode_loss, -float(distance))
    joint_loss = float(distance + float(mode_loss_weight) * mode_loss)
    tier = (4.0 if bool(mode_sign_tier)
            and bool(mode.get("matrix_sign_consistent")) else 3.0)
    return (tier, -joint_loss, -float(distance), -mode_loss)


def _mode_summary(mode):
    keys = (
        "status", "n_events", "required_events", "required_events_per_cluster",
        "cluster_counts", "min_cluster_count", "minority_fraction",
        "support_eligible", "prototype_correlation", "similarity_matrix",
        "matched_correlations", "crossed_correlations", "matched_mean",
        "crossed_mean", "matrix_contrast", "matrix_sign_consistent",
        "target_similarity_matrix", "mode_matrix_loss",
    )
    out = {}
    for key in keys:
        if key not in mode:
            continue
        value = mode[key]
        out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


def score_candidate(
        results, axial, reference, data_prototypes,
        mode_loss_weight=MODE_LOSS_WEIGHT, mode_sign_tier=False):
    good = [row for row in results if "error" not in row]
    events = [event for row in good for event in row.get("events", [])]
    curves = rank_curve_table(events, axial)
    distance = (
        fixed_count_sliced_distance(curves, reference, OBJECTIVE_N_EVENTS)
        if len(curves) >= OBJECTIVE_N_EVENTS else float("nan")
    )
    mode = fixed_count_kmeans_mode_loss(
        curves, data_prototypes, reference,
        n_events=MODE_OBJECTIVE_N_EVENTS,
        min_cluster_events=MODE_MIN_CLUSTER_EVENTS,
    )
    participant_credit = float(sum(
        row.get("participant_credit", 0.0) for row in good))
    row = dict(
        distance=None if not np.isfinite(distance) else float(distance),
        mode=_mode_summary(mode),
        joint_loss=(
            None if not np.isfinite(distance) or mode.get("status") != "ok"
            else float(distance + float(mode_loss_weight)
                       * mode["mode_matrix_loss"])
        ),
        mode_loss_weight=float(mode_loss_weight),
        mode_sign_tier=bool(mode_sign_tier),
        n_usable=int(len(curves)),
        n_detected=int(sum(
            value.get("n_detected", len(value.get("events", []))) for value in good)),
        participant_credit=participant_credit,
        max_n_part=int(max((value.get("max_n_part", 0) for value in good), default=0)),
        n_networks=int(len(good)),
        n_failed=int(len(results) - len(good)),
    )
    return rev8_candidate_fitness(
        distance, mode, len(curves), participant_credit,
        mode_loss_weight=mode_loss_weight,
        mode_sign_tier=mode_sign_tier), row


def training_elite_warm_start(checkpoint_path, K, mode_loss_weight):
    """Choose a training-only, cluster-supported elite under the new scalar."""
    if checkpoint_path is None:
        return None, None
    checkpoint = json.load(open(checkpoint_path))
    if int(checkpoint.get("K", -1)) != int(K):
        raise RuntimeError("warm-start checkpoint K does not match the requested K")
    eligible = [
        row for row in checkpoint.get("history", [])
        if row.get("distance") is not None
        and row.get("mode", {}).get("support_eligible")
        and row.get("mode", {}).get("mode_matrix_loss") is not None
    ]
    if not eligible:
        raise RuntimeError("warm-start checkpoint has no supported training elite")
    weight = float(mode_loss_weight)
    selected = min(eligible, key=lambda row: (
        float(row["distance"]) + weight * float(row["mode"]["mode_matrix_loss"]),
        float(row["mode"]["mode_matrix_loss"]),
        float(row["distance"]),
    ))
    latent = np.asarray(selected["latent"], float)
    if latent.size != n_free(K) or not np.isfinite(latent).all():
        raise RuntimeError("warm-start elite has an invalid latent vector")
    descriptor = dict(
        path=str(checkpoint_path), sha256=_sha256(checkpoint_path),
        source_objective=checkpoint.get("objective"),
        source_git_commit=checkpoint.get("provenance", {}).get("git_commit"),
        source_generation=int(selected["generation"]),
        source_distance=float(selected["distance"]),
        source_mode_loss=float(selected["mode"]["mode_matrix_loss"]),
        source_joint_loss=float(
            selected["distance"] + weight * selected["mode"]["mode_matrix_loss"]),
        source_cluster_counts=[int(value) for value in
                               selected["mode"]["cluster_counts"]],
    )
    return latent, descriptor


def validate_objective_configuration(args):
    """Bind revision IDs to their scientific scoring contracts."""
    if args.objective_id == OBJECTIVE_ID:
        expected = dict(
            initializer_id=REV8_INIT_ID,
            mode_loss_weight=MODE_LOSS_WEIGHT,
            mode_sign_tier=False,
            warm_start_checkpoint=None,
        )
    elif args.objective_id == REV81_OBJECTIVE_ID:
        expected = dict(
            initializer_id=REV81_INIT_ID,
            mode_loss_weight=REV81_MODE_LOSS_WEIGHT,
            mode_sign_tier=True,
        )
        if int(args.K) != 3 or not args.warm_start_checkpoint:
            raise SystemExit("rev8.1 requires K=3 and a training checkpoint warm start")
    else:
        raise SystemExit(f"unknown objective-id: {args.objective_id}")
    mismatches = []
    for key, value in expected.items():
        actual = getattr(args, key)
        matches = (np.isclose(actual, value) if isinstance(value, float)
                   else actual == value)
        if not matches:
            mismatches.append(f"{key}={actual!r}, expected {value!r}")
    if mismatches:
        raise SystemExit(
            "objective-id/scoring contract mismatch:\n- " + "\n- ".join(mismatches))


def _run_contract(args, reference_sha, target_sha, config_checksum,
                  numeric_contract_sha, warm_start):
    return dict(
        objective_id=str(args.objective_id),
        initializer_id=str(args.initializer_id),
        K=int(args.K),
        restart=int(args.restart),
        popsize=int(args.popsize),
        seeds_per_candidate=int(args.seeds_per_candidate),
        distance_event_count=OBJECTIVE_N_EVENTS,
        mode_event_count=MODE_OBJECTIVE_N_EVENTS,
        min_events_per_mode=MODE_MIN_CLUSTER_EVENTS,
        mode_loss_weight=float(args.mode_loss_weight),
        mode_sign_tier=bool(args.mode_sign_tier),
        sigma0=float(args.sigma0),
        warm_start=warm_start,
        train_seed_pool=list(TRAIN_SEED_POOL),
        selection_seed_pool=list(SELECTION_SEED_POOL),
        final_confirm_seed_pool=list(FINAL_CONFIRM_SEED_POOL),
        reference_sha256=str(reference_sha),
        target_sha256=str(target_sha),
        config_checksum=str(config_checksum),
        numeric_contract_sha256=str(numeric_contract_sha),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, choices=(2, 3), required=True)
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seeds-per-candidate", type=int, choices=(4,), default=4)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--max-gens", type=int, default=4)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--pilot-stop-dead-fraction", type=float, default=0.50)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--objective-id", default=OBJECTIVE_ID)
    parser.add_argument("--initializer-id", default=REV8_INIT_ID)
    parser.add_argument("--mode-loss-weight", type=float, default=MODE_LOSS_WEIGHT)
    parser.add_argument("--mode-sign-tier", action="store_true")
    parser.add_argument("--sigma0", type=float, default=SIGMA0)
    parser.add_argument("--warm-start-checkpoint")
    args = parser.parse_args()
    if args.mode_loss_weight <= 0.0 or args.sigma0 <= 0.0:
        raise SystemExit("mode-loss-weight and sigma0 must be positive")
    validate_objective_configuration(args)

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    axial = axial_map()
    reference = load_reference(REFERENCE_PATH)
    target = load_training_target(TARGET_PATH)
    data_prototypes = target["patient_train_mode_prototypes"]
    reference_sha = _reference_sha256(REFERENCE_PATH)
    target_sha = _sha256(TARGET_PATH)
    prov = provenance()
    numeric_contract_sha = _numeric_contract_sha256(prov)
    warm_latent, warm_start = training_elite_warm_start(
        args.warm_start_checkpoint, args.K, args.mode_loss_weight)
    run_contract = _run_contract(
        args, reference_sha, target_sha, cfg["checksum"], numeric_contract_sha,
        warm_start)

    tag = f"K{args.K}_r{args.restart}"
    checkpoint_path = os.path.join(args.out, f"checkpoint_{tag}.json")
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(checkpoint_path):
        checkpoint = json.load(open(checkpoint_path))
        mismatches = _resume_mismatches(checkpoint, run_contract)
        if mismatches:
            raise SystemExit(
                "checkpoint contract mismatch; refusing to resume:\n- "
                + "\n- ".join(mismatches))
        optimizer = CMAES.from_state(checkpoint["optimizer"])
        history = checkpoint["history"]
        generation_summary = checkpoint["generation_summary"]
        print(f"[{tag}] resuming at generation {optimizer.generation}", flush=True)
    else:
        initial_latent = (_initial_latent(args.K, args.restart)
                          if warm_latent is None else warm_latent)
        optimizer = CMAES(
            initial_latent, float(args.sigma0),
            seed=42000 + 100 * args.K + args.restart,
            popsize=args.popsize,
        )
        history, generation_summary = [], []
        print(
            f"[{tag}] fresh {args.initializer_id}, dim={n_free(args.K)}, "
            f"popsize={args.popsize}", flush=True)

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    cache = os.path.join(STAGE2, "network_cache")
    started = time.time()
    stop_reason = None
    while (optimizer.generation < args.max_gens
           and (time.time() - started) / 3600.0 < args.hours):
        latent = optimizer.ask()
        physical = [
            latent_to_theta(value, args.K, float(cfg["engine"]["L"]))
            for value in latent
        ]
        rng = np.random.default_rng(
            43000 + 100 * args.K + 10 * args.restart + optimizer.generation)
        seeds = [int(value) for value in rng.choice(
            TRAIN_SEED_POOL, size=args.seeds_per_candidate, replace=False)]
        cache_jobs = _unique_seed_cache_jobs(seeds, cfg, cache)
        with WORKER_CONTEXT.Pool(
                min(args.workers, len(cache_jobs)), maxtasksperchild=1) as pool:
            cache_rows = pool.map(_precache_network, cache_jobs)
        errors = [row for row in cache_rows if "error" in row]
        if errors:
            raise RuntimeError(f"network cache prewarm failed: {errors}")
        jobs = [
            (theta, seed, cfg, cache, args.K, MIN_PARTICIPANTS)
            for theta in physical for seed in seeds
        ]
        with WORKER_CONTEXT.Pool(args.workers, maxtasksperchild=1) as pool:
            raw = pool.map(_evaluate, jobs)

        keys, rows = [], []
        for index, (latent_value, theta) in enumerate(zip(latent, physical)):
            chunk = raw[
                index * len(seeds):(index + 1) * len(seeds)]
            key, row = score_candidate(
                chunk, axial, reference, data_prototypes,
                mode_loss_weight=args.mode_loss_weight,
                mode_sign_tier=args.mode_sign_tier)
            row.update(
                latent=[float(value) for value in latent_value],
                theta=[float(value) for value in theta],
                seeds=seeds,
                generation=int(optimizer.generation),
                fitness_key=[float(value) for value in key],
            )
            keys.append(key)
            rows.append(row)
        optimizer.tell(latent, keys)
        history.extend(rows)

        eligible = [row for row in rows if row["mode"].get("support_eligible")]
        best = max(rows, key=lambda row: tuple(row["fitness_key"]))
        summary = dict(
            generation=int(optimizer.generation),
            seeds=seeds,
            network_cache_hits=int(sum(row["cache_hit"] for row in cache_rows)),
            network_cache_builds=int(sum(not row["cache_hit"] for row in cache_rows)),
            sigma=float(optimizer.sigma),
            distance_feasible_fraction=float(np.mean([
                row["distance"] is not None for row in rows])),
            mode_supported_fraction=float(len(eligible) / len(rows)),
            mode_sign_consistent_fraction=float(np.mean([
                bool(row["mode"].get("matrix_sign_consistent"))
                for row in rows])),
            mode_sign_supported_fraction=float(np.mean([
                bool(row["mode"].get("support_eligible"))
                and bool(row["mode"].get("matrix_sign_consistent"))
                for row in rows])),
            zero_usable_fraction=float(np.mean([
                row["n_usable"] == 0 for row in rows])),
            median_usable=float(np.median([row["n_usable"] for row in rows])),
            best_joint_loss=best["joint_loss"],
            best_distance=best["distance"],
            best_mode_loss=best["mode"].get("mode_matrix_loss"),
            best_min_cluster_count=best["mode"].get("min_cluster_count"),
        )
        generation_summary.append(summary)
        print(
            f"[{tag}] gen {optimizer.generation:2d} sigma={optimizer.sigma:.3f} "
            f"D-feasible={summary['distance_feasible_fraction']:.0%} "
            f"mode-supported={summary['mode_supported_fraction']:.0%} "
            f"sign+supported={summary['mode_sign_supported_fraction']:.0%} "
            f"median-events={summary['median_usable']:.1f} "
            f"best-joint={summary['best_joint_loss']}", flush=True)

        atomic_write_json(dict(
            objective=str(args.objective_id),
            initializer=str(args.initializer_id),
            run_contract=run_contract,
            K=int(args.K), restart=int(args.restart), popsize=int(args.popsize),
            seeds_per_candidate=int(args.seeds_per_candidate),
            reference_path=REFERENCE_PATH, target_path=TARGET_PATH,
            reference_sha256=reference_sha, target_sha256=target_sha,
            config_checksum=cfg["checksum"],
            optimizer=optimizer.get_state(), history=history,
            generation_summary=generation_summary,
            stop_reason=stop_reason, warm_start=warm_start, provenance=prov,
        ), checkpoint_path)

        if len(generation_summary) >= 3:
            recent = generation_summary[-3:]
            if all(row["zero_usable_fraction"] > args.pilot_stop_dead_fraction
                   for row in recent):
                stop_reason = (
                    "three consecutive generations exceeded the frozen "
                    "zero-usable dead-zone threshold")
                print(f"[{tag}] STOP: {stop_reason}", flush=True)
                break

    if stop_reason is None and optimizer.generation >= args.max_gens:
        stop_reason = "max_generations"
    elif stop_reason is None:
        stop_reason = "wall_clock_limit"
    checkpoint = json.load(open(checkpoint_path))
    checkpoint["stop_reason"] = stop_reason
    atomic_write_json(checkpoint, checkpoint_path)
    print(f"[{tag}] finished: {stop_reason}; checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
