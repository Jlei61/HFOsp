"""Select and confirm the Stage 3 rev8 KMeans-assisted field.

``select`` evaluates a training-derived shortlist on a dedicated model seed
pool and never reads patient held-out curves.  ``confirm`` consumes the frozen
selection, opens the patient held-out recording blocks once, and evaluates one
candidate on a disjoint final network pool.
"""
from __future__ import annotations

import argparse
import glob
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
    OPPOSITION_MIN_CLUSTER_EVENTS,
    SPLIT_SEED,
    _patient_curves,
)
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _evaluate  # noqa: E402
from scripts.run_topic4_core_field_stage3_joint_confirm import (  # noqa: E402
    CALIBRATION,
    HELDOUT_REFERENCE_SEED,
    _atomic_npz,
    _bootstrap_to_target,
    _control_curve_sets,
    _control_distributions,
    _distance_to_target,
    _kmeans_robustness,
    _kmeans_summary,
    _posthoc_diagnostic,
    reconcile_confirmation,
)
from scripts.run_topic4_core_field_stage3_joint_fit import (  # noqa: E402
    REFERENCE_PATH,
    _precache_network,
    _reference_sha256,
    _unique_seed_cache_jobs,
    load_reference,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_core_field_stage3_rev8_kmeans_fit import (  # noqa: E402
    FINAL_CONFIRM_SEED_POOL,
    MODE_LOSS_WEIGHT,
    OBJECTIVE_ID,
    SELECTION_SEED_POOL,
    TARGET_PATH,
    load_training_target,
    score_candidate,
)
from src.topic4_core_field_profile import (  # noqa: E402
    MODE_MIN_CLUSTER_EVENTS,
    MODE_OBJECTIVE_N_EVENTS,
    OBJECTIVE_N_EVENTS,
    PROFILE_REFERENCE_N,
    fixed_count_kmeans_mode_loss,
    kmeans_data_consistency,
    normalized_rank_curve,
    split_by_block,
    transform_rank_curves,
)
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_core_field_stage3 import latent_to_theta  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
FIT_DIR = f"{ROOT}/joint_fit_kmeans_rev8"
OUT_DIR = f"{ROOT}/joint_confirmation_rev8"
SELECTION_OUT = f"{OUT_DIR}/selection.json"
CONFIRM_OUT = f"{OUT_DIR}/final_confirmation.json"
PROFILES_OUT = f"{OUT_DIR}/final_event_profiles.npz"


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _theta_sha256(theta):
    return hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()


def _candidate_rows(checkpoint_path, sheet_length, top_per_checkpoint=2):
    checkpoint = json.load(open(checkpoint_path))
    if checkpoint.get("objective") != OBJECTIVE_ID:
        raise RuntimeError(f"not a rev8 checkpoint: {checkpoint_path}")
    history = checkpoint.get("history", [])
    if not history:
        raise RuntimeError(f"empty checkpoint: {checkpoint_path}")
    ranked = sorted(
        history, key=lambda row: tuple(row.get("fitness_key", ())), reverse=True)
    selected = ranked[:int(top_per_checkpoint)]
    final_generation = max(int(row["generation"]) for row in history)
    final_rows = [row for row in ranked if int(row["generation"]) == final_generation]
    if final_rows:
        selected.append(final_rows[0])
    mean_theta = latent_to_theta(
        checkpoint["optimizer"]["mean"], int(checkpoint["K"]), sheet_length)

    rows = []
    for source_role, theta, source in [
            *[("training_ranked", row["theta"], row) for row in selected],
            ("final_optimizer_mean", mean_theta, None)]:
        rows.append(dict(
            K=int(checkpoint["K"]),
            restart=int(checkpoint["restart"]),
            theta=np.asarray(theta, float).tolist(),
            theta_sha256=_theta_sha256(theta),
            source_role=source_role,
            source_checkpoint=checkpoint_path,
            source_checkpoint_sha256=_sha256(checkpoint_path),
            source_training=(None if source is None else dict(
                generation=int(source["generation"]),
                joint_loss=source.get("joint_loss"),
                distance=source.get("distance"),
                mode_loss=source.get("mode", {}).get("mode_matrix_loss"),
                min_cluster_count=source.get("mode", {}).get("min_cluster_count"),
                n_usable=int(source["n_usable"]),
                seeds=[int(seed) for seed in source["seeds"]],
            )),
            fit_seeds=sorted({
                int(seed) for row in history for seed in row.get("seeds", [])
            }),
            run_contract=checkpoint["run_contract"],
        ))
    return rows


def build_shortlist(fit_dir, sheet_length):
    checkpoint_paths = sorted(glob.glob(os.path.join(fit_dir, "checkpoint_K*_r*.json")))
    if not checkpoint_paths:
        raise RuntimeError(f"no rev8 checkpoints in {fit_dir}")
    rows = [
        row for path in checkpoint_paths
        for row in _candidate_rows(path, sheet_length)
    ]
    unique = {}
    for row in rows:
        key = (row["K"], row["theta_sha256"])
        if key not in unique:
            unique[key] = row
        elif row["source_role"] not in unique[key]["source_role"]:
            unique[key]["source_role"] += f"+{row['source_role']}"
    out = list(unique.values())
    for index, row in enumerate(out):
        row["candidate_id"] = f"rev8_candidate_{index}"
    return out


def _prewarm(seeds, cfg, cache, workers):
    jobs = _unique_seed_cache_jobs(seeds, cfg, cache)
    with Pool(min(int(workers), len(jobs)), maxtasksperchild=1) as pool:
        rows = pool.map(_precache_network, jobs)
    errors = [row for row in rows if "error" in row]
    if errors:
        raise RuntimeError(f"network cache prewarm failed: {errors}")
    return rows


def _evaluate_candidates(candidates, seeds, cfg, cache, workers,
                         axial, reference, patient_prototypes):
    jobs = [
        (candidate["theta"], int(seed), cfg, cache, int(candidate["K"]), 6)
        for candidate in candidates for seed in seeds
    ]
    with Pool(int(workers), maxtasksperchild=1) as pool:
        raw = pool.map(_evaluate, jobs)
    failures = []
    results = []
    for candidate_index, candidate in enumerate(candidates):
        chunk = raw[
            candidate_index * len(seeds):(candidate_index + 1) * len(seeds)]
        for seed, row in zip(seeds, chunk):
            if "error" in row:
                failures.append(dict(
                    candidate_id=candidate["candidate_id"], seed=int(seed),
                    error=str(row["error"])))
        key, metrics = score_candidate(
            chunk, axial, reference, patient_prototypes)
        results.append(dict(
            **candidate,
            selection_fitness_key=[float(value) for value in key],
            selection_metrics=metrics,
            selection_event_count_by_seed={
                str(seed): int(len(row.get("events", [])))
                for seed, row in zip(seeds, chunk)
            },
        ))
    if failures:
        raise RuntimeError(f"selection failed closed: {failures}")
    return results, raw


def run_selection(args, cfg, axial, reference, patient_prototypes):
    candidates = build_shortlist(args.fit_dir, float(cfg["engine"]["L"]))
    fit_seeds = sorted({seed for row in candidates for seed in row["fit_seeds"]})
    seeds = list(SELECTION_SEED_POOL)
    if set(seeds) & set(fit_seeds):
        raise RuntimeError("selection seed pool overlaps optimization seeds")
    cache = os.path.join(STAGE2, "network_cache")
    cache_rows = _prewarm(seeds, cfg, cache, args.workers)
    rows, _ = _evaluate_candidates(
        candidates, seeds, cfg, cache, args.workers,
        axial, reference, patient_prototypes)
    selected = max(rows, key=lambda row: tuple(row["selection_fitness_key"]))
    output = dict(
        status="REV8_CANDIDATE_FROZEN_BEFORE_FINAL_CONFIRMATION",
        scientific_role=(
            "model-side candidate selection using patient-training target only; "
            "patient held-out recordings and final network seeds were not read"
        ),
        selected_candidate_id=selected["candidate_id"],
        selected_theta_sha256=selected["theta_sha256"],
        selected_candidate=selected,
        ranked_candidates=sorted(
            rows, key=lambda row: tuple(row["selection_fitness_key"]), reverse=True),
        optimization_network_seeds=fit_seeds,
        selection_network_seeds=seeds,
        final_confirm_network_seeds=list(FINAL_CONFIRM_SEED_POOL),
        patient_heldout_read=False,
        patient_heldout_scores_computed=False,
        reference=dict(path=args.reference, sha256=_sha256(args.reference)),
        target=dict(path=args.target, sha256=_sha256(args.target)),
        network_cache=dict(
            hits=int(sum(row["cache_hit"] for row in cache_rows)),
            builds=int(sum(not row["cache_hit"] for row in cache_rows)),
        ),
        provenance=provenance(),
    )
    atomic_write_json(output, args.selection_out)
    print(json.dumps({
        "status": output["status"],
        "selected_candidate_id": output["selected_candidate_id"],
        "K": selected["K"],
        "joint_loss": selected["selection_metrics"]["joint_loss"],
        "distance": selected["selection_metrics"]["distance"],
        "mode_loss": selected["selection_metrics"]["mode"].get("mode_matrix_loss"),
        "min_cluster_count": selected["selection_metrics"]["mode"].get(
            "min_cluster_count"),
    }, indent=2))


def _usable_event_arrays(chunk, seeds, axial, grid):
    curves, seed_ids, local_indices, participants, rank_columns = [], [], [], [], []
    contact_names = sorted(axial, key=axial.get)
    for seed, row in zip(seeds, chunk):
        for local_index, event in enumerate(row.get("events", [])):
            curve = normalized_rank_curve(event.get("ranks"), axial, grid=grid)
            if curve is None:
                continue
            curves.append(curve)
            seed_ids.append(int(seed))
            local_indices.append(int(local_index))
            participants.append(int(event.get("n_part", 0)))
            rank_columns.append([
                np.nan if event.get("ranks", {}).get(name) is None
                else float(event["ranks"][name])
                for name in contact_names
            ])
    return (
        np.asarray(curves, float),
        np.asarray(seed_ids, np.int64),
        np.asarray(local_indices, np.int64),
        np.asarray(participants, np.int64),
        np.asarray(rank_columns, float).T,
        np.asarray(contact_names, dtype="U32"),
    )


def _representative_events(labels, seed_ids, local_indices, participants, seeds):
    candidates = []
    for seed in seeds:
        selected = seed_ids == int(seed)
        counts = np.bincount(labels[selected], minlength=2)
        candidates.append((int(counts.min()), int(counts.sum()), -int(seed), int(seed)))
    representative_seed = max(candidates)[-1]
    event_indices = {}
    for mode in (0, 1):
        eligible = np.flatnonzero(
            (seed_ids == representative_seed) & (labels == mode))
        if len(eligible):
            best = max(eligible, key=lambda index: (
                int(participants[index]), -int(local_indices[index])))
            event_indices[str(mode)] = int(local_indices[best])
        else:
            event_indices[str(mode)] = None
    return representative_seed, event_indices


def run_confirmation(args, cfg, axial, reference, target):
    selection = json.load(open(args.selection_out))
    if selection.get("status") != "REV8_CANDIDATE_FROZEN_BEFORE_FINAL_CONFIRMATION":
        raise RuntimeError("final confirmation requires a frozen rev8 selection")
    if selection["reference"]["sha256"] != _sha256(args.reference):
        raise RuntimeError("selection/reference hash mismatch")
    if selection["target"]["sha256"] != _sha256(args.target):
        raise RuntimeError("selection/target hash mismatch")
    candidate = selection["selected_candidate"]
    seeds = list(FINAL_CONFIRM_SEED_POOL)
    used = set(selection["optimization_network_seeds"]) | set(
        selection["selection_network_seeds"])
    if set(seeds) & used:
        raise RuntimeError("final confirmation seeds overlap fit or selection")

    cache = os.path.join(STAGE2, "network_cache")
    cache_rows = _prewarm(seeds, cfg, cache, args.workers)
    jobs = [
        (candidate["theta"], int(seed), cfg, cache, int(candidate["K"]), 6)
        for seed in seeds
    ]
    with Pool(int(args.workers), maxtasksperchild=1) as pool:
        chunk = pool.map(_evaluate, jobs)
    errors = [
        dict(seed=int(seed), error=str(row["error"]))
        for seed, row in zip(seeds, chunk) if "error" in row
    ]
    if errors:
        atomic_write_json(dict(
            status="FAIL_CLOSED_SIMULATION_ERRORS", errors=errors,
            selection=dict(path=args.selection_out, sha256=_sha256(args.selection_out)),
            provenance=provenance(),
        ), args.confirm_out)
        raise RuntimeError(f"final confirmation failed closed: {errors}")

    reference_file = np.load(args.reference)
    grid = np.asarray(reference_file["grid"], float)
    patient_prototypes = np.asarray(
        target["patient_train_mode_prototypes"], float)
    curves, seed_ids, local_indices, participants, ranks, contact_names = (
        _usable_event_arrays(chunk, seeds, axial, grid))
    consistency_full = kmeans_data_consistency(
        curves, patient_prototypes, reference,
        min_cluster_events=OPPOSITION_MIN_CLUSTER_EVENTS)
    if consistency_full.get("status") != "ok":
        raise RuntimeError("final model events do not define two KMeans modes")
    consistency_full["robustness"] = _kmeans_robustness(
        curves, seed_ids, seeds, patient_prototypes, reference)
    labels = np.asarray(consistency_full["labels"], np.int8)
    representative_seed, representative_indices = _representative_events(
        labels, seed_ids, local_indices, participants, seeds)

    # Patient held-out is opened only after the candidate and all thresholds are frozen.
    patient_curves, block_ids = _patient_curves(axial, grid)
    train_index, heldout_index = split_by_block(
        block_ids, HELD_OUT_FRAC, SPLIT_SEED)
    patient_heldout = patient_curves[heldout_index]
    heldout_z_all = transform_rank_curves(patient_heldout, reference)
    rng = np.random.default_rng(HELDOUT_REFERENCE_SEED)
    heldout_take = min(PROFILE_REFERENCE_N, len(heldout_z_all))
    heldout_z = heldout_z_all[
        rng.choice(len(heldout_z_all), size=heldout_take, replace=False)]

    calibration = json.load(open(args.calibration))
    control_curves = _control_curve_sets(
        calibration, patient_heldout, axial, grid)
    controls = _control_distributions(control_curves, reference)
    kmeans_controls_full = {
        key: kmeans_data_consistency(
            values, patient_prototypes, reference,
            min_cluster_events=OPPOSITION_MIN_CLUSTER_EVENTS)
        for key, values in control_curves.items()
    }
    kmeans_controls = {
        key: _kmeans_summary(value)
        for key, value in kmeans_controls_full.items()
    }

    _, train_metrics = score_candidate(
        chunk, axial, reference, patient_prototypes)
    train_bootstrap = _bootstrap_to_target(
        curves, reference, reference["reference_z"], DISTANCE_BOOTSTRAP_SEED + 300)
    heldout_bootstrap = _bootstrap_to_target(
        curves, reference, heldout_z, DISTANCE_BOOTSTRAP_SEED + 301)
    mode_full = fixed_count_kmeans_mode_loss(
        curves, patient_prototypes, reference,
        n_events=MODE_OBJECTIVE_N_EVENTS,
        min_cluster_events=MODE_MIN_CLUSTER_EVENTS)

    _atomic_npz(
        args.profiles_out,
        grid=np.asarray(grid, np.float32),
        contact_names=contact_names,
        model_curves=np.asarray(curves, np.float32),
        model_labels=labels,
        model_seed_ids=seed_ids,
        model_local_event_indices=local_indices,
        model_participants=participants,
        model_rank_matrix=np.asarray(ranks, np.float32),
        model_mode_prototypes=np.asarray(consistency_full["prototypes"], np.float32),
        patient_train_mode_prototypes=np.asarray(patient_prototypes, np.float32),
        patient_train_target_similarity_matrix=np.asarray(
            target["patient_train_target_similarity_matrix"], np.float32),
    )

    result_candidate = dict(
        candidate_id=candidate["candidate_id"],
        roles=["rev8_selection_winner"],
        K=int(candidate["K"]), restart=int(candidate["restart"]),
        theta=candidate["theta"], theta_sha256=candidate["theta_sha256"],
        training=candidate.get("source_training"),
        selection=candidate["selection_metrics"],
        confirm=dict(
            deterministic_distance_patient_train=train_metrics["distance"],
            bootstrap_distance_patient_train=train_bootstrap,
            deterministic_distance_patient_heldout=_distance_to_target(
                curves, reference, heldout_z),
            bootstrap_distance_patient_heldout=heldout_bootstrap,
            training_objective_joint_loss=train_metrics["joint_loss"],
            training_objective_mode=_kmeans_summary(mode_full),
            n_usable=int(len(curves)),
            n_detected=int(train_metrics["n_detected"]),
            n_failed_networks=0,
            usable_event_count_by_seed={
                str(seed): int(np.sum(seed_ids == seed)) for seed in seeds
            },
            posthoc_prototypes=_posthoc_diagnostic(curves, reference),
            kmeans_data_consistency=_kmeans_summary(consistency_full),
        ),
    )
    output = reconcile_confirmation(dict(
        status="REV8_FINAL_CONFIRMATION_MEASUREMENT_COMPLETE",
        scientific_role=(
            "one frozen rev8 candidate evaluated on patient held-out recordings "
            "and a final unseen-network pool"
        ),
        selection=dict(path=args.selection_out, sha256=_sha256(args.selection_out)),
        reference=dict(path=args.reference, sha256=_sha256(args.reference)),
        target=dict(path=args.target, sha256=_sha256(args.target)),
        event_profiles=dict(
            path=args.profiles_out, sha256=_sha256(args.profiles_out),
            contract="same model event pool for direct-waveform and KMeans figures",
        ),
        patient_split=dict(
            unit="recording block", frac=HELD_OUT_FRAC, seed=SPLIT_SEED,
            n_train=int(len(train_index)), n_heldout=int(len(heldout_index)),
            heldout_reference_n=int(heldout_take),
            heldout_reference_seed=HELDOUT_REFERENCE_SEED,
        ),
        fit_network_seeds=selection["optimization_network_seeds"],
        selection_network_seeds=selection["selection_network_seeds"],
        confirm_network_seeds=seeds,
        representative_run=dict(
            seed=int(representative_seed),
            local_event_index_by_mode=representative_indices,
            has_both_modes=bool(all(
                value is not None for value in representative_indices.values())),
        ),
        network_cache=dict(
            hits=int(sum(row["cache_hit"] for row in cache_rows)),
            builds=int(sum(not row["cache_hit"] for row in cache_rows)),
        ),
        objective_event_count=OBJECTIVE_N_EVENTS,
        mode_objective_event_count=MODE_OBJECTIVE_N_EVENTS,
        mode_min_cluster_events_train=MODE_MIN_CLUSTER_EVENTS,
        mode_loss_weight=MODE_LOSS_WEIGHT,
        patient_floor_train=calibration["optimization_patient_floor"],
        optimization_controls_n20=controls,
        kmeans_patient_train=dict(
            n_events=int(np.asarray(target["patient_train_mode_counts"]).sum()),
            cluster_counts=np.asarray(
                target["patient_train_mode_counts"], int).tolist(),
            prototype_correlation=float(
                target["patient_train_target_similarity_matrix"][0, 1]),
        ),
        kmeans_controls=kmeans_controls,
        candidates=[result_candidate],
        limitations=[
            "single-subject optimization and confirmation",
            "event bootstrap does not replace the reported leave-one-network-out diagnostic",
            "passing this screen would support field capacity, not a causal patient mechanism",
            "finite seizure lifecycle and connectivity-equivalent core remain separate downstream gates",
        ],
        provenance=provenance(),
    ))
    atomic_write_json(output, args.confirm_out)
    print(json.dumps({
        "status": output["status"],
        "verdict": output["candidates"][0]["confirm"]["verdict"],
        "n_usable": len(curves),
        "cluster_counts": consistency_full["cluster_counts"].tolist(),
        "matched_mean": consistency_full["matched_mean"],
        "representative_run": output["representative_run"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "confirm"), required=True)
    parser.add_argument("--fit-dir", default=FIT_DIR)
    parser.add_argument("--reference", default=REFERENCE_PATH)
    parser.add_argument("--target", default=TARGET_PATH)
    parser.add_argument("--calibration", default=CALIBRATION)
    parser.add_argument("--selection-out", default=SELECTION_OUT)
    parser.add_argument("--confirm-out", default=CONFIRM_OUT)
    parser.add_argument("--profiles-out", default=PROFILES_OUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    reference = load_reference(args.reference)
    target = load_training_target(args.target)
    if _reference_sha256(args.reference) != _sha256(args.reference):
        raise RuntimeError("reference hashing contract drifted")
    target_grid = np.asarray(target["grid"], float)
    reference_grid = np.asarray(np.load(args.reference)["grid"], float)
    if not np.array_equal(target_grid, reference_grid):
        raise RuntimeError("rev8 target/reference grids differ")
    axial = axial_map()
    patient_prototypes = np.asarray(
        target["patient_train_mode_prototypes"], float)

    if args.phase == "select":
        run_selection(args, cfg, axial, reference, patient_prototypes)
    else:
        run_confirmation(args, cfg, axial, reference, target)


if __name__ == "__main__":
    main()
