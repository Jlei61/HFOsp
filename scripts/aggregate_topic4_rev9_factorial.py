"""Aggregate rev9 four-arm spontaneous workers without refitting frozen modes."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_joint_fit import load_reference  # noqa: E402
from scripts.run_topic4_rev9_factorial_worker import _load_contract  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    fit_profile_modes,
    kmeans_data_consistency,
    profile_template_similarity,
    sliced_rank_curve_distance,
)
from src.topic4_core_field_rev9 import assign_frozen_modes  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_rev9_factorial import (  # noqa: E402
    ARM_ORDER,
    event_equal_density,
    factorial_effects,
    normalized_event_ranks,
    pairwise_precedence,
)


DEFAULT_CONFIG = "config/topic4_rev9_factorial.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


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


def _jsonable(row, *, drop=()):
    output = {}
    for key, value in row.items():
        if key in drop:
            continue
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def _worker_paths(root, arm, seed):
    slug = arm.lower().replace("+", "_")
    stem = Path(root) / "workers" / f"{slug}_seed{int(seed)}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _load_worker(json_path, npz_path, *, arm, seed, config_sha):
    payload = json.loads(Path(json_path).read_text())
    if payload["arm"] != arm or int(payload["seed"]) != int(seed):
        raise RuntimeError(f"worker identity mismatch: {json_path}")
    if payload["capture_lfp"]:
        raise RuntimeError(f"capture artifact cannot enter factorial pool: {json_path}")
    if payload["inputs"]["factorial_config"]["sha256"] != config_sha:
        raise RuntimeError(f"worker factorial-config hash mismatch: {json_path}")
    if payload["arrays"]["sha256"] != _sha256(npz_path):
        raise RuntimeError(f"worker array hash mismatch: {json_path}")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return payload, arrays


def _mode_prototypes(curves, labels):
    curves = np.asarray(curves, float)
    labels = np.asarray(labels, int)
    output = np.full((2, curves.shape[1]), np.nan)
    for mode in (0, 1):
        if np.any(labels == mode):
            output[mode] = curves[labels == mode].mean(axis=0)
    return output


def _mode_density(histograms, labels):
    histograms = np.asarray(histograms, float)
    labels = np.asarray(labels, int)
    output, counts = [], []
    for mode in (0, 1):
        selected = histograms[labels == mode]
        if len(selected):
            density, count = event_equal_density(selected)
        else:
            density = np.zeros(histograms.shape[1:], float)
            count = 0
        output.append(density)
        counts.append(count)
    return np.asarray(output), counts


def _consensus(curves, reference, states, n_init):
    rows = [fit_profile_modes(curves, reference, seed=int(seed), n_init=n_init)
            for seed in states]
    labels = [np.asarray(row["labels"], int) for row in rows
              if row.get("status") == "ok"]
    ami = [float(adjusted_mutual_info_score(left, right))
           for left, right in itertools.combinations(labels, 2)]
    return dict(
        random_states=[int(value) for value in states], n_init=int(n_init),
        n_successful=int(len(labels)),
        cluster_counts=[np.bincount(value, minlength=2).tolist()
                        for value in labels],
        pairwise_ami_median=(None if not ami else float(np.median(ami))),
        pairwise_ami_min=(None if not ami else float(np.min(ami))),
        pairwise_ami_max=(None if not ami else float(np.max(ami))),
    )


def _representative_seed(seeds, seed_ids, local_indices, participants,
                         frozen_labels, ood):
    candidates = []
    for seed in seeds:
        selected = np.asarray(seed_ids) == int(seed)
        counts = np.bincount(np.asarray(frozen_labels)[selected], minlength=2)
        ood_fraction = float(np.mean(np.asarray(ood)[selected])) if selected.any() else 1.0
        candidates.append((int(counts.min()), int(counts.sum()),
                           -ood_fraction, -int(seed), int(seed)))
    seed = max(candidates)[-1]
    event_indices = {}
    for mode in (0, 1):
        eligible = np.flatnonzero(
            (np.asarray(seed_ids) == seed) &
            (np.asarray(frozen_labels) == mode))
        if len(eligible):
            best = max(eligible, key=lambda index: (
                int(participants[index]), -int(local_indices[index])))
            event_indices[str(mode)] = int(local_indices[best])
        else:
            event_indices[str(mode)] = None
    return seed, event_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-json")
    parser.add_argument("--out-npz")
    args = parser.parse_args()

    factorial, base, frozen_summary, _ = _load_contract(args.config)
    config_sha = _sha256(args.config)
    output_root = Path(factorial["output_root"])
    output_json = Path(args.out_json or output_root / "factorial_summary.json")
    output_npz = Path(args.out_npz or output_root / "factorial_summary.npz")
    seeds = [int(value) for value in factorial["seeds"]]
    arms = [str(value) for value in factorial["arms"]]
    if tuple(arms) != ARM_ORDER:
        raise RuntimeError("factorial arm order changed from the frozen 2x2 design")

    reference_path = frozen_summary["inputs"]["reference"]["path"]
    if _sha256(reference_path) != frozen_summary["inputs"]["reference"]["sha256"]:
        raise RuntimeError("frozen rank-curve reference hash mismatch")
    reference = load_reference(reference_path)
    with np.load(factorial["frozen_readouts"]["npz"], allow_pickle=False) as frozen:
        classifier = dict(
            embedding_centroids=np.asarray(
                frozen["classifier_embedding_centroids"], float),
            ood_distance_thresholds=np.asarray(
                frozen["classifier_ood_thresholds"], float))
    profiles_path = base["inputs"]["profiles"]
    with np.load(profiles_path, allow_pickle=False) as profiles:
        patient_prototypes = np.asarray(
            profiles["patient_train_mode_prototypes"], float)
        grid = np.asarray(profiles["grid"], float)
    diagnostic_path = factorial["figure_inputs"]["patient_block_diagnostics"]
    with np.load(diagnostic_path, allow_pickle=False) as diagnostics:
        patient_band_low = np.asarray(diagnostics["patient_block_band_low"], float)
        patient_band_high = np.asarray(diagnostics["patient_block_band_high"], float)

    mode_cfg = factorial["mode_readout"]["de_novo_kmeans"]
    arrays_out = dict(
        arms=np.asarray(arms, dtype="U16"), seeds=np.asarray(seeds, np.int64),
        grid=np.asarray(grid, np.float32),
        patient_train_mode_prototypes=np.asarray(patient_prototypes, np.float32),
        patient_block_band_low=np.asarray(patient_band_low, np.float32),
        patient_block_band_high=np.asarray(patient_band_high, np.float32),
    )
    worker_inputs, arm_summaries, scalar_by_endpoint = [], {}, {}
    per_arm_scalars = {}

    for arm in arms:
        payloads, bundles = [], []
        for seed in seeds:
            json_path, npz_path = _worker_paths(output_root, arm, seed)
            payload, arrays = _load_worker(
                json_path, npz_path, arm=arm, seed=seed, config_sha=config_sha)
            payloads.append(payload)
            bundles.append(arrays)
            worker_inputs.append(dict(
                arm=arm, seed=seed,
                json=dict(path=str(json_path), sha256=_sha256(json_path)),
                npz=dict(path=str(npz_path), sha256=_sha256(npz_path))))

        contact_names = np.asarray(bundles[0]["contact_names"]).astype(str)
        density_edges = np.asarray(bundles[0]["density_edges"], float)
        for bundle in bundles[1:]:
            if not np.array_equal(bundle["contact_names"].astype(str), contact_names):
                raise RuntimeError(f"contact order changed within {arm}")
            if not np.array_equal(bundle["grid"], bundles[0]["grid"]):
                raise RuntimeError(f"profile grid changed within {arm}")

        curves, ranks, histograms = [], [], []
        seed_ids, local_indices, participants = [], [], []
        for seed, bundle in zip(seeds, bundles):
            valid = np.isfinite(bundle["event_curves"]).all(axis=1)
            curves.append(np.asarray(bundle["event_curves"][valid], float))
            ranks.append(np.asarray(bundle["event_ranks"][valid], float))
            histograms.append(np.asarray(bundle["event_early_density"][valid], float))
            seed_ids.extend([seed] * int(valid.sum()))
            local_indices.extend(bundle["event_local_indices"][valid].astype(int).tolist())
            participants.extend(bundle["event_n_part"][valid].astype(int).tolist())
        curves = np.concatenate(curves, axis=0) if curves else np.empty((0, len(grid)))
        ranks = np.concatenate(ranks, axis=0) if ranks else np.empty((0, len(contact_names)))
        histograms = (np.concatenate(histograms, axis=0) if histograms else
                      np.empty((0, len(density_edges) - 1, len(density_edges) - 1)))
        seed_ids = np.asarray(seed_ids, np.int64)
        local_indices = np.asarray(local_indices, np.int64)
        participants = np.asarray(participants, np.int64)

        if len(curves):
            frozen = assign_frozen_modes(curves, classifier, reference)
            frozen_labels = np.asarray(frozen["labels"], int)
            ood = np.asarray(frozen["ood"], bool)
            distances = np.asarray(frozen["distance_matrix"], float)
            margins = np.max(distances, axis=1) - np.min(distances, axis=1)
            consistency = kmeans_data_consistency(
                curves, patient_prototypes, reference,
                min_cluster_events=1,
                seed=int(mode_cfg["primary_random_state"]),
                n_init=int(mode_cfg["n_init"]))
        else:
            frozen_labels = np.empty(0, int)
            ood = np.empty(0, bool)
            margins = np.empty(0, float)
            consistency = dict(status="insufficient", n_events=0)

        if consistency.get("status") == "ok":
            de_novo_labels = np.asarray(consistency["labels"], int)
            de_novo_prototypes = np.asarray(consistency["prototypes"], float)
            de_novo_matrix = np.asarray(consistency["similarity_matrix"], float)
            frozen_de_novo_ami = float(adjusted_mutual_info_score(
                frozen_labels, de_novo_labels))
        else:
            de_novo_labels = np.full(len(curves), -1, int)
            de_novo_prototypes = np.full((2, len(grid)), np.nan)
            de_novo_matrix = np.full((2, 2), np.nan)
            frozen_de_novo_ami = None
        consensus = _consensus(
            curves, reference, mode_cfg["consensus_random_states"],
            int(mode_cfg["n_init"])) if len(curves) >= 2 else dict(
                random_states=mode_cfg["consensus_random_states"],
                n_init=int(mode_cfg["n_init"]), n_successful=0,
                cluster_counts=[], pairwise_ami_median=None,
                pairwise_ami_min=None, pairwise_ami_max=None)

        frozen_prototypes = _mode_prototypes(curves, frozen_labels)
        frozen_matrix = (profile_template_similarity(
            frozen_prototypes, patient_prototypes)
            if np.isfinite(frozen_prototypes).all() else np.full((2, 2), np.nan))
        frozen_density, frozen_density_counts = _mode_density(
            histograms, frozen_labels) if len(histograms) else (
                np.zeros((2, len(density_edges) - 1, len(density_edges) - 1)),
                [0, 0])
        de_novo_density, de_novo_density_counts = _mode_density(
            histograms, de_novo_labels) if len(histograms) and np.all(de_novo_labels >= 0) else (
                np.zeros_like(frozen_density), [0, 0])
        normalized_ranks = normalized_event_ranks(ranks)
        recruitment = (np.mean(np.isfinite(ranks), axis=0) if len(ranks)
                       else np.full(len(contact_names), np.nan))
        mean_rank = np.full(len(contact_names), np.nan)
        for contact_index in range(len(contact_names)):
            finite = normalized_ranks[:, contact_index][
                np.isfinite(normalized_ranks[:, contact_index])]
            if len(finite):
                mean_rank[contact_index] = float(finite.mean())
        precedence, precedence_support = pairwise_precedence(ranks)

        seed_scalars = {key: [] for key in (
            "event_rate_hz", "usable_event_rate_hz",
            "frozen_mode_b_proportion", "ood_fraction",
            "assignment_margin_median", "duration_median_ms",
            "active_neurons_median", "participants_median", "return_fraction")}
        for seed, payload, bundle in zip(seeds, payloads, bundles):
            duration_s = float(payload["simulation"]["simulated_until_ms"]) / 1000.0
            selected = seed_ids == seed
            events = payload["events"]
            seed_scalars["event_rate_hz"].append(len(events) / duration_s)
            seed_scalars["usable_event_rate_hz"].append(int(selected.sum()) / duration_s)
            seed_scalars["frozen_mode_b_proportion"].append(
                float(np.mean(frozen_labels[selected] == 1)) if selected.any() else np.nan)
            seed_scalars["ood_fraction"].append(
                float(np.mean(ood[selected])) if selected.any() else np.nan)
            seed_scalars["assignment_margin_median"].append(
                float(np.median(margins[selected])) if selected.any() else np.nan)
            seed_scalars["duration_median_ms"].append(
                float(np.median([row["duration_ms"] for row in events]))
                if events else np.nan)
            seed_scalars["active_neurons_median"].append(
                float(np.median([row["n_active_neurons"] for row in events]))
                if events else np.nan)
            seed_scalars["participants_median"].append(
                float(np.median(participants[selected])) if selected.any() else np.nan)
            seed_scalars["return_fraction"].append(
                float(np.mean([row["returned"] for row in events]))
                if events else np.nan)
        seed_scalars = {key: np.asarray(value, float)
                        for key, value in seed_scalars.items()}
        per_arm_scalars[arm] = seed_scalars

        representative_seed, representative_events = _representative_seed(
            seeds, seed_ids, local_indices, participants, frozen_labels, ood)
        slug = arm.lower().replace("+", "_")
        arm_key = slug
        arrays_out.update({
            f"{arm_key}_contact_names": np.asarray(contact_names, dtype="U32"),
            f"{arm_key}_density_edges": np.asarray(density_edges, np.float32),
            f"{arm_key}_curves": np.asarray(curves, np.float32),
            f"{arm_key}_ranks": np.asarray(ranks, np.float32),
            f"{arm_key}_normalized_ranks": np.asarray(normalized_ranks, np.float32),
            f"{arm_key}_seed_ids": seed_ids,
            f"{arm_key}_local_event_indices": local_indices,
            f"{arm_key}_participants": participants,
            f"{arm_key}_frozen_labels": np.asarray(frozen_labels, np.int8),
            f"{arm_key}_frozen_ood": np.asarray(ood, bool),
            f"{arm_key}_frozen_margin": np.asarray(margins, np.float32),
            f"{arm_key}_de_novo_labels": np.asarray(de_novo_labels, np.int8),
            f"{arm_key}_frozen_prototypes": np.asarray(frozen_prototypes, np.float32),
            f"{arm_key}_de_novo_prototypes": np.asarray(de_novo_prototypes, np.float32),
            f"{arm_key}_frozen_similarity": np.asarray(frozen_matrix, np.float32),
            f"{arm_key}_de_novo_similarity": np.asarray(de_novo_matrix, np.float32),
            f"{arm_key}_frozen_onset_density": np.asarray(frozen_density, np.float32),
            f"{arm_key}_de_novo_onset_density": np.asarray(de_novo_density, np.float32),
            f"{arm_key}_recruitment_probability": np.asarray(recruitment, np.float32),
            f"{arm_key}_mean_normalized_rank": np.asarray(mean_rank, np.float32),
            f"{arm_key}_precedence_probability": np.asarray(precedence, np.float32),
            f"{arm_key}_precedence_support": np.asarray(precedence_support, np.int32),
        })
        for name, values in seed_scalars.items():
            arrays_out[f"{arm_key}_seed_{name}"] = np.asarray(values, np.float32)

        arm_summaries[arm] = dict(
            n_networks=len(seeds),
            n_runaway=int(sum(row["simulation"]["runaway_early_stop_ms"] is not None
                              for row in payloads)),
            n_detected=int(sum(row["simulation"]["n_detected"] for row in payloads)),
            n_usable=int(len(curves)),
            usable_by_seed={str(seed): int(np.sum(seed_ids == seed)) for seed in seeds},
            frozen=dict(
                counts=np.bincount(frozen_labels, minlength=2).tolist(),
                ood_fraction=(None if not len(ood) else float(np.mean(ood))),
                assignment_margin_median=(None if not len(margins) else
                                          float(np.median(margins))),
                similarity_matrix=frozen_matrix.tolist(),
                onset_density_event_counts=frozen_density_counts),
            de_novo=dict(
                **_jsonable(consistency, drop=("labels", "prototypes")),
                frozen_assignment_ami=frozen_de_novo_ami,
                onset_density_event_counts=de_novo_density_counts,
                consensus=consensus),
            event_cloud_distance_patient_train=(
                None if not len(curves) else
                float(sliced_rank_curve_distance(curves, reference))),
            scalar_seed_summary={
                key: dict(
                    median=(None if not np.isfinite(value).any() else
                            float(np.nanmedian(value))),
                    n_finite=int(np.isfinite(value).sum()))
                for key, value in seed_scalars.items()},
            representative=dict(
                seed=int(representative_seed),
                local_event_index_by_frozen_mode=representative_events,
                worker_json=str(_worker_paths(
                    output_root, arm, representative_seed)[0]),
                capture_json=str(output_root / "representative" /
                                 f"{slug}_seed{representative_seed}_capture.json"),
                capture_npz=str(output_root / "representative" /
                                f"{slug}_seed{representative_seed}_capture.npz")),
        )

    for endpoint in next(iter(per_arm_scalars.values())):
        values = {arm: per_arm_scalars[arm][endpoint] for arm in arms}
        scalar_by_endpoint[endpoint] = factorial_effects(
            values, seed=int(factorial["bootstrap_seed"]) +
            100 * len(scalar_by_endpoint),
            repeats=int(factorial["bootstrap_repeats"]))
        arrays_out[f"seed_endpoint_{endpoint}"] = np.stack(
            [values[arm] for arm in arms]).astype(np.float32)

    _atomic_npz(output_npz, **arrays_out)
    payload = dict(
        status="REV9_FACTORIAL_AGGREGATION_COMPLETE",
        scientific_role=(
            "exploratory network-seed paired Node-Edge factorization; frozen "
            "patient-training readout only, no patient held-out or blind validation"),
        arms=arms, seeds=seeds,
        alpha_star=float(factorial["alpha_reference"]["value"]),
        alpha_role=factorial["alpha_reference"]["role"],
        arm_summaries=arm_summaries,
        paired_factorial_endpoints=scalar_by_endpoint,
        endpoint_contract=(
            "paired mean effects and seed bootstrap 95% intervals; missing event-level "
            "values remain missing and may make an endpoint unidentifiable"),
        worker_inputs=worker_inputs,
        arrays=dict(path=str(output_npz), sha256=_sha256(output_npz)),
        inputs=dict(
            factorial_config=dict(path=args.config, sha256=config_sha),
            frozen_readouts=factorial["frozen_readouts"],
            rank_curve_reference=dict(path=reference_path, sha256=_sha256(reference_path)),
            patient_training_profiles=dict(path=profiles_path, sha256=_sha256(profiles_path)),
            patient_block_diagnostics=dict(
                path=diagnostic_path, sha256=_sha256(diagnostic_path))),
        limitations=[
            "alpha was selected by a response objective on different network seeds; site-resolved equivalence was not established",
            "the same seeds were used for the out-of-selection local-response audit and this exploratory factorial",
            "de novo KMeans is descriptive and does not replace the frozen classifier",
            "no new patient blind unit was read",
        ],
        provenance=dict(
            **provenance(), git_status_porcelain=_git("status", "--porcelain"),
            producer_sha256=_sha256(__file__),
            config_sha256=config_sha, python_executable=sys.executable,
            python_version=platform.python_version(),
            systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
            network_seed=seeds, readout_seed=int(factorial["bootstrap_seed"])),
    )
    atomic_write_json(payload, output_json)
    print(json.dumps(dict(
        status=payload["status"], alpha_star=payload["alpha_star"],
        arms={arm: dict(
            n_detected=arm_summaries[arm]["n_detected"],
            n_usable=arm_summaries[arm]["n_usable"],
            frozen_counts=arm_summaries[arm]["frozen"]["counts"],
            de_novo_counts=arm_summaries[arm]["de_novo"].get("cluster_counts"),
            ood_fraction=arm_summaries[arm]["frozen"]["ood_fraction"])
            for arm in arms}, arrays_sha256=payload["arrays"]["sha256"]),
        indent=2), flush=True)


if __name__ == "__main__":
    main()
