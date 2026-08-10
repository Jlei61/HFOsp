"""Freeze zero-simulation component and mode readouts for rev9 interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from src.topic4_core_field_profile import fit_profile_modes  # noqa: E402
from src.topic4_core_field_rev9 import (  # noqa: E402
    assign_frozen_modes,
    component_responsibilities,
    fit_frozen_mode_classifier,
    node_reconstruction_error,
    reconstruct_frozen_node,
)
from src.topic4_core_field_runner import (  # noqa: E402
    CONNECTIVITY_FIELDS,
    _placement,
    atomic_write_json,
    canonical_checksum,
    provenance,
)
from src.topic4_core_field_stage3 import unpack  # noqa: E402


ROOT = Path("results/topic4_sef_hfo/data_driven_core_field_stage3")
REV8 = ROOT / "joint_confirmation_rev8_1"
OUT_ROOT = Path("results/topic4_sef_hfo/data_driven_core_field_rev9")


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


def _null_summary(responsibilities, labels, *, n_resamples, seed):
    responsibilities = np.asarray(responsibilities, float)
    labels = np.asarray(labels, int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("null summary requires two event modes")
    rng = np.random.default_rng(int(seed))
    n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    observed = (responsibilities[labels == 1].mean(axis=0)
                - responsibilities[labels == 0].mean(axis=0))
    permutation = np.empty((int(n_resamples), responsibilities.shape[1]))
    pooled = np.empty_like(permutation)
    for index in range(int(n_resamples)):
        shuffled = rng.permutation(labels)
        permutation[index] = (
            responsibilities[shuffled == 1].mean(axis=0)
            - responsibilities[shuffled == 0].mean(axis=0))
        draw0 = rng.integers(0, len(responsibilities), size=n0)
        draw1 = rng.integers(0, len(responsibilities), size=n1)
        pooled[index] = (responsibilities[draw1].mean(axis=0)
                         - responsibilities[draw0].mean(axis=0))

    def describe(samples):
        p_two_sided = (
            1 + np.sum(np.abs(samples) >= np.abs(observed), axis=0)
        ) / (len(samples) + 1)
        return dict(
            p_two_sided=p_two_sided.tolist(),
            interval_95=np.quantile(samples, (0.025, 0.975), axis=0).T.tolist(),
        )

    return dict(
        observed_mode1_minus_mode0=observed.tolist(),
        permutation=describe(permutation),
        pooled_location_resampling=describe(pooled),
    ), permutation, pooled


def _consensus_summary(curves, reference):
    labels = []
    counts = []
    for seed in range(10):
        modes = fit_profile_modes(curves, reference, seed=seed, n_init=100)
        if modes.get("status") != "ok":
            raise RuntimeError(f"de novo KMeans failed at random_state={seed}")
        labels.append(np.asarray(modes["labels"], int))
        counts.append(np.asarray(modes["cluster_counts"], int))
    pairwise = [
        adjusted_mutual_info_score(labels[left], labels[right])
        for left in range(len(labels)) for right in range(left + 1, len(labels))
    ]
    return dict(
        random_states=list(range(10)), n_init=100,
        cluster_counts=np.asarray(counts).tolist(),
        pairwise_ami_median=float(np.median(pairwise)),
        pairwise_ami_min=float(np.min(pairwise)),
        pairwise_ami_max=float(np.max(pairwise)),
    )


def _representative_network_cache(stage, confirmation, cache_dir):
    params_cls = __import__("params").Params
    engine = stage["engine"]
    seed = int(confirmation["representative_run"]["seed"])
    params = params_cls(
        g=engine["g"], L=engine["L"], density=engine["density"],
        T=stage["duration_ms"], dt=engine["dt"], nu_ext_ratio=0.6,
        seed=seed)
    reg = _placement(stage)
    cache_config = {field: getattr(params, field) for field in CONNECTIVITY_FIELDS}
    cache_config.update(
        theta_EE_deg=float(reg["theta_deg"]), AR=float(engine["AR"]),
        numpy_version=confirmation["provenance"]["numpy_version"],
        rng_bit_generator="PCG64",
        git_commit=confirmation["provenance"]["git_commit"],
    )
    key = canonical_checksum(cache_config, drop=())
    path = Path(cache_dir) / f"{key}.pkl"
    if not path.exists():
        raise RuntimeError(f"representative upstream network cache is missing: {path}")
    return path, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default=str(REV8 / "final_confirmation.json"))
    parser.add_argument("--profiles", default=str(REV8 / "final_event_profiles.npz"))
    parser.add_argument("--onsets", default=str(REV8 / "all_event_onset_diagnostics.npz"))
    parser.add_argument(
        "--figdata", default=str(REV8 / "representative_figdata.npz"))
    parser.add_argument(
        "--stage-config",
        default="results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json")
    parser.add_argument(
        "--network-cache",
        default="results/topic4_sef_hfo/data_driven_core_field/network_cache")
    parser.add_argument("--out-json", default=str(OUT_ROOT / "frozen_readouts.json"))
    parser.add_argument("--out-npz", default=str(OUT_ROOT / "frozen_readouts.npz"))
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--null-seed", type=int, default=2026081001)
    args = parser.parse_args()

    confirmation = json.loads(Path(args.confirmation).read_text())
    if confirmation["event_profiles"]["sha256"] != _sha256(args.profiles):
        raise RuntimeError("confirmation/event profile hash mismatch")
    candidate = confirmation["candidates"][0]
    profiles = np.load(args.profiles)
    onsets = np.load(args.onsets)
    if len(profiles["model_curves"]) != len(onsets["source_centroids"]):
        raise RuntimeError("event profile and onset pools differ in size")
    keys_profiles = np.column_stack((
        profiles["model_seed_ids"], profiles["model_local_event_indices"]))
    keys_onsets = np.column_stack((
        onsets["event_seed_ids"], onsets["event_local_indices"]))
    if not np.array_equal(keys_profiles, keys_onsets):
        raise RuntimeError("event profile and onset pools differ in ordering")
    labels = np.asarray(profiles["model_labels"], int)
    if not np.array_equal(labels, np.asarray(onsets["event_modes"], int)):
        raise RuntimeError("event mode labels drifted between frozen artifacts")

    theta = np.asarray(candidate["theta"], float)
    source_centroids = np.asarray(onsets["source_centroids"], float)
    soft = component_responsibilities(
        theta, source_centroids, K=int(candidate["K"]), L=20.0)
    nulls, permutation, pooled = _null_summary(
        soft["responsibilities"], labels,
        n_resamples=args.n_resamples, seed=args.null_seed)

    reference_path = confirmation["reference"]["path"]
    if confirmation["reference"]["sha256"] != _sha256(reference_path):
        raise RuntimeError("confirmation/reference hash mismatch")
    reference_file = np.load(reference_path)
    reference = {key: np.asarray(reference_file[key]) for key in reference_file.files}
    curves = np.asarray(profiles["model_curves"], float)
    classifier = fit_frozen_mode_classifier(curves, labels, reference)
    baseline_assignment = assign_frozen_modes(curves, classifier, reference)
    if not np.array_equal(baseline_assignment["labels"], labels):
        raise RuntimeError("frozen classifier does not reproduce baseline labels")
    consensus = _consensus_summary(curves, reference)

    stage = json.loads(Path(args.stage_config).read_text())
    figdata = np.load(args.figdata, allow_pickle=False)
    if not np.array_equal(np.asarray(figdata["theta"], float), theta):
        raise RuntimeError("representative figdata theta differs from candidate")
    network_cache_path, network_cache_key = _representative_network_cache(
        stage, confirmation, args.network_cache)
    with open(network_cache_path, "rb") as handle:
        cached = pickle.load(handle)
    original_pos_e = np.asarray(cached["net"]["pos"][:cached["NE"]], float)
    if not np.array_equal(original_pos_e.astype(np.float32), figdata["posE"]):
        raise RuntimeError("representative network neuron order differs from figdata")
    representative_node = reconstruct_frozen_node(
        theta, original_pos_e,
        n_total=len(original_pos_e), target_count=stage["N_core_manual"],
        quantile_seed=stage["quantile_seed"],
        core_mean=stage["engine"]["core_mean"],
        core_std=stage["engine"]["core_std"],
        v_base=stage["engine"]["v_base"], K=int(candidate["K"]),
        L=stage["engine"]["L"])
    frozen_h = np.asarray(figdata["h"])
    frozen_vtheta = np.asarray(figdata["vth"])
    reconstructed_h = np.asarray(representative_node["h"], dtype=frozen_h.dtype)
    reconstructed_vtheta = np.asarray(
        representative_node["vtheta"], dtype=frozen_vtheta.dtype)
    if not np.array_equal(reconstructed_h, frozen_h):
        raise RuntimeError("representative h does not reconstruct exactly at frozen dtype")
    reconstruction = node_reconstruction_error(reconstructed_vtheta, frozen_vtheta)
    if not reconstruction["exact"]:
        raise RuntimeError("representative Vtheta does not reconstruct exactly")

    components = unpack(theta, int(candidate["K"]), 20.0)
    _atomic_npz(
        args.out_npz,
        source_centroids=np.asarray(source_centroids, np.float32),
        event_modes=np.asarray(labels, np.int8),
        event_seed_ids=np.asarray(profiles["model_seed_ids"], np.int64),
        event_local_indices=np.asarray(profiles["model_local_event_indices"], np.int64),
        component_contributions=np.asarray(soft["contributions"], np.float32),
        component_responsibilities=np.asarray(soft["responsibilities"], np.float32),
        component_assignments=np.asarray(soft["assignments"], np.int8),
        maximum_responsibility=np.asarray(soft["maximum_responsibility"], np.float32),
        permutation_mode_difference=np.asarray(permutation, np.float32),
        pooled_mode_difference=np.asarray(pooled, np.float32),
        classifier_embedding_centroids=np.asarray(
            classifier["embedding_centroids"], np.float64),
        classifier_ood_thresholds=np.asarray(
            classifier["ood_distance_thresholds"], np.float64),
        baseline_embedded=np.asarray(classifier["baseline_embedded"], np.float32),
        baseline_assigned_distance=np.asarray(
            classifier["baseline_assigned_distance"], np.float32),
        reference_center=np.asarray(reference["center"], np.float64),
        reference_components=np.asarray(reference["components"], np.float64),
        reference_score_center=np.asarray(reference["score_center"], np.float64),
        reference_score_scale=np.asarray(reference["score_scale"], np.float64),
        representative_h=np.asarray(representative_node["h"], np.float32),
        representative_d=np.asarray(representative_node["d"], np.float32),
        representative_vtheta=np.asarray(
            representative_node["vtheta"], np.float32),
    )

    mode_means = np.asarray([
        soft["responsibilities"][labels == mode].mean(axis=0) for mode in (0, 1)
    ])
    nearest_agreement = float(np.mean(
        soft["assignments"] == np.asarray(onsets["nearest_components"], int)))
    package_lock = "requirements.txt"
    payload = dict(
        status="REV9_FROZEN_READOUTS_COMPLETE",
        scientific_role=(
            "zero-simulation frozen readouts for causal interventions; no candidate "
            "selection and no new patient validation"),
        candidate=dict(
            id=candidate["candidate_id"], K=int(candidate["K"]),
            theta_sha256=candidate["theta_sha256"],
            components=[dict(
                center=np.asarray(row["center"]).tolist(),
                sigma_par=float(row["sigma_par"]),
                sigma_perp=float(row["sigma_perp"]),
                phi=float(row["phi"]), weight=float(row["weight"]))
                for row in components],
        ),
        component_responsibility=dict(
            definition="raw q_c(source centroid) / sum_j raw q_j(source centroid)",
            mode_counts=np.bincount(labels, minlength=2).astype(int).tolist(),
            mean_by_mode=mode_means.tolist(),
            median_maximum_by_mode=[
                float(np.median(soft["maximum_responsibility"][labels == mode]))
                for mode in (0, 1)],
            nearest_center_agreement=nearest_agreement,
            null_seed=int(args.null_seed), n_resamples=int(args.n_resamples),
            nulls=nulls,
        ),
        frozen_mode_classifier=dict(
            space="frozen patient-training PCA z-space",
            assignment="nearest Euclidean centroid; no refit after intervention",
            baseline_counts=classifier["baseline_counts"].astype(int).tolist(),
            ood_quantile=float(classifier["ood_quantile"]),
            ood_distance_thresholds=classifier["ood_distance_thresholds"].tolist(),
            baseline_ood_count=int(np.sum(baseline_assignment["ood"])),
            de_novo_kmeans=consensus,
        ),
        node_reconstruction_preflight=dict(
            exact_at_frozen_dtype=True,
            max_abs_error=reconstruction["max_abs_error"],
            h_vector_sha256=representative_node["hashes"]["h_vector_sha256"],
            d_vector_sha256=representative_node["hashes"]["d_vector_sha256"],
            vtheta_reconstructed_sha256=(
                representative_node["hashes"]["vtheta_reconstructed_sha256"]),
            vtheta_frozen_dtype_sha256=reconstruction["frozen_sha256"],
        ),
        inputs=dict(
            confirmation=dict(path=args.confirmation, sha256=_sha256(args.confirmation)),
            profiles=dict(path=args.profiles, sha256=_sha256(args.profiles)),
            onsets=dict(path=args.onsets, sha256=_sha256(args.onsets)),
            reference=dict(path=reference_path, sha256=_sha256(reference_path)),
            representative_figdata=dict(
                path=args.figdata, sha256=_sha256(args.figdata)),
            stage_config=dict(
                path=args.stage_config, sha256=_sha256(args.stage_config)),
            representative_network_cache=dict(
                path=str(network_cache_path), cache_key=network_cache_key,
                sha256=_sha256(network_cache_path)),
        ),
        arrays=dict(path=args.out_npz, sha256=_sha256(args.out_npz)),
        provenance=dict(
            **provenance(),
            git_status_porcelain=_git("status", "--porcelain"),
            producer_sha256=_sha256(__file__),
            python_executable=sys.executable,
            python_version=platform.python_version(),
            package_lock=dict(path=package_lock, sha256=_sha256(package_lock)),
        ),
    )
    atomic_write_json(payload, args.out_json)
    print(json.dumps(dict(
        status=payload["status"], mode_mean_responsibility=mode_means.tolist(),
        ood_thresholds=classifier["ood_distance_thresholds"].tolist(),
        consensus_ami_median=consensus["pairwise_ami_median"],
        arrays_sha256=payload["arrays"]["sha256"],
    ), indent=2))


if __name__ == "__main__":
    main()
