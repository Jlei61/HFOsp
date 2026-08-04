"""Target-sealed spatial readout for the shared-scaffold Topic 5 RNN.

This module deliberately does not load checkpoints, interictal event files, or
ictal targets.  It owns the small, easily-audited boundary between those
stages:

1. derive one target-free diffusion coordinate and two endpoint source pools
   from a *training-only*, symmetric effective operator;
2. reduce each directional first-arrival rollout to exactly one contact field;
3. fingerprint the frozen fields before any ictal value is read; and
4. score those two fields against an already exact-name-joined early-ictal
   contact field, repeating the two-direction maximum inside every null draw.

The unique directional field is participation-weighted first-arrival
earliness.  If ``q[t, i] = P(T_i=t+1)`` for rollout steps ``1..H``, then

``F[i] = sum_t q[t, i] * (1 - (t + 1) / H)``.

The observed source pool is assigned first arrival at step zero and therefore
has field value one.  Equivalently, for a non-source contact the field is its
participation probability multiplied by conditional mean first-arrival
earliness.  There is no menu of alternative rollout summaries.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from src.propagation_skeleton_geometry import parse_shaft


# The v0.2 symmetric-only run is a stopped diagnostic and must never share a
# field pool with v0.3.  Carrying the model version in the record contract
# makes an accidental mix fail validation instead of averaging silently.
FIELD_RECORD_CONTRACT = "topic5_source_conditioned_scaffold_directional_field_v0_3"
FIELD_MANIFEST_CONTRACT = "topic5_source_conditioned_scaffold_field_manifest_v0_3"
FINGERPRINT_ALGORITHM = "sha256_canonical_json_float64_v1"
FIELD_DEFINITION = "participation_weighted_first_arrival_earliness_v1"
SOURCE_POOL_RULE = "normalized_laplacian_first_nontrivial_coordinate_endpoint_quartiles"
LEARNED_AXIS_SOURCE_POOL_RULE = (
    "learned_signed_axis_endpoint_quantiles_seed_sign_aligned"
)


def _finite_float_array(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != int(ndim):
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _canonical_jsonable(value: Any) -> Any:
    """Convert NumPy containers to a finite, deterministic JSON structure."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return _canonical_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("frozen manifests cannot contain NaN or infinity")
        return number
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        _canonical_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: object) -> str:
    """Hash a numerical array with explicit shape and little-endian float64."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def normalized_laplacian_source_pools(
    operator: np.ndarray,
    *,
    contact_names: Sequence[str],
    endpoint_fraction: float = 0.25,
    symmetry_atol: float = 1.0e-7,
    zero_eigenvalue_atol: float = 1.0e-8,
) -> dict[str, object]:
    """Derive a deterministic first diffusion coordinate and endpoint pools.

    ``operator`` is the frozen training-only structured ``W``.  Self loops are
    removed before constructing ``L = I - D^-1/2 A D^-1/2`` because source
    sides are defined by between-contact propagation, not self persistence.
    The graph must be connected; otherwise a unique first non-trivial
    diffusion coordinate is not identifiable.

    The arbitrary eigenvector sign is canonicalized by making its largest
    absolute loading positive.  Exactly ``ceil(endpoint_fraction * N)``
    contacts are assigned to each side using a stable sort, which guarantees
    non-empty, disjoint pools even when coordinates tie numerically.
    """

    matrix = _finite_float_array(operator, name="operator", ndim=2)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("operator must be a square matrix with at least two contacts")
    names = tuple(map(str, contact_names))
    if len(names) != matrix.shape[0] or len(set(names)) != len(names):
        raise ValueError("contact_names must be unique and align with operator")
    if not 0.0 < float(endpoint_fraction) <= 0.5:
        raise ValueError("endpoint_fraction must lie in (0, 0.5]")
    if not np.allclose(matrix, matrix.T, atol=float(symmetry_atol), rtol=0.0):
        raise ValueError("structured operator must be symmetric")
    adjacency = 0.5 * (matrix + matrix.T)
    if np.min(adjacency) < -float(symmetry_atol):
        raise ValueError("structured operator must be non-negative")
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    if np.any(degree <= np.finfo(float).eps):
        raise ValueError("operator graph contains an isolated contact")
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    normalized_adjacency = (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    laplacian = np.eye(len(names), dtype=np.float64) - normalized_adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    zero_count = int(np.count_nonzero(eigenvalues <= float(zero_eigenvalue_atol)))
    if zero_count != 1:
        raise ValueError(
            "operator graph is disconnected; first non-trivial coordinate is not unique"
        )
    coordinate_index = 1
    coordinate = np.asarray(eigenvectors[:, coordinate_index], dtype=np.float64)
    anchor_index = int(np.flatnonzero(np.abs(coordinate) == np.max(np.abs(coordinate)))[0])
    if coordinate[anchor_index] < 0:
        coordinate = -coordinate

    n_pool = max(1, int(np.ceil(float(endpoint_fraction) * len(names))))
    order = np.argsort(coordinate, kind="stable")
    source_minus = np.sort(order[:n_pool])
    source_plus = np.sort(order[-n_pool:])
    if np.intersect1d(source_minus, source_plus).size:
        raise RuntimeError("endpoint source pools overlap")
    return {
        "diffusion_coordinate": coordinate,
        "normalized_laplacian": laplacian,
        "laplacian_eigenvalues": eigenvalues,
        "first_nontrivial_eigenvalue": float(eigenvalues[coordinate_index]),
        "source_minus_indices": source_minus,
        "source_plus_indices": source_plus,
        "source_minus_contacts": [names[index] for index in source_minus],
        "source_plus_contacts": [names[index] for index in source_plus],
        "endpoint_fraction": float(endpoint_fraction),
        "operator_sha256": array_sha256(matrix),
        "source_pool_rule": SOURCE_POOL_RULE,
    }


def learned_axis_source_pools(
    axis_coordinate_by_seed: Mapping[str, Sequence[float]],
    *,
    contact_names: Sequence[str],
    endpoint_fraction: float = 0.25,
) -> dict[str, object]:
    """Derive endpoint pools from the structured model's own signed axis.

    The v0.2 rule split the diffusion graph of ``W`` into two communities.
    That graph mixes the fixed shaft/geometry term with the learned axis, so
    its first non-trivial coordinate need not track the axis the direction
    state actually reads; when it does not, both pools drive the model the
    same way and the two directional fields collapse onto each other.

    The event likelihood is exactly invariant under flipping one patient's
    signed coordinate, so each training seed fixes its own sign arbitrarily.
    Seeds are therefore flipped onto the first sorted seed before averaging,
    and the ensemble is then canonicalized the same way as the diffusion
    rule, by making its largest absolute loading positive.  ``minus`` is the
    negative-coordinate end, which is the end with the larger ``a``
    membership and therefore the end that drives a positive direction state.
    """

    names = tuple(map(str, contact_names))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("contact_names must be unique and contain two or more contacts")
    if not 0.0 < float(endpoint_fraction) <= 0.5:
        raise ValueError("endpoint_fraction must lie in (0, 0.5]")
    seed_keys = sorted(axis_coordinate_by_seed, key=lambda key: (len(str(key)), str(key)))
    if not seed_keys:
        raise ValueError("at least one seed axis coordinate is required")
    axes = []
    for key in seed_keys:
        axis = _finite_float_array(
            axis_coordinate_by_seed[key], name=f"axis_coordinate[{key}]", ndim=1
        )
        if axis.shape != (len(names),):
            raise ValueError("axis coordinates must align with contact_names")
        if not np.any(np.abs(axis - axis.mean()) > 0):
            raise ValueError("axis coordinate must vary across contacts")
        axes.append(axis)

    reference = axes[0]
    aligned, flipped = [], {}
    for key, axis in zip(seed_keys, axes):
        flip = bool(float(np.dot(axis, reference)) < 0.0)
        flipped[str(key)] = flip
        aligned.append(-axis if flip else axis)
    ensemble = np.mean(aligned, axis=0)
    if not np.any(np.abs(ensemble - ensemble.mean()) > 0):
        raise ValueError("seed-aligned axis ensemble is constant across contacts")
    anchor = int(np.flatnonzero(np.abs(ensemble) == np.max(np.abs(ensemble)))[0])
    if ensemble[anchor] < 0:
        ensemble = -ensemble
        flipped = {key: (not value) for key, value in flipped.items()}

    # Reported, not gated: a low value means the seeds did not agree on one
    # axis, so their ensemble mean is not a meaningful single scaffold.
    pairwise = [
        float(np.corrcoef(aligned[i], aligned[j])[0, 1])
        for i in range(len(aligned))
        for j in range(i + 1, len(aligned))
    ]

    n_pool = max(1, int(np.ceil(float(endpoint_fraction) * len(names))))
    order = np.argsort(ensemble, kind="stable")
    source_minus = np.sort(order[:n_pool])
    source_plus = np.sort(order[-n_pool:])
    if np.intersect1d(source_minus, source_plus).size:
        raise RuntimeError("endpoint source pools overlap")
    return {
        "diffusion_coordinate": ensemble,
        "seed_order": [str(key) for key in seed_keys],
        "seed_axis_sign_flipped": flipped,
        "seed_axis_pairwise_pearson": pairwise,
        "min_seed_axis_pairwise_pearson": float(min(pairwise)) if pairwise else float("nan"),
        "source_minus_indices": source_minus,
        "source_plus_indices": source_plus,
        "source_minus_contacts": [names[index] for index in source_minus],
        "source_plus_contacts": [names[index] for index in source_plus],
        "endpoint_fraction": float(endpoint_fraction),
        "source_pool_rule": LEARNED_AXIS_SOURCE_POOL_RULE,
    }


def participation_weighted_first_arrival_earliness(
    first_arrival_mass: np.ndarray,
    *,
    source_indices: Sequence[int],
    probability_atol: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Reduce one absorbing rollout to its unique directional contact field.

    Rows of ``first_arrival_mass`` are future rank steps ``1..H`` and columns
    are contacts.  The source pool is observed at rank step zero and must not
    carry future first-arrival mass.  The returned field is in ``[0, 1]``.
    """

    mass = _finite_float_array(
        first_arrival_mass, name="first_arrival_mass", ndim=2
    )
    horizon, n_contacts = mass.shape
    if horizon < 1 or n_contacts < 2:
        raise ValueError("first_arrival_mass must have shape [H>=1, N>=2]")
    if np.min(mass) < -float(probability_atol):
        raise ValueError("first-arrival mass cannot be negative")
    mass = np.maximum(mass, 0.0)
    source = np.asarray(source_indices, dtype=np.int64)
    if source.ndim != 1 or source.size < 1 or len(np.unique(source)) != len(source):
        raise ValueError("source_indices must be a non-empty unique vector")
    if np.any((source < 0) | (source >= n_contacts)):
        raise ValueError("source index is outside the contact range")
    participation = mass.sum(axis=0)
    if np.any(participation > 1.0 + float(probability_atol)):
        raise ValueError("first-arrival probability exceeds one")
    if np.any(participation[source] > float(probability_atol)):
        raise ValueError("observed sources cannot have future first-arrival mass")

    # Step zero (the source) has earliness 1; step H has earliness 0.
    step = np.arange(1, horizon + 1, dtype=np.float64)
    weights = 1.0 - step / float(horizon)
    field = mass.T @ weights
    field[source] = 1.0
    participation_with_source = participation.copy()
    participation_with_source[source] = 1.0
    conditional_mean_step = np.divide(
        step @ mass,
        participation,
        out=np.full(n_contacts, np.nan, dtype=np.float64),
        where=participation > float(probability_atol),
    )
    conditional_mean_step[source] = 0.0
    return {
        "field": field,
        "participation_probability": participation_with_source,
        "conditional_mean_first_arrival_step": conditional_mean_step,
        "earliness_weights": weights,
    }


def bidirectional_rollout_fields(
    *,
    first_arrival_minus: np.ndarray,
    first_arrival_plus: np.ndarray,
    source_minus_indices: Sequence[int],
    source_plus_indices: Sequence[int],
) -> dict[str, object]:
    """Construct exactly two fields from the two frozen source-side rollouts."""

    minus = participation_weighted_first_arrival_earliness(
        first_arrival_minus, source_indices=source_minus_indices
    )
    plus = participation_weighted_first_arrival_earliness(
        first_arrival_plus, source_indices=source_plus_indices
    )
    if minus["field"].shape != plus["field"].shape:
        raise ValueError("directional rollouts must share a contact denominator")
    if np.asarray(first_arrival_minus).shape[0] != np.asarray(first_arrival_plus).shape[0]:
        raise ValueError("directional rollouts must share a frozen horizon")
    return {
        "field_minus": minus["field"],
        "field_plus": plus["field"],
        "minus_diagnostics": minus,
        "plus_diagnostics": plus,
        "horizon": int(np.asarray(first_arrival_minus).shape[0]),
        "field_definition": FIELD_DEFINITION,
    }


def build_frozen_subject_field_record(
    *,
    subject_id: str,
    model_name: str,
    contact_names: Sequence[str],
    operator: np.ndarray,
    diffusion_result: Mapping[str, object],
    field_minus: Sequence[float],
    field_plus: Sequence[float],
    horizon: int,
    checkpoint_sha256_by_seed: Mapping[str, str],
    training_split_sha256: str,
) -> dict[str, object]:
    """Build and fingerprint one target-free subject/model field record."""

    names = tuple(map(str, contact_names))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("contact_names must contain at least two unique contacts")
    matrix = _finite_float_array(operator, name="operator", ndim=2)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("operator and contact_names are not aligned")
    minus = _finite_float_array(field_minus, name="field_minus", ndim=1)
    plus = _finite_float_array(field_plus, name="field_plus", ndim=1)
    coordinate = _finite_float_array(
        diffusion_result["diffusion_coordinate"],
        name="diffusion_coordinate",
        ndim=1,
    )
    if minus.shape != (len(names),) or plus.shape != (len(names),):
        raise ValueError("directional fields must be contact-aligned")
    if coordinate.shape != (len(names),):
        raise ValueError("diffusion coordinate must be contact-aligned")
    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    checkpoints = {str(k): str(v) for k, v in checkpoint_sha256_by_seed.items()}
    if not checkpoints or any(not value for value in checkpoints.values()):
        raise ValueError("at least one checkpoint SHA256 is required")
    if not str(training_split_sha256):
        raise ValueError("training_split_sha256 is required")
    source_minus = np.asarray(diffusion_result["source_minus_indices"], dtype=int)
    source_plus = np.asarray(diffusion_result["source_plus_indices"], dtype=int)
    if np.intersect1d(source_minus, source_plus).size:
        raise ValueError("source pools overlap")
    record: dict[str, object] = {
        "contract": FIELD_RECORD_CONTRACT,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "subject_id": str(subject_id),
        "model_name": str(model_name),
        "operator_source": "training_only_structured_seed_ensemble_no_ictal_target",
        "operator_role": "shared_source_pool_definition",
        "rollout_model_name": str(model_name),
        "target_values_read": False,
        "contact_order": list(names),
        "training_split_sha256": str(training_split_sha256),
        "checkpoint_sha256_by_seed": checkpoints,
        "operator_sha256": array_sha256(matrix),
        "horizon": int(horizon),
        # The rule travels with the pools, so a record can never claim a
        # source definition it was not actually built from.
        "source_pool_rule": str(
            diffusion_result.get("source_pool_rule", SOURCE_POOL_RULE)
        ),
        "source_minus_indices": source_minus.tolist(),
        "source_plus_indices": source_plus.tolist(),
        "source_minus_contacts": [names[index] for index in source_minus],
        "source_plus_contacts": [names[index] for index in source_plus],
        "diffusion_coordinate": coordinate.tolist(),
        "field_definition": FIELD_DEFINITION,
        "field_minus": minus.tolist(),
        "field_plus": plus.tolist(),
    }
    for optional in (
        "first_nontrivial_eigenvalue",
        "min_seed_axis_pairwise_pearson",
        "seed_axis_sign_flipped",
    ):
        if optional in diffusion_result:
            value = diffusion_result[optional]
            record[optional] = (
                dict(value) if isinstance(value, Mapping) else float(value)
            )
    record["fingerprint_sha256"] = frozen_subject_field_fingerprint(record)
    return record


def frozen_subject_field_fingerprint(record: Mapping[str, object]) -> str:
    """Recompute the immutable fingerprint for one frozen field record."""

    if record.get("contract") != FIELD_RECORD_CONTRACT:
        raise ValueError("unsupported frozen field record contract")
    if record.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("unsupported frozen field fingerprint algorithm")
    payload = dict(record)
    payload.pop("fingerprint_sha256", None)
    return _canonical_sha256(payload)


def validate_frozen_subject_field_record(record: Mapping[str, object]) -> None:
    """Fail closed on target leakage, schema drift, or fingerprint mismatch."""

    if record.get("target_values_read") is not False:
        raise ValueError("field record was not frozen under the target seal")
    if record.get("operator_source") != "training_only_structured_seed_ensemble_no_ictal_target":
        raise ValueError("field operator is not declared training-only")
    if record.get("operator_role") != "shared_source_pool_definition":
        raise ValueError("frozen operator role is not the shared source-pool definition")
    if record.get("rollout_model_name") != record.get("model_name"):
        raise ValueError("rollout model identity differs from the field record identity")
    if record.get("field_definition") != FIELD_DEFINITION:
        raise ValueError("field record does not contain the unique frozen field")
    expected = str(record.get("fingerprint_sha256", ""))
    observed = frozen_subject_field_fingerprint(record)
    if not expected or observed != expected:
        raise ValueError("frozen subject field fingerprint mismatch")


def build_frozen_field_manifest(
    records: Sequence[Mapping[str, object]],
    *,
    created_utc: str,
    code_sha256: str,
) -> dict[str, object]:
    """Build the cohort manifest that must exist before target unlock."""

    if not records:
        raise ValueError("at least one frozen field record is required")
    normalized = []
    identities = set()
    for item in records:
        validate_frozen_subject_field_record(item)
        record = _canonical_jsonable(item)
        identity = (record["subject_id"], record["model_name"])
        if identity in identities:
            raise ValueError(f"duplicate frozen field identity: {identity}")
        identities.add(identity)
        normalized.append(record)
    normalized.sort(key=lambda item: (item["subject_id"], item["model_name"]))
    manifest: dict[str, object] = {
        "contract": FIELD_MANIFEST_CONTRACT,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "created_utc": str(created_utc),
        "code_sha256": str(code_sha256),
        "target_values_read": False,
        "target_values_sealed": True,
        "n_records": len(normalized),
        "records": normalized,
    }
    manifest["manifest_sha256"] = frozen_field_manifest_fingerprint(manifest)
    return manifest


def frozen_field_manifest_fingerprint(manifest: Mapping[str, object]) -> str:
    if manifest.get("contract") != FIELD_MANIFEST_CONTRACT:
        raise ValueError("unsupported frozen field manifest contract")
    if manifest.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("unsupported manifest fingerprint algorithm")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _canonical_sha256(payload)


def validate_frozen_field_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("target_values_read") is not False:
        raise ValueError("manifest was not frozen before target values were read")
    if manifest.get("target_values_sealed") is not True:
        raise ValueError("target seal is not active in frozen field manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or int(manifest.get("n_records", -1)) != len(records):
        raise ValueError("frozen field manifest record count mismatch")
    for record in records:
        validate_frozen_subject_field_record(record)
    expected = str(manifest.get("manifest_sha256", ""))
    observed = frozen_field_manifest_fingerprint(manifest)
    if not expected or expected != observed:
        raise ValueError("frozen field manifest fingerprint mismatch")


def write_frozen_field_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    """Atomically write an already validated target-sealed field manifest."""

    validate_frozen_field_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_canonical_jsonable(manifest), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary.replace(destination)


def _centered_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    return ranks - np.mean(ranks)


def contact_label_permutations(
    contact_names: Sequence[str],
    *,
    n_draws: int = 5000,
    seed: int,
    mode: str,
) -> np.ndarray:
    """Return all-contact or within-shaft label permutations."""

    names = tuple(map(str, contact_names))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("contact names must be unique")
    if int(n_draws) < 1:
        raise ValueError("n_draws must be positive")
    if mode not in {"all_contact", "within_shaft"}:
        raise ValueError("mode must be all_contact or within_shaft")
    rng = np.random.default_rng(int(seed))
    base = np.arange(len(names), dtype=np.int64)
    if mode == "all_contact":
        groups = [base]
    else:
        by_shaft: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(names):
            shaft, _ = parse_shaft(name)
            by_shaft[str(shaft) if shaft is not None else name].append(index)
        groups = [np.asarray(indices, dtype=np.int64) for indices in by_shaft.values()]
    draws = np.tile(base, (int(n_draws), 1))
    for draw in range(int(n_draws)):
        for group in groups:
            if group.size > 1:
                draws[draw, group] = rng.permutation(group)
    return draws


def score_two_direction_max_abs_spearman(
    *,
    field_minus: Sequence[float],
    field_plus: Sequence[float],
    target: Sequence[float],
    contact_names: Sequence[str],
    n_draws: int = 5000,
    all_contact_seed: int = 0,
    within_shaft_seed: int = 1,
    min_contacts: int = 6,
) -> dict[str, object]:
    """Score one seizure and rerun abs/two-direction max in every null draw."""

    minus = np.asarray(field_minus, dtype=np.float64)
    plus = np.asarray(field_plus, dtype=np.float64)
    energy = np.asarray(target, dtype=np.float64)
    names = np.asarray(list(map(str, contact_names)), dtype=str)
    if minus.ndim != 1 or plus.shape != minus.shape or energy.shape != minus.shape:
        raise ValueError("two fields and target must be aligned vectors")
    if names.shape != minus.shape or len(set(names.tolist())) != len(names):
        raise ValueError("contact_names must be unique and field-aligned")
    finite = np.isfinite(minus) & np.isfinite(plus) & np.isfinite(energy)
    minus, plus, energy, names = minus[finite], plus[finite], energy[finite], names[finite]
    if len(energy) < int(min_contacts):
        raise ValueError(f"fewer than {int(min_contacts)} finite exact-joined contacts")
    field_ranks = np.column_stack([_centered_ranks(minus), _centered_ranks(plus)])
    target_rank = _centered_ranks(energy)
    field_norm = np.linalg.norm(field_ranks, axis=0)
    target_norm = float(np.linalg.norm(target_rank))
    if target_norm <= 1.0e-12 or np.any(field_norm <= 1.0e-12):
        raise ValueError("target and both directional fields must be non-constant")
    direction_rho = (target_rank @ field_ranks) / (target_norm * field_norm)
    observed = float(np.max(np.abs(direction_rho)))
    selected_index = int(np.argmax(np.abs(direction_rho)))

    nulls: dict[str, np.ndarray] = {}
    for mode, seed in (
        ("all_contact", int(all_contact_seed)),
        ("within_shaft", int(within_shaft_seed)),
    ):
        permutations = contact_label_permutations(
            names, n_draws=int(n_draws), seed=seed, mode=mode
        )
        permuted_target_rank = target_rank[permutations]
        # Shape is draw x direction.  Taking max on this axis is deliberately
        # inside the draw loop/vectorization, never inherited from observed.
        correlation = (
            permuted_target_rank @ field_ranks
        ) / (target_norm * field_norm[None, :])
        nulls[mode] = np.max(np.abs(correlation), axis=1)

    result: dict[str, object] = {
        "n_contacts": int(len(energy)),
        "observed_max_abs_rho": observed,
        "minus_signed_rho": float(direction_rho[0]),
        "plus_signed_rho": float(direction_rho[1]),
        "selected_direction": "minus" if selected_index == 0 else "plus",
        "n_null_draws": int(n_draws),
        "all_contact_null": nulls["all_contact"],
        "within_shaft_null": nulls["within_shaft"],
    }
    for mode, values in nulls.items():
        median = float(np.median(values))
        result[f"{mode}_null_median"] = median
        result[f"{mode}_null_p95"] = float(np.percentile(values, 95.0))
        result[f"{mode}_margin"] = observed - median
        result[f"{mode}_empirical_p"] = float(
            (1 + np.count_nonzero(values >= observed - 1.0e-15))
            / (len(values) + 1)
        )
    return result


def score_frozen_field_against_ictal(
    record: Mapping[str, object],
    *,
    seizure_id: str,
    target_contact_names: Sequence[str],
    target_values: Sequence[float],
    n_draws: int = 5000,
    all_contact_seed: int = 0,
    within_shaft_seed: int = 1,
    min_contacts: int = 6,
) -> dict[str, object]:
    """Exact-name join one frozen field record to one early-ictal seizure."""

    validate_frozen_subject_field_record(record)
    model_names = list(map(str, record["contact_order"]))
    target_names = list(map(str, target_contact_names))
    if len(target_names) != len(target_values) or len(set(target_names)) != len(target_names):
        raise ValueError("target contact names must be unique and align with target values")
    target_lookup = {name: index for index, name in enumerate(target_names)}
    matched_model_indices = [index for index, name in enumerate(model_names) if name in target_lookup]
    matched_names = [model_names[index] for index in matched_model_indices]
    matched_target = np.asarray(
        [target_values[target_lookup[name]] for name in matched_names], dtype=float
    )
    minus = np.asarray(record["field_minus"], dtype=float)[matched_model_indices]
    plus = np.asarray(record["field_plus"], dtype=float)[matched_model_indices]
    finite = np.isfinite(matched_target) & np.isfinite(minus) & np.isfinite(plus)
    matched_names = np.asarray(matched_names, dtype=str)[finite].tolist()
    matched_target = matched_target[finite]
    minus = minus[finite]
    plus = plus[finite]
    result = score_two_direction_max_abs_spearman(
        field_minus=minus,
        field_plus=plus,
        target=matched_target,
        contact_names=matched_names,
        n_draws=n_draws,
        all_contact_seed=all_contact_seed,
        within_shaft_seed=within_shaft_seed,
        min_contacts=min_contacts,
    )
    result.update(
        {
            "subject": str(record["subject_id"]),
            "model": str(record["model_name"]),
            "seizure_id": str(seizure_id),
            "field_fingerprint_sha256": str(record["fingerprint_sha256"]),
            "matched_contact_names": matched_names,
        }
    )
    return result


def _bootstrap_median_ci(
    values: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(n_boot), len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _exact_one_sided_signed_rank(values: Sequence[float], *, tie_atol: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > float(tie_atol)]
    if not len(values):
        return 1.0
    ranks = rankdata(np.abs(values), method="average")
    observed = float(np.sum(ranks[values > 0]))
    if len(values) <= 20:
        codes = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        positive = ((codes >> np.arange(len(values), dtype=np.uint64)) & 1).astype(bool)
        null = positive @ ranks
        return float(np.mean(null >= observed - 1.0e-12))
    # The registered primary cohort has 15 subjects.  This deterministic
    # normal approximation is only a defensive path for larger sensitivity
    # cohorts.
    expected = 0.5 * np.sum(ranks)
    variance = 0.25 * np.sum(np.square(ranks))
    if variance <= 0:
        return 1.0
    from scipy.stats import norm

    return float(norm.sf((observed - 0.5 - expected) / np.sqrt(variance)))


def seizure_first_patient_first_summary(
    seizure_scores: Sequence[Mapping[str, object]],
    *,
    supportive_subject: str = "epilepsiae_1146",
    n_boot: int = 5000,
    bootstrap_seed: int = 0,
    tie_atol: float = 1.0e-12,
) -> dict[str, object]:
    """Fold seizure scores within patient, then compute patient-first stats.

    Null draws are folded seizure-by-seizure at matched draw indices before
    their median, p95, margin, and empirical p are computed.  Seizures are
    never pooled as independent cohort observations.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in seizure_scores:
        grouped[(str(row["subject"]), str(row["model"]))].append(row)
    if not grouped:
        raise ValueError("no seizure scores supplied")

    patient_rows: list[dict[str, object]] = []
    for (subject, model), rows in sorted(grouped.items()):
        observed = np.asarray([row["observed_max_abs_rho"] for row in rows], dtype=float)
        patient: dict[str, object] = {
            "subject": subject,
            "model": model,
            "n_seizures": len(rows),
            "observed_max_abs_rho": float(np.median(observed)),
            "supportive_only": subject == str(supportive_subject),
        }
        for mode in ("all_contact", "within_shaft"):
            arrays = [np.asarray(row[f"{mode}_null"], dtype=float) for row in rows]
            if any(array.ndim != 1 for array in arrays):
                raise ValueError("each seizure null must be one-dimensional")
            if len({len(array) for array in arrays}) != 1:
                raise ValueError("seizure nulls need a common draw denominator")
            folded_null = np.median(np.row_stack(arrays), axis=0)
            null_median = float(np.median(folded_null))
            patient[f"{mode}_null"] = folded_null
            patient[f"{mode}_null_median"] = null_median
            patient[f"{mode}_null_p95"] = float(np.percentile(folded_null, 95.0))
            patient[f"{mode}_margin"] = patient["observed_max_abs_rho"] - null_median
            patient[f"{mode}_empirical_p"] = float(
                (1 + np.count_nonzero(folded_null >= patient["observed_max_abs_rho"] - 1.0e-15))
                / (len(folded_null) + 1)
            )
            patient[f"{mode}_exceeds_p95"] = bool(
                patient["observed_max_abs_rho"] > patient[f"{mode}_null_p95"]
            )
        patient_rows.append(patient)

    cohort: dict[str, object] = {}
    models = sorted({str(row["model"]) for row in patient_rows})
    for model_index, model in enumerate(models):
        primary = [
            row for row in patient_rows
            if row["model"] == model and not bool(row["supportive_only"])
        ]
        model_summary: dict[str, object] = {
            "n_primary_patients": len(primary),
            "supportive_subject_excluded": str(supportive_subject),
        }
        for mode in ("all_contact", "within_shaft"):
            margin = np.asarray([row[f"{mode}_margin"] for row in primary], dtype=float)
            low, high = _bootstrap_median_ci(
                margin,
                n_boot=int(n_boot),
                seed=int(bootstrap_seed) + model_index * 10 + (mode == "within_shaft"),
            )
            model_summary[mode] = {
                "median_margin": float(np.median(margin)) if len(margin) else None,
                "bootstrap_95ci": [low, high],
                "exact_wilcoxon_greater_p": _exact_one_sided_signed_rank(
                    margin, tie_atol=float(tie_atol)
                ),
                "n_positive": int(np.count_nonzero(margin > float(tie_atol))),
                "n_negative": int(np.count_nonzero(margin < -float(tie_atol))),
                "n_tied": int(np.count_nonzero(np.abs(margin) <= float(tie_atol))),
                "n_exceeds_patient_p95": int(
                    np.count_nonzero([row[f"{mode}_exceeds_p95"] for row in primary])
                ),
            }
        cohort[model] = model_summary
    return {"patients": patient_rows, "cohort": cohort}


def paired_model_patient_statistics(
    patient_rows: Sequence[Mapping[str, object]],
    *,
    model_a: str,
    model_b: str,
    null_mode: str = "all_contact",
    supportive_subject: str = "epilepsiae_1146",
    tie_atol: float = 1.0e-12,
    n_boot: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Patient-first paired contrast of null-corrected model margins."""

    if null_mode not in {"all_contact", "within_shaft"}:
        raise ValueError("unknown null_mode")
    lookup = {
        (str(row["subject"]), str(row["model"])): row
        for row in patient_rows
    }
    subjects = sorted(
        subject for subject, model in lookup
        if model == str(model_a)
        and subject != str(supportive_subject)
        and (subject, str(model_b)) in lookup
    )
    delta = np.asarray(
        [
            float(lookup[(subject, str(model_a))][f"{null_mode}_margin"])
            - float(lookup[(subject, str(model_b))][f"{null_mode}_margin"])
            for subject in subjects
        ],
        dtype=float,
    )
    low, high = _bootstrap_median_ci(delta, n_boot=int(n_boot), seed=int(bootstrap_seed))
    return {
        "model_a": str(model_a),
        "model_b": str(model_b),
        "null_mode": null_mode,
        "n_paired_primary_patients": len(subjects),
        "subjects": subjects,
        "paired_delta": delta,
        "median_delta": float(np.median(delta)) if len(delta) else None,
        "bootstrap_95ci": [low, high],
        "exact_wilcoxon_greater_p": _exact_one_sided_signed_rank(
            delta, tie_atol=float(tie_atol)
        ),
        "n_positive": int(np.count_nonzero(delta > float(tie_atol))),
        "n_negative": int(np.count_nonzero(delta < -float(tie_atol))),
        "n_tied": int(np.count_nonzero(np.abs(delta) <= float(tie_atol))),
        "supportive_subject_excluded": str(supportive_subject),
    }


__all__ = [
    "FIELD_DEFINITION",
    "FIELD_MANIFEST_CONTRACT",
    "FIELD_RECORD_CONTRACT",
    "FINGERPRINT_ALGORITHM",
    "LEARNED_AXIS_SOURCE_POOL_RULE",
    "SOURCE_POOL_RULE",
    "array_sha256",
    "bidirectional_rollout_fields",
    "build_frozen_field_manifest",
    "build_frozen_subject_field_record",
    "contact_label_permutations",
    "frozen_field_manifest_fingerprint",
    "frozen_subject_field_fingerprint",
    "learned_axis_source_pools",
    "normalized_laplacian_source_pools",
    "paired_model_patient_statistics",
    "participation_weighted_first_arrival_earliness",
    "score_frozen_field_against_ictal",
    "score_two_direction_max_abs_spearman",
    "seizure_first_patient_first_summary",
    "validate_frozen_field_manifest",
    "validate_frozen_subject_field_record",
    "write_frozen_field_manifest",
]
