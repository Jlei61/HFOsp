"""Shared Propagation Field (SPF-RNN) primitives for Topic 5.

The scientific object in this module is a patient-specific autonomous latent
field shared by all interictal rank events from that patient.  Event identity,
clinical labels, geometry, A/B labels, and future observed rank sets are not
inputs to the field.

The primary task conditions on:

* the observed first rank set;
* the number of later rank sets; and
* the cardinality of each later rank set.

The model must generate the contact identities of the complete suffix without
teacher-forcing observed contacts into its recurrent state.  The only use of
the observed prefix during likelihood evaluation is the deterministic
``already recruited`` support mask: a contact cannot be recruited twice.

This file intentionally does not call raw latent weights an effective
connectivity matrix.  An autonomous emission model does not define a general
``do(contact_i at time t)`` intervention.  Observable source-conditioned
responses and full generated-event distributions are the admissible v0.1
objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - audit utilities remain importable
    torch = None
    Tensor = Any
    nn = None


CONTRACT_NAME = "topic5_shared_propagation_field_v0_1"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 fingerprint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SubjectRankEvents:
    """One patient from the frozen masked rank-event dataset."""

    subject: str
    dataset: str
    path: Path
    contact_names: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    event_split: np.ndarray
    event_abs_time: np.ndarray
    event_source_index: np.ndarray
    input_sha256: str
    target_values_read: bool

    @property
    def train80_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 0)

    @property
    def old_heldout20_indices(self) -> np.ndarray:
        return np.flatnonzero(self.event_split == 1)

    def inner_split(self, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """Chronologically split only the old train80 pool.

        The old heldout20 has already been read by earlier RNN development and
        is therefore not used for SPF-RNN model selection.
        """
        fraction = float(validation_fraction)
        if not 0.0 < fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0, 0.5)")
        pool = self.train80_indices
        cut = int(np.floor((1.0 - fraction) * len(pool)))
        if cut < 1 or cut >= len(pool):
            raise ValueError("inner split would create an empty partition")
        return pool[:cut], pool[cut:]

    def development_split(
        self,
        validation_fraction: float,
        test_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronologically split old train80 into train / monitor / test.

        ``validation`` is used only for optimization monitoring and early
        stopping. ``test`` stays untouched until the best checkpoint has been
        selected. The previously read outer heldout20 remains excluded from
        all three partitions.
        """
        validation = float(validation_fraction)
        test = float(test_fraction)
        if validation <= 0.0 or test <= 0.0 or validation + test >= 0.5:
            raise ValueError(
                "validation_fraction and test_fraction must be positive and "
                "sum to less than 0.5"
            )
        pool = self.train80_indices
        train_cut = int(np.floor((1.0 - validation - test) * len(pool)))
        validation_cut = int(np.floor((1.0 - test) * len(pool)))
        if train_cut < 1 or validation_cut <= train_cut or validation_cut >= len(pool):
            raise ValueError("development split would create an empty partition")
        return pool[:train_cut], pool[train_cut:validation_cut], pool[validation_cut:]


def load_subject_rank_events(
    dataset_dir: Path,
    subject: str,
    *,
    verify_fingerprint: bool = True,
) -> SubjectRankEvents:
    """Load one subject and enforce the frozen target-sealed input contract."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("target_values_read", True)):
        raise RuntimeError("input manifest does not certify sealed ictal targets")
    if bool(manifest.get("ab_or_kmeans_labels_read", True)):
        raise RuntimeError("input manifest does not certify A/B-label blindness")
    if "masked" not in str(manifest.get("contract", "")).lower() and not (
        root / "PHASE0_DONE.json"
    ).exists():
        raise RuntimeError("masked rank-event provenance is not certified")

    npz_path = root / "per_subject" / f"{subject}.npz"
    metadata_path = npz_path.with_suffix(".json")
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing frozen subject artifact: {subject}")
    metadata = json.loads(metadata_path.read_text())
    actual_sha = sha256_file(npz_path)
    if "dataset_npz_sha256" not in metadata:
        raise RuntimeError(f"{subject}: metadata does not freeze dataset_npz_sha256")
    expected_sha = str(metadata["dataset_npz_sha256"])
    if verify_fingerprint and actual_sha != expected_sha:
        raise RuntimeError(f"{subject}: dataset fingerprint mismatch")

    with np.load(npz_path, allow_pickle=False) as artifact:
        required = {
            "contact_names",
            "event_group_ids",
            "event_group_count",
            "event_split",
            "event_abs_time",
            "event_source_index",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise RuntimeError(f"{subject}: missing arrays {sorted(missing)}")
        record = SubjectRankEvents(
            subject=str(subject),
            dataset=str(metadata["dataset"]),
            path=npz_path,
            contact_names=np.asarray(artifact["contact_names"]),
            group_ids=np.asarray(artifact["event_group_ids"], dtype=np.int16),
            group_count=np.asarray(artifact["event_group_count"], dtype=np.int16),
            event_split=np.asarray(artifact["event_split"], dtype=np.uint8),
            event_abs_time=np.asarray(artifact["event_abs_time"], dtype=np.float64),
            event_source_index=np.asarray(
                artifact["event_source_index"], dtype=np.int64
            ),
            input_sha256=actual_sha,
            target_values_read=bool(manifest.get("target_values_read", True)),
        )
    if record.contact_names.shape != (record.group_ids.shape[1],):
        raise RuntimeError(f"{subject}: contact_names do not align with rank columns")
    if len(np.unique(record.contact_names)) != len(record.contact_names):
        raise RuntimeError(f"{subject}: contact_names are not unique")
    if str(metadata.get("subject", "")) != record.subject:
        raise RuntimeError(f"{subject}: subject metadata mismatch")
    if str(metadata.get("dataset", "")) != record.dataset:
        raise RuntimeError(f"{subject}: dataset metadata mismatch")
    if int(metadata.get("n_contacts", -1)) != len(record.contact_names):
        raise RuntimeError(f"{subject}: metadata contact count mismatch")
    if int(metadata.get("n_events", -1)) != len(record.group_ids):
        raise RuntimeError(f"{subject}: metadata event count mismatch")
    validate_rank_event_arrays(
        record.group_ids,
        record.group_count,
        record.event_split,
        record.event_abs_time,
    )
    return record


def load_frozen_cohort(dataset_dir: Path) -> Dict[str, SubjectRankEvents]:
    """Load every subject named by the frozen manifest."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    subjects = list(map(str, manifest.get("cohort_subjects", [])))
    if not subjects:
        raise RuntimeError("dataset manifest has no cohort_subjects")
    records = {
        subject: load_subject_rank_events(root, subject) for subject in subjects
    }
    if len(records) != int(manifest.get("n_subjects_ok", len(records))):
        raise RuntimeError("loaded cohort does not match manifest subject count")
    return records


def validate_rank_event_arrays(
    group_ids: np.ndarray,
    group_count: np.ndarray,
    event_split: Optional[np.ndarray] = None,
    event_abs_time: Optional[np.ndarray] = None,
) -> None:
    """Fail closed on non-contiguous, phantom, or non-chronological events."""
    groups = np.asarray(group_ids)
    counts = np.asarray(group_count)
    if groups.ndim != 2 or counts.shape != (groups.shape[0],):
        raise ValueError("group_ids/group_count shapes are not aligned")
    if groups.shape[0] == 0 or groups.shape[1] < 2:
        raise ValueError("rank-event array is empty or has fewer than two contacts")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("group_ids must be an integer masked representation")
    if np.any(groups < -1):
        raise ValueError("group_ids contain values below the -1 nonparticipant mask")
    if np.any(counts < 1):
        raise ValueError("every event must contain at least one rank set")
    for event_index, (event, count) in enumerate(zip(groups, counts)):
        observed = np.unique(event[event >= 0])
        expected = np.arange(int(count), dtype=observed.dtype)
        if not np.array_equal(observed, expected):
            raise ValueError(
                f"event {event_index}: non-contiguous or empty rank-set ids"
            )
    if event_split is not None:
        split = np.asarray(event_split)
        if split.shape != counts.shape or np.any(~np.isin(split, [0, 1])):
            raise ValueError("event_split must be a binary event-aligned vector")
        if not np.any(split == 0) or not np.any(split == 1):
            raise ValueError("event_split must contain non-empty train and heldout sets")
        transition = np.flatnonzero(np.diff(split.astype(int)) != 0)
        if len(transition) > 1 or (
            len(transition) == 1
            and not (split[transition[0]] == 0 and split[transition[0] + 1] == 1)
        ):
            raise ValueError("event_split is not one chronological train-to-heldout cut")
    if event_abs_time is not None:
        time = np.asarray(event_abs_time, dtype=float)
        if time.shape != counts.shape or not np.all(np.isfinite(time)):
            raise ValueError("event_abs_time must be finite and event aligned")
        if np.any(np.diff(time) < 0):
            raise ValueError("events are not chronologically ordered")


def first_rank_mask(group_ids: np.ndarray) -> np.ndarray:
    """Return the observed first rank set for every event."""
    groups = np.asarray(group_ids)
    if groups.ndim != 2:
        raise ValueError("group_ids must be [event, contact]")
    first = groups == 0
    if not np.all(first.any(axis=1)):
        raise ValueError("every event must contain a first rank set")
    return first


def rank_cardinality_schedule(
    group_ids: np.ndarray,
    group_count: np.ndarray,
) -> np.ndarray:
    """Return padded cardinalities for suffix ranks 1..T-1.

    Zero denotes padding after the observed event length.  The first rank set
    is omitted because it is the conditioning variable.
    """
    groups = np.asarray(group_ids)
    counts = np.asarray(group_count)
    max_suffix = max(0, int(np.max(counts)) - 1)
    schedule = np.zeros((groups.shape[0], max_suffix), dtype=np.int16)
    for step in range(1, max_suffix + 1):
        active = counts > step
        schedule[active, step - 1] = np.sum(
            groups[active] == step, axis=1
        ).astype(np.int16)
    return schedule


def estimate_static_participation_bias(
    group_ids: np.ndarray,
    indices: Sequence[int],
    *,
    alpha: float = 0.5,
    clamp: float = 8.0,
) -> np.ndarray:
    """Train-only smoothed logit of event-level contact participation."""
    selected = np.asarray(group_ids)[np.asarray(indices, dtype=int)]
    if selected.ndim != 2 or selected.shape[0] == 0:
        raise ValueError("static bias requires non-empty training events")
    if float(alpha) <= 0:
        raise ValueError("alpha must be positive")
    count = np.sum(selected >= 0, axis=0, dtype=np.float64)
    probability = (count + float(alpha)) / (
        selected.shape[0] + 2.0 * float(alpha)
    )
    logit = np.log(probability) - np.log1p(-probability)
    # A common additive constant cancels under fixed-cardinality likelihood.
    logit -= float(np.mean(logit))
    return np.clip(logit, -float(clamp), float(clamp)).astype(np.float32)


def training_adequacy_verdict(
    validation_nll: Sequence[float],
    *,
    patience: int = 3,
    tolerance: float = 0.002,
    window: int = 3,
    initial_validation_nll: Optional[float] = None,
    minimum_relative_improvement: float = 0.001,
    minimum_epochs: int = 5,
    minimum_best_epoch: int = 3,
    stopped_by_patience: Optional[bool] = None,
) -> Dict[str, Any]:
    """Decide whether one run may contribute to a G1 likelihood comparison.

    A run whose inner-validation curve is still descending at the final epoch
    has not finished training.  Its likelihood gap against another model then
    measures optimization budget rather than mechanism, which is exactly how
    the first smoke run produced an uninterpretable ``SPF worse than Markov``
    reading from six gradient updates.  Runs flagged ``NOT_CONVERGED`` are
    development artifacts only.
    """
    values = [float(value) for value in validation_nll]
    if not values:
        raise ValueError("training adequacy needs at least one validation epoch")
    if not all(np.isfinite(values)):
        raise ValueError("validation curve contains non-finite values")
    initial = (
        float(values[0])
        if initial_validation_nll is None
        else float(initial_validation_nll)
    )
    if not np.isfinite(initial):
        raise ValueError("initial validation NLL is not finite")
    best_epoch = int(min(range(len(values)), key=lambda index: values[index]))
    epochs_since_best = len(values) - 1 - best_epoch
    span = min(int(window), len(values) - 1)
    if span >= 1:
        reference = values[-1 - span]
        recent = (reference - values[-1]) / max(abs(reference), 1e-9)
    else:
        recent = float("inf")
    relative_improvement = (initial - values[best_epoch]) / max(abs(initial), 1e-9)
    learned = bool(relative_improvement >= float(minimum_relative_improvement))
    plateaued = bool(
        epochs_since_best >= int(patience) or recent <= float(tolerance)
    )
    enough_epochs = bool(len(values) - 1 >= int(minimum_epochs))
    early_optimum = bool(best_epoch < int(minimum_best_epoch))
    stopped = plateaued if stopped_by_patience is None else bool(stopped_by_patience)
    converged = bool(
        learned
        and not early_optimum
        and plateaued
        and enough_epochs
        and stopped
    )
    if not learned:
        verdict = "NO_LEARNING_PROGRESS"
    elif early_optimum:
        verdict = "EARLY_OPTIMUM_UNVERIFIED"
    elif not enough_epochs:
        verdict = "INSUFFICIENT_EPOCHS"
    elif not plateaued or not stopped:
        verdict = "NOT_CONVERGED"
    else:
        verdict = "CONVERGED"
    return {
        "converged": converged,
        "verdict": verdict,
        "n_epochs": len(values),
        "best_epoch": best_epoch,
        "best_validation_nll": values[best_epoch],
        "final_validation_nll": values[-1],
        "initial_validation_nll": initial,
        "relative_improvement_from_initial": float(relative_improvement),
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "minimum_epochs": int(minimum_epochs),
        "minimum_best_epoch": int(minimum_best_epoch),
        "early_optimum": early_optimum,
        "stopped_by_patience": bool(stopped),
        "epochs_since_best": int(epochs_since_best),
        "recent_relative_improvement": float(recent),
        "patience": int(patience),
        "tolerance": float(tolerance),
    }


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return -(value * np.log2(value) + (1.0 - value) * np.log2(1.0 - value))


def train_precedence_entropy(group_ids: np.ndarray, indices: Sequence[int]) -> float:
    """Target-blind repertoire-diversity descriptor for pilot stratification."""
    groups = np.asarray(group_ids)[np.asarray(indices, dtype=int)]
    values = []
    for left in range(groups.shape[1]):
        for right in range(left + 1, groups.shape[1]):
            valid = (groups[:, left] >= 0) & (groups[:, right] >= 0)
            if np.sum(valid) < 10:
                continue
            delta = groups[valid, left] - groups[valid, right]
            probability = np.mean(
                (delta < 0).astype(float) + 0.5 * (delta == 0)
            )
            values.append(float(_binary_entropy(np.asarray(probability))))
    return float(np.mean(values)) if values else float("nan")


def audit_subject(record: SubjectRankEvents) -> Dict[str, Any]:
    """Compute the Phase-0 contract and identifiability inventory."""
    groups = record.group_ids
    counts = record.group_count
    participants = np.sum(groups >= 0, axis=1)
    schedule = rank_cardinality_schedule(groups, counts)
    first = first_rank_mask(groups)
    first_keys = [np.packbits(row).tobytes() for row in first]
    _, first_frequency = np.unique(first_keys, return_counts=True)
    tied_sets = 0
    all_sets = 0
    for event, count in zip(groups, counts):
        for step in range(int(count)):
            all_sets += 1
            tied_sets += int(np.sum(event == step) > 1)
    train = record.train80_indices
    inner_train, inner_validation, inner_test = record.development_split(0.15, 0.15)
    support = np.mean(groups[inner_train] >= 0, axis=0)
    return {
        "contract": CONTRACT_NAME,
        "subject": record.subject,
        "dataset": record.dataset,
        "input_sha256": record.input_sha256,
        "target_values_read": record.target_values_read,
        "n_events": int(len(groups)),
        "n_contacts": int(groups.shape[1]),
        "n_train80_events": int(len(train)),
        "n_old_heldout20_events": int(len(record.old_heldout20_indices)),
        "n_inner_train_events": int(len(inner_train)),
        "n_inner_validation_events": int(len(inner_validation)),
        "n_inner_test_events": int(len(inner_test)),
        "rank_count_median": float(np.median(counts)),
        "rank_count_q10": float(np.quantile(counts, 0.10)),
        "rank_count_q90": float(np.quantile(counts, 0.90)),
        "participant_count_median": float(np.median(participants)),
        "participant_count_q10": float(np.quantile(participants, 0.10)),
        "participant_count_q90": float(np.quantile(participants, 0.90)),
        "rank_set_size_median": float(
            np.median(schedule[schedule > 0]) if np.any(schedule > 0) else np.nan
        ),
        "rank_set_size_max": int(np.max(schedule)) if schedule.size else 0,
        "tied_rank_set_fraction": float(tied_sets / max(all_sets, 1)),
        "n_unique_first_rank_sets": int(len(first_frequency)),
        "max_events_same_first_rank_set": int(np.max(first_frequency)),
        "fraction_events_with_repeated_first_rank_set": float(
            np.sum(first_frequency[first_frequency >= 2]) / len(groups)
        ),
        "first_rank_set_size_median": float(np.median(np.sum(first, axis=1))),
        "precedence_entropy_train80": train_precedence_entropy(groups, train),
        "n_zero_support_contacts_inner_train": int(np.sum(support == 0.0)),
        "n_full_support_contacts_inner_train": int(np.sum(support == 1.0)),
        "contact_reappears_within_event": False,
        "lag_raw_available": True,
        "lag_semantics": "within-event spectrogram centroid time; not certified peak time",
        "precise_peak_time_available": False,
        "old_heldout20_status": "PREVIOUSLY_READ_NOT_CONFIRMATORY_FOR_RNNV2",
        "structural_n_min_status": "PENDING_SNN_CALIBRATION",
        "rank_event_contract": "PASS",
    }


def audit_legacy_snn_lagpat(path: Path) -> Dict[str, Any]:
    """Audit one virtual-SEEG legacy artifact without pooling SNN conditions.

    ``chnNames`` is an object array in most legacy files.  The Phase-0 audit
    deliberately does not unpickle it: the numeric rank/participation contract
    can be checked safely, while contact-name provenance remains an explicit
    unresolved field for the future frozen SNN benchmark.
    """
    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {"lagPatRank", "eventsBool"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"missing legacy SNN keys: {sorted(missing)}")
        has_channel_name_key = "chnNames" in artifact.files
        ranks = np.asarray(artifact["lagPatRank"])
        participation = np.asarray(artifact["eventsBool"], dtype=bool)
    if ranks.shape != participation.shape or ranks.ndim != 2:
        raise ValueError("legacy SNN rank and participation matrices are misaligned")
    masked = np.where(participation, ranks, np.nan)
    participant_count = participation.sum(axis=0)
    return {
        "path": str(artifact_path),
        "input_sha256": sha256_file(artifact_path),
        "n_contacts": int(ranks.shape[0]),
        "n_events": int(ranks.shape[1]),
        "channel_name_key_present": bool(has_channel_name_key),
        "channel_name_count_status": "NOT_READ_OBJECT_ARRAY",
        "participant_count_median": (
            float(np.median(participant_count)) if participant_count.size else np.nan
        ),
        "n_events_min_two_participants": int(np.sum(participant_count >= 2)),
        "finite_nonparticipant_rank_fraction_raw": float(
            np.mean(np.isfinite(ranks[~participation]))
            if np.any(~participation)
            else 0.0
        ),
        "consumer_mask_required": True,
        "masked_rank_finite_participants": bool(
            np.all(np.isfinite(masked[participation]))
        ),
    }


if nn is not None:

    def _require_bool(name: str, value: Tensor) -> None:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be bool")


    def log_elementary_symmetric(
        logits: Tensor,
        candidate_mask: Tensor,
        cardinality: Tensor,
    ) -> Tensor:
        """Exact log elementary symmetric polynomial for variable ``k``.

        For each batch row this returns

        ``log sum_{S subset candidates, |S|=k} exp(sum_{c in S} logits[c])``.
        """
        if logits.ndim != 2 or candidate_mask.shape != logits.shape:
            raise ValueError("logits/candidate_mask must be aligned [batch, contact]")
        _require_bool("candidate_mask", candidate_mask)
        k = cardinality.to(dtype=torch.long, device=logits.device)
        if k.shape != (logits.shape[0],):
            raise ValueError("cardinality must be [batch]")
        candidate_count = candidate_mask.sum(1)
        if torch.any(k < 0) or torch.any(k > candidate_count):
            raise ValueError("cardinality exceeds the candidate set")
        max_k = int(k.max().item()) if k.numel() else 0
        # ``logaddexp(-inf, -inf)`` has an undefined backward split in PyTorch
        # and can inject NaN gradients even when the corresponding impossible
        # state is not selected later.  A dtype-scaled finite sentinel is
        # numerically identical here (its exponential underflows to zero) and
        # keeps the dynamic program differentiable.
        impossible = torch.tensor(
            torch.finfo(logits.dtype).min / float(logits.shape[1] + 2),
            dtype=logits.dtype,
            device=logits.device,
        )
        if max_k <= 1:
            # Every subject in the frozen cohort has median rank-set size 1 and
            # a tied-set fraction below 1.4e-4, so this branch carries almost
            # all of the work.  e_1 is a plain logsumexp over the candidate
            # set, which removes the per-contact Python loop while staying
            # exact; the contact loop cost is what made the 26-contact subject
            # an order of magnitude slower than the rest.
            masked = torch.where(candidate_mask, logits, impossible)
            first = torch.logsumexp(masked, dim=1)
            return torch.where(k == 0, torch.zeros_like(first), first)
        dp = logits.new_full((logits.shape[0], max_k + 1), impossible)
        dp[:, 0] = 0.0
        for contact in range(logits.shape[1]):
            score = torch.where(
                candidate_mask[:, contact],
                logits[:, contact],
                impossible,
            )
            if max_k:
                include = dp[:, :-1] + score[:, None]
                dp = torch.cat(
                    [
                        dp[:, :1],
                        torch.logaddexp(dp[:, 1:], include),
                    ],
                    dim=1,
                )
        return dp.gather(1, k[:, None]).squeeze(1)


    def conditional_k_subset_log_prob(
        logits: Tensor,
        target_set: Tensor,
        candidate_mask: Tensor,
        cardinality: Optional[Tensor] = None,
    ) -> Tensor:
        """Exact unordered fixed-cardinality subset log likelihood."""
        if target_set.shape != logits.shape:
            raise ValueError("target_set must align with logits")
        _require_bool("target_set", target_set)
        _require_bool("candidate_mask", candidate_mask)
        if torch.any(target_set & ~candidate_mask):
            raise ValueError("target_set contains an ineligible contact")
        observed_k = target_set.sum(1)
        if cardinality is None:
            cardinality = observed_k
        k = cardinality.to(dtype=torch.long, device=logits.device)
        if not torch.equal(observed_k.to(torch.long), k):
            raise ValueError("target-set size disagrees with cardinality")
        log_normalizer = log_elementary_symmetric(logits, candidate_mask, k)
        numerator = torch.where(
            target_set, logits, torch.zeros_like(logits)
        ).sum(1)
        return numerator - log_normalizer


    @torch.no_grad()
    def sample_conditional_k_subset(
        logits: Tensor,
        candidate_mask: Tensor,
        cardinality: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Draw exactly from the product-weight fixed-cardinality subset law."""
        if logits.ndim != 2 or candidate_mask.shape != logits.shape:
            raise ValueError("logits/candidate_mask must be aligned [batch, contact]")
        _require_bool("candidate_mask", candidate_mask)
        k = cardinality.to(dtype=torch.long, device=logits.device)
        candidate_count = candidate_mask.sum(1)
        if torch.any(k < 0) or torch.any(k > candidate_count):
            raise ValueError("cardinality exceeds the candidate set")
        batch, contacts = logits.shape
        max_k = int(k.max().item()) if k.numel() else 0
        impossible = torch.tensor(
            torch.finfo(logits.dtype).min / float(contacts + 2),
            dtype=logits.dtype,
            device=logits.device,
        )
        suffix = logits.new_full(
            (contacts + 1, batch, max_k + 1), impossible
        )
        suffix[contacts, :, 0] = 0.0
        for contact in range(contacts - 1, -1, -1):
            suffix[contact, :, 0] = 0.0
            score = torch.where(
                candidate_mask[:, contact],
                logits[:, contact],
                impossible,
            )
            for remaining in range(1, max_k + 1):
                suffix[contact, :, remaining] = torch.logaddexp(
                    suffix[contact + 1, :, remaining],
                    score + suffix[contact + 1, :, remaining - 1],
                )

        selected = torch.zeros_like(candidate_mask)
        remaining = k.clone()
        for contact in range(contacts):
            active = remaining > 0
            available = candidate_mask[:, contact:].sum(1)
            forced = active & (remaining == available) & candidate_mask[:, contact]
            lookup = (remaining - 1).clamp_min(0)
            log_numerator = (
                logits[:, contact]
                + suffix[contact + 1].gather(1, lookup[:, None]).squeeze(1)
            )
            log_denominator = suffix[contact].gather(
                1, remaining.clamp_max(max_k)[:, None]
            ).squeeze(1)
            probability = torch.exp(log_numerator - log_denominator)
            probability = torch.where(
                active & candidate_mask[:, contact],
                probability.clamp(0.0, 1.0),
                torch.zeros_like(probability),
            )
            draw = torch.rand(
                batch,
                dtype=logits.dtype,
                device=logits.device,
                generator=generator,
            )
            include = forced | (
                active & candidate_mask[:, contact] & (draw < probability)
            )
            selected[:, contact] = include
            remaining = remaining - include.to(torch.long)
        if torch.any(remaining != 0):
            raise RuntimeError("exact subset sampler did not satisfy cardinality")
        return selected


    def suffix_log_likelihood(
        logit_fn,
        group_ids: Tensor,
        group_count: Tensor,
    ) -> Dict[str, Tensor]:
        """Exact complete-suffix likelihood shared by every comparison model.

        ``logit_fn(step, previous, active)`` returns ``[batch, contact]`` logits
        for suffix step ``step`` and may keep its own internal state.  Every
        model in the ladder is scored through this one loop so that an M0-M4
        difference is a difference in mechanism rather than in how the
        likelihood was assembled.

        The observed prefix enters here in exactly two ways: the deterministic
        ``already recruited`` support mask (a contact cannot be recruited
        twice), and the ``previous`` rank set handed to autoregressive
        baselines.  An autonomous model simply ignores ``previous``.
        """
        recruited = group_ids == 0
        previous = recruited.clone()
        max_groups = int(group_count.max().item())
        event_log_probability: Optional[Tensor] = None
        step_log_probability: list[Tensor] = []
        step_active: list[Tensor] = []
        decision_count = torch.zeros(
            group_ids.shape[0], dtype=torch.long, device=group_ids.device
        )
        for step in range(1, max_groups):
            active = group_count > step
            logits = logit_fn(step, previous, active)
            if event_log_probability is None:
                event_log_probability = logits.new_zeros(logits.shape[0])
            target = group_ids == step
            value = conditional_k_subset_log_prob(
                logits, target, ~recruited, target.sum(1)
            )
            event_log_probability = event_log_probability + torch.where(
                active, value, torch.zeros_like(value)
            )
            step_log_probability.append(
                torch.where(active, value, torch.zeros_like(value))
            )
            step_active.append(active)
            decision_count = decision_count + active.to(torch.long)
            recruited = recruited | target
            previous = target
        if event_log_probability is None:
            event_log_probability = torch.zeros(
                group_ids.shape[0],
                dtype=torch.get_default_dtype(),
                device=group_ids.device,
            )
        if step_log_probability:
            step_values = torch.stack(step_log_probability, dim=1)
            active_values = torch.stack(step_active, dim=1)
        else:
            step_values = event_log_probability.new_zeros(
                (group_ids.shape[0], 0)
            )
            active_values = torch.zeros(
                (group_ids.shape[0], 0),
                dtype=torch.bool,
                device=group_ids.device,
            )
        return {
            "event_log_probability": event_log_probability,
            "step_log_probability": step_values,
            "step_active": active_values,
            "decision_count": decision_count,
            "nll_per_event": -event_log_probability.mean(),
            "nll_per_decision": (
                -event_log_probability.sum() / decision_count.sum().clamp_min(1)
            ),
        }


    @torch.no_grad()
    def generate_from_step_logits(
        logit_fn,
        conditioning_group_ids: Tensor,
        group_count: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Free-running generation under the observed nuisance schedule.

        The first rank set, event length and per-step cardinalities are kept
        from the conditioning event; contact identity is sampled.  ``previous``
        is the model's OWN emitted set, never the observed one, so no model can
        recover from a divergent trajectory by peeking at the data.
        """
        first = conditioning_group_ids == 0
        if not torch.all(first.any(1)):
            raise ValueError("every event needs a non-empty first rank set")
        generated = torch.full_like(conditioning_group_ids, -1)
        generated[first] = 0
        recruited = first.clone()
        previous = first.clone()
        for step in range(1, int(group_count.max().item())):
            active = group_count > step
            logits = logit_fn(step, previous, active)
            cardinality = (conditioning_group_ids == step).sum(1)
            target = sample_conditional_k_subset(
                logits,
                ~recruited,
                torch.where(active, cardinality, torch.zeros_like(cardinality)),
                generator=generator,
            )
            generated[target] = int(step)
            recruited = recruited | target
            previous = target
        return generated


    def diagonal_gaussian_kl(
        q_mean: Tensor,
        q_log_variance: Tensor,
        p_mean: Tensor,
        p_log_variance: Tensor,
    ) -> Tensor:
        """Elementwise KL(q || p) for diagonal Gaussian distributions."""
        return 0.5 * (
            p_log_variance
            - q_log_variance
            + (
                torch.exp(q_log_variance)
                + (q_mean - p_mean).square()
            )
            / torch.exp(p_log_variance)
            - 1.0
        )


    def diagonal_gaussian_log_prob(
        value: Tensor,
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        """Summed diagonal-Gaussian log density."""
        return -0.5 * (
            np.log(2.0 * np.pi)
            + log_variance
            + (value - mean).square() / torch.exp(log_variance)
        ).sum(-1)


    class _GaussianHead(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), 2 * int(latent_dim)),
            )
            self.latent_dim = int(latent_dim)

        def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
            mean, log_variance = self.network(value).split(self.latent_dim, dim=-1)
            return mean, log_variance.clamp(-8.0, 4.0)


    class LatentEventModelBase(nn.Module):
        """Encoder / decoder / ELBO machinery shared by M3 and M4.

        Subclasses define only how the latent state evolves across suffix
        steps.  The future-blind initial-state prior, the training-only
        full-event posterior, the frozen static scaffold, the exact
        fixed-cardinality decoder and free-running generation are identical, so
        an M3-vs-M4 difference isolates the trajectory generator rather than
        encoder capacity or decoder family.
        """

        def __init__(
            self,
            n_contacts: int,
            static_bias: np.ndarray | Tensor,
            *,
            latent_dim: int = 4,
            encoder_hidden: int = 32,
        ):
            super().__init__()
            if int(n_contacts) < 2:
                raise ValueError("SPF-RNN requires at least two contacts")
            self.n_contacts = int(n_contacts)
            self.latent_dim = int(latent_dim)
            bias = torch.as_tensor(static_bias, dtype=torch.float32)
            if bias.shape != (self.n_contacts,):
                raise ValueError("static_bias must align with contacts")
            self.register_buffer("static_bias", bias.clone())
            self.prior_head = _GaussianHead(
                self.n_contacts, int(encoder_hidden), self.latent_dim
            )
            self.posterior_head = _GaussianHead(
                3 * self.n_contacts, int(encoder_hidden), self.latent_dim
            )
            self.contact_loading = nn.Parameter(
                torch.randn(self.n_contacts, self.latent_dim) * 0.05
            )

        def state_factory(self, initial_state: Tensor, group_count: Tensor):
            """Return ``state_at(step, active)`` for suffix steps 1..T-1."""
            raise NotImplementedError(
                "latent event models must define a trajectory generator"
            )

        def dynamics_penalty(self, states: Tensor) -> Tensor:
            """Optional stability regularizer; flat for non-recurrent models."""
            return states.new_zeros(())

        def dynamics_parameters(self) -> list[Tensor]:
            """Parameters carrying the explicit weight penalty."""
            return [self.contact_loading]

        def prior_parameters(self, first_set: Tensor) -> tuple[Tensor, Tensor]:
            _require_bool("first_set", first_set)
            if first_set.shape[-1] != self.n_contacts:
                raise ValueError("first_set does not align with contacts")
            return self.prior_head(first_set.to(self.static_bias.dtype))

        def posterior_parameters(
            self,
            group_ids: Tensor,
            group_count: Tensor,
        ) -> tuple[Tensor, Tensor]:
            if group_ids.ndim != 2 or group_ids.shape[1] != self.n_contacts:
                raise ValueError("group_ids must be [batch, contact]")
            participant = group_ids >= 0
            first = group_ids == 0
            denominator = (group_count - 1).clamp_min(1).to(torch.float32)
            normalized_rank = torch.where(
                participant,
                group_ids.to(torch.float32) / denominator[:, None],
                torch.zeros_like(group_ids, dtype=torch.float32),
            )
            feature = torch.cat(
                [
                    first.to(torch.float32),
                    participant.to(torch.float32),
                    normalized_rank,
                ],
                dim=1,
            )
            return self.posterior_head(feature)

        @staticmethod
        def _sample_gaussian(
            mean: Tensor,
            log_variance: Tensor,
            *,
            generator: Optional[torch.Generator] = None,
        ) -> Tensor:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            return mean + torch.exp(0.5 * log_variance) * noise

        def contact_logits(self, state: Tensor) -> Tensor:
            # Centering removes an unidentifiable common loading direction:
            # fixed-k subset probabilities are invariant to a common logit.
            loading = self.contact_loading - self.contact_loading.mean(
                dim=0, keepdim=True
            )
            dynamic = state @ loading.T / np.sqrt(float(self.latent_dim))
            return self.static_bias[None, :] + dynamic

        def conditional_log_likelihood(
            self,
            initial_state: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
        ) -> Dict[str, Tensor]:
            """Complete suffix log likelihood without recurrent teacher forcing."""
            if group_ids.shape != (initial_state.shape[0], self.n_contacts):
                raise ValueError("initial_state and group_ids batch/contact mismatch")
            state_at = self.state_factory(initial_state, group_count)

            def logit_fn(step, previous, active):
                # ``previous`` is deliberately unused: the trajectory is
                # autonomous given the initial state.
                return self.contact_logits(state_at(step, active))

            return suffix_log_likelihood(logit_fn, group_ids, group_count)

        def elbo_loss(
            self,
            group_ids: Tensor,
            group_count: Tensor,
            *,
            beta: float,
            free_bits: float,
            jacobian_weight: float,
            weight_decay: float,
        ) -> Dict[str, Tensor]:
            """Training-only posterior ELBO; formal generation uses the prior."""
            first = group_ids == 0
            p_mean, p_log_variance = self.prior_parameters(first)
            q_mean, q_log_variance = self.posterior_parameters(
                group_ids, group_count
            )
            initial_state = self._sample_gaussian(q_mean, q_log_variance)
            likelihood = self.conditional_log_likelihood(
                initial_state, group_ids, group_count
            )
            reconstruction = -likelihood["event_log_probability"]
            raw_kl_dim = diagonal_gaussian_kl(
                q_mean, q_log_variance, p_mean, p_log_variance
            )
            effective_kl = torch.clamp(raw_kl_dim, min=float(free_bits)).sum(1)
            jacobian = self.dynamics_penalty(initial_state)
            l2 = torch.stack(
                [value.square().mean() for value in self.dynamics_parameters()]
            ).sum()
            loss = (
                reconstruction.mean()
                + float(beta) * effective_kl.mean()
                + float(jacobian_weight) * jacobian
                + float(weight_decay) * l2
            )
            return {
                "loss": loss,
                "reconstruction_nll_per_event": reconstruction.mean(),
                "reconstruction_nll_per_decision": (
                    reconstruction.sum()
                    / likelihood["decision_count"].sum().clamp_min(1)
                ),
                "raw_kl": raw_kl_dim.sum(1).mean(),
                "effective_kl": effective_kl.mean(),
                "jacobian_penalty": jacobian,
                "weight_penalty": l2,
            }

        @torch.no_grad()
        def prior_predictive_log_likelihood(
            self,
            group_ids: Tensor,
            group_count: Tensor,
            *,
            n_samples: int = 32,
            seed: int = 0,
        ) -> Dict[str, Tensor]:
            """Monte-Carlo prior-predictive complete-event likelihood."""
            if int(n_samples) < 1:
                raise ValueError("n_samples must be positive")
            generator = torch.Generator(device=group_ids.device)
            generator.manual_seed(int(seed))
            first = group_ids == 0
            mean, log_variance = self.prior_parameters(first)
            samples = []
            step_samples = []
            active_steps: Optional[Tensor] = None
            for _ in range(int(n_samples)):
                initial = self._sample_gaussian(
                    mean, log_variance, generator=generator
                )
                conditional = self.conditional_log_likelihood(
                    initial, group_ids, group_count
                )
                samples.append(conditional["event_log_probability"])
                step_samples.append(conditional["step_log_probability"])
                active_steps = conditional["step_active"]
            sample_log_probability = torch.stack(samples, dim=0)
            event_log_probability = (
                torch.logsumexp(sample_log_probability, dim=0)
                - np.log(float(n_samples))
            )
            sample_step_log_probability = torch.stack(step_samples, dim=0)
            marginal_step_log_probability = (
                torch.logsumexp(sample_step_log_probability, dim=0)
                - np.log(float(n_samples))
            )
            assert active_steps is not None
            active_per_step = active_steps.sum(0)
            step_nll = -(
                marginal_step_log_probability * active_steps
            ).sum(0) / active_per_step.clamp_min(1)
            decisions = (group_count - 1).clamp_min(0)
            return {
                "event_log_probability": event_log_probability,
                "step_log_probability_diagnostic": marginal_step_log_probability,
                "step_active": active_steps,
                "step_nll_per_decision_diagnostic": step_nll,
                "nll_per_event": -event_log_probability.mean(),
                "nll_per_decision": (
                    -event_log_probability.sum()
                    / decisions.sum().clamp_min(1)
                ),
            }

        @torch.no_grad()
        def importance_weighted_log_likelihood(
            self,
            group_ids: Tensor,
            group_count: Tensor,
            *,
            n_samples: int = 32,
            seed: int = 0,
        ) -> Dict[str, Tensor]:
            """IWAE estimate for likelihood reporting, not for generation."""
            generator = torch.Generator(device=group_ids.device)
            generator.manual_seed(int(seed))
            first = group_ids == 0
            p_mean, p_log_variance = self.prior_parameters(first)
            q_mean, q_log_variance = self.posterior_parameters(
                group_ids, group_count
            )
            weights = []
            for _ in range(int(n_samples)):
                initial = self._sample_gaussian(
                    q_mean, q_log_variance, generator=generator
                )
                log_likelihood = self.conditional_log_likelihood(
                    initial, group_ids, group_count
                )["event_log_probability"]
                log_prior = diagonal_gaussian_log_prob(
                    initial, p_mean, p_log_variance
                )
                log_posterior = diagonal_gaussian_log_prob(
                    initial, q_mean, q_log_variance
                )
                weights.append(log_likelihood + log_prior - log_posterior)
            stacked = torch.stack(weights, dim=0)
            event_log_probability = (
                torch.logsumexp(stacked, dim=0) - np.log(float(n_samples))
            )
            decisions = (group_count - 1).clamp_min(0)
            return {
                "event_log_probability": event_log_probability,
                "nll_per_event": -event_log_probability.mean(),
                "nll_per_decision": (
                    -event_log_probability.sum()
                    / decisions.sum().clamp_min(1)
                ),
            }

        @torch.no_grad()
        def generate_conditioned(
            self,
            conditioning_group_ids: Tensor,
            group_count: Tensor,
            *,
            seed: int,
        ) -> Tensor:
            """Generate complete suffixes from the future-blind initial prior."""
            if conditioning_group_ids.ndim != 2:
                raise ValueError("conditioning_group_ids must be [event, contact]")
            first = conditioning_group_ids == 0
            generator = torch.Generator(device=conditioning_group_ids.device)
            generator.manual_seed(int(seed))
            mean, log_variance = self.prior_parameters(first)
            initial_state = self._sample_gaussian(
                mean, log_variance, generator=generator
            )
            state_at = self.state_factory(initial_state, group_count)

            def logit_fn(step, previous, active):
                return self.contact_logits(state_at(step, active))

            return generate_from_step_logits(
                logit_fn,
                conditioning_group_ids,
                group_count,
                generator=generator,
            )


    class SharedPropagationFieldRNN(LatentEventModelBase):
        """M4: patient-specific autonomous latent field with a k-subset decoder.

        The state advances by its own flow only; the emitted rank set is never
        written back.  That is what makes this the strong reading of the
        hypothesis -- the whole suffix must follow from the seed through one
        shared field -- and it is also why raw ``field_weight`` / ``contact_loading``
        do not define a contact intervention (spec v0.1 section 2.1).
        """

        def __init__(
            self,
            n_contacts: int,
            static_bias: np.ndarray | Tensor,
            *,
            latent_dim: int = 4,
            encoder_hidden: int = 32,
            jacobian_soft_cap: float = 1.5,
        ):
            super().__init__(
                n_contacts,
                static_bias,
                latent_dim=latent_dim,
                encoder_hidden=encoder_hidden,
            )
            self.jacobian_soft_cap = float(jacobian_soft_cap)
            scale = 1.0 / np.sqrt(max(self.latent_dim, 1))
            self.field_weight = nn.Parameter(
                torch.randn(self.latent_dim, self.latent_dim) * scale
            )
            self.field_bias = nn.Parameter(torch.zeros(self.latent_dim))
            self.raw_alpha = nn.Parameter(torch.tensor(0.0))

        @property
        def alpha(self) -> Tensor:
            return 0.05 + 0.90 * torch.sigmoid(self.raw_alpha)

        def field_step(self, state: Tensor) -> Tensor:
            proposal = torch.tanh(
                state @ self.field_weight.T + self.field_bias
            )
            return (1.0 - self.alpha) * state + self.alpha * proposal

        def state_factory(self, initial_state: Tensor, group_count: Tensor):
            carried = {"value": initial_state}

            def state_at(step: int, active: Tensor) -> Tensor:
                advanced = self.field_step(carried["value"])
                carried["value"] = torch.where(
                    active[:, None], advanced, carried["value"]
                )
                return carried["value"]

            return state_at

        def dynamics_parameters(self) -> list[Tensor]:
            return [
                self.field_weight,
                self.field_bias,
                self.raw_alpha,
                self.contact_loading,
            ]

        def dynamics_penalty(self, states: Tensor) -> Tensor:
            return self.jacobian_penalty(states)

        def jacobian_penalty(self, states: Tensor) -> Tensor:
            """Penalize only extreme local expansion, not all non-contraction."""
            preactivation = states @ self.field_weight.T + self.field_bias
            derivative = 1.0 - torch.tanh(preactivation).square()
            identity = torch.eye(
                self.latent_dim, dtype=states.dtype, device=states.device
            )
            jacobian = (
                (1.0 - self.alpha) * identity[None, :, :]
                + self.alpha
                * derivative[:, :, None]
                * self.field_weight[None, :, :]
            )
            spectral = torch.linalg.matrix_norm(jacobian, ord=2, dim=(-2, -1))
            return torch.relu(spectral - self.jacobian_soft_cap).square().mean()


    class PhaseConditionedPropagationFieldRNN(SharedPropagationFieldRNN):
        """Diagnostic M4 variant with the same normalized clock available to M3.

        This model is intentionally *not* autonomous.  It answers whether an
        M3 advantage is explained by access to event progress.  Its success
        cannot rescue the stronger autonomous-field claim.
        """

        def __init__(
            self,
            n_contacts: int,
            static_bias: np.ndarray | Tensor,
            *,
            latent_dim: int = 4,
            encoder_hidden: int = 32,
            jacobian_soft_cap: float = 1.5,
            phase_order: int = 2,
        ):
            super().__init__(
                n_contacts,
                static_bias,
                latent_dim=latent_dim,
                encoder_hidden=encoder_hidden,
                jacobian_soft_cap=jacobian_soft_cap,
            )
            if int(phase_order) < 1:
                raise ValueError("phase_order must be positive")
            self.phase_order = int(phase_order)
            self.phase_drive = nn.Parameter(
                torch.zeros(self.latent_dim, self.phase_order)
            )

        def field_step_with_phase(self, state: Tensor, phase: Tensor) -> Tensor:
            basis = torch.stack(
                [phase.pow(order) for order in range(1, self.phase_order + 1)],
                dim=1,
            )
            proposal = torch.tanh(
                state @ self.field_weight.T
                + basis @ self.phase_drive.T
                + self.field_bias
            )
            return (1.0 - self.alpha) * state + self.alpha * proposal

        def state_factory(self, initial_state: Tensor, group_count: Tensor):
            carried = {"value": initial_state}
            denominator = (group_count - 1).clamp_min(1).to(initial_state.dtype)

            def state_at(step: int, active: Tensor) -> Tensor:
                phase = (float(step) / denominator).clamp(0.0, 1.0)
                advanced = self.field_step_with_phase(carried["value"], phase)
                carried["value"] = torch.where(
                    active[:, None], advanced, carried["value"]
                )
                return carried["value"]

            return state_at

        def dynamics_parameters(self) -> list[Tensor]:
            return super().dynamics_parameters() + [self.phase_drive]


    class LatentTemplateModel(LatentEventModelBase):
        """M3: event-latent time-indexed template, no autonomous recurrence.

        The state is an arbitrary learned function of the initial code and
        normalized progress, ``h_t = g(z0, t/(T-1))``.  It can express any
        smooth low-dimensional template an event might follow, but it cannot
        carry state forward.  If M4 does not beat M3, the shared-field claim
        reduces to "events follow a low-dimensional time template", which is a
        materially weaker statement than "a shared flow generates them".

        M3 reads normalized progress, which M4 never sees.  That makes it the
        stronger control on purpose: ``T`` is a conditioning variable under the
        v0.1 contract, so using it is admissible.
        """

        def __init__(
            self,
            n_contacts: int,
            static_bias: np.ndarray | Tensor,
            *,
            latent_dim: int = 4,
            encoder_hidden: int = 32,
            template_hidden: int = 32,
        ):
            super().__init__(
                n_contacts,
                static_bias,
                latent_dim=latent_dim,
                encoder_hidden=encoder_hidden,
            )
            self.template = nn.Sequential(
                nn.Linear(self.latent_dim + 1, int(template_hidden)),
                nn.SiLU(),
                nn.Linear(int(template_hidden), int(template_hidden)),
                nn.SiLU(),
                nn.Linear(int(template_hidden), self.latent_dim),
            )

        def state_factory(self, initial_state: Tensor, group_count: Tensor):
            denominator = (
                (group_count - 1).clamp_min(1).to(initial_state.dtype)
            )

            def state_at(step: int, active: Tensor) -> Tensor:
                phase = (float(step) / denominator).clamp(0.0, 1.0)
                return self.template(
                    torch.cat([initial_state, phase[:, None]], dim=1)
                )

            return state_at

        def dynamics_parameters(self) -> list[Tensor]:
            return [self.contact_loading, *list(self.template.parameters())]


    @torch.no_grad()
    def generate_static_conditioned(
        static_bias: Tensor,
        conditioning_group_ids: Tensor,
        group_count: Tensor,
        *,
        seed: int,
    ) -> Tensor:
        """M0 fixed-scaffold generator under the identical schedule."""
        if static_bias.ndim != 1:
            raise ValueError("static_bias must be [contact]")
        first = conditioning_group_ids == 0
        generated = torch.full_like(conditioning_group_ids, -1)
        generated[first] = 0
        recruited = first.clone()
        generator = torch.Generator(device=conditioning_group_ids.device)
        generator.manual_seed(int(seed))
        logits = static_bias[None, :].expand(conditioning_group_ids.shape[0], -1)
        max_groups = int(group_count.max().item())
        for step in range(1, max_groups):
            active = group_count > step
            cardinality = (conditioning_group_ids == step).sum(1)
            target = sample_conditional_k_subset(
                logits,
                ~recruited,
                torch.where(active, cardinality, torch.zeros_like(cardinality)),
                generator=generator,
            )
            generated[target] = int(step)
            recruited = recruited | target
        return generated


    def baseline_conditioned_log_likelihood(
        static_bias: Tensor,
        group_ids: Tensor,
        group_count: Tensor,
        *,
        transition_residual: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Exact M0/M1 complete-suffix likelihood under the fixed schedule.

        With ``transition_residual=None`` this is M0 (static scaffold).  With a
        train-only contact-by-contact residual matrix it is M1 (first-order
        Markov).  The Markov likelihood reads only the immediately preceding
        observed rank set; it is a baseline/null, not the SPF recurrent state.
        """
        if static_bias.ndim != 1 or group_ids.ndim != 2:
            raise ValueError("static_bias/group_ids dimensions are invalid")
        contacts = group_ids.shape[1]
        if static_bias.shape != (contacts,):
            raise ValueError("static_bias does not align with group_ids")
        if transition_residual is not None and transition_residual.shape != (
            contacts,
            contacts,
        ):
            raise ValueError("transition_residual does not align with contacts")
        recruited = group_ids == 0
        previous = recruited.clone()
        event_log_probability = static_bias.new_zeros(group_ids.shape[0])
        decision_count = torch.zeros(
            group_ids.shape[0], dtype=torch.long, device=group_ids.device
        )
        max_groups = int(group_count.max().item())
        for step in range(1, max_groups):
            active = group_count > step
            logits = static_bias[None, :].expand(group_ids.shape[0], -1)
            if transition_residual is not None:
                weight = previous.to(static_bias.dtype)
                logits = logits + (
                    weight @ transition_residual
                    / weight.sum(1, keepdim=True).clamp_min(1.0)
                )
            target = group_ids == step
            step_log_probability = conditional_k_subset_log_prob(
                logits, target, ~recruited, target.sum(1)
            )
            event_log_probability += torch.where(
                active,
                step_log_probability,
                torch.zeros_like(step_log_probability),
            )
            decision_count += active.to(torch.long)
            recruited = recruited | target
            previous = target
        return {
            "event_log_probability": event_log_probability,
            "nll_per_event": -event_log_probability.mean(),
            "nll_per_decision": (
                -event_log_probability.sum() / decision_count.sum().clamp_min(1)
            ),
            "decision_count": decision_count,
        }


    @torch.no_grad()
    def generate_first_order_markov_conditioned(
        static_bias: Tensor,
        transition_residual: Tensor,
        conditioning_group_ids: Tensor,
        group_count: Tensor,
        *,
        seed: int,
    ) -> Tensor:
        """M1 matched Markov null with fixed length and rank-set sizes.

        ``transition_residual`` must be estimated from training events only.
        The generator preserves the observed first rank set and nuisance
        schedule, while contact identity after each step depends only on the
        immediately preceding generated rank set.
        """
        contacts = conditioning_group_ids.shape[1]
        if static_bias.shape != (contacts,) or transition_residual.shape != (
            contacts,
            contacts,
        ):
            raise ValueError("Markov parameters do not align with contacts")
        first = conditioning_group_ids == 0
        generated = torch.full_like(conditioning_group_ids, -1)
        generated[first] = 0
        recruited = first.clone()
        previous = first.clone()
        generator = torch.Generator(device=conditioning_group_ids.device)
        generator.manual_seed(int(seed))
        max_groups = int(group_count.max().item())
        for step in range(1, max_groups):
            active = group_count > step
            weight = previous.to(static_bias.dtype)
            drive = (
                weight @ transition_residual
                / weight.sum(1, keepdim=True).clamp_min(1.0)
            )
            logits = static_bias[None, :] + drive
            cardinality = (conditioning_group_ids == step).sum(1)
            target = sample_conditional_k_subset(
                logits,
                ~recruited,
                torch.where(active, cardinality, torch.zeros_like(cardinality)),
                generator=generator,
            )
            generated[target] = int(step)
            recruited = recruited | target
            previous = target
        return generated


    class MarkovMixtureModel(nn.Module):
        """M0 / M1 / M2 family under the identical k-subset suffix likelihood.

        ``n_components=1, use_transition=False`` is M0 (frozen scaffold only),
        ``n_components=1`` is M1 (first-order Markov residual), and
        ``n_components>1`` is M2 (a small repertoire of discrete first-order
        routes).  All three are fitted by maximizing the same objective used to
        score M3/M4, so a likelihood gap reflects mechanism rather than the
        estimator that produced the parameters.
        """

        def __init__(
            self,
            n_contacts: int,
            static_bias: np.ndarray | Tensor,
            *,
            n_components: int = 1,
            use_transition: bool = True,
            learn_bias_offset: Optional[bool] = None,
            phase_order: int = 0,
        ):
            super().__init__()
            if int(n_components) < 1:
                raise ValueError("n_components must be positive")
            self.n_contacts = int(n_contacts)
            self.n_components = int(n_components)
            self.use_transition = bool(use_transition)
            self.phase_order = int(phase_order)
            if self.phase_order < 0:
                raise ValueError("phase_order cannot be negative")
            bias = torch.as_tensor(static_bias, dtype=torch.float32)
            if bias.shape != (self.n_contacts,):
                raise ValueError("static_bias must align with contacts")
            self.register_buffer("static_bias", bias.clone())
            if learn_bias_offset is None:
                # Only a mixture needs component-specific participation
                # profiles; M0/M1 must keep exactly the shared frozen scaffold
                # so that M1's gain is attributable to the transition alone.
                learn_bias_offset = self.n_components > 1
            # Identically initialized mixture components receive identical
            # gradients forever, so a K>1 model would silently collapse to K=1
            # and be reported as "a mixture did not help".  Break the symmetry
            # at initialization; K=1 keeps the well-conditioned zero start that
            # begins exactly at the frozen scaffold.
            symmetry_break = 0.05 if self.n_components > 1 else 0.0
            self.bias_offset = nn.Parameter(
                torch.randn(self.n_components, self.n_contacts) * symmetry_break,
                requires_grad=bool(learn_bias_offset),
            )
            self.transition = nn.Parameter(
                torch.randn(
                    self.n_components, self.n_contacts, self.n_contacts
                )
                * symmetry_break,
                requires_grad=self.use_transition,
            )
            self.mixture_logit = nn.Parameter(
                torch.zeros(self.n_components),
                requires_grad=self.n_components > 1,
            )
            self.phase_basis = nn.Parameter(
                torch.zeros(
                    self.n_components,
                    self.phase_order,
                    self.n_contacts,
                ),
                requires_grad=self.phase_order > 0,
            )

        def component_logits(
            self,
            component: int,
            previous: Tensor,
            *,
            step: Optional[int] = None,
            group_count: Optional[Tensor] = None,
        ) -> Tensor:
            logits = (
                self.static_bias[None, :] + self.bias_offset[component][None, :]
            ).expand(previous.shape[0], -1)
            if self.use_transition:
                weight = previous.to(self.static_bias.dtype)
                drive = weight @ self.transition[component] / weight.sum(
                    1, keepdim=True
                ).clamp_min(1.0)
                logits = logits + drive
            if self.phase_order > 0:
                if step is None or group_count is None:
                    raise ValueError(
                        "phase-conditioned Markov logits require step/group_count"
                    )
                denominator = (group_count - 1).clamp_min(1).to(logits.dtype)
                phase = (float(step) / denominator).clamp(0.0, 1.0)
                basis = torch.stack(
                    [
                        phase.pow(order)
                        for order in range(1, self.phase_order + 1)
                    ],
                    dim=1,
                )
                logits = logits + basis @ self.phase_basis[component]
            return logits

        def event_log_probability(
            self, group_ids: Tensor, group_count: Tensor
        ) -> Tensor:
            return self._likelihood(group_ids, group_count)[
                "event_log_probability"
            ]

        def _likelihood(
            self, group_ids: Tensor, group_count: Tensor
        ) -> Dict[str, Tensor]:
            per_component = []
            per_component_steps = []
            step_active: Optional[Tensor] = None
            for component in range(self.n_components):

                def logit_fn(step, previous, active, component=component):
                    return self.component_logits(
                        component,
                        previous,
                        step=step,
                        group_count=group_count,
                    )

                likelihood = suffix_log_likelihood(
                    logit_fn, group_ids, group_count
                )
                per_component.append(likelihood["event_log_probability"])
                per_component_steps.append(likelihood["step_log_probability"])
                step_active = likelihood["step_active"]
            stacked = torch.stack(per_component, dim=0)
            log_weight = torch.log_softmax(self.mixture_logit, dim=0)
            event_log_probability = torch.logsumexp(
                stacked + log_weight[:, None], dim=0
            )
            stacked_steps = torch.stack(per_component_steps, dim=0)
            marginal_steps = torch.logsumexp(
                stacked_steps + log_weight[:, None, None], dim=0
            )
            assert step_active is not None
            active_per_step = step_active.sum(0)
            step_nll = -(marginal_steps * step_active).sum(
                0
            ) / active_per_step.clamp_min(1)
            return {
                "event_log_probability": event_log_probability,
                "step_log_probability_diagnostic": marginal_steps,
                "step_active": step_active,
                "step_nll_per_decision_diagnostic": step_nll,
            }

        def conditional_nll(
            self, group_ids: Tensor, group_count: Tensor
        ) -> Dict[str, Tensor]:
            likelihood = self._likelihood(group_ids, group_count)
            event_log_probability = likelihood["event_log_probability"]
            decisions = (group_count - 1).clamp_min(0)
            return {
                "event_log_probability": event_log_probability,
                "step_log_probability_diagnostic": likelihood[
                    "step_log_probability_diagnostic"
                ],
                "step_active": likelihood["step_active"],
                "step_nll_per_decision_diagnostic": likelihood[
                    "step_nll_per_decision_diagnostic"
                ],
                "nll_per_event": -event_log_probability.mean(),
                "nll_per_decision": (
                    -event_log_probability.sum() / decisions.sum().clamp_min(1)
                ),
            }

        @torch.no_grad()
        def generate_conditioned(
            self,
            conditioning_group_ids: Tensor,
            group_count: Tensor,
            *,
            seed: int,
        ) -> Tensor:
            """Free-running generation; the component is drawn per event."""
            generator = torch.Generator(device=conditioning_group_ids.device)
            generator.manual_seed(int(seed))
            batch = conditioning_group_ids.shape[0]
            if self.n_components == 1:
                assignment = torch.zeros(
                    batch, dtype=torch.long, device=conditioning_group_ids.device
                )
            else:
                assignment = torch.multinomial(
                    torch.softmax(self.mixture_logit, dim=0),
                    batch,
                    replacement=True,
                    generator=generator,
                )

            def logit_fn(step, previous, active):
                stacked = torch.stack(
                    [
                        self.component_logits(
                            component,
                            previous,
                            step=step,
                            group_count=group_count,
                        )
                        for component in range(self.n_components)
                    ],
                    dim=0,
                )
                index = assignment[None, :, None].expand(1, batch, self.n_contacts)
                return stacked.gather(0, index).squeeze(0)

            return generate_from_step_logits(
                logit_fn,
                conditioning_group_ids,
                group_count,
                generator=generator,
            )


    def fit_static_scaffold_ml(
        group_ids: np.ndarray,
        group_count: np.ndarray,
        indices: Sequence[int],
        *,
        steps: int = 300,
        learning_rate: float = 0.05,
        batch_events: int = 4096,
        seed: int = 0,
        device: Optional[Any] = None,
    ) -> np.ndarray:
        """Maximum-likelihood static scaffold under the exact k-subset model.

        ``estimate_static_participation_bias`` returns an event-level
        participation logit.  That is a moment estimate, not the maximizer of
        the fixed-cardinality suffix likelihood every model is scored with, so
        using it as the shared scaffold would leave an estimator confound in
        every M0/M1/M2/M3/M4 gap.  This fits the scaffold under the scoring
        objective and returns it centered, since a common additive logit
        cancels at fixed cardinality.
        """
        selected = np.asarray(indices, dtype=int)
        if selected.size == 0:
            raise ValueError("scaffold fitting requires non-empty training events")
        target = torch.device("cpu") if device is None else torch.device(device)
        groups = torch.as_tensor(
            np.asarray(group_ids)[selected], dtype=torch.long, device=target
        )
        counts = torch.as_tensor(
            np.asarray(group_count)[selected], dtype=torch.long, device=target
        )
        start = torch.as_tensor(
            estimate_static_participation_bias(group_ids, selected),
            dtype=torch.float32,
            device=target,
        )
        bias = torch.nn.Parameter(start.clone())
        optimizer = torch.optim.Adam([bias], lr=float(learning_rate))
        rng = np.random.default_rng(int(seed))
        size = min(int(batch_events), int(groups.shape[0]))
        for _ in range(int(steps)):
            batch = torch.as_tensor(
                rng.choice(groups.shape[0], size=size, replace=False),
                dtype=torch.long,
                device=target,
            )
            optimizer.zero_grad(set_to_none=True)

            def logit_fn(step, previous, active):
                return bias[None, :].expand(previous.shape[0], -1)

            loss = suffix_log_likelihood(
                logit_fn, groups[batch], counts[batch]
            )["nll_per_event"]
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            bias -= bias.mean()
        return bias.detach().cpu().numpy().astype(np.float32)


else:  # pragma: no cover - exercised only outside the torch environment

    class SharedPropagationFieldRNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SPF-RNN")
