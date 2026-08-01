"""Topic 5 RNN training-sufficiency and objective-sufficiency primitives.

Two questions only:

* **optimization sufficiency** -- is the frozen one-step teacher-forced model
  trained to a validation plateau?
* **objective sufficiency** -- is one-step teacher forcing enough to support
  free-running generation, or does exposure bias explain the whole-event
  negative?

Nothing here changes the model class, the observation encoder, the candidate
mask or the next-set/STOP likelihood.  The self-fed objectives replace only the
history token that enters the recurrent state; the supervised target, the
candidate mask and the decision denominator stay byte-identical to teacher
forcing so that one-step NLL remains directly comparable across objectives.
"""
from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.topic5_minimal_sequence_kernel import decomposed_next_set_stop_loss


OBJECTIVES = (
    "teacher_forced_one_step",
    "self_fed_2step",
    "self_fed_3step",
    "scheduled_sampling",
)
OPTIMIZERS = ("adamw", "adam")
#: Frozen end point of the scheduled-sampling ramp (linear in optimizer update).
SCHEDULED_SAMPLING_FINAL_PROBABILITY = 0.5


@dataclass(frozen=True)
class Objective:
    """Which history token is fed into the recurrent state at each rank step."""

    name: str

    def feeds_model_at_step(self, step: int) -> bool:
        """Rank step 0 always receives the true first rank set."""
        if self.name == "teacher_forced_one_step":
            return False
        if self.name == "self_fed_2step":
            return step % 2 == 1
        if self.name == "self_fed_3step":
            return step % 3 != 0
        if self.name == "scheduled_sampling":
            return step >= 1
        raise ValueError(f"unknown objective: {self.name}")

    def self_feed_probability(self, progress: float) -> float:
        """``progress`` is the fraction of the optimizer budget already spent."""
        if self.name == "teacher_forced_one_step":
            return 0.0
        if self.name == "scheduled_sampling":
            fraction = float(min(max(progress, 0.0), 1.0))
            return SCHEDULED_SAMPLING_FINAL_PROBABILITY * fraction
        return 1.0

    @property
    def max_consecutive_model_steps(self) -> Optional[int]:
        if self.name == "self_fed_2step":
            return 1
        if self.name == "self_fed_3step":
            return 2
        if self.name == "teacher_forced_one_step":
            return 0
        return None


def objective_from_name(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective: {name}")
    return Objective(name=str(name))


def run_environment() -> Dict[str, object]:
    """Provenance recorded in every ``run_state.json``."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover
        commit = "unknown"
    payload: Dict[str, object] = {
        "git_commit": commit,
        "hostname": platform.node(),
        "python": platform.python_version(),
    }
    if torch is not None:
        payload["torch"] = torch.__version__
        payload["cuda"] = (
            torch.version.cuda if torch.cuda.is_available() else "unavailable"
        )
        payload["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    return payload


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def development_records(records: Mapping[str, object], fraction: float):
    """Split only the chronological first 80%; keep the outer 20% sealed.

    ``event_split`` becomes 0 = inner training, 1 = inner validation and
    2 = sealed outer heldout.  ``SubjectRecord.train_indices`` /
    ``eval_indices`` therefore never touch the outer heldout again.
    """
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("inner validation fraction must lie in (0, 1)")
    out = {}
    audit = []
    for subject, record in records.items():
        first80 = record.train_indices
        n_validation = max(1, int(round(len(first80) * float(fraction))))
        n_training = len(first80) - n_validation
        if n_training < 1:
            raise RuntimeError(f"{subject}: insufficient inner-training events")
        split = np.full(record.event_split.shape, 2, dtype=np.uint8)
        split[first80[:n_training]] = 0
        split[first80[n_training:]] = 1
        out[subject] = replace(record, event_split=split)
        audit.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_inner_train": int(n_training),
                "n_inner_validation": int(n_validation),
                "n_outer_heldout_sealed": int(len(record.eval_indices)),
            }
        )
    return out, audit


# --------------------------------------------------------------------------
# forward pass with an optional self-fed history token
# --------------------------------------------------------------------------


if torch is not None:

    def scheduled_forward(
        model,
        contact_features: "torch.Tensor",
        contact_mask: "torch.Tensor",
        group_ids: "torch.Tensor",
        group_count: "torch.Tensor",
        local_offset: "torch.Tensor",
        *,
        objective: Objective,
        self_feed_probability: float,
        generator: Optional["torch.Generator"] = None,
    ) -> Dict[str, object]:
        """Teacher-forced likelihood with an optionally self-fed history token.

        The decision set, the candidate mask and the target are exactly those of
        the frozen teacher-forced contract.  Only the token that advances the
        recurrent state may be replaced by the model's own sample, and a
        replaced token is drawn from contacts the model has not already
        emitted.
        """
        embedding, encoder_input = model._encode(contact_features, local_offset)
        hidden = model._initial_hidden(embedding, contact_mask)
        true_recruited = torch.zeros_like(contact_mask)
        fed_recruited = torch.zeros_like(contact_mask)
        model_emitted = torch.zeros_like(contact_mask)
        batch = contact_mask.shape[0]
        rows = torch.arange(batch, device=contact_mask.device)
        model_fed_per_row = torch.zeros(
            batch, dtype=torch.long, device=contact_mask.device
        )
        max_groups = int(group_count.max().item())

        action_logits = []
        stop_logits = []
        candidate_masks = []
        # counters stay on device and are read once at the end; a per-step
        # ``.item()`` would force a synchronisation on every rank step
        zero = torch.zeros((), dtype=torch.long, device=contact_mask.device)
        n_model_fed = zero.clone()
        n_tie_fallback = zero.clone()
        n_eligible_fed = zero.clone()

        for step in range(max_groups + 1):
            candidate = contact_mask & ~true_recruited
            action, stop = model._decode(
                embedding, encoder_input, hidden, candidate
            )
            action_logits.append(action)
            stop_logits.append(stop)
            candidate_masks.append(candidate)
            if step == max_groups:
                break

            current_true = (group_ids == step) & contact_mask
            active = group_count > step
            current_fed = current_true

            if objective.feeds_model_at_step(step) and float(
                self_feed_probability
            ) > 0.0:
                singleton = current_true.sum(1) == 1
                sample_candidate = contact_mask & ~fed_recruited
                use_model = active & singleton & sample_candidate.any(1)
                n_eligible_fed = n_eligible_fed + active.sum()
                n_tie_fallback = n_tie_fallback + (active & ~singleton).sum()
                if float(self_feed_probability) < 1.0:
                    draw = torch.rand(
                        (batch,),
                        generator=generator,
                        device=contact_mask.device,
                        dtype=torch.float32,
                    )
                    use_model = use_model & (draw < float(self_feed_probability))
                # the sampling branch runs unconditionally; testing
                # ``use_model.any()`` first would synchronise every rank step
                with torch.no_grad():
                    sample_logits, _ = model._decode(
                        embedding, encoder_input, hidden, sample_candidate
                    )
                    sampled = torch.multinomial(
                        torch.softmax(sample_logits, dim=1),
                        1,
                        generator=generator,
                    ).squeeze(1)
                model_set = torch.zeros_like(contact_mask)
                model_set[rows, sampled] = True
                # never re-emit a contact the model already produced
                model_set = model_set & sample_candidate
                current_fed = torch.where(
                    use_model.unsqueeze(1), model_set, current_true
                )
                n_model_fed = n_model_fed + use_model.sum()
                active_use = use_model & active
                model_emitted = model_emitted | (
                    model_set & active_use.unsqueeze(1)
                )
                model_fed_per_row = model_fed_per_row + active_use.long()

            active_column = active.unsqueeze(1)
            updated_fed = fed_recruited | current_fed
            updated_true = true_recruited | current_true
            # ``_advance`` reads ``recruited`` only through its cardinality, and
            # a genuine free run advances by exactly one contact per rank step.
            # Passing the true prefix therefore reproduces the free-running
            # progress signal even when a self-fed contact coincides with a
            # contact the true path uses later.
            updated_hidden = model._advance(
                embedding, current_fed, updated_true, hidden, contact_mask
            )
            hidden = torch.where(active_column, updated_hidden, hidden)
            fed_recruited = torch.where(active_column, updated_fed, fed_recruited)
            true_recruited = torch.where(
                active_column, updated_true, true_recruited
            )

        return {
            "contact_logits": torch.stack(action_logits, dim=1),
            "stop_logits": torch.stack(stop_logits, dim=1),
            "candidate_mask": torch.stack(candidate_masks, dim=1),
            "self_feed_counters": torch.stack(
                [n_model_fed, n_tie_fallback, n_eligible_fed]
            ),
            "fed_recruited": fed_recruited,
            "true_recruited": true_recruited,
            "model_emitted": model_emitted,
            "model_fed_steps_per_event": model_fed_per_row,
        }

    @torch.no_grad()
    def paired_native_rollout(
        model,
        contact_features: "torch.Tensor",
        contact_mask: "torch.Tensor",
        local_offset: "torch.Tensor",
        source_mask: np.ndarray,
        uniforms: np.ndarray,
    ):
        """The model's *own* free run from a revealed first rank set.

        The constructive generator samples from ``static scaffold + frozen
        ordered residual``; a self-fed training objective instead makes the
        model robust to samples from its own next-contact head.  This rollout
        closes that gap: it is the same source-conditioned protocol, but every
        contact and the STOP decision come from the model's own joint
        distribution, so the training and evaluation input distributions match.

        Sampling is inverse-CDF from ``uniforms`` so that different models
        consume identical random numbers.
        """
        from src.topic5_constructive_event_generator import categorical_from_uniform

        source = np.asarray(source_mask, dtype=bool)
        random_uniforms = np.asarray(uniforms, dtype=np.float64)
        n_events, n_contacts = source.shape
        if random_uniforms.shape != (n_events, n_contacts):
            raise ValueError("uniforms must be event x contact")
        if not np.all(source.any(axis=1)):
            raise ValueError("every event needs a non-empty revealed source set")

        model.eval()
        device = contact_features.device
        groups = np.full((n_events, n_contacts), -1, dtype=np.int16)
        groups[source] = 0
        counts = np.ones(n_events, dtype=np.int16)

        features = contact_features[:1].expand(n_events, -1, -1)
        mask = contact_mask[:1].expand(n_events, -1)
        embedding, encoder_input = model._encode(features, local_offset)
        hidden = model._initial_hidden(embedding, mask)
        recruited = torch.as_tensor(source, dtype=torch.bool, device=device).clone()
        hidden = model._advance(embedding, recruited, recruited, hidden, mask)
        alive = np.ones(n_events, dtype=bool)
        count = np.ones(n_events, dtype=np.int64)

        for step in range(n_contacts - 1):
            candidate = mask & ~recruited
            alive = alive & candidate.any(1).cpu().numpy()
            if not alive.any():
                break
            contact_logits, stop_logit = model._decode(
                embedding, encoder_input, hidden, candidate
            )
            action_logits = torch.cat([stop_logit[:, None], contact_logits], dim=1)
            probability = torch.softmax(action_logits, dim=1).cpu().numpy()
            # a finished event keeps choosing STOP, which is action index 0
            terminated = np.zeros_like(probability)
            terminated[:, 0] = 1.0
            probability = np.where(alive[:, None], probability, terminated)
            action = categorical_from_uniform(
                probability, random_uniforms[:, step]
            )
            action = np.where(alive, action, 0)
            chose_contact = action > 0
            new_set = torch.zeros_like(recruited)
            if chose_contact.any():
                rows = np.flatnonzero(chose_contact)
                picked = action[rows] - 1
                groups[rows, picked] = count[rows].astype(np.int16)
                new_set[
                    torch.as_tensor(rows, dtype=torch.long, device=device),
                    torch.as_tensor(picked, dtype=torch.long, device=device),
                ] = True
                count[rows] += 1
            alive = alive & chose_contact
            recruited = recruited | new_set
            updated = model._advance(embedding, new_set, recruited, hidden, mask)
            hidden = torch.where(
                torch.as_tensor(alive, device=device).unsqueeze(1), updated, hidden
            )
        counts = count.astype(np.int16)
        if not np.all(groups[source] == 0):
            raise RuntimeError("a revealed source was not retained at rank zero")
        return groups, counts

    def _build_optimizer(
        groups: Sequence[dict],
        optimizer_name: str,
    ):
        name = str(optimizer_name).lower()
        if name == "adamw":
            return torch.optim.AdamW(list(groups))
        if name == "adam":
            for group in groups:
                if float(group.get("weight_decay", 0.0)) != 0.0:
                    raise ValueError("the Adam sensitivity is frozen at weight decay 0")
            return torch.optim.Adam(list(groups))
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    def _flat_parameters(parameters) -> "torch.Tensor":
        return torch.cat([p.detach().reshape(-1) for p in parameters])

    def _dataset_balanced_patient_order(records, rng):
        """Return every patient once, interleaving datasets when possible."""
        pools: Dict[str, list] = {}
        for record in records:
            pools.setdefault(record.dataset, []).append(record)
        for dataset, pool in pools.items():
            order = rng.permutation(len(pool))
            pools[dataset] = [pool[int(index)] for index in order]
        datasets = sorted(pools)
        ordered = []
        for position in range(max(len(pool) for pool in pools.values())):
            for dataset in datasets:
                if position < len(pools[dataset]):
                    ordered.append(pools[dataset][position])
        return ordered

    def train_coverage_instrumented(
        model,
        records: Sequence[object],
        *,
        coverage_cycles: int,
        updates_per_patient: int,
        batch_size: int,
        learning_rate: float,
        local_learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        local_offset_dim: int,
        device: "torch.device",
        seed: int,
        objective: Objective,
        optimizer_name: str = "adamw",
        batch_builder: Optional[Callable] = None,
        on_cycle_end: Optional[Callable] = None,
    ):
        """Coverage-cycle shared training with full optimizer instrumentation.

        Identical event coverage and update boundaries to the frozen
        ``train_shared_coverage``; it additionally records per-step update norm,
        clipping, and the self-feeding diagnostics, and can call back at the end
        of every coverage cycle so that nested cycle budgets are read from one
        run.
        """
        if batch_builder is None:
            from scripts.train_topic5_interictal_rank_distribution import _batch

            batch_builder = _batch
        model.to(device)
        rng = np.random.default_rng(int(seed))
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 90_000_011)
        offsets = {
            record.subject: torch.nn.Parameter(
                torch.zeros(
                    (record.contact_features.shape[0], int(local_offset_dim)),
                    dtype=torch.float32,
                    device=device,
                )
            )
            for record in records
        }
        optimizer = _build_optimizer(
            [
                {
                    "params": list(model.parameters()),
                    "lr": float(learning_rate),
                    "weight_decay": float(weight_decay),
                },
                {
                    "params": list(offsets.values()),
                    "lr": float(local_learning_rate),
                    "weight_decay": float(weight_decay),
                },
            ],
            optimizer_name,
        )
        total_updates = int(coverage_cycles) * len(records) * int(updates_per_patient)
        rows: list[dict] = []
        cycle_snapshots: Dict[int, dict] = {}
        global_update = 0
        start = time.time()
        for cycle in range(int(coverage_cycles)):
            for record in _dataset_balanced_patient_order(records, rng):
                indices = rng.permutation(record.train_indices)
                segments = [
                    segment
                    for segment in np.array_split(indices, int(updates_per_patient))
                    if len(segment)
                ]
                for segment_index, segment in enumerate(segments):
                    optimizer.zero_grad(set_to_none=True)
                    # scalars are accumulated on device and read back once per
                    # optimizer step; a per-chunk ``.cpu()`` would synchronise
                    losses = torch.zeros(3, device=device)
                    counters = torch.zeros(3, dtype=torch.long, device=device)
                    n_chunks = 0
                    probability = objective.self_feed_probability(
                        global_update / max(total_updates - 1, 1)
                    )
                    for batch_start in range(0, len(segment), int(batch_size)):
                        chunk = segment[batch_start : batch_start + int(batch_size)]
                        batch = batch_builder(
                            record,
                            chunk,
                            device,
                            rank_shuffle=False,
                            rng=rng,
                        )
                        outputs = scheduled_forward(
                            model,
                            batch["contact_features"],
                            batch["contact_mask"],
                            batch["group_ids"],
                            batch["group_count"],
                            offsets[record.subject],
                            objective=objective,
                            self_feed_probability=probability,
                            generator=generator,
                        )
                        loss = decomposed_next_set_stop_loss(
                            outputs, batch["group_ids"], batch["group_count"]
                        )
                        weight = len(chunk) / len(segment)
                        (loss["total"] * weight).backward()
                        losses = losses + weight * torch.stack(
                            [
                                loss["total"].detach(),
                                loss["event_contact_choice_nll"].mean().detach(),
                                loss["event_stop_contribution_nll"].mean().detach(),
                            ]
                        )
                        counters = counters + outputs["self_feed_counters"]
                        n_chunks += 1
                    trainable = [*model.parameters(), offsets[record.subject]]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, float(gradient_clip)
                    )
                    before = _flat_parameters(trainable)
                    optimizer.step()
                    update_norm = torch.linalg.vector_norm(
                        _flat_parameters(trainable) - before
                    )
                    scalars = torch.cat(
                        [losses, torch.stack([grad_norm.detach(), update_norm])]
                    ).cpu()
                    weighted_loss = float(scalars[0])
                    weighted_contact = float(scalars[1])
                    weighted_stop = float(scalars[2])
                    gradient_norm = float(scalars[3])
                    update_norm = float(scalars[4])
                    model_fed, tie_fallback, eligible = (
                        int(value) for value in counters.cpu()
                    )
                    global_update += 1
                    rows.append(
                        {
                            "phase": "shared_full_coverage",
                            "coverage_cycle": cycle + 1,
                            "patient_update": segment_index + 1,
                            "global_update": global_update,
                            "subject": record.subject,
                            "dataset": record.dataset,
                            "n_events": int(len(segment)),
                            "n_backward_chunks": n_chunks,
                            "loss": weighted_loss,
                            "contact_choice_nll": weighted_contact,
                            "stop_contribution_nll": weighted_stop,
                            "gradient_norm": gradient_norm,
                            "clipped": bool(gradient_norm > float(gradient_clip)),
                            "parameter_update_norm": update_norm,
                            "self_feed_probability": float(probability),
                            "n_model_fed_steps": model_fed,
                            "n_tie_fallback_steps": tie_fallback,
                            "n_self_feed_eligible_steps": eligible,
                            "elapsed_seconds": time.time() - start,
                        }
                    )
            snapshot = {
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "offsets": {
                    subject: value.detach().cpu().clone()
                    for subject, value in offsets.items()
                },
                "global_update": global_update,
            }
            cycle_snapshots[cycle + 1] = snapshot
            if on_cycle_end is not None:
                on_cycle_end(cycle + 1, model, offsets)
        coverage = {
            record.subject: {
                "events_available": int(record.train_indices.size),
                "drawn": int(record.train_indices.size * int(coverage_cycles)),
                "completed_cycles": int(coverage_cycles),
                "fraction_of_first_cycle": 1.0,
            }
            for record in records
        }
        return cycle_snapshots, rows, coverage

    def calibrate_offset_instrumented(
        model,
        record,
        *,
        coverage_cycles: int,
        updates_per_cycle: int,
        batch_size: int,
        local_learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        local_offset_dim: int,
        device: "torch.device",
        seed: int,
        objective: Objective,
        snapshot_cycles: Sequence[int] = (),
        batch_builder: Optional[Callable] = None,
    ):
        """Freeze the core and fit only the held-out patient's local offsets.

        Snapshots at the requested cycles let one calibration run report several
        calibration budgets without retraining the shared core.
        """
        if batch_builder is None:
            from scripts.train_topic5_interictal_rank_distribution import _batch

            batch_builder = _batch
        model.to(device)
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        offset = torch.nn.Parameter(
            torch.zeros(
                (record.contact_features.shape[0], int(local_offset_dim)),
                dtype=torch.float32,
                device=device,
            )
        )
        optimizer = torch.optim.AdamW(
            [offset],
            lr=float(local_learning_rate),
            weight_decay=float(weight_decay),
        )
        rng = np.random.default_rng(int(seed))
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 40_000_003)
        wanted = {int(value) for value in snapshot_cycles} | {int(coverage_cycles)}
        snapshots: Dict[int, "torch.Tensor"] = {}
        rows: list[dict] = []
        global_update = 0
        total_updates = int(coverage_cycles) * int(updates_per_cycle)
        start = time.time()
        for cycle in range(int(coverage_cycles)):
            indices = rng.permutation(record.train_indices)
            segments = [
                segment
                for segment in np.array_split(indices, int(updates_per_cycle))
                if len(segment)
            ]
            for segment_index, segment in enumerate(segments):
                optimizer.zero_grad(set_to_none=True)
                accumulated = torch.zeros((), device=device)
                probability = objective.self_feed_probability(
                    global_update / max(total_updates - 1, 1)
                )
                for batch_start in range(0, len(segment), int(batch_size)):
                    chunk = segment[batch_start : batch_start + int(batch_size)]
                    batch = batch_builder(
                        record, chunk, device, rank_shuffle=False, rng=rng
                    )
                    outputs = scheduled_forward(
                        model,
                        batch["contact_features"],
                        batch["contact_mask"],
                        batch["group_ids"],
                        batch["group_count"],
                        offset,
                        objective=objective,
                        self_feed_probability=probability,
                        generator=generator,
                    )
                    loss = decomposed_next_set_stop_loss(
                        outputs, batch["group_ids"], batch["group_count"]
                    )
                    weight = len(chunk) / len(segment)
                    (loss["total"] * weight).backward()
                    accumulated = accumulated + weight * loss["total"].detach()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [offset], float(gradient_clip)
                )
                before = _flat_parameters([offset])
                optimizer.step()
                scalars = torch.stack(
                    [
                        accumulated,
                        grad_norm.detach(),
                        torch.linalg.vector_norm(_flat_parameters([offset]) - before),
                    ]
                ).cpu()
                weighted_loss = float(scalars[0])
                gradient_norm = float(scalars[1])
                update_norm = float(scalars[2])
                global_update += 1
                rows.append(
                    {
                        "phase": "heldout_offset_full_coverage",
                        "coverage_cycle": cycle + 1,
                        "patient_update": segment_index + 1,
                        "global_update": global_update,
                        "subject": record.subject,
                        "dataset": record.dataset,
                        "n_events": int(len(segment)),
                        "loss": weighted_loss,
                        "gradient_norm": gradient_norm,
                        "clipped": bool(gradient_norm > float(gradient_clip)),
                        "parameter_update_norm": update_norm,
                        "elapsed_seconds": time.time() - start,
                    }
                )
            if cycle + 1 in wanted:
                snapshots[cycle + 1] = offset.detach().clone()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        coverage = {
            "events_available": int(record.train_indices.size),
            "drawn": int(record.train_indices.size * int(coverage_cycles)),
            "completed_cycles": int(coverage_cycles),
            "fraction_of_first_cycle": 1.0,
        }
        return snapshots, rows, coverage

    @torch.no_grad()
    def evaluate_decomposed(
        model,
        record,
        offset: "torch.Tensor",
        *,
        device: "torch.device",
        batch_size: int = 256,
        indices: Optional[np.ndarray] = None,
        batch_builder: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Teacher-forced held-out likelihood, split into contact choice and STOP.

        Evaluation is always teacher forced and identical for every training
        objective, so the same-denominator contract holds across Phase C.
        """
        if batch_builder is None:
            from scripts.train_topic5_interictal_rank_distribution import _batch

            batch_builder = _batch
        model.eval()
        evaluation_indices = (
            record.eval_indices if indices is None else np.asarray(indices, int)
        )
        totals = []
        contacts = []
        stops = []
        decisions = 0
        nonterminal = 0
        for start in range(0, evaluation_indices.size, int(batch_size)):
            chunk = evaluation_indices[start : start + int(batch_size)]
            batch = batch_builder(
                record, chunk, device, rank_shuffle=False, rng=np.random.default_rng(0)
            )
            outputs = model(**batch, local_offset=offset)
            loss = decomposed_next_set_stop_loss(
                outputs, batch["group_ids"], batch["group_count"]
            )
            totals.append(loss["event_total_nll"].detach().cpu().numpy())
            contacts.append(
                loss["event_contact_choice_nll"].detach().cpu().numpy()
            )
            stops.append(
                loss["event_stop_contribution_nll"].detach().cpu().numpy()
            )
            decisions += int(loss["decision_mask"].sum().item())
            nonterminal += int(loss["nonterminal_mask"].sum().item())
        return {
            "n_events": int(evaluation_indices.size),
            "n_decisions": decisions,
            "n_nonterminal_decisions": nonterminal,
            "event_total_nll": float(np.mean(np.concatenate(totals))),
            "contact_choice_nll": float(np.mean(np.concatenate(contacts))),
            "stop_contribution_nll": float(np.mean(np.concatenate(stops))),
        }


# --------------------------------------------------------------------------
# patient-first aggregation and the frozen plateau rule
# --------------------------------------------------------------------------


def plan_cells(cells: Sequence[str], root: Path, *, done_name: str = "DONE.json"):
    """Resume plan for a manifest of run cells.

    A cell that already carries ``DONE.json`` is complete and must not be
    re-run.  A directory that exists without ``DONE.json`` is a partial run and
    blocks resume loudly rather than being silently overwritten.
    """
    root = Path(root)
    complete, pending, blocked = [], [], []
    for cell in cells:
        path = root / str(cell)
        if (path / done_name).is_file():
            complete.append(str(cell))
        elif path.exists():
            blocked.append(str(cell))
        else:
            pending.append(str(cell))
    return {
        "n_cells": len(list(cells)),
        "complete": complete,
        "pending": pending,
        "blocked": blocked,
    }


def aggregate_patient_metric(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    patient_key: str = "subject",
    seed_key: str = "seed",
) -> Dict[str, object]:
    """Seed-mean inside a patient first, then unweighted patient statistics.

    Patients contribute exactly one value regardless of how many events or
    seeds they carry, so event-rich patients cannot dominate the cohort number.
    """
    per_patient: Dict[str, list] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        per_patient.setdefault(str(row[patient_key]), []).append((row.get(seed_key), value))
    collapsed = {
        subject: float(np.mean([value for _, value in entries]))
        for subject, entries in per_patient.items()
    }
    summary = patient_first_summary(collapsed)
    summary["per_patient"] = collapsed
    summary["n_seeds_per_patient"] = {
        subject: len(entries) for subject, entries in per_patient.items()
    }
    return summary


def patient_first_summary(values: Mapping[str, float]) -> Dict[str, float]:
    """One number per patient first, then unweighted cohort statistics.

    Patients with many events must not dominate, so every cohort statistic is
    computed over the per-patient values and never over pooled events.
    """
    array = np.asarray([float(value) for value in values.values()], dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n_patients": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "sd": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "n_patients": int(finite.size),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def plateau_verdict(
    cycle_medians: Sequence[float],
    *,
    threshold: float = 0.002,
) -> Dict[str, object]:
    """Frozen convergence rule on the patient-median contact-choice NLL.

    A plateau requires two consecutive coverage cycles whose patient-median
    validation improvement is below ``threshold`` nats/decision.  Improvement is
    positive when the loss falls.
    """
    values = [float(value) for value in cycle_medians]
    if len(values) < 2:
        return {
            "improvements": [],
            "plateau_reached": False,
            "reason": "fewer than two coverage cycles",
            "threshold": float(threshold),
        }
    improvements = [values[i - 1] - values[i] for i in range(1, len(values))]
    plateau = False
    plateau_at = None
    for index in range(1, len(improvements)):
        if (
            improvements[index - 1] < float(threshold)
            and improvements[index] < float(threshold)
        ):
            plateau = True
            plateau_at = index + 2  # cycle number of the second quiet cycle
            break
    return {
        "improvements": improvements,
        "plateau_reached": bool(plateau),
        "plateau_at_cycle": plateau_at,
        "final_improvement": improvements[-1],
        "threshold": float(threshold),
        "reason": (
            "two consecutive cycles below threshold"
            if plateau
            else "still improving at the final cycle"
        ),
    }
