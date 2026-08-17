"""Repair-only stateful event RNN v2.7.

This module deliberately reuses the frozen v2.6 model/data implementation and
changes only early-stopping bookkeeping.  Patience is measured against the best
*trained* epoch; the epoch-minus-one static initialization remains available only
as the nested fallback.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from src.topic5_stateful_event_rnn_v2_6 import (
    FamilyScales,
    StatefulEventRNN,
    StatefulFitResult,
    StatefulProfile,
    StatefulTrainingTrace,
    _gradient_norm,
    detach_hidden,
    family_scales_from_sequences,
    fit_continuous_ewma_ridge,
    fit_normalization,
    mean_future_descriptor,
    make_optimizer,
    profile_from_mapping,
    score_model,
    set_seed,
    stateful_family_loss,
    trace_to_dict,
    build_stateful_sequences,
)


def trained_patience_step(
    best_score: float | None,
    current_score: float,
    stale: int,
    *,
    tolerance: float = 1e-7,
) -> tuple[float, int, bool]:
    """Update patience using trained epochs only.

    The first trained epoch always initializes the tracker, regardless of how it
    compares with the untrained static initialization.
    """

    current = float(current_score)
    if best_score is None or current < float(best_score) - float(tolerance):
        return current, 0, True
    return float(best_score), int(stale) + 1, False


def checkpoint_selection_from_trace(
    trained_scores,
    *,
    static_score: float,
    minimum_epochs: int,
    patience: int,
) -> dict[str, int | float]:
    """Pure reference implementation of the v2.7 checkpoint contract.

    It is used by regression tests to exercise complete early-stopping traces
    without coupling the contract test to optimizer noise.
    """

    best_trained = None
    best_trained_epoch = -1
    best_nested = float(static_score)
    best_nested_epoch = -1
    stale = 0
    stopped_epoch = -1
    for epoch, value in enumerate(trained_scores):
        previous = best_trained
        best_trained, stale, improved = trained_patience_step(
            best_trained, float(value), stale
        )
        if improved:
            best_trained_epoch = int(epoch)
        if float(value) < best_nested - 1e-7:
            best_nested = float(value)
            best_nested_epoch = int(epoch)
        stopped_epoch = int(epoch)
        if epoch + 1 >= int(minimum_epochs) and stale >= int(patience):
            break
        del previous
    if best_trained is None:
        raise ValueError("trained_scores must contain at least one epoch")
    return {
        "best_trained_score": float(best_trained),
        "best_trained_epoch": int(best_trained_epoch),
        "best_nested_score": float(best_nested),
        "best_nested_epoch": int(best_nested_epoch),
        "stopped_epoch": int(stopped_epoch),
        "trained_stale": int(stale),
    }


def fit_stateful_event_rnn(
    train_sequences,
    validation_sequences,
    *,
    profile: StatefulProfile,
    scales: FamilyScales,
    n_modes: int,
    n_contacts: int,
    seed: int,
    maximum_epochs: int,
    minimum_epochs: int,
    patience: int,
    carry_state: bool = True,
) -> StatefulFitResult:
    """Fit the frozen v2.6 model with repaired trained-epoch patience."""

    set_seed(seed)
    mean, scale = fit_normalization(train_sequences, profile.normalization)
    output_mean = mean_future_descriptor(train_sequences)
    input_dim = train_sequences[0].tokens.shape[1]
    target_dim = train_sequences[0].targets.shape[1]
    model = StatefulEventRNN(
        input_dim, target_dim, n_modes, profile, output_mean
    )
    optimizer = make_optimizer(model, profile)
    initial_state = deepcopy(model.state_dict())
    initial_score = score_model(
        model,
        validation_sequences,
        mean=mean,
        scale=scale,
        profile=profile,
        carry_state=carry_state,
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
        formal=False,
    )

    best_trained_score = None
    best_trained_state = None
    best_trained_epoch = -1
    best_nested_score = initial_score
    best_nested_state = deepcopy(initial_state)
    best_nested_epoch = -1
    trained_stale = 0
    trace = StatefulTrainingTrace(
        train_loss=[],
        validation_trained_propagation=[],
        validation_nested_propagation=[],
        gradient_norm_mean=[],
        gradient_norm_max=[],
        clipped_fraction=[],
        state_norm_mean=[],
        best_trained_epoch=-1,
        best_nested_epoch=-1,
        stopped_epoch=-1,
        finite=True,
    )
    rng = np.random.default_rng(int(seed))

    for epoch in range(int(maximum_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        losses = []
        state_norms = []
        gradient_norms = []
        clipped = []

        for source_index in rng.permutation(len(train_sequences)):
            sequence = train_sequences[int(source_index)]
            hidden = None
            for start in range(0, len(sequence.tokens), int(profile.tbptt_length)):
                stop = min(start + int(profile.tbptt_length), len(sequence.tokens))
                values = (
                    sequence.tokens[start:stop] - mean[None, :]
                ) / scale[None, :]
                prediction, next_hidden, states = model(
                    torch.as_tensor(values[None, :, :], dtype=torch.float32),
                    hidden if carry_state else None,
                )
                hidden = detach_hidden(next_hidden) if carry_state else None
                state_norms.append(
                    float(torch.linalg.vector_norm(states.detach(), dim=-1).mean())
                )
                mask = sequence.valid_mask[start:stop]
                if np.any(mask):
                    selected = torch.as_tensor(
                        np.flatnonzero(mask), dtype=torch.long
                    )
                    target = torch.as_tensor(
                        sequence.targets[start:stop][mask], dtype=torch.float32
                    )
                    loss = stateful_family_loss(
                        prediction[0].index_select(0, selected),
                        target,
                        n_modes=n_modes,
                        n_contacts=n_contacts,
                        scales=scales,
                        participation_weight=profile.participation_weight,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite stateful RNN loss")
                    (loss / float(profile.update_chunks)).backward()
                    losses.append(float(loss.detach()))
                    pending += 1
                if pending >= int(profile.update_chunks):
                    norm = _gradient_norm(model.parameters())
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(profile.gradient_clip)
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    gradient_norms.append(norm)
                    clipped.append(float(norm > float(profile.gradient_clip)))
                    pending = 0
            hidden = None

        if pending:
            norm = _gradient_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(profile.gradient_clip)
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gradient_norms.append(norm)
            clipped.append(float(norm > float(profile.gradient_clip)))

        validation_score = score_model(
            model,
            validation_sequences,
            mean=mean,
            scale=scale,
            profile=profile,
            carry_state=carry_state,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
            formal=False,
        )
        trace.train_loss.append(float(np.mean(losses)))
        trace.validation_trained_propagation.append(
            validation_score.propagation
        )
        trace.gradient_norm_mean.append(float(np.mean(gradient_norms)))
        trace.gradient_norm_max.append(float(np.max(gradient_norms)))
        trace.clipped_fraction.append(float(np.mean(clipped)))
        trace.state_norm_mean.append(float(np.mean(state_norms)))

        tracked_score, trained_stale, trained_improved = trained_patience_step(
            None if best_trained_score is None else best_trained_score.propagation,
            validation_score.propagation,
            trained_stale,
        )
        del tracked_score
        if trained_improved:
            best_trained_score = validation_score
            best_trained_state = deepcopy(model.state_dict())
            best_trained_epoch = epoch

        if validation_score.propagation < best_nested_score.propagation - 1e-7:
            best_nested_score = validation_score
            best_nested_state = deepcopy(model.state_dict())
            best_nested_epoch = epoch

        trace.validation_nested_propagation.append(
            best_nested_score.propagation
        )
        if (
            epoch + 1 >= int(minimum_epochs)
            and trained_stale >= int(patience)
        ):
            break

    if best_trained_state is None or best_trained_score is None:
        raise RuntimeError("no trained recurrent checkpoint")
    trace.best_trained_epoch = int(best_trained_epoch)
    trace.best_nested_epoch = int(best_nested_epoch)
    trace.stopped_epoch = int(epoch)
    trace.finite = bool(
        all(torch.isfinite(parameter).all() for parameter in model.parameters())
    )

    trained_model = deepcopy(model)
    trained_model.load_state_dict(best_trained_state)
    nested_model = deepcopy(model)
    nested_model.load_state_dict(best_nested_state)
    return StatefulFitResult(
        trained_model=trained_model,
        nested_model=nested_model,
        feature_mean=mean,
        feature_scale=scale,
        profile=profile,
        trace=trace,
        trained_validation_score=best_trained_score,
        nested_validation_score=best_nested_score,
        n_modes=int(n_modes),
        n_contacts=int(n_contacts),
        n_parameters=int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        carry_state=bool(carry_state),
    )


__all__ = [
    "StatefulProfile",
    "build_stateful_sequences",
    "checkpoint_selection_from_trace",
    "family_scales_from_sequences",
    "fit_continuous_ewma_ridge",
    "fit_stateful_event_rnn",
    "mean_future_descriptor",
    "profile_from_mapping",
    "trace_to_dict",
    "trained_patience_step",
]
