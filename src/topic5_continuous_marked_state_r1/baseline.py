"""Deterministic-history timing and exact tied-mark baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .design import SplitDesign
from .mark_likelihood import TiedMarkTerms, tied_group_mark_log_prob


class HistoryIntensity(nn.Module):
    """Log-linear conditional intensity on deterministic causal history."""

    def __init__(self, history_dim: int, *, history_visible: bool = True):
        super().__init__()
        self.history_visible = bool(history_visible)
        self.intercept = nn.Parameter(torch.zeros(()))
        self.weight = nn.Parameter(torch.zeros(history_dim))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if not self.history_visible:
            return self.intercept.expand(history.shape[:-1])
        return self.intercept + torch.matmul(history, self.weight)


@dataclass(frozen=True)
class IntensityMetrics:
    nll_per_event: float
    event_log_intensity_per_event: float
    survival_per_event: float
    predicted_events: float
    observed_events: int
    recorded_hours: float


def intensity_loss(model: HistoryIntensity, design: SplitDesign,
                   *, device: torch.device | str = "cpu",
                   quadrature_chunk: int = 131072) -> torch.Tensor:
    event_history = torch.as_tensor(design.event_history, device=device)
    event_term = model(event_history).sum()
    survival = event_term.new_zeros(())
    for lo in range(0, len(design.quadrature_history), int(quadrature_chunk)):
        hi = min(lo + int(quadrature_chunk), len(design.quadrature_history))
        history = torch.as_tensor(design.quadrature_history[lo:hi], device=device)
        weight = torch.as_tensor(
            design.quadrature_weight_seconds[lo:hi],
            dtype=history.dtype, device=device,
        )
        survival = survival + torch.sum(weight * torch.exp(
            torch.clamp(model(history), max=20.0)
        ))
    return (survival - event_term) / max(len(design.event_index), 1)


@torch.no_grad()
def intensity_metrics(model: HistoryIntensity, design: SplitDesign,
                      *, device: torch.device | str = "cpu") -> IntensityMetrics:
    event_history = torch.as_tensor(design.event_history, device=device)
    event_term = model(event_history).sum()
    survival = event_term.new_zeros(())
    for lo in range(0, len(design.quadrature_history), 131072):
        hi = min(lo + 131072, len(design.quadrature_history))
        history = torch.as_tensor(design.quadrature_history[lo:hi], device=device)
        weight = torch.as_tensor(
            design.quadrature_weight_seconds[lo:hi],
            dtype=history.dtype, device=device,
        )
        survival += torch.sum(weight * torch.exp(torch.clamp(model(history), max=20.0)))
    n = max(len(design.event_index), 1)
    return IntensityMetrics(
        nll_per_event=float((survival - event_term) / n),
        event_log_intensity_per_event=float(event_term / n),
        survival_per_event=float(survival / n),
        predicted_events=float(survival),
        observed_events=int(len(design.event_index)),
        recorded_hours=float(design.recorded_seconds / 3600.0),
    )


def fit_history_intensity(train: SplitDesign, *, history_visible: bool = True,
                          l2: float = 1e-3, max_iter: int = 120,
                          device: torch.device | str = "cpu") -> HistoryIntensity:
    torch.manual_seed(0)
    model = HistoryIntensity(train.event_history.shape[1],
                             history_visible=history_visible).to(device)
    rate = len(train.event_index) / max(train.recorded_seconds, 1.0)
    with torch.no_grad():
        model.intercept.fill_(math.log(max(rate, 1e-8)))
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=int(max_iter), history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = intensity_loss(model, train, device=device)
        if model.history_visible:
            loss = loss + float(l2) * model.weight.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model.eval()


class ExactHistoryMarkDecoder(nn.Module):
    """Teacher-forced sequential mark decoder using the exact subset law."""

    def __init__(self, history_dim: int, n_contacts: int,
                 adjacency: np.ndarray, *, history_visible: bool = True):
        super().__init__()
        self.n_contacts = int(n_contacts)
        self.history_visible = bool(history_visible)
        relation = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
        if relation.ndim != 3 or relation.shape[1:] != (n_contacts, n_contacts):
            raise ValueError("adjacency shape disagrees with contacts")
        self.register_buffer("adjacency", relation)
        self.static_contact = nn.Parameter(torch.zeros(n_contacts))
        self.history_contact = nn.Linear(history_dim, n_contacts, bias=False)
        self.size_head = nn.Linear(history_dim + 2, n_contacts + 1)
        self.prefix_weight = nn.Parameter(torch.zeros(relation.shape[0]))
        self.step_contact = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.history_contact.weight)
        nn.init.zeros_(self.size_head.weight)
        nn.init.zeros_(self.size_head.bias)

    def logits(self, history: torch.Tensor, group_ids: torch.Tensor,
               group_count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_contacts = group_ids.shape
        if n_contacts != self.n_contacts or history.shape[0] != batch:
            raise ValueError("mark decoder batch shape disagrees")
        if not self.history_visible:
            history = torch.zeros_like(history)
        n_steps = int(group_count.max().item()) + 1
        contact_base = self.static_contact.unsqueeze(0) + self.history_contact(history)
        size_rows = []
        contact_rows = []
        participating = group_ids >= 0
        for step in range(n_steps):
            recruited = participating & (group_ids < step)
            recruited_fraction = recruited.float().sum(-1) / float(n_contacts)
            step_fraction = torch.full_like(
                recruited_fraction, float(step) / max(n_contacts, 1)
            )
            size_input = torch.cat([
                history, step_fraction.unsqueeze(-1), recruited_fraction.unsqueeze(-1)
            ], dim=-1)
            size_rows.append(self.size_head(size_input))
            neighbour = torch.einsum(
                "bn,rnm->brm", recruited.float(), self.adjacency
            )
            prefix = torch.einsum("brm,r->bm", neighbour, self.prefix_weight)
            contact_rows.append(
                contact_base + prefix
                + self.step_contact[0] * step_fraction.unsqueeze(-1)
                + self.step_contact[1] * recruited_fraction.unsqueeze(-1)
            )
        return torch.stack(size_rows, dim=1), torch.stack(contact_rows, dim=1)

    def forward(self, history: torch.Tensor, group_ids: torch.Tensor,
                group_count: torch.Tensor) -> TiedMarkTerms:
        size_logits, contact_logits = self.logits(history, group_ids, group_count)
        return tied_group_mark_log_prob(
            group_ids, group_count, size_logits, contact_logits
        )


@dataclass(frozen=True)
class MarkMetrics:
    event_nll: float
    group_size_nll: float
    subset_nll: float
    n_events: int


@torch.no_grad()
def mark_metrics(model: ExactHistoryMarkDecoder, history: np.ndarray,
                 group_ids: np.ndarray, group_count: np.ndarray,
                 *, device: torch.device | str = "cpu",
                 batch_size: int = 512) -> MarkMetrics:
    total = np.zeros(3, dtype=np.float64)
    n = len(history)
    for lo in range(0, n, int(batch_size)):
        hi = min(lo + int(batch_size), n)
        terms = model(
            torch.as_tensor(history[lo:hi], device=device),
            torch.as_tensor(group_ids[lo:hi], dtype=torch.long, device=device),
            torch.as_tensor(group_count[lo:hi], dtype=torch.long, device=device),
        )
        total += np.asarray([
            -float(terms.event_log_prob.sum()),
            -float(terms.group_size_log_prob.sum()),
            -float(terms.subset_log_prob.sum()),
        ])
    return MarkMetrics(
        event_nll=float(total[0] / max(n, 1)),
        group_size_nll=float(total[1] / max(n, 1)),
        subset_nll=float(total[2] / max(n, 1)),
        n_events=int(n),
    )


def fit_mark_decoder(history: np.ndarray, group_ids: np.ndarray,
                     group_count: np.ndarray, adjacency: np.ndarray,
                     *, history_visible: bool = True, seed: int = 0,
                     epochs: int = 30, batch_size: int = 256,
                     learning_rate: float = 3e-3,
                     device: torch.device | str = "cpu") -> ExactHistoryMarkDecoder:
    torch.manual_seed(int(seed))

    def new_model(initial_index: np.ndarray) -> ExactHistoryMarkDecoder:
        torch.manual_seed(int(seed))
        model = ExactHistoryMarkDecoder(
            history.shape[1], group_ids.shape[1], adjacency,
            history_visible=history_visible,
        ).to(device)
        # TRAIN-only marginal initialisation makes all arms start from the same
        # sensible contact/size baseline.
        participation = group_ids[initial_index] >= 0
        rate = (participation.sum(0) + 1.0) / (len(participation) + 2.0)
        selected_group = group_ids[initial_index]
        selected_count = group_count[initial_index]
        with torch.no_grad():
            model.static_contact.copy_(torch.as_tensor(
                np.log(rate) - np.log1p(-rate), dtype=torch.float32, device=device
            ))
            counts = np.bincount(
                np.concatenate([
                    np.asarray([np.sum(row == step) for step in range(int(k))])
                    for row, k in zip(selected_group, selected_count)
                ]), minlength=group_ids.shape[1] + 1,
            ).astype(np.float64)
            counts[0] += len(initial_index)  # one terminal STOP per event
            probability = (counts + 1.0) / (counts.sum() + len(counts))
            model.size_head.bias.copy_(torch.as_tensor(
                np.log(probability), dtype=torch.float32, device=device
            ))
        return model

    def train_for(model: ExactHistoryMarkDecoder, train_index: np.ndarray,
                  n_epochs: int, run_seed: int,
                  selection_index: np.ndarray | None = None
                  ) -> tuple[ExactHistoryMarkDecoder, int]:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=1e-2
        )
        rng = np.random.default_rng(int(run_seed))
        best_epoch = 0
        best_value = float("inf")
        best_state = None
        order = np.array(train_index, copy=True)
        for epoch in range(int(n_epochs)):
            rng.shuffle(order)
            model.train()
            for lo in range(0, len(order), int(batch_size)):
                take = order[lo:lo + int(batch_size)]
                terms = model(
                    torch.as_tensor(history[take], device=device),
                    torch.as_tensor(group_ids[take], dtype=torch.long, device=device),
                    torch.as_tensor(group_count[take], dtype=torch.long, device=device),
                )
                loss = -terms.event_log_prob.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if selection_index is not None:
                value = mark_metrics(
                    model, history[selection_index], group_ids[selection_index],
                    group_count[selection_index], device=device,
                    batch_size=batch_size,
                ).event_nll
                if value < best_value:
                    best_value = value
                    best_epoch = epoch
                    best_state = {
                        key: tensor.detach().cpu().clone()
                        for key, tensor in model.state_dict().items()
                    }
        if best_state is not None:
            model.load_state_dict(best_state)
        return model.eval(), best_epoch

    n = len(history)
    if n < 20:
        final = new_model(np.arange(n))
        final, _ = train_for(final, np.arange(n), int(epochs), int(seed))
        final.selected_epochs = int(epochs)
        return final
    cut = int(np.clip(math.floor(0.8 * n), 1, n - 1))
    inner_train = np.arange(cut)
    inner_validation = np.arange(cut, n)
    selection_model = new_model(inner_train)
    _, best_epoch = train_for(
        selection_model, inner_train, int(epochs), int(seed), inner_validation
    )
    # Refit on all TRAIN rows for the selected number of epochs.  This prevents
    # the small Yuquan pilot from losing 20% of its already limited events.
    all_index = np.arange(n)
    final = new_model(all_index)
    final, _ = train_for(
        final, all_index, best_epoch + 1, int(seed) + 1000
    )
    final.selected_epochs = int(best_epoch + 1)
    return final
