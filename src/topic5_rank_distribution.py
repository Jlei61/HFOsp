"""Self-supervised within-event rank-distribution models for Topic 5.

The recurrent axis is recruitment pseudo-time inside one group event.  No
state is carried between events.  The only supervised action is the next
recruitment set or STOP.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object
    nn = None


if nn is not None:

    class _ContactSequenceBase(nn.Module):
        """Shared contact encoder and contact-query action decoder."""

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            hidden_size: int = 32,
            contact_embedding_dim: int = 32,
            contact_encoder_hidden: int = 32,
            local_offset_dim: int = 8,
        ):
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.contact_embedding_dim = int(contact_embedding_dim)
            self.local_offset_dim = int(local_offset_dim)
            encoder_input = int(contact_feature_dim) + self.local_offset_dim
            self.contact_encoder = nn.Sequential(
                nn.Linear(encoder_input, int(contact_encoder_hidden)),
                nn.SiLU(),
                nn.Linear(int(contact_encoder_hidden), self.contact_embedding_dim),
                nn.LayerNorm(self.contact_embedding_dim),
            )
            self.initial_state = nn.Linear(
                2 * self.contact_embedding_dim, self.hidden_size
            )
            self.action_query = nn.Linear(
                self.hidden_size, self.contact_embedding_dim
            )
            self.action_bias = nn.Linear(encoder_input, 1)
            self.stop_head = nn.Linear(self.hidden_size, 1)

        @staticmethod
        def _masked_pool(embedding: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
            weight = mask.to(embedding.dtype).unsqueeze(-1)
            mean = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            floor = torch.finfo(embedding.dtype).min
            maximum = embedding.masked_fill(~mask.unsqueeze(-1), floor).max(1).values
            # ``finfo.min`` is finite, so ``isfinite`` cannot identify an
            # empty set.  Test the mask directly to keep empty-prefix controls
            # at an exact zero vector.
            has_member = mask.any(1, keepdim=True)
            maximum = torch.where(has_member, maximum, torch.zeros_like(maximum))
            return mean, maximum

        def _encode(
            self,
            contact_features: Tensor,
            local_offset: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            if contact_features.ndim != 3:
                raise ValueError("contact_features must be [batch, contact, feature]")
            if local_offset.ndim == 2:
                local_offset = local_offset.unsqueeze(0).expand(
                    contact_features.shape[0], -1, -1
                )
            if local_offset.shape[:2] != contact_features.shape[:2]:
                raise ValueError("local_offset must align with batch/contact axes")
            encoder_input = torch.cat([contact_features, local_offset], dim=-1)
            return self.contact_encoder(encoder_input), encoder_input

        def _initial_hidden(self, embedding: Tensor, contact_mask: Tensor) -> Tensor:
            mean, maximum = self._masked_pool(embedding, contact_mask)
            return torch.tanh(self.initial_state(torch.cat([mean, maximum], dim=-1)))

        def _decode(
            self,
            embedding: Tensor,
            encoder_input: Tensor,
            hidden: Tensor,
            candidate_mask: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            scale = float(np.sqrt(self.contact_embedding_dim))
            logits = (
                torch.einsum(
                    "bce,be->bc", embedding, self.action_query(hidden)
                )
                / scale
            )
            logits = logits + self.action_bias(encoder_input).squeeze(-1)
            logits = logits.masked_fill(~candidate_mask, -1e9)
            return logits, self.stop_head(hidden).squeeze(-1)

        def _rollout_hidden(
            self,
            embedding: Tensor,
            contact_mask: Tensor,
            recruited: Tensor,
            last_set: Tensor,
            hidden: Optional[Tensor],
        ) -> Tensor:
            raise NotImplementedError

        @torch.no_grad()
        def rollout(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            local_offset: Tensor,
            *,
            n_events: int,
            seed: int,
            batch_size: int = 512,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Free-run events using a categorical next-contact/STOP action.

            Exact tied sets are extremely rare in the frozen primary encoding.
            The rollout therefore emits one sampled contact per generated rank
            set while the training likelihood remains set-invariant.
            """
            self.eval()
            device = contact_features.device
            all_groups = []
            all_counts = []
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            remaining = int(n_events)
            while remaining:
                current_batch = min(int(batch_size), remaining)
                features = contact_features[:1].expand(current_batch, -1, -1)
                mask = contact_mask[:1].expand(current_batch, -1)
                embedding, encoder_input = self._encode(features, local_offset)
                recruited = torch.zeros_like(mask)
                last_set = torch.zeros_like(mask)
                hidden: Optional[Tensor] = None
                groups = torch.full(
                    mask.shape, -1, dtype=torch.int16, device=device
                )
                alive = torch.ones(current_batch, dtype=torch.bool, device=device)
                count = torch.zeros(
                    current_batch, dtype=torch.int16, device=device
                )
                for step in range(mask.shape[1]):
                    hidden = self._rollout_hidden(
                        embedding, mask, recruited, last_set, hidden
                    )
                    candidate = mask & ~recruited
                    contact_logits, stop_logit = self._decode(
                        embedding, encoder_input, hidden, candidate
                    )
                    action_logits = torch.cat(
                        [stop_logit[:, None], contact_logits], dim=1
                    )
                    action = torch.multinomial(
                        torch.softmax(action_logits, dim=1),
                        1,
                        generator=generator,
                    ).squeeze(1)
                    action = torch.where(alive, action, torch.zeros_like(action))
                    chose_stop = action == 0
                    chose_contact = alive & ~chose_stop
                    last_set = torch.zeros_like(mask)
                    if torch.any(chose_contact):
                        row = torch.where(chose_contact)[0]
                        contact = action[row] - 1
                        groups[row, contact] = count[row]
                        last_set[row, contact] = True
                        recruited[row, contact] = True
                        count[row] += 1
                    alive = alive & ~chose_stop
                    if not torch.any(alive):
                        break
                all_groups.append(groups.cpu().numpy())
                all_counts.append(count.cpu().numpy())
                remaining -= current_batch
            return np.row_stack(all_groups), np.concatenate(all_counts)


    class FullHistorySequenceGRU(_ContactSequenceBase):
        """Permutation-invariant recruitment-set tokens followed by a GRU."""

        def __init__(self, contact_feature_dim: int, **kwargs):
            super().__init__(contact_feature_dim, **kwargs)
            self.gru = nn.GRUCell(
                self.contact_embedding_dim + 2, self.hidden_size
            )

        def _advance(
            self,
            embedding: Tensor,
            current_set: Tensor,
            recruited: Tensor,
            hidden: Tensor,
            contact_mask: Tensor,
        ) -> Tensor:
            weight = current_set.to(embedding.dtype).unsqueeze(-1)
            token = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            denominator = contact_mask.sum(1).clamp_min(1).to(embedding.dtype)
            progress = recruited.sum(1).to(embedding.dtype) / denominator
            new_fraction = current_set.sum(1).to(embedding.dtype) / denominator
            return self.gru(
                torch.cat([token, progress[:, None], new_fraction[:, None]], dim=1),
                hidden,
            )

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
        ) -> Dict[str, Tensor]:
            embedding, encoder_input = self._encode(contact_features, local_offset)
            hidden = self._initial_hidden(embedding, contact_mask)
            recruited = torch.zeros_like(contact_mask)
            max_groups = int(group_count.max().item())
            action_logits = []
            stop_logits = []
            candidate_masks = []
            for step in range(max_groups + 1):
                candidate = contact_mask & ~recruited
                action, stop = self._decode(
                    embedding, encoder_input, hidden, candidate
                )
                action_logits.append(action)
                stop_logits.append(stop)
                candidate_masks.append(candidate)
                if step == max_groups:
                    break
                current = (group_ids == step) & contact_mask
                active = (group_count > step).unsqueeze(1)
                updated_recruited = recruited | current
                updated_hidden = self._advance(
                    embedding,
                    current,
                    updated_recruited,
                    hidden,
                    contact_mask,
                )
                hidden = torch.where(active, updated_hidden, hidden)
                recruited = torch.where(active, updated_recruited, recruited)
            return {
                "contact_logits": torch.stack(action_logits, dim=1),
                "stop_logits": torch.stack(stop_logits, dim=1),
                "candidate_mask": torch.stack(candidate_masks, dim=1),
            }

        def _rollout_hidden(
            self,
            embedding: Tensor,
            contact_mask: Tensor,
            recruited: Tensor,
            last_set: Tensor,
            hidden: Optional[Tensor],
        ) -> Tensor:
            if hidden is None:
                return self._initial_hidden(embedding, contact_mask)
            return self._advance(
                embedding, last_set, recruited, hidden, contact_mask
            )


    class LowRankLeakySequenceRNN(_ContactSequenceBase):
        """Leaky RNN with stable diagonal decay plus rank-r recurrence."""

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            recurrent_rank: int,
            **kwargs,
        ):
            super().__init__(contact_feature_dim, **kwargs)
            rank = int(recurrent_rank)
            if rank < 0 or rank > self.hidden_size:
                raise ValueError("recurrent_rank must lie in [0, hidden_size]")
            self.recurrent_rank = rank
            self.input_projection = nn.Linear(
                self.contact_embedding_dim + 2, self.hidden_size
            )
            # softplus(raw_decay) enters with a negative sign, providing an
            # explicitly stable self term before the low-rank interaction.
            self.raw_decay = nn.Parameter(torch.full((self.hidden_size,), -0.5))
            self.raw_alpha = nn.Parameter(torch.tensor(0.0))
            if rank:
                scale = 1.0 / np.sqrt(max(self.hidden_size, 1))
                self.mode_u = nn.Parameter(
                    torch.randn(self.hidden_size, rank) * scale
                )
                self.mode_v = nn.Parameter(
                    torch.randn(self.hidden_size, rank) * scale
                )
            else:
                self.register_parameter("mode_u", None)
                self.register_parameter("mode_v", None)

        @property
        def alpha(self) -> Tensor:
            return 0.05 + 0.90 * torch.sigmoid(self.raw_alpha)

        @property
        def decay(self) -> Tensor:
            return torch.nn.functional.softplus(self.raw_decay)

        def recurrent_drive(self, hidden: Tensor) -> Tensor:
            drive = -self.decay * hidden
            if self.recurrent_rank:
                coordinate = hidden @ self.mode_v
                drive = drive + (coordinate @ self.mode_u.T) / np.sqrt(
                    float(self.recurrent_rank)
                )
            return drive

        def _advance(
            self,
            embedding: Tensor,
            current_set: Tensor,
            recruited: Tensor,
            hidden: Tensor,
            contact_mask: Tensor,
        ) -> Tensor:
            weight = current_set.to(embedding.dtype).unsqueeze(-1)
            token = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            denominator = contact_mask.sum(1).clamp_min(1).to(embedding.dtype)
            progress = recruited.sum(1).to(embedding.dtype) / denominator
            new_fraction = current_set.sum(1).to(embedding.dtype) / denominator
            external = self.input_projection(
                torch.cat(
                    [token, progress[:, None], new_fraction[:, None]], dim=1
                )
            )
            proposal = torch.tanh(self.recurrent_drive(hidden) + external)
            return (1.0 - self.alpha) * hidden + self.alpha * proposal

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
        ) -> Dict[str, Tensor]:
            embedding, encoder_input = self._encode(contact_features, local_offset)
            hidden = self._initial_hidden(embedding, contact_mask)
            recruited = torch.zeros_like(contact_mask)
            max_groups = int(group_count.max().item())
            action_logits = []
            stop_logits = []
            candidate_masks = []
            for step in range(max_groups + 1):
                candidate = contact_mask & ~recruited
                action, stop = self._decode(
                    embedding, encoder_input, hidden, candidate
                )
                action_logits.append(action)
                stop_logits.append(stop)
                candidate_masks.append(candidate)
                if step == max_groups:
                    break
                current = (group_ids == step) & contact_mask
                active = (group_count > step).unsqueeze(1)
                updated_recruited = recruited | current
                updated_hidden = self._advance(
                    embedding,
                    current,
                    updated_recruited,
                    hidden,
                    contact_mask,
                )
                hidden = torch.where(active, updated_hidden, hidden)
                recruited = torch.where(active, updated_recruited, recruited)
            return {
                "contact_logits": torch.stack(action_logits, dim=1),
                "stop_logits": torch.stack(stop_logits, dim=1),
                "candidate_mask": torch.stack(candidate_masks, dim=1),
            }

        def _rollout_hidden(
            self,
            embedding: Tensor,
            contact_mask: Tensor,
            recruited: Tensor,
            last_set: Tensor,
            hidden: Optional[Tensor],
        ) -> Tensor:
            if hidden is None:
                return self._initial_hidden(embedding, contact_mask)
            return self._advance(
                embedding, last_set, recruited, hidden, contact_mask
            )

        @torch.no_grad()
        def hidden_trajectory(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
        ) -> Dict[str, Tensor]:
            """Teacher-force events and return padded hidden/mode trajectories."""
            embedding, _ = self._encode(contact_features, local_offset)
            hidden = self._initial_hidden(embedding, contact_mask)
            recruited = torch.zeros_like(contact_mask)
            max_groups = int(group_count.max().item())
            states = [hidden]
            for step in range(max_groups):
                current = (group_ids == step) & contact_mask
                active = (group_count > step).unsqueeze(1)
                updated_recruited = recruited | current
                updated_hidden = self._advance(
                    embedding,
                    current,
                    updated_recruited,
                    hidden,
                    contact_mask,
                )
                hidden = torch.where(active, updated_hidden, hidden)
                recruited = torch.where(active, updated_recruited, recruited)
                states.append(hidden)
            hidden_states = torch.stack(states, dim=1)
            step = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            )
            state_mask = step[None, :] <= group_count[:, None]
            if self.recurrent_rank:
                mode_coordinates = torch.einsum(
                    "bth,hr->btr", hidden_states, self.mode_v
                )
            else:
                mode_coordinates = hidden_states.new_zeros(
                    (*hidden_states.shape[:2], 0)
                )
            return {
                "hidden_states": hidden_states,
                "mode_coordinates": mode_coordinates,
                "state_mask": state_mask,
            }

        @torch.no_grad()
        def contact_mode_loadings(
            self,
            contact_features: Tensor,
            local_offset: Tensor,
        ) -> Dict[str, Tensor]:
            """Map recurrent modes into the held-out patient's contact logits."""
            if contact_features.ndim == 2:
                contact_features = contact_features.unsqueeze(0)
            embedding, _ = self._encode(contact_features, local_offset)
            if not self.recurrent_rank:
                empty = embedding.new_zeros((embedding.shape[1], 0))
                return {"u_output_loading": empty, "v_output_loading": empty}
            # action_query(hidden) = W h + b; contact logit is e_c^T W h.
            hidden_to_contact = (
                embedding[0] @ self.action_query.weight
            ) / np.sqrt(float(self.contact_embedding_dim))
            return {
                "u_output_loading": hidden_to_contact @ self.mode_u,
                "v_output_loading": hidden_to_contact @ self.mode_v,
            }


    class StaticSequenceContactQuery(_ContactSequenceBase):
        """Matched non-recurrent controls.

        Modes are ``static`` (no prefix), ``unordered`` (all recruited
        contacts), and ``last_set`` (only the latest recruitment set).
        """

        VALID_MODES = {"static", "unordered", "last_set"}

        def __init__(self, contact_feature_dim: int, *, mode: str, **kwargs):
            super().__init__(contact_feature_dim, **kwargs)
            if mode not in self.VALID_MODES:
                raise ValueError(f"unknown static mode: {mode}")
            self.mode = str(mode)
            n_pools = 2 if mode == "static" else 4
            self.prefix_mlp = nn.Sequential(
                nn.Linear(n_pools * self.contact_embedding_dim + 1, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        def _static_hidden(
            self,
            embedding: Tensor,
            contact_mask: Tensor,
            recruited: Tensor,
            last_set: Tensor,
        ) -> Tensor:
            global_mean, global_max = self._masked_pool(embedding, contact_mask)
            pools = [global_mean, global_max]
            if self.mode == "unordered":
                prefix_mean, prefix_max = self._masked_pool(embedding, recruited)
                pools.extend([prefix_mean, prefix_max])
            elif self.mode == "last_set":
                last_mean, last_max = self._masked_pool(embedding, last_set)
                pools.extend([last_mean, last_max])
            progress = (
                recruited.sum(1).to(embedding.dtype)
                / contact_mask.sum(1).clamp_min(1).to(embedding.dtype)
            )
            return self.prefix_mlp(torch.cat([*pools, progress[:, None]], dim=1))

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
        ) -> Dict[str, Tensor]:
            embedding, encoder_input = self._encode(contact_features, local_offset)
            recruited = torch.zeros_like(contact_mask)
            last_set = torch.zeros_like(contact_mask)
            max_groups = int(group_count.max().item())
            action_logits = []
            stop_logits = []
            candidate_masks = []
            for step in range(max_groups + 1):
                hidden = self._static_hidden(
                    embedding, contact_mask, recruited, last_set
                )
                candidate = contact_mask & ~recruited
                action, stop = self._decode(
                    embedding, encoder_input, hidden, candidate
                )
                action_logits.append(action)
                stop_logits.append(stop)
                candidate_masks.append(candidate)
                if step == max_groups:
                    break
                current = (group_ids == step) & contact_mask
                active = (group_count > step).unsqueeze(1)
                recruited = torch.where(active, recruited | current, recruited)
                last_set = torch.where(active, current, last_set)
            return {
                "contact_logits": torch.stack(action_logits, dim=1),
                "stop_logits": torch.stack(stop_logits, dim=1),
                "candidate_mask": torch.stack(candidate_masks, dim=1),
            }

        def _rollout_hidden(
            self,
            embedding: Tensor,
            contact_mask: Tensor,
            recruited: Tensor,
            last_set: Tensor,
            hidden: Optional[Tensor],
        ) -> Tensor:
            del hidden
            return self._static_hidden(
                embedding, contact_mask, recruited, last_set
            )


    def next_set_stop_loss(
        outputs: Mapping[str, Tensor],
        group_ids: Tensor,
        group_count: Tensor,
    ) -> Dict[str, Tensor]:
        """Event-balanced next-set/STOP categorical negative log likelihood."""
        contact_logits = outputs["contact_logits"]
        stop_logits = outputs["stop_logits"]
        max_steps = contact_logits.shape[1]
        per_step = []
        valid_step = []
        for step in range(max_steps):
            active = group_count >= step
            terminal = group_count == step
            denominator = torch.logsumexp(
                torch.cat(
                    [stop_logits[:, step, None], contact_logits[:, step]], dim=1
                ),
                dim=1,
            )
            target_set = group_ids == step
            target_contact = torch.logsumexp(
                contact_logits[:, step].masked_fill(~target_set, -1e9), dim=1
            )
            numerator = torch.where(
                terminal, stop_logits[:, step], target_contact
            )
            per_step.append(denominator - numerator)
            valid_step.append(active)
        step_nll = torch.stack(per_step, dim=1)
        step_mask = torch.stack(valid_step, dim=1)
        event_nll = (
            (step_nll * step_mask).sum(1)
            / step_mask.sum(1).clamp_min(1)
        )
        return {
            "total": event_nll.mean(),
            "event_nll": event_nll,
            "step_nll": step_nll,
            "step_mask": step_mask,
        }


def contact_rank_distribution(
    group_ids: np.ndarray,
    group_count: np.ndarray,
    *,
    bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Summarize participation and conditional normalized-rank distributions."""
    groups = np.asarray(group_ids, int)
    counts = np.asarray(group_count, int)
    if groups.ndim != 2 or counts.shape != (groups.shape[0],):
        raise ValueError("group arrays are not aligned")
    n_events, n_contacts = groups.shape
    participation = np.mean(groups >= 0, axis=0) if n_events else np.zeros(n_contacts)
    histogram = np.zeros((n_contacts, int(bins)), dtype=float)
    mean_rank = np.full(n_contacts, np.nan)
    variance = np.full(n_contacts, np.nan)
    quantiles = np.full((n_contacts, 3), np.nan)
    rank_values = []
    denominator = np.maximum(counts - 1, 1)
    normalized = np.where(groups >= 0, groups / denominator[:, None], np.nan)
    for contact in range(n_contacts):
        values = normalized[:, contact]
        values = values[np.isfinite(values)]
        rank_values.append(values)
        if not values.size:
            continue
        histogram[contact], _ = np.histogram(
            values, bins=np.linspace(0.0, 1.0, int(bins) + 1)
        )
        histogram[contact] /= histogram[contact].sum()
        mean_rank[contact] = float(np.mean(values))
        variance[contact] = float(np.var(values))
        quantiles[contact] = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "participation_probability": participation,
        "rank_histogram": histogram,
        "mean_rank": mean_rank,
        "rank_variance": variance,
        "rank_quantiles": quantiles,
        "normalized_rank_values": np.asarray(rank_values, dtype=object),
    }


def pairwise_precedence(group_ids: np.ndarray) -> np.ndarray:
    """P(i precedes j | both participate); ties contribute one half."""
    groups = np.asarray(group_ids, int)
    n_contacts = groups.shape[1]
    out = np.full((n_contacts, n_contacts), np.nan)
    for left in range(n_contacts):
        for right in range(n_contacts):
            if left == right:
                continue
            valid = (groups[:, left] >= 0) & (groups[:, right] >= 0)
            if not np.any(valid):
                continue
            delta = groups[valid, left] - groups[valid, right]
            out[left, right] = float(
                np.mean((delta < 0).astype(float) + 0.5 * (delta == 0))
            )
    return out


def distribution_errors(
    predicted_groups: np.ndarray,
    predicted_count: np.ndarray,
    observed_groups: np.ndarray,
    observed_count: np.ndarray,
    *,
    bins: int = 10,
) -> Dict[str, float]:
    """Patient-level errors between generated and observed event fields."""
    predicted = contact_rank_distribution(
        predicted_groups, predicted_count, bins=bins
    )
    observed = contact_rank_distribution(observed_groups, observed_count, bins=bins)
    part_mae = float(
        np.mean(
            np.abs(
                predicted["participation_probability"]
                - observed["participation_probability"]
            )
        )
    )
    valid = (
        np.isfinite(predicted["mean_rank"])
        & np.isfinite(observed["mean_rank"])
    )
    if np.any(valid):
        pred_cdf = np.cumsum(predicted["rank_histogram"][valid], axis=1)
        obs_cdf = np.cumsum(observed["rank_histogram"][valid], axis=1)
        # Mean across equally spaced bins already includes the bin width
        # (sum / n_bins); dividing by ``bins`` again would understate W1.
        rank_w1 = float(np.mean(np.abs(pred_cdf - obs_cdf)))
    else:
        rank_w1 = float("nan")
    pred_precedence = pairwise_precedence(predicted_groups)
    obs_precedence = pairwise_precedence(observed_groups)
    pair_valid = np.isfinite(pred_precedence) & np.isfinite(obs_precedence)
    precedence_mae = (
        float(np.mean(np.abs(pred_precedence[pair_valid] - obs_precedence[pair_valid])))
        if np.any(pair_valid)
        else float("nan")
    )
    if np.sum(pair_valid) >= 3:
        precedence_correlation = float(
            np.corrcoef(
                pred_precedence[pair_valid], obs_precedence[pair_valid]
            )[0, 1]
        )
    else:
        precedence_correlation = float("nan")
    return {
        "participation_mae": part_mae,
        "rank_wasserstein": rank_w1,
        "precedence_mae": precedence_mae,
        "precedence_correlation": precedence_correlation,
        "participant_count_mean_error": float(
            np.mean(np.sum(predicted_groups >= 0, axis=1))
            - np.mean(np.sum(observed_groups >= 0, axis=1))
        ),
        "event_length_mean_error": float(
            np.mean(predicted_count) - np.mean(observed_count)
        ),
    }
