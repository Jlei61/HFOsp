"""Patient-axis constrained graph RNN for interictal propagation events.

The model deliberately has no dense contact-to-contact recurrent matrix.
Patient-specific recurrence is supplied by a fixed, train-only pair of
forward/reverse axis graphs.  The two directional channels share every
trainable parameter, making predictions invariant to the arbitrary sign of
the patient axis and to swapping the two template labels.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object
    nn = None
    F = None


if nn is not None:

    class AxisStructuredGraphRNN(nn.Module):
        """Minimal graph RNN whose rank indexes biologically typed channels.

        Rank meanings are fixed:

        * 0: static contact hazard with no recurrent state;
        * 1: one direction-agnostic symmetric-axis channel;
        * 2: paired forward/reverse channels with shared parameters;
        * 3: rank 2 plus a global recruitment channel;
        * 4: rank 3 plus a local surround-suppression channel.

        Local patient offsets enter only the static contact hazard.  They
        cannot change the fixed patient graph or the shared transition.
        """

        VALID_LESIONS = {
            "none",
            "endpoints",
            "direction_forward",
            "direction_reverse",
            "inhibition",
        }

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            structured_rank: int,
            local_offset_dim: int = 1,
        ):
            super().__init__()
            rank = int(structured_rank)
            if rank < 0 or rank > 4:
                raise ValueError("structured_rank must lie in [0, 4]")
            if int(local_offset_dim) < 1:
                raise ValueError("local_offset_dim must be positive")
            self.structured_rank = rank
            self.local_offset_dim = int(local_offset_dim)

            # A linear static hazard keeps the patient-specific nuisance path
            # deliberately weaker than the graph transition.
            self.static_feature = nn.Linear(int(contact_feature_dim), 1)
            self.local_readout = nn.Linear(
                self.local_offset_dim, 1, bias=False
            )
            # Endpoint source enrichment is cohort-level but not universal.
            # Start near zero and let train-only events decide whether it is
            # useful; an endpoint-source auxiliary is an optional sensitivity.
            self.raw_endpoint_gain = nn.Parameter(torch.tensor(-3.0))

            # Type 0 is shared by both directional channels.  Types 1 and 2
            # are the optional global and surround channels.
            n_types = 0 if rank == 0 else 1 + int(rank >= 3) + int(rank >= 4)
            self.n_channel_types = n_types
            if n_types:
                self.raw_alpha = nn.Parameter(torch.zeros(n_types))
                self.raw_input_gain = nn.Parameter(torch.zeros(n_types))
                self.raw_propagation_gain = nn.Parameter(torch.zeros(n_types))
                self.raw_decay = nn.Parameter(torch.full((n_types,), -0.5))
                self.raw_inhibition_gain = nn.Parameter(
                    torch.full((n_types,), -1.0)
                )
                self.channel_bias = nn.Parameter(torch.zeros(n_types))
                self.raw_output_gain = nn.Parameter(torch.zeros(n_types))

                self.raw_inhibitory_alpha = nn.Parameter(torch.tensor(-0.5))
                self.raw_inhibitory_drive = nn.Parameter(torch.tensor(-0.5))
                if rank >= 2:
                    # Symmetric competition prevents the paired directional
                    # channels from collapsing algebraically to their sum.
                    self.raw_direction_competition = nn.Parameter(
                        torch.tensor(-0.5)
                    )
                else:
                    self.register_parameter(
                        "raw_direction_competition", None
                    )
            else:
                for name in (
                    "raw_alpha",
                    "raw_input_gain",
                    "raw_propagation_gain",
                    "raw_decay",
                    "raw_inhibition_gain",
                    "channel_bias",
                    "raw_output_gain",
                    "raw_inhibitory_alpha",
                    "raw_inhibitory_drive",
                    "raw_direction_competition",
                ):
                    self.register_parameter(name, None)

            self.stop_bias = nn.Parameter(torch.tensor(-1.0))
            self.raw_stop_progress_gain = nn.Parameter(torch.tensor(0.0))
            self.raw_stop_inhibition_gain = nn.Parameter(torch.tensor(-0.5))
            self.raw_continue_state_gain = nn.Parameter(torch.tensor(-0.5))

        @property
        def endpoint_gain(self) -> Tensor:
            return F.softplus(self.raw_endpoint_gain)

        @property
        def alpha_by_type(self) -> Tensor:
            if self.raw_alpha is None:
                return torch.empty(0, device=self.stop_bias.device)
            return 0.05 + 0.90 * torch.sigmoid(self.raw_alpha)

        @property
        def input_gain_by_type(self) -> Tensor:
            if self.raw_input_gain is None:
                return torch.empty(0, device=self.stop_bias.device)
            return F.softplus(self.raw_input_gain)

        @property
        def propagation_gain_by_type(self) -> Tensor:
            if self.raw_propagation_gain is None:
                return torch.empty(0, device=self.stop_bias.device)
            return F.softplus(self.raw_propagation_gain)

        @property
        def decay_by_type(self) -> Tensor:
            if self.raw_decay is None:
                return torch.empty(0, device=self.stop_bias.device)
            return F.softplus(self.raw_decay)

        @property
        def inhibition_gain_by_type(self) -> Tensor:
            if self.raw_inhibition_gain is None:
                return torch.empty(0, device=self.stop_bias.device)
            return F.softplus(self.raw_inhibition_gain)

        @property
        def output_gain_by_type(self) -> Tensor:
            if self.raw_output_gain is None:
                return torch.empty(0, device=self.stop_bias.device)
            return F.softplus(self.raw_output_gain)

        @property
        def inhibitory_alpha(self) -> Tensor:
            if self.raw_inhibitory_alpha is None:
                return torch.zeros((), device=self.stop_bias.device)
            return 0.05 + 0.90 * torch.sigmoid(self.raw_inhibitory_alpha)

        @property
        def inhibitory_drive(self) -> Tensor:
            if self.raw_inhibitory_drive is None:
                return torch.zeros((), device=self.stop_bias.device)
            return F.softplus(self.raw_inhibitory_drive)

        @property
        def direction_competition(self) -> Tensor:
            if self.raw_direction_competition is None:
                return torch.zeros((), device=self.stop_bias.device)
            return F.softplus(self.raw_direction_competition)

        @staticmethod
        def _expand_contact(
            value: Tensor, batch_size: int, n_contacts: int, name: str
        ) -> Tensor:
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(batch_size, -1)
            if value.shape != (batch_size, n_contacts):
                raise ValueError(f"{name} must be [batch, contact] or [contact]")
            return value

        @staticmethod
        def _expand_graph(
            value: Tensor, batch_size: int, n_contacts: int, name: str
        ) -> Tensor:
            if value.ndim == 2:
                value = value.unsqueeze(0).expand(batch_size, -1, -1)
            if value.shape != (batch_size, n_contacts, n_contacts):
                raise ValueError(
                    f"{name} must be [batch, contact, contact] or "
                    "[contact, contact]"
                )
            return value

        @staticmethod
        def _row_normalize(graph: Tensor) -> Tensor:
            denominator = graph.sum(-1, keepdim=True)
            return torch.where(
                denominator > 0,
                graph / denominator.clamp_min(1e-8),
                torch.zeros_like(graph),
            )

        def _validate_inputs(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            local_offset: Tensor,
            axis_coordinate: Tensor,
            forward_graph: Tensor,
            reverse_graph: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            if contact_features.ndim != 3:
                raise ValueError(
                    "contact_features must be [batch, contact, feature]"
                )
            batch_size, n_contacts, _ = contact_features.shape
            if contact_mask.shape != (batch_size, n_contacts):
                raise ValueError("contact_mask is not aligned")
            if local_offset.ndim == 2:
                local_offset = local_offset.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            if local_offset.shape != (
                batch_size,
                n_contacts,
                self.local_offset_dim,
            ):
                raise ValueError("local_offset is not aligned")
            axis = self._expand_contact(
                axis_coordinate, batch_size, n_contacts, "axis_coordinate"
            )
            forward = self._expand_graph(
                forward_graph, batch_size, n_contacts, "forward_graph"
            )
            reverse = self._expand_graph(
                reverse_graph, batch_size, n_contacts, "reverse_graph"
            )
            left = self._expand_contact(
                left_endpoint, batch_size, n_contacts, "left_endpoint"
            ).bool()
            right = self._expand_contact(
                right_endpoint, batch_size, n_contacts, "right_endpoint"
            ).bool()
            valid_pair = (
                contact_mask.unsqueeze(1) & contact_mask.unsqueeze(2)
            ).to(forward.dtype)
            # Graph producers own normalization.  In v0.7, columns encode
            # source-to-next-contact weights and the paired operators are
            # spectral-normalized exact transposes.  Row-normalizing here
            # would erase that conditional transition information.
            forward = forward.clamp_min(0.0) * valid_pair
            reverse = reverse.clamp_min(0.0) * valid_pair
            axis = axis.clamp(-1.0, 1.0) * contact_mask.to(axis.dtype)
            return local_offset, axis, forward, reverse, left, right

        def _channel_type_indices(self, device: torch.device) -> Tensor:
            if self.structured_rank == 0:
                return torch.empty(0, dtype=torch.long, device=device)
            indices = [0]
            if self.structured_rank >= 2:
                indices.append(0)
            if self.structured_rank >= 3:
                indices.append(1)
            if self.structured_rank >= 4:
                indices.append(2)
            return torch.as_tensor(indices, dtype=torch.long, device=device)

        def _channel_graphs(
            self,
            forward: Tensor,
            reverse: Tensor,
            contact_mask: Tensor,
        ) -> Tensor:
            if self.structured_rank == 0:
                return forward.new_zeros(
                    (*forward.shape[:2], forward.shape[2], 0)
                )
            # The average of two norm-bounded transpose operators is itself
            # norm-bounded and preserves their edge weights.
            symmetric = 0.5 * (forward + reverse)
            if self.structured_rank == 1:
                graphs = [symmetric]
            else:
                graphs = [forward, reverse]
            if self.structured_rank >= 3:
                valid = contact_mask.to(forward.dtype)
                global_graph = valid.unsqueeze(2) * valid.unsqueeze(1)
                global_graph = self._row_normalize(global_graph)
                graphs.append(global_graph)
            if self.structured_rank >= 4:
                graphs.append(symmetric)
            return torch.stack(graphs, dim=-1)

        def _channel_input(
            self,
            current_set: Tensor,
            recruited: Tensor,
            axis: Tensor,
            contact_mask: Tensor,
        ) -> Tensor:
            dtype = axis.dtype
            current = current_set.to(dtype)
            if self.structured_rank == 0:
                return current.new_zeros((*current.shape, 0))
            if self.structured_rank == 1:
                inputs = [current]
            else:
                # A contact near the negative endpoint preferentially launches
                # the increasing-axis branch; the opposite endpoint launches
                # its transpose branch.
                inputs = [
                    current * (1.0 - axis) * 0.5,
                    current * (1.0 + axis) * 0.5,
                ]
            if self.structured_rank >= 3:
                denominator = (
                    contact_mask.sum(1, keepdim=True).clamp_min(1).to(dtype)
                )
                progress = recruited.sum(1, keepdim=True).to(dtype) / denominator
                inputs.append(progress.expand_as(current))
            if self.structured_rank >= 4:
                inputs.append(current)
            return torch.stack(inputs, dim=-1)

        def _advance(
            self,
            state: Tensor,
            inhibition: Tensor,
            current_set: Tensor,
            recruited: Tensor,
            axis: Tensor,
            graphs: Tensor,
            contact_mask: Tensor,
            *,
            lesion: str,
        ) -> Tuple[Tensor, Tensor]:
            if self.structured_rank == 0:
                return state, inhibition
            channel_input = self._channel_input(
                current_set, recruited, axis, contact_mask
            )
            # Recruitment pseudo-time has one observation per rank set.  The
            # newly observed set must therefore seed the outgoing graph in the
            # same update; using A@state alone delays graph information by one
            # rank and cannot help predict the immediately following set.
            propagated = torch.einsum(
                "bijr,bjr->bir", graphs, state + channel_input
            )
            type_index = self._channel_type_indices(state.device)
            alpha = self.alpha_by_type[type_index]
            input_gain = self.input_gain_by_type[type_index]
            propagation_gain = self.propagation_gain_by_type[type_index]
            decay = self.decay_by_type[type_index]
            inhibitory_gain = self.inhibition_gain_by_type[type_index]
            bias = self.channel_bias[type_index]
            if lesion == "inhibition":
                inhibitory_term = torch.zeros_like(state)
            else:
                inhibitory_term = (
                    inhibitory_gain.view(1, 1, -1)
                    * inhibition[:, None, None]
                )
            drive = (
                input_gain.view(1, 1, -1) * channel_input
                + propagation_gain.view(1, 1, -1) * propagated
                - decay.view(1, 1, -1) * state
                - inhibitory_term
                + bias.view(1, 1, -1)
            )
            if self.structured_rank >= 2:
                valid = contact_mask.to(state.dtype)
                denominator = valid.sum(1).clamp_min(1.0)
                directional_energy = (
                    F.relu(state[:, :, :2] + channel_input[:, :, :2])
                    * valid.unsqueeze(-1)
                ).sum(1) / denominator[:, None]
                opposite = torch.stack(
                    [directional_energy[:, 1], directional_energy[:, 0]],
                    dim=1,
                )
                drive = drive.clone()
                drive[:, :, :2] = (
                    drive[:, :, :2]
                    - self.direction_competition * opposite[:, None, :]
                )
            proposal = torch.tanh(drive)
            updated = (
                (1.0 - alpha.view(1, 1, -1)) * state
                + alpha.view(1, 1, -1) * proposal
            )
            if lesion == "direction_forward" and self.structured_rank >= 2:
                updated = updated.clone()
                updated[:, :, 0] = 0.0
            if lesion == "direction_reverse" and self.structured_rank >= 2:
                updated = updated.clone()
                updated[:, :, 1] = 0.0
            updated = updated * contact_mask.unsqueeze(-1).to(updated.dtype)

            if lesion == "inhibition":
                updated_inhibition = torch.zeros_like(inhibition)
            else:
                excitatory_channels = min(self.structured_rank, 3)
                valid = contact_mask.unsqueeze(-1).to(updated.dtype)
                denominator = valid.sum((1, 2)).clamp_min(1.0)
                mean_excitation = (
                    F.relu(updated[:, :, :excitatory_channels]) * valid
                ).sum((1, 2)) / (
                    denominator * float(excitatory_channels)
                )
                proposal_inhibition = torch.tanh(
                    self.inhibitory_drive * mean_excitation
                )
                updated_inhibition = (
                    (1.0 - self.inhibitory_alpha) * inhibition
                    + self.inhibitory_alpha * proposal_inhibition
                )
            return updated, updated_inhibition

        def _decode(
            self,
            contact_features: Tensor,
            local_offset: Tensor,
            state: Tensor,
            inhibition: Tensor,
            recruited: Tensor,
            contact_mask: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
            *,
            lesion: str,
        ) -> Tuple[Tensor, Tensor]:
            logits = self.static_feature(contact_features).squeeze(-1)
            logits = logits + self.local_readout(local_offset).squeeze(-1)
            progress = (
                recruited.sum(1).to(logits.dtype)
                / contact_mask.sum(1).clamp_min(1).to(logits.dtype)
            )
            if lesion != "endpoints":
                endpoint = (left_endpoint | right_endpoint).to(logits.dtype)
                logits = logits + (
                    (progress == 0).to(logits.dtype)[:, None]
                    * self.endpoint_gain
                    * endpoint
                )
            mean_excitation = logits.new_zeros(logits.shape[0])
            if self.structured_rank:
                type_index = self._channel_type_indices(state.device)
                output_gain = self.output_gain_by_type[type_index]
                sign = torch.ones_like(output_gain)
                if self.structured_rank >= 4:
                    sign[-1] = -1.0
                contribution = (
                    state
                    * output_gain.view(1, 1, -1)
                    * sign.view(1, 1, -1)
                )
                if lesion == "direction_forward" and self.structured_rank >= 2:
                    contribution = contribution.clone()
                    contribution[:, :, 0] = 0.0
                if lesion == "direction_reverse" and self.structured_rank >= 2:
                    contribution = contribution.clone()
                    contribution[:, :, 1] = 0.0
                logits = logits + contribution.sum(-1)
                excitatory_channels = min(self.structured_rank, 3)
                valid = contact_mask.unsqueeze(-1).to(state.dtype)
                mean_excitation = (
                    F.relu(state[:, :, :excitatory_channels]) * valid
                ).sum((1, 2)) / (
                    valid.sum((1, 2)).clamp_min(1.0)
                    * float(excitatory_channels)
                )
            stop = (
                self.stop_bias
                + F.softplus(self.raw_stop_progress_gain) * progress
                - F.softplus(self.raw_continue_state_gain) * mean_excitation
            )
            if lesion != "inhibition":
                stop = (
                    stop
                    + F.softplus(self.raw_stop_inhibition_gain) * inhibition
                )
            candidate = contact_mask & ~recruited
            logits = logits.masked_fill(~candidate, -1e9)
            return logits, stop

        def _prepare(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            local_offset: Tensor,
            axis_coordinate: Tensor,
            forward_graph: Tensor,
            reverse_graph: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            local_offset, axis, forward, reverse, left, right = (
                self._validate_inputs(
                    contact_features,
                    contact_mask,
                    local_offset,
                    axis_coordinate,
                    forward_graph,
                    reverse_graph,
                    left_endpoint,
                    right_endpoint,
                )
            )
            graphs = self._channel_graphs(forward, reverse, contact_mask)
            return local_offset, axis, graphs, left, right

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
            axis_coordinate: Tensor,
            forward_graph: Tensor,
            reverse_graph: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
            *,
            lesion: str = "none",
        ) -> Dict[str, Tensor]:
            if lesion not in self.VALID_LESIONS:
                raise ValueError(f"unknown lesion: {lesion}")
            local_offset, axis, graphs, left, right = self._prepare(
                contact_features,
                contact_mask,
                local_offset,
                axis_coordinate,
                forward_graph,
                reverse_graph,
                left_endpoint,
                right_endpoint,
            )
            batch_size, n_contacts = contact_mask.shape
            state = contact_features.new_zeros(
                (batch_size, n_contacts, self.structured_rank)
            )
            inhibition = contact_features.new_zeros(batch_size)
            recruited = torch.zeros_like(contact_mask)
            max_groups = int(group_count.max().item())
            contact_logits = []
            stop_logits = []
            candidate_masks = []
            states = []
            inhibitions = []
            for step in range(max_groups + 1):
                action, stop = self._decode(
                    contact_features,
                    local_offset,
                    state,
                    inhibition,
                    recruited,
                    contact_mask,
                    left,
                    right,
                    lesion=lesion,
                )
                contact_logits.append(action)
                stop_logits.append(stop)
                candidate_masks.append(contact_mask & ~recruited)
                states.append(state)
                inhibitions.append(inhibition)
                if step == max_groups:
                    break
                current = (group_ids == step) & contact_mask
                active = group_count > step
                updated_recruited = recruited | current
                updated_state, updated_inhibition = self._advance(
                    state,
                    inhibition,
                    current,
                    updated_recruited,
                    axis,
                    graphs,
                    contact_mask,
                    lesion=lesion,
                )
                state = torch.where(
                    active[:, None, None], updated_state, state
                )
                inhibition = torch.where(
                    active, updated_inhibition, inhibition
                )
                recruited = torch.where(
                    active[:, None], updated_recruited, recruited
                )
            return {
                "contact_logits": torch.stack(contact_logits, dim=1),
                "stop_logits": torch.stack(stop_logits, dim=1),
                "candidate_mask": torch.stack(candidate_masks, dim=1),
                "latent_state": torch.stack(states, dim=1),
                "inhibitory_state": torch.stack(inhibitions, dim=1),
                "endpoint_union": left | right,
            }

        @torch.no_grad()
        def rollout(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            local_offset: Tensor,
            axis_coordinate: Tensor,
            forward_graph: Tensor,
            reverse_graph: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
            *,
            n_events: int,
            seed: int,
            batch_size: int = 512,
            lesion: str = "none",
        ) -> Tuple[np.ndarray, np.ndarray]:
            if lesion not in self.VALID_LESIONS:
                raise ValueError(f"unknown lesion: {lesion}")
            self.eval()
            device = contact_features.device
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            all_groups = []
            all_counts = []
            remaining = int(n_events)
            while remaining:
                current_batch = min(int(batch_size), remaining)
                features = contact_features[:1].expand(
                    current_batch, -1, -1
                )
                mask = contact_mask[:1].expand(current_batch, -1)
                offset = local_offset
                if offset.ndim == 3:
                    offset = offset[:1]
                prepared = self._prepare(
                    features,
                    mask,
                    offset,
                    axis_coordinate,
                    forward_graph,
                    reverse_graph,
                    left_endpoint,
                    right_endpoint,
                )
                offset, axis, graphs, left, right = prepared
                n_contacts = mask.shape[1]
                state = features.new_zeros(
                    (current_batch, n_contacts, self.structured_rank)
                )
                inhibition = features.new_zeros(current_batch)
                recruited = torch.zeros_like(mask)
                groups = torch.full(
                    mask.shape, -1, dtype=torch.int16, device=device
                )
                counts = torch.zeros(
                    current_batch, dtype=torch.int16, device=device
                )
                alive = torch.ones(
                    current_batch, dtype=torch.bool, device=device
                )
                for _ in range(n_contacts):
                    action_logits, stop_logit = self._decode(
                        features,
                        offset,
                        state,
                        inhibition,
                        recruited,
                        mask,
                        left,
                        right,
                        lesion=lesion,
                    )
                    logits = torch.cat(
                        [stop_logit[:, None], action_logits], dim=1
                    )
                    action = torch.multinomial(
                        torch.softmax(logits, dim=1),
                        1,
                        generator=generator,
                    ).squeeze(1)
                    action = torch.where(
                        alive, action, torch.zeros_like(action)
                    )
                    chose_stop = action == 0
                    chose_contact = alive & ~chose_stop
                    current = torch.zeros_like(mask)
                    if torch.any(chose_contact):
                        row = torch.where(chose_contact)[0]
                        contact = action[row] - 1
                        groups[row, contact] = counts[row]
                        current[row, contact] = True
                        recruited[row, contact] = True
                        counts[row] += 1
                    state, inhibition = self._advance(
                        state,
                        inhibition,
                        current,
                        recruited,
                        axis,
                        graphs,
                        mask,
                        lesion=lesion,
                    )
                    alive = alive & ~chose_stop
                    if not torch.any(alive):
                        break
                all_groups.append(groups.cpu().numpy())
                all_counts.append(counts.cpu().numpy())
                remaining -= current_batch
            return np.row_stack(all_groups), np.concatenate(all_counts)


    def structured_next_set_stop_loss(
        outputs: Mapping[str, Tensor],
        group_ids: Tensor,
        group_count: Tensor,
        *,
        stop_calibration_weight: float = 0.1,
        endpoint_source_weight: float = 0.05,
    ) -> Dict[str, Tensor]:
        """Next-set/STOP loss plus weak, pre-specified structural terms."""
        from src.topic5_rank_distribution import next_set_stop_loss

        base = next_set_stop_loss(outputs, group_ids, group_count)
        stop_logits = outputs["stop_logits"]
        steps = torch.arange(
            stop_logits.shape[1], device=stop_logits.device
        )[None, :]
        valid = steps <= group_count[:, None]
        stop_target = steps == group_count[:, None]
        stop_brier = (
            (
                torch.sigmoid(stop_logits)
                - stop_target.to(stop_logits.dtype)
            ).square()
            * valid
        ).sum() / valid.sum().clamp_min(1)

        endpoint = outputs["endpoint_union"]
        initial_contacts = outputs["contact_logits"][:, 0]
        initial_stop = outputs["stop_logits"][:, 0]
        denominator = torch.logsumexp(
            torch.cat([initial_stop[:, None], initial_contacts], dim=1),
            dim=1,
        )
        endpoint_numerator = torch.logsumexp(
            initial_contacts.masked_fill(~endpoint, -1e9), dim=1
        )
        endpoint_source_nll = (denominator - endpoint_numerator).mean()
        total = (
            base["total"]
            + float(stop_calibration_weight) * stop_brier
            + float(endpoint_source_weight) * endpoint_source_nll
        )
        return {
            **base,
            "total": total,
            "next_set_stop": base["total"],
            "stop_calibration": stop_brier,
            "endpoint_source": endpoint_source_nll,
        }
