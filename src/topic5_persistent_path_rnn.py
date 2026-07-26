"""Event-persistent latent path-mode graph RNN.

The patient graph bases are fixed train80-only inputs.  A latent component is
one (path mode, direction) pair selected once per event.  Training marginalizes
over components; prefix predictions update their posterior using past ranks
only.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

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

    class PersistentPathModeRNN(nn.Module):
        """Shared scalar dynamics on fixed patient-specific graph components."""

        VALID_LESIONS = {"none", "inhibition", "graph"}

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            local_offset_dim: int = 1,
            use_recurrence: bool = True,
        ):
            super().__init__()
            self.local_offset_dim = int(local_offset_dim)
            self.use_recurrence = bool(use_recurrence)
            if self.local_offset_dim < 1:
                raise ValueError("local_offset_dim must be positive")
            self.static_feature = nn.Linear(int(contact_feature_dim), 1)
            self.local_readout = nn.Linear(
                self.local_offset_dim, 1, bias=False
            )
            self.raw_endpoint_gain = nn.Parameter(torch.tensor(-3.0))
            if self.use_recurrence:
                self.raw_alpha = nn.Parameter(torch.tensor(0.0))
                self.raw_input_gain = nn.Parameter(torch.tensor(0.0))
                self.raw_propagation_gain = nn.Parameter(torch.tensor(0.0))
                self.raw_decay = nn.Parameter(torch.tensor(-0.5))
                self.raw_inhibition_gain = nn.Parameter(torch.tensor(-1.0))
                self.state_bias = nn.Parameter(torch.tensor(0.0))
                self.raw_output_gain = nn.Parameter(torch.tensor(0.0))
                self.raw_inhibitory_alpha = nn.Parameter(torch.tensor(-0.5))
                self.raw_inhibitory_drive = nn.Parameter(torch.tensor(-0.5))
            else:
                for name in (
                    "raw_alpha",
                    "raw_input_gain",
                    "raw_propagation_gain",
                    "raw_decay",
                    "raw_inhibition_gain",
                    "state_bias",
                    "raw_output_gain",
                    "raw_inhibitory_alpha",
                    "raw_inhibitory_drive",
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
        def alpha(self) -> Tensor:
            if self.raw_alpha is None:
                return self.stop_bias.new_zeros(())
            return 0.05 + 0.90 * torch.sigmoid(self.raw_alpha)

        @property
        def input_gain(self) -> Tensor:
            if self.raw_input_gain is None:
                return self.stop_bias.new_zeros(())
            return F.softplus(self.raw_input_gain)

        @property
        def propagation_gain(self) -> Tensor:
            if self.raw_propagation_gain is None:
                return self.stop_bias.new_zeros(())
            return F.softplus(self.raw_propagation_gain)

        @property
        def decay(self) -> Tensor:
            if self.raw_decay is None:
                return self.stop_bias.new_zeros(())
            return F.softplus(self.raw_decay)

        @property
        def inhibition_gain(self) -> Tensor:
            if self.raw_inhibition_gain is None:
                return self.stop_bias.new_zeros(())
            return F.softplus(self.raw_inhibition_gain)

        @property
        def output_gain(self) -> Tensor:
            if self.raw_output_gain is None:
                return self.stop_bias.new_zeros(())
            return F.softplus(self.raw_output_gain)

        @property
        def inhibitory_alpha(self) -> Tensor:
            if self.raw_inhibitory_alpha is None:
                return self.stop_bias.new_zeros(())
            return 0.05 + 0.90 * torch.sigmoid(self.raw_inhibitory_alpha)

        @property
        def inhibitory_drive(self) -> Tensor:
            if self.raw_inhibitory_drive is None:
                return self.stop_bias.new_zeros(())
            return F.softplus(self.raw_inhibitory_drive)

        @staticmethod
        def _expand_contact(
            value: Tensor, batch_size: int, n_contacts: int, name: str
        ) -> Tensor:
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(batch_size, -1)
            if value.shape != (batch_size, n_contacts):
                raise ValueError(f"{name} must be [contact] or [batch, contact]")
            return value

        @staticmethod
        def _expand_graphs(
            value: Tensor, batch_size: int, n_contacts: int
        ) -> Tensor:
            if value.ndim == 3:
                value = value.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if (
                value.ndim != 4
                or value.shape[0] != batch_size
                or value.shape[2:] != (n_contacts, n_contacts)
            ):
                raise ValueError(
                    "component_graphs must be [component, contact, contact] "
                    "or [batch, component, contact, contact]"
                )
            return value

        @staticmethod
        def _expand_prior(value: Tensor, batch_size: int, n_components: int) -> Tensor:
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(batch_size, -1)
            if value.shape != (batch_size, n_components):
                raise ValueError(
                    "component_prior must be [component] or [batch, component]"
                )
            if torch.any(value < 0):
                raise ValueError("component_prior must be nonnegative")
            denominator = value.sum(1, keepdim=True)
            if torch.any(denominator <= 0):
                raise ValueError("component_prior must have positive mass")
            return value / denominator

        def _prepare(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            local_offset: Tensor,
            component_graphs: Tensor,
            component_prior: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            if contact_features.ndim != 3 or contact_mask.ndim != 2:
                raise ValueError("contact inputs must be batched")
            batch_size, n_contacts, _ = contact_features.shape
            if contact_mask.shape != (batch_size, n_contacts):
                raise ValueError("contact mask shape mismatch")
            if local_offset.ndim == 2:
                local_offset = local_offset.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            if local_offset.shape != (
                batch_size,
                n_contacts,
                self.local_offset_dim,
            ):
                raise ValueError("local_offset shape mismatch")
            graphs = self._expand_graphs(
                component_graphs, batch_size, n_contacts
            )
            if torch.any(graphs < 0):
                raise ValueError("component graphs must be nonnegative")
            prior = self._expand_prior(
                component_prior, batch_size, graphs.shape[1]
            )
            left = self._expand_contact(
                left_endpoint, batch_size, n_contacts, "left_endpoint"
            ).bool()
            right = self._expand_contact(
                right_endpoint, batch_size, n_contacts, "right_endpoint"
            ).bool()
            valid_pair = (
                contact_mask[:, None, :, None]
                & contact_mask[:, None, None, :]
            ).to(graphs.dtype)
            graphs = graphs * valid_pair
            return local_offset, graphs, prior, left, right

        def _decode(
            self,
            contact_features: Tensor,
            local_offset: Tensor,
            state: Tensor,
            inhibition: Tensor,
            recruited: Tensor,
            contact_mask: Tensor,
            left: Tensor,
            right: Tensor,
            *,
            lesion: str,
        ) -> Tuple[Tensor, Tensor]:
            static = self.static_feature(contact_features).squeeze(-1)
            static = static + self.local_readout(local_offset).squeeze(-1)
            progress = (
                recruited.sum(1).to(static.dtype)
                / contact_mask.sum(1).clamp_min(1).to(static.dtype)
            )
            endpoint = (left | right).to(static.dtype)
            static = static + (
                (progress == 0).to(static.dtype)[:, None]
                * self.endpoint_gain
                * endpoint
            )
            logits = static[:, None, :].expand(-1, state.shape[1], -1)
            mean_excitation = logits.new_zeros(
                (logits.shape[0], logits.shape[1])
            )
            if self.use_recurrence:
                logits = logits + self.output_gain * state
                valid = contact_mask[:, None, :].to(state.dtype)
                mean_excitation = (
                    F.relu(state) * valid
                ).sum(2) / valid.sum(2).clamp_min(1.0)
            stop = (
                self.stop_bias
                + F.softplus(self.raw_stop_progress_gain) * progress[:, None]
                - F.softplus(self.raw_continue_state_gain) * mean_excitation
            )
            if lesion != "inhibition" and self.use_recurrence:
                stop = (
                    stop
                    + F.softplus(self.raw_stop_inhibition_gain) * inhibition
                )
            candidate = contact_mask & ~recruited
            logits = logits.masked_fill(~candidate[:, None, :], -1e9)
            return logits, stop

        def _advance(
            self,
            state: Tensor,
            inhibition: Tensor,
            current: Tensor,
            graphs: Tensor,
            contact_mask: Tensor,
            *,
            lesion: str,
        ) -> Tuple[Tensor, Tensor]:
            if not self.use_recurrence:
                return state, inhibition
            active_graphs = (
                torch.zeros_like(graphs) if lesion == "graph" else graphs
            )
            component_input = current[:, None, :].to(state.dtype)
            propagated = torch.einsum(
                "bmij,bmj->bmi",
                active_graphs,
                state + component_input,
            )
            inhibitory_term = (
                state.new_zeros(state.shape)
                if lesion == "inhibition"
                else self.inhibition_gain * inhibition[:, :, None]
            )
            proposal = torch.tanh(
                self.input_gain * component_input
                + self.propagation_gain * propagated
                - self.decay * state
                - inhibitory_term
                + self.state_bias
            )
            updated = (1.0 - self.alpha) * state + self.alpha * proposal
            updated = updated * contact_mask[:, None, :].to(updated.dtype)
            if lesion == "inhibition":
                updated_inhibition = torch.zeros_like(inhibition)
            else:
                valid = contact_mask[:, None, :].to(updated.dtype)
                mean_excitation = (
                    F.relu(updated) * valid
                ).sum(2) / valid.sum(2).clamp_min(1.0)
                proposal_inhibition = torch.tanh(
                    self.inhibitory_drive * mean_excitation
                )
                updated_inhibition = (
                    (1.0 - self.inhibitory_alpha) * inhibition
                    + self.inhibitory_alpha * proposal_inhibition
                )
            return updated, updated_inhibition

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
            component_graphs: Tensor,
            component_prior: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
            *,
            lesion: str = "none",
        ) -> Dict[str, Tensor]:
            if lesion not in self.VALID_LESIONS:
                raise ValueError(f"unknown lesion: {lesion}")
            local_offset, graphs, prior, left, right = self._prepare(
                contact_features,
                contact_mask,
                local_offset,
                component_graphs,
                component_prior,
                left_endpoint,
                right_endpoint,
            )
            batch_size, n_contacts = contact_mask.shape
            n_components = graphs.shape[1]
            state = contact_features.new_zeros(
                (batch_size, n_components, n_contacts)
            )
            inhibition = contact_features.new_zeros(
                (batch_size, n_components)
            )
            recruited = torch.zeros_like(contact_mask)
            max_groups = int(group_count.max().item())
            contact_logits = []
            stop_logits = []
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
                states.append(state)
                inhibitions.append(inhibition)
                if step == max_groups:
                    break
                current = (group_ids == step) & contact_mask
                active = group_count > step
                updated_state, updated_inhibition = self._advance(
                    state,
                    inhibition,
                    current,
                    graphs,
                    contact_mask,
                    lesion=lesion,
                )
                state = torch.where(
                    active[:, None, None], updated_state, state
                )
                inhibition = torch.where(
                    active[:, None], updated_inhibition, inhibition
                )
                recruited = torch.where(
                    active[:, None], recruited | current, recruited
                )
            return {
                "component_contact_logits": torch.stack(
                    contact_logits, dim=2
                ),
                "component_stop_logits": torch.stack(stop_logits, dim=2),
                "component_prior": prior,
                "latent_state": torch.stack(states, dim=2),
                "inhibitory_state": torch.stack(inhibitions, dim=2),
                "endpoint_union": left | right,
            }

        @torch.no_grad()
        def rollout(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            local_offset: Tensor,
            component_graphs: Tensor,
            component_prior: Tensor,
            left_endpoint: Tensor,
            right_endpoint: Tensor,
            *,
            n_events: int,
            seed: int,
            batch_size: int = 512,
            lesion: str = "none",
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if lesion not in self.VALID_LESIONS:
                raise ValueError(f"unknown lesion: {lesion}")
            self.eval()
            device = contact_features.device
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            all_groups = []
            all_counts = []
            all_components = []
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
                    component_graphs,
                    component_prior,
                    left_endpoint,
                    right_endpoint,
                )
                offset, graphs, prior, left, right = prepared
                selected = torch.multinomial(
                    prior,
                    1,
                    replacement=True,
                    generator=generator,
                ).squeeze(1)
                row = torch.arange(current_batch, device=device)
                graph = graphs[row, selected][:, None, :, :]
                n_contacts = mask.shape[1]
                state = features.new_zeros((current_batch, 1, n_contacts))
                inhibition = features.new_zeros((current_batch, 1))
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
                        [stop_logit[:, 0, None], action_logits[:, 0]], dim=1
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
                        active_row = torch.where(chose_contact)[0]
                        contact = action[active_row] - 1
                        groups[active_row, contact] = counts[active_row]
                        current[active_row, contact] = True
                        recruited[active_row, contact] = True
                        counts[active_row] += 1
                    state, inhibition = self._advance(
                        state,
                        inhibition,
                        current,
                        graph,
                        mask,
                        lesion=lesion,
                    )
                    alive = alive & ~chose_stop
                    if not torch.any(alive):
                        break
                all_groups.append(groups.cpu().numpy())
                all_counts.append(counts.cpu().numpy())
                all_components.append(selected.cpu().numpy())
                remaining -= current_batch
            return (
                np.row_stack(all_groups),
                np.concatenate(all_counts),
                np.concatenate(all_components),
            )


    def persistent_mixture_loss(
        outputs: Mapping[str, Tensor],
        group_ids: Tensor,
        group_count: Tensor,
        *,
        stop_calibration_weight: float = 0.1,
        endpoint_source_weight: float = 0.0,
    ) -> Dict[str, Tensor]:
        """Prefix-causal marginal likelihood over event-persistent components."""
        contact_logits = outputs["component_contact_logits"]
        stop_logits = outputs["component_stop_logits"]
        prior = outputs["component_prior"]
        batch_size, n_components, n_steps, _ = contact_logits.shape
        if stop_logits.shape != (batch_size, n_components, n_steps):
            raise ValueError("component stop-logit shape mismatch")
        steps = torch.arange(n_steps, device=group_ids.device)[None, :]
        valid = steps <= group_count[:, None]
        terminal = steps == group_count[:, None]
        denominator = torch.logsumexp(
            torch.cat([stop_logits[..., None], contact_logits], dim=3),
            dim=3,
        )
        target_set = group_ids[:, None, None, :] == steps[:, None, :, None]
        target_contact = torch.logsumexp(
            contact_logits.masked_fill(~target_set, -1e9), dim=3
        )
        numerator = torch.where(terminal[:, None, :], stop_logits, target_contact)
        component_log_probability = numerator - denominator
        component_action_probability = torch.softmax(
            torch.cat([stop_logits[..., None], contact_logits], dim=3),
            dim=3,
        )
        log_posterior = torch.log(prior.clamp_min(1e-12))
        predictive_probabilities = []
        mixture_step_nll = []
        posterior_trajectory = []
        for step in range(n_steps):
            posterior = torch.softmax(log_posterior, dim=1)
            posterior_trajectory.append(posterior)
            predictive = torch.sum(
                posterior[:, :, None]
                * component_action_probability[:, :, step],
                dim=1,
            )
            predictive_probabilities.append(predictive)
            target_probability = torch.where(
                terminal[:, step],
                predictive[:, 0],
                (
                    predictive[:, 1:]
                    * target_set[:, 0, step].to(predictive.dtype)
                ).sum(1),
            )
            mixture_step_nll.append(
                -torch.log(target_probability.clamp_min(1e-12))
            )
            updated = log_posterior + component_log_probability[:, :, step]
            updated = updated - torch.logsumexp(updated, dim=1, keepdim=True)
            log_posterior = torch.where(
                valid[:, step, None], updated, log_posterior
            )
        predictive = torch.stack(predictive_probabilities, dim=1)
        mixture_step_nll = torch.stack(mixture_step_nll, dim=1)
        posterior_trajectory = torch.stack(posterior_trajectory, dim=1)
        event_nll = (
            (mixture_step_nll * valid.to(mixture_step_nll.dtype)).sum(1)
            / valid.sum(1).clamp_min(1)
        )
        stop_target = terminal.to(predictive.dtype)
        stop_brier = (
            (predictive[:, :, 0] - stop_target).square()
            * valid.to(predictive.dtype)
        ).sum() / valid.sum().clamp_min(1)
        endpoint = outputs["endpoint_union"]
        initial_contact_probability = predictive[:, 0, 1:]
        endpoint_probability = (
            initial_contact_probability * endpoint.to(predictive.dtype)
        ).sum(1)
        endpoint_source_nll = -torch.log(
            endpoint_probability.clamp_min(1e-12)
        ).mean()
        total = (
            event_nll.mean()
            + float(stop_calibration_weight) * stop_brier
            + float(endpoint_source_weight) * endpoint_source_nll
        )
        return {
            "total": total,
            "event_nll": event_nll,
            "step_nll": mixture_step_nll,
            "step_mask": valid,
            "next_set_stop": event_nll.mean(),
            "stop_calibration": stop_brier,
            "endpoint_source": endpoint_source_nll,
            "predictive_action_probability": predictive,
            "component_posterior_trajectory": posterior_trajectory,
            "final_component_posterior": torch.softmax(log_posterior, dim=1),
            "component_log_probability": component_log_probability,
        }
