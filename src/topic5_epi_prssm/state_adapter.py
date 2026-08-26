"""Adapter capacity ladder for the state-conditioned event readout.

Each adapter is run with matched state and no-state controls so that an
improvement can be attributed to the state rather than to the adapter's own
capacity.
"""
from __future__ import annotations

import torch
from torch import nn

#: ``edge_gate`` carries the node FiLM terms as well, so it cannot answer whether
#: edge coupling contributes on its own.  ``edge_gate_only`` is the separable arm:
#: the edge gate without any per-node shift or scale.
ADAPTERS = ("no_state", "initial_state", "node_film", "edge_gate", "edge_gate_only")


class StateAdapter(nn.Module):
    """Maps ``(H^-, r)`` into node scores, global biases and optional edge gates."""

    def __init__(self, mode: str, state_dim: int, *, rank: int = 2):
        super().__init__()
        if mode not in ADAPTERS:
            raise ValueError(f"unknown adapter {mode!r}")
        self.mode = mode
        self.state_dim = state_dim
        self.global_head = nn.Linear(state_dim + 1, 3)     # step bias, stop bias, participation bias
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        if mode in ("node_film", "edge_gate"):
            self.node_head = nn.Linear(state_dim + 1, 2)   # per-node score shift and scale
            nn.init.zeros_(self.node_head.weight)
            nn.init.zeros_(self.node_head.bias)
        if mode in ("edge_gate", "edge_gate_only"):
            self.left = nn.Linear(state_dim, rank, bias=False)
            self.right = nn.Linear(state_dim, rank, bias=False)
            nn.init.zeros_(self.left.weight)
            nn.init.zeros_(self.right.weight)

    def forward(self, state: torch.Tensor, resource: torch.Tensor
                ) -> dict[str, torch.Tensor | None]:
        """``state`` (T, N, D); ``resource`` (T,)."""
        if self.mode == "no_state":
            zeros_g = state.new_zeros(state.shape[0], 3)
            # ``state_visible`` is False here, and the decoder is required to honour
            # it: without that flag the STOP head and the participation head would
            # still receive H, so the "no state" reference would not be state-free.
            return {"node_shift": None, "node_scale": None, "global": zeros_g,
                    "edge_gate": None, "state_visible": False}

        r = resource.reshape(-1, 1)
        pooled = torch.cat([state.mean(dim=-2), r], dim=-1)
        global_terms = self.global_head(pooled)
        if self.mode == "initial_state":
            return {"node_shift": None, "node_scale": None, "global": global_terms,
                    "edge_gate": None, "state_visible": True}

        node_shift = node_scale = None
        if self.mode in ("node_film", "edge_gate"):
            node_in = torch.cat([state, r.unsqueeze(-2).expand(-1, state.shape[-2], -1)],
                                dim=-1)
            node_terms = self.node_head(node_in)
            node_shift = node_terms[..., 0]
            node_scale = 1.0 + torch.tanh(node_terms[..., 1])
        edge_gate = None
        if self.mode in ("edge_gate", "edge_gate_only"):
            left = self.left(state)
            right = self.right(state)
            edge_gate = torch.sigmoid(torch.einsum("tir,tjr->tij", left, right)).unsqueeze(1)
        return {"node_shift": node_shift, "node_scale": node_scale,
                "global": global_terms, "edge_gate": edge_gate, "state_visible": True}
