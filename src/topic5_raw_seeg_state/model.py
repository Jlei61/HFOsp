"""Raw-SEEG encoder + stable dynamics + shared linear decoder (spec section 5).

Data flow for one training sample (all sizes from ``contract``)::

    raw (C, M*15360)                       M = CONTEXT_MINUTES minutes at 256 Hz
      -> reshape        (C, M, 12, 20, 64)     minute / 5 s window / 250 ms patch
      -> sanitize+mask                          non-finite -> 0, invalid -> 0
      -> shared Conv1D patch projection         64 samples -> d=128
      -> [2] temporal Transformer               sequence = 20 patches of one window
      -> mean over patches                      one token per (contact, window)
      -> + shaft_emb + shaft_index_emb + coord_proj*coord_valid   position floor
      -> [3] spatial Transformer                sequence = C contacts, masked
      -> [4] attention pool over 12 windows     one token per (contact, minute)
      -> [4] attention pool over contacts       one token per minute
      -> [5] causal context Transformer         sequence = M minute tokens
      -> [6] linear head                        z_t in R^32
      -> DampedRotationDynamics(z, h)           closed form, h in minutes
      -> shared linear decoder                  (C, 12) log-power field

Three things in here are load-bearing and easy to get wrong:

1. **Invalid data can never reach the state.**  Non-finite samples are mapped to
   0 *before* the validity mask is applied (``inf * 0 = NaN`` otherwise), whole
   invalid contacts and artefacted contact-minutes are zeroed, and the same mask
   is used as the spatial attention key-padding mask and as the contact-pooling
   mask.  ``tests/test_raw_seeg_state_model.py::test_8_*`` pins this.  The same
   ordering rule applies to the mm coordinate, which is sanitised before it is
   projected and gated to zero after, so an unlocalised contact cannot pick up a
   phantom position from the projection bias.
2. **Nothing mixes across minutes before the causal context Transformer**, so
   ``encode_sequence(...)[:, m]`` is a state built only from minutes ``<= m``.
3. **The horizon enters only through ``B(h)``** -- one decoder, four horizons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.utils.checkpoint as ckpt
from torch import Tensor, nn

from . import contract
from .dynamics import DampedRotationDynamics

__all__ = ["RawSeegStateModel", "AttentionPool", "benchmark"]


def _transformer(n_layers: int, d_model: int, n_heads: int, dropout: float) -> nn.Module:
    """Pre-LN Transformer encoder stack (pre-LN: stable without a warmup schedule)."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * contract.FFN_MULT,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
    )


class AttentionPool(nn.Module):
    """Learned-query attention pooling over the second-to-last axis.

    ``x`` is ``(..., L, D)`` and ``mask`` is ``(..., L)`` with True = keep.
    Rows whose mask is empty return exactly zero instead of NaN (a softmax over
    an all ``-inf`` row is NaN, and hard-invalid condition 6 of the spec forbids
    letting that into a metric).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Parameter(torch.randn(d_model) * d_model ** -0.5)
        self.scale = d_model ** -0.5

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        scores = (self.key(x) @ self.query) * self.scale            # (..., L)
        if mask is None:
            weights = torch.softmax(scores, dim=-1)
            return (weights.unsqueeze(-1) * x).sum(dim=-2)
        keep = mask.to(torch.bool)
        neg = torch.finfo(scores.dtype).min
        scores = torch.where(keep, scores, torch.full_like(scores, neg))
        empty = ~keep.any(dim=-1, keepdim=True)                     # (..., 1)
        scores = torch.where(empty, torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(keep, weights, torch.zeros_like(weights))
        out = (weights.unsqueeze(-1) * x).sum(dim=-2)
        return torch.where(empty, torch.zeros_like(out), out)


class RawSeegStateModel(nn.Module):
    """R0.1 model: raw SEEG -> z_t -> open-loop spectral field at four horizons.

    Parameters
    ----------
    n_contacts, n_shafts:
        Per-patient. ``n_contacts`` fixes the shared decoder's output width.
    identity_dynamics:
        Baseline #4 ("raw encoder + identity dynamics").  Same encoder, same
        parameter count, ``B(h) = I``.
    use_checkpoint:
        Gradient checkpointing over the per-patch stage (patch conv + temporal
        Transformer).  Those two stages are the only ones that see the full
        ``B * C * M * 12 * 20`` patch tensor (~240 k tokens per sample at
        C = 100), so they are the memory lever; everything downstream has
        already collapsed the patch axis.  Checkpointing is applied in chunks of
        ``checkpoint_chunk`` window-sequences, so the backward re-materialisation
        peak is one chunk rather than the whole stage.
    """

    def __init__(
        self,
        n_contacts: int,
        n_shafts: int,
        *,
        latent_dim: int = contract.LATENT_DIM,
        d_model: int = contract.D_MODEL,
        n_freq_bins: int = contract.N_FREQ_BINS,
        context_minutes: int = contract.CONTEXT_MINUTES,
        horizons: Sequence[int] = contract.HORIZONS_MIN,
        encoder_kind: str = "transformer",
        conformer_kernel_temporal: int = 7,
        conformer_kernel_context: int = 3,
        n_heads: int = contract.N_HEADS,
        dropout: float = contract.DROPOUT,
        max_shaft_index: int = 32,
        identity_dynamics: bool = False,
        use_checkpoint: bool = False,
        checkpoint_chunk: int = 2048,
    ) -> None:
        super().__init__()
        self.n_contacts = int(n_contacts)
        self.n_shafts = int(max(1, n_shafts))
        self.max_shaft_index = int(max(0, max_shaft_index))
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.n_freq_bins = int(n_freq_bins)
        self.context_minutes = int(context_minutes)
        self.horizons: Tuple[int, ...] = tuple(int(h) for h in horizons)
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_chunk = int(checkpoint_chunk)

        self.patch_samples = contract.PATCH_SAMPLES
        self.patches_per_window = contract.PATCHES_PER_WINDOW
        self.windows_per_minute = contract.WINDOWS_PER_MINUTE
        self.minute_samples = contract.MINUTE_SAMPLES

        # [1] shared raw patch projection: 64 samples -> d_model.
        # Two strided convolutions (stride 4 each, 64 -> 16 -> 5) instead of the
        # stride-2/stride-2 variant sketched in the plan: same 2xConv+GELU shape
        # and the same (1 -> d_model) contract, but 8x fewer FLOPs and 4x less
        # activation memory on the ~240 k-token-per-sample patch tensor, which is
        # the only place in this model where memory is actually tight.  Padding
        # is chosen so every one of the 64 input samples reaches at least one
        # surviving output position (conv1: 7/4/3 -> 16, conv2: 5/4/3 -> 5).
        self.patch_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(32, self.d_model, kernel_size=5, stride=4, padding=3),
            nn.GELU(),
        )
        self.patch_pos = nn.Parameter(
            torch.randn(self.patches_per_window, self.d_model) * 0.02
        )

        # [2] within-contact stack over the 20 patches of a window.
        #
        # encoder_kind="conformer" puts a depthwise convolution inside every
        # block here. This is the stage that has to read local waveform
        # morphology out of twenty 250 ms patches, and self-attention over
        # twenty tokens is the wrong tool for it -- the R0.1 pure-Transformer
        # encoder lost to a 1008-coefficient ridge on spectral history at all
        # four horizons after converging, so a representation bottleneck at this
        # stage is the leading remaining explanation.
        self.encoder_kind = str(encoder_kind)
        if self.encoder_kind not in ("transformer", "conformer"):
            raise ValueError(f"unknown encoder_kind {encoder_kind!r}")
        if self.encoder_kind == "conformer":
            from .conformer import ConformerEncoder
            self.temporal = ConformerEncoder(
                contract.N_TEMPORAL_LAYERS, self.d_model, n_heads,
                ffn_mult=contract.FFN_MULT, dropout=dropout,
                kernel_size=int(conformer_kernel_temporal), causal=False)
        else:
            self.temporal = _transformer(contract.N_TEMPORAL_LAYERS, self.d_model,
                                         n_heads, dropout)

        # [3] across-contact spatial Transformer within a 5 s window.
        # Position encoding has a floor that always exists: shaft identity and
        # within-shaft index are known for every contact, while the mm
        # coordinate is added only where coord_valid says we actually localised
        # it.  Five Yuquan subjects have usable recordings but no localisation
        # artifact at all, and some Epilepsiae subjects are localised only in
        # part, so contact_valid (electrically usable) and coord_valid (we know
        # where it is) are independent axes -- see contract ALLOWED_INPUT_KEYS.
        self.coord_proj = nn.Linear(3, self.d_model)
        self.shaft_embedding = nn.Embedding(self.n_shafts, self.d_model)
        self.shaft_index_embedding = nn.Embedding(self.max_shaft_index + 1, self.d_model)
        self.spatial = _transformer(contract.N_SPATIAL_LAYERS, self.d_model, n_heads, dropout)

        # [4] minute pooling: 12 window tokens -> minute token, then over contacts.
        self.window_pool = AttentionPool(self.d_model)
        self.contact_pool = AttentionPool(self.d_model)

        # [5] causal context Transformer over the minute tokens.
        self.context_pos = nn.Parameter(torch.randn(self.context_minutes, self.d_model) * 0.02)
        if self.encoder_kind == "conformer":
            from .conformer import ConformerEncoder
            # CAUSAL. Minute t may not see minute t+1; a symmetric depthwise
            # convolution here would carry the future around the attention mask
            # and quietly invalidate every open-loop number.
            self.context = ConformerEncoder(
                contract.N_CONTEXT_LAYERS, self.d_model, n_heads,
                ffn_mult=contract.FFN_MULT, dropout=dropout,
                kernel_size=int(conformer_kernel_context), causal=True)
        else:
            self.context = _transformer(contract.N_CONTEXT_LAYERS, self.d_model,
                                        n_heads, dropout)

        # [6] state head, dynamics, shared decoder.
        self.head = nn.Linear(self.d_model, self.latent_dim)
        self.dynamics = DampedRotationDynamics(
            latent_dim=self.latent_dim, identity_mode=identity_dynamics
        )
        self.decoder = nn.Linear(self.latent_dim, self.n_contacts * self.n_freq_bins)

    # -- input gate ----------------------------------------------------------

    def _gate(
        self,
        raw: Tensor,
        coords_mm: Tensor,
        coord_valid: Tensor,
        shaft_id: Tensor,
        shaft_index: Tensor,
        contact_valid: Tensor,
        minute_valid: Tensor,
        check_inputs: bool,
        extra_inputs: Dict[str, object],
    ) -> None:
        """Route the encoder inputs through ``contract.assert_no_forbidden_inputs``.

        Unexpected keyword arguments are ALWAYS checked, even when
        ``check_inputs=False``: turning the gate off must not become the way an
        IED / SOZ / lagPatRank vector reaches the encoder.
        """
        if not check_inputs and not extra_inputs:
            return
        payload: Dict[str, object] = {
            "raw": raw,
            "coords_mm": coords_mm,
            "coord_valid": coord_valid,
            "shaft_id": shaft_id,
            "shaft_index": shaft_index,
            "contact_valid": contact_valid,
            "minute_valid": minute_valid,
        }
        payload.update(extra_inputs)
        contract.assert_no_forbidden_inputs(payload)

    # -- encoder -------------------------------------------------------------

    def _patch_stage(self, seq: Tensor) -> Tensor:
        """(N, P, S) raw patches -> (N, D) window tokens (steps 1 and 2).

        The spec fixes steps 1-2 (patch conv, then a Transformer over the 20
        patches of one 5 s window) and step 3 (a Transformer over contacts, one
        token per contact per window) but is silent on how the 20 patch tokens
        become that one window token.  Mean over the patch axis is used: the
        temporal Transformer has already mixed the 20 positions, so a learned
        pooling would add parameters without adding reachable functions, and an
        unweighted mean keeps the window token invariant to patch order effects
        that the positional embedding did not already encode.
        """
        n, p, s = seq.shape
        h = self.patch_conv(seq.reshape(n * p, 1, s))       # (N*P, D, L)
        h = h.mean(dim=-1).reshape(n, p, self.d_model)      # (N, P, D)
        h = h + self.patch_pos
        h = self.temporal(h)                                # (N, P, D)
        return h.mean(dim=1)                                # (N, D)

    #: A CUDA kernel launch may not exceed 65535 blocks in grid dim y/z, and
    #: ``scaled_dot_product_attention`` maps the batch axis onto one of them. The
    #: patch stage flattens (batch x contacts x minutes x windows) into that
    #: axis, so batch 4 at 139 contacts is 4*139*10*12 = 66720 sequences and the
    #: kernel dies with "invalid configuration argument" -- not an OOM, a hard
    #: launch failure that no batch-size ladder would have recovered from.
    #: Twelve cohort subjects have >=137 contacts, so this is the common case,
    #: not an edge case. Splitting the stage is free: the sequences are
    #: independent, and the split is exact.
    MAX_ATTENTION_ROWS = 32768

    def _patch_stage_chunked(self, seq: Tensor) -> Tensor:
        checkpointing = self.use_checkpoint and torch.is_grad_enabled()
        if checkpointing:
            step = max(1, min(self.checkpoint_chunk, self.MAX_ATTENTION_ROWS))
        elif seq.shape[0] <= self.MAX_ATTENTION_ROWS:
            return self._patch_stage(seq)
        else:
            step = self.MAX_ATTENTION_ROWS
        parts: List[Tensor] = []
        for chunk in seq.split(step, dim=0):
            parts.append(ckpt.checkpoint(self._patch_stage, chunk, use_reentrant=False)
                         if checkpointing else self._patch_stage(chunk))
        return torch.cat(parts, dim=0)

    def encode_minute_tokens(
        self,
        raw: Tensor,
        coords_mm: Tensor,
        coord_valid: Tensor,
        shaft_id: Tensor,
        shaft_index: Tensor,
        contact_valid: Tensor,
        minute_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Steps 1-4.  Returns ``(minute_tokens (B,M,D), per_contact (B,C,M,D))``.

        ``M`` is inferred from ``raw`` and may be larger than
        ``context_minutes`` -- that is what ``encode_consistency_pair`` exploits
        to get ``z(t)`` and ``z(t+1)`` from a single pass over the shared
        minutes.  Only the causal context Transformer is tied to
        ``context_minutes``.
        """
        if raw.ndim != 3:
            raise ValueError(f"raw must be (B, C, T); got {tuple(raw.shape)}")
        b, c, t = raw.shape
        if c != self.n_contacts:
            raise ValueError(f"model built for {self.n_contacts} contacts, got {c}")
        if t % self.minute_samples != 0:
            raise ValueError(
                f"raw length {t} is not a whole number of {self.minute_samples}-sample minutes"
            )
        m = t // self.minute_samples
        w, p, s = self.windows_per_minute, self.patches_per_window, self.patch_samples
        if minute_valid.shape != (b, c, m):
            raise ValueError(
                f"minute_valid must be {(b, c, m)}; got {tuple(minute_valid.shape)}"
            )
        for name, arr in (
            ("contact_valid", contact_valid),
            ("coord_valid", coord_valid),
            ("shaft_id", shaft_id),
            ("shaft_index", shaft_index),
        ):
            if tuple(arr.shape) != (b, c):
                raise ValueError(f"{name} must be {(b, c)}; got {tuple(arr.shape)}")
        if coords_mm.shape != (b, c, 3):
            raise ValueError(f"coords_mm must be {(b, c, 3)}; got {tuple(coords_mm.shape)}")

        cm_valid = contact_valid.to(torch.bool).unsqueeze(-1) & minute_valid.to(torch.bool)

        # sanitize BEFORE masking: inf * 0 would be NaN.
        x = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.reshape(b, c, m, w, p, s) * cm_valid.to(x.dtype)[:, :, :, None, None, None]

        tokens = self._patch_stage_chunked(x.reshape(b * c * m * w, p, s))
        tokens = tokens.reshape(b, c, m, w, self.d_model)

        # [3] spatial Transformer, one sequence of C contacts per (sample, window).
        coords = torch.nan_to_num(coords_mm, nan=0.0, posinf=0.0, neginf=0.0)
        shaft = shaft_id.to(torch.long).clamp(0, self.n_shafts - 1)
        s_index = shaft_index.to(torch.long).clamp(0, self.max_shaft_index)
        # The mm term is gated to EXACTLY zero after the projection, not before:
        # coord_proj carries a bias, so feeding zeros to an unlocalised contact
        # would still hand it a constant phantom location.  nan_to_num runs
        # first for the same reason the raw path sanitises before masking --
        # coord_proj(NaN) * 0 would be NaN, not 0.
        coord_term = self.coord_proj(coords.to(tokens.dtype)) * coord_valid.to(
            tokens.dtype
        ).unsqueeze(-1)
        pos = coord_term + self.shaft_embedding(shaft) + self.shaft_index_embedding(s_index)
        tokens = tokens + pos.to(tokens.dtype)[:, :, None, None, :]
        seq = tokens.permute(0, 2, 3, 1, 4).reshape(b * m * w, c, self.d_model)
        # An artefacted contact-minute is padded out of the spatial attention as
        # well, not just out of the pooling: the spec names contact_valid here,
        # this is the same mask AND the per-minute artefact mask, which can only
        # remove leakage, never add it.
        pad = ~cm_valid.permute(0, 2, 1)[:, :, None, :].expand(b, m, w, c).reshape(-1, c)
        # All-masked row guard: softmax over an all -inf row is NaN, and a NaN
        # here would reach z and every metric downstream.  The window index makes
        # a >=70%-valid minute a precondition for usability, so this cannot occur
        # on a usable minute -- but "cannot occur" is how silent NaNs enter a
        # cohort, so a fully-masked window falls back to uniform attention over
        # all contacts.  Its minute token is zeroed by the contact pool anyway
        # (see below), so the fallback only keeps the graph finite; it does not
        # let masked data through.
        pad = torch.where(pad.all(dim=-1, keepdim=True), torch.zeros_like(pad), pad)
        seq = self.spatial(seq, src_key_padding_mask=pad)
        tokens = seq.reshape(b, m, w, c, self.d_model).permute(0, 3, 1, 2, 4)

        # [4] 12 window tokens -> one minute token per contact, then over contacts.
        # The window pool takes no mask: validity is defined per contact-MINUTE,
        # so the 12 windows inside one contact-minute are valid or invalid
        # together and the whole pooled token is zeroed on the next line.  The
        # contact pool does take a mask, and returns an exact zero token for a
        # minute in which every contact is masked (AttentionPool empty-row rule).
        per_contact = self.window_pool(tokens)                       # (B, C, M, D)
        per_contact = per_contact * cm_valid.unsqueeze(-1).to(per_contact.dtype)
        minute_tokens = self.contact_pool(
            per_contact.permute(0, 2, 1, 3), mask=cm_valid.permute(0, 2, 1)
        )                                                            # (B, M, D)
        return minute_tokens, per_contact

    def _context_head(self, minute_tokens: Tensor) -> Tensor:
        """Step 5-6: causal context Transformer -> z at every context position."""
        b, m, _ = minute_tokens.shape
        if m != self.context_minutes:
            raise ValueError(
                f"context Transformer expects {self.context_minutes} minute tokens, got {m}"
            )
        h = minute_tokens + self.context_pos
        causal = torch.nn.Transformer.generate_square_subsequent_mask(
            m, device=h.device, dtype=h.dtype
        )
        h = self.context(h, mask=causal, is_causal=False)
        return self.head(h)

    def encode_sequence(
        self,
        raw: Tensor,
        coords_mm: Tensor,
        coord_valid: Tensor,
        shaft_id: Tensor,
        shaft_index: Tensor,
        contact_valid: Tensor,
        minute_valid: Tensor,
        *,
        check_inputs: bool = True,
        return_tokens: bool = False,
        **extra_inputs: object,
    ):
        """z at every one of the ``context_minutes`` positions -> (B, M, latent).

        This is the ``encode_all_positions`` mode.  Position ``m`` is causal: it
        sees minutes ``0..m`` only, so it is a *shorter-context* state than
        ``encode`` (position ``m`` has ``m+1`` minutes of history, the last one
        has the full 10).  For the consistency term, whose two states must have
        the same context length by contract, use ``encode_consistency_pair``
        (or two ``encode`` calls on shifted windows); use these intermediate
        positions for diagnostics only.
        """
        self._gate(raw, coords_mm, coord_valid, shaft_id, shaft_index,
                   contact_valid, minute_valid, check_inputs, extra_inputs)
        minute_tokens, per_contact = self.encode_minute_tokens(
            raw, coords_mm, coord_valid, shaft_id, shaft_index, contact_valid, minute_valid
        )
        z_seq = self._context_head(minute_tokens)
        if return_tokens:
            return z_seq, minute_tokens, per_contact
        return z_seq

    def encode(
        self,
        raw: Tensor,
        coords_mm: Tensor,
        coord_valid: Tensor,
        shaft_id: Tensor,
        shaft_index: Tensor,
        contact_valid: Tensor,
        minute_valid: Tensor,
        *,
        check_inputs: bool = True,
        **extra_inputs: object,
    ) -> Tensor:
        """State at the END of the context window -> (B, latent). Worker D entry point."""
        z_seq = self.encode_sequence(
            raw, coords_mm, coord_valid, shaft_id, shaft_index, contact_valid, minute_valid,
            check_inputs=check_inputs, **extra_inputs,
        )
        return z_seq[:, -1]

    def encode_consistency_pair(
        self,
        raw: Tensor,
        coords_mm: Tensor,
        coord_valid: Tensor,
        shaft_id: Tensor,
        shaft_index: Tensor,
        contact_valid: Tensor,
        minute_valid: Tensor,
        *,
        check_inputs: bool = True,
        **extra_inputs: object,
    ) -> Tuple[Tensor, Tensor]:
        """``(z_enc(t), z_enc(t+1))`` from ONE pass over ``context_minutes + 1`` minutes.

        Both states are produced by exactly the encoder the contract specifies:
        a full ``context_minutes`` window ending at ``t`` and a full
        ``context_minutes`` window ending at ``t+1``.  The expensive per-patch
        stage runs once over the union of the two windows (11 minutes instead of
        2 x 10), and only the cheap causal context Transformer runs twice.
        """
        self._gate(raw, coords_mm, coord_valid, shaft_id, shaft_index,
                   contact_valid, minute_valid, check_inputs, extra_inputs)
        m_expected = self.context_minutes + 1
        if raw.shape[-1] != m_expected * self.minute_samples:
            raise ValueError(
                f"encode_consistency_pair needs {m_expected} minutes of raw, got "
                f"{raw.shape[-1] / self.minute_samples:.3f}"
            )
        minute_tokens, _ = self.encode_minute_tokens(
            raw, coords_mm, coord_valid, shaft_id, shaft_index, contact_valid, minute_valid
        )
        z_now = self._context_head(minute_tokens[:, :-1])[:, -1]
        z_next = self._context_head(minute_tokens[:, 1:])[:, -1]
        return z_now, z_next

    # -- decoder / forward ---------------------------------------------------

    def decode(self, z: Tensor) -> Tensor:
        """Shared linear decoder: (..., latent) -> (..., n_contacts, n_freq_bins)."""
        out = self.decoder(z)
        return out.reshape(*z.shape[:-1], self.n_contacts, self.n_freq_bins)

    def forward(
        self,
        raw: Tensor,
        coords_mm: Tensor,
        coord_valid: Tensor,
        shaft_id: Tensor,
        shaft_index: Tensor,
        contact_valid: Tensor,
        minute_valid: Tensor,
        *,
        horizons: Optional[Sequence[int]] = None,
        check_inputs: bool = True,
        return_diagnostics: bool = False,
        **extra_inputs: object,
    ) -> Dict[str, object]:
        """Encode, roll the state forward to each horizon, decode.

        Returns ``{"z": (B, latent), "pred": {h: (B, C, F)}}``; with
        ``return_diagnostics`` also ``minute_tokens`` and
        ``per_contact_minute_tokens`` (the decoder does not use them).

        ``horizons`` overrides the constructor list for this call, which is what
        the phase-B pilot ladder needs (1 min only, then 1/5/10, then +100).  It
        is a *decoding* argument, not an encoder input, so it is deliberately
        outside the gated payload -- nothing about it reaches the encoder.
        """
        z_seq, minute_tokens, per_contact = self.encode_sequence(
            raw, coords_mm, coord_valid, shaft_id, shaft_index, contact_valid, minute_valid,
            check_inputs=check_inputs, return_tokens=True, **extra_inputs,
        )
        z = z_seq[:, -1]
        out: Dict[str, object] = {
            "z": z,
            "pred": {
                int(h): self.decode(self.dynamics(z, float(h)))
                for h in (self.horizons if horizons is None else horizons)
            },
        }
        if return_diagnostics:
            out["z_sequence"] = z_seq
            out["minute_tokens"] = minute_tokens
            out["per_contact_minute_tokens"] = per_contact
        return out

    # -- reporting -----------------------------------------------------------

    def param_count(self) -> Dict[str, int]:
        """Parameter count per top-level submodule, plus ``total``."""
        out: Dict[str, int] = {}
        for name, p in self.named_parameters():
            out[name.split(".")[0]] = out.get(name.split(".")[0], 0) + int(p.numel())
        out["total"] = int(sum(p.numel() for p in self.parameters()))
        return out


# ---------------------------------------------------------------------------
# memory / throughput probe
# ---------------------------------------------------------------------------


def benchmark(
    n_contacts: int = 100,
    batch_size: int = 2,
    n_shafts: int = 12,
    context_minutes: int = contract.CONTEXT_MINUTES,
    device: str = "cuda",
    use_checkpoint: bool = False,
    amp: bool = True,
    n_iters: int = 3,
) -> Dict[str, object]:
    """One forward+backward step: parameter count, peak memory, milliseconds."""
    import gc
    import time

    from .losses import total_loss

    dev = torch.device(device)
    model = RawSeegStateModel(
        n_contacts=n_contacts,
        n_shafts=n_shafts,
        context_minutes=context_minutes,
        use_checkpoint=use_checkpoint,
    ).to(dev)
    counts = model.param_count()
    result: Dict[str, object] = {
        "n_contacts": n_contacts,
        "batch_size": batch_size,
        "context_minutes": context_minutes,
        "use_checkpoint": use_checkpoint,
        "amp": amp,
        "param_count": counts,
    }
    t_total = context_minutes * contract.MINUTE_SAMPLES
    buf: Dict[str, object] = {}
    try:
        buf["raw"] = torch.randn(batch_size, n_contacts, t_total, device=dev)
        buf["coords"] = torch.randn(batch_size, n_contacts, 3, device=dev) * 20.0
        buf["shaft"] = torch.arange(n_contacts, device=dev).remainder(n_shafts).expand(
            batch_size, n_contacts
        ).contiguous()
        buf["coord_valid"] = torch.ones(batch_size, n_contacts, dtype=torch.bool, device=dev)
        buf["shaft_index"] = torch.arange(n_contacts, device=dev).div(
            n_shafts, rounding_mode="floor"
        ).expand(batch_size, n_contacts).contiguous()
        buf["cv"] = torch.ones(batch_size, n_contacts, dtype=torch.bool, device=dev)
        buf["mv"] = torch.ones(
            batch_size, n_contacts, context_minutes, dtype=torch.bool, device=dev
        )
        buf["target"] = {
            h: torch.randn(batch_size, n_contacts, contract.N_FREQ_BINS, device=dev)
            for h in contract.HORIZONS_MIN
        }
        buf["mask"] = {h: torch.ones_like(v, dtype=torch.bool) for h, v in buf["target"].items()}
        buf["opt"] = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def _step() -> float:
            opt = buf["opt"]
            opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                out = model(
                    buf["raw"], buf["coords"], buf["coord_valid"], buf["shaft"],
                    buf["shaft_index"], buf["cv"], buf["mv"],
                )
                loss, _ = total_loss(out["pred"], buf["target"], buf["mask"])
            loss.backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e3

        _step()  # warm-up (allocator + autotune)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        times = [_step() for _ in range(n_iters)]
        times.sort()
        result["status"] = "ok"
        result["fwd_bwd_ms_median"] = times[len(times) // 2]
        result["fwd_bwd_ms_all"] = times
        if dev.type == "cuda":
            result["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
            result["peak_reserved_gb"] = torch.cuda.max_memory_reserved(dev) / 1024 ** 3
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        result["status"] = "oom"
        result["error"] = str(exc).splitlines()[0]
    finally:
        buf.clear()
        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _main() -> None:  # pragma: no cover - manual probe
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Raw-SEEG R0.1 memory/throughput probe")
    ap.add_argument("--contacts", type=int, default=100)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--shafts", type=int, default=12)
    ap.add_argument("--context-minutes", type=int, default=contract.CONTEXT_MINUTES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    rows = []
    for bs in args.batch_sizes:
        row = benchmark(
            n_contacts=args.contacts,
            batch_size=bs,
            n_shafts=args.shafts,
            context_minutes=args.context_minutes,
            device=args.device,
            use_checkpoint=args.checkpoint,
            amp=not args.no_amp,
        )
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))
        if row.get("status") == "oom":
            print("stopping: OOM (not retrying the same configuration)")
            break
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":  # pragma: no cover
    _main()
