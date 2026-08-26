"""Conformer encoder (R0.2): shape, causality, and the switch not perturbing R0.1.

The load-bearing test here is causality. The context stage runs over minute
tokens and a symmetric depthwise convolution would let minute t read minute
t+1 -- silently, through the convolution, around the attention mask -- which
would make every open-loop number meaningless while looking perfectly healthy.
It is checked by perturbation, not by inspecting the padding arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import contract  # noqa: E402
from src.topic5_raw_seeg_state.conformer import (  # noqa: E402
    ConformerBlock, ConformerEncoder, ConvModule)
from src.topic5_raw_seeg_state.model import RawSeegStateModel  # noqa: E402

D = 32
H = 4


def test_conv_module_and_block_preserve_shape():
    x = torch.randn(3, 20, D)
    assert ConvModule(D, kernel_size=7).eval()(x).shape == x.shape
    assert ConformerBlock(D, H, kernel_size=7).eval()(x).shape == x.shape


def test_non_causal_kernel_must_be_odd():
    with pytest.raises(ValueError):
        ConvModule(D, kernel_size=8, causal=False)
    ConvModule(D, kernel_size=8, causal=True)          # fine when left-padded


def test_causal_conv_module_cannot_see_the_future():
    """Perturb token t; every output at t' < t must be bit-identical."""
    mod = ConvModule(D, kernel_size=5, dropout=0.0, causal=True).eval()
    x = torch.randn(1, 12, D)
    with torch.no_grad():
        a = mod(x)
        x2 = x.clone()
        x2[0, 7] += 10.0
        b = mod(x2)
    assert torch.equal(a[0, :7], b[0, :7]), "a causal conv leaked the future"
    assert not torch.equal(a[0, 7:], b[0, 7:]), "the perturbation had no effect at all"


def test_symmetric_conv_module_does_see_the_future_so_the_test_bites():
    mod = ConvModule(D, kernel_size=5, dropout=0.0, causal=False).eval()
    x = torch.randn(1, 12, D)
    with torch.no_grad():
        a = mod(x)
        x2 = x.clone()
        x2[0, 7] += 10.0
        b = mod(x2)
    assert not torch.equal(a[0, :7], b[0, :7]), (
        "a symmetric conv must leak backwards -- if it does not, the causal test "
        "above proves nothing")


def test_causal_conformer_stack_is_causal_end_to_end():
    enc = ConformerEncoder(3, D, H, dropout=0.0, kernel_size=3, causal=True).eval()
    n = 10
    mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
    x = torch.randn(1, n, D)
    with torch.no_grad():
        a = enc(x, mask=mask)
        x2 = x.clone()
        x2[0, 6] += 5.0
        b = enc(x2, mask=mask)
    assert torch.equal(a[0, :6], b[0, :6]), "the causal stack leaked the future"


def test_non_causal_stack_refuses_an_attention_mask():
    enc = ConformerEncoder(2, D, H, causal=False)
    mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
    with pytest.raises(ValueError, match="causal"):
        enc(torch.randn(1, 4, D), mask=mask)


def _batch(n_c=5, b=2):
    L = contract.CONTEXT_MINUTES * contract.MINUTE_SAMPLES
    return dict(
        raw=torch.randn(b, n_c, L),
        coords_mm=torch.randn(b, n_c, 3),
        coord_valid=torch.ones(b, n_c, dtype=torch.bool),
        shaft_id=torch.zeros(b, n_c, dtype=torch.long),
        shaft_index=torch.arange(n_c).expand(b, n_c).contiguous(),
        contact_valid=torch.ones(b, n_c, dtype=torch.bool),
        minute_valid=torch.ones(b, n_c, contract.CONTEXT_MINUTES, dtype=torch.bool),
    )


def test_conformer_model_runs_and_stays_causal_over_minutes():
    net = RawSeegStateModel(n_contacts=5, n_shafts=2, encoder_kind="conformer").eval()
    assert net.encoder_kind == "conformer"
    b = _batch()
    with torch.no_grad():
        out = net(**b)
    for h in contract.HORIZONS_MIN:
        assert tuple(out["pred"][h].shape) == (2, 5, contract.N_FREQ_BINS)
        assert torch.isfinite(out["pred"][h]).all()

    # perturbing the LAST minute must move the last state and leave the first alone
    with torch.no_grad():
        z0 = net.encode_sequence(**b)
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["raw"][:, :, -contract.MINUTE_SAMPLES:] += 3.0
        z1 = net.encode_sequence(**b2)
    assert torch.allclose(z0[:, 0], z1[:, 0], atol=1e-5), (
        "the first minute's state moved when only the last minute changed")
    assert not torch.allclose(z0[:, -1], z1[:, -1], atol=1e-5)


def test_switching_encoder_kind_leaves_the_transformer_path_untouched():
    torch.manual_seed(0)
    a = RawSeegStateModel(n_contacts=5, n_shafts=2).eval()
    torch.manual_seed(0)
    b = RawSeegStateModel(n_contacts=5, n_shafts=2, encoder_kind="transformer").eval()
    batch = _batch()
    with torch.no_grad():
        za, zb = a.encode(**batch), b.encode(**batch)
    assert torch.equal(za, zb), "the default path changed when the switch was added"


def test_conformer_costs_a_declared_amount_of_extra_capacity():
    t = RawSeegStateModel(n_contacts=100, n_shafts=12)
    c = RawSeegStateModel(n_contacts=100, n_shafts=12, encoder_kind="conformer")
    nt = sum(p.numel() for p in t.parameters())
    nc = sum(p.numel() for p in c.parameters())
    # The comparison against the Transformer arm is only interpretable if the
    # capacity difference is stated, so pin it: a silent 5x would make any win
    # unreadable.
    assert 0.5 < nc / nt < 2.0, f"conformer/transformer parameter ratio {nc/nt:.2f}"
    print(f"transformer {nt} vs conformer {nc} ({nc / nt:.2f}x)")
