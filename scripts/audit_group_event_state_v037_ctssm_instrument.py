#!/usr/bin/env python3
"""Machine audit for the v0.3.7 irregular-time dual-stream observer.

This is an architecture/instrument check, not a human scientific result.  It
tests the exact properties needed before H1/H2 may use the observer: scan and
gradient parity, physical-time memory, event-density/null-event invariance,
finite 72-hour trajectories and practical long-chain execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_group_event_state.v037 import (
    DualStreamEventCTSSM,
    affine_associative_scan,
    affine_sequential_scan,
)
from src.topic5_group_event_state.v037.contracts import atomic_json


TAUS = (600.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0, 115200.0)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _parity(device: torch.device) -> dict:
    torch.manual_seed(3701)
    phi_a = (0.8 + 0.199 * torch.rand(1024, 2, 24, device=device)).requires_grad_()
    q_a = torch.randn(1024, 2, 24, device=device, requires_grad=True)
    phi_b = phi_a.detach().clone().requires_grad_()
    q_b = q_a.detach().clone().requires_grad_()
    pa, qa = affine_associative_scan(phi_a, q_a)
    pb, qb = affine_sequential_scan(phi_b, q_b)
    la = qa[-1].square().mean() + pa[-1].mean()
    lb = qb[-1].square().mean() + pb[-1].mean()
    la.backward(); lb.backward()
    return {
        "length": 1024,
        "value_max_abs_error": float((qa - qb).abs().max()),
        "transition_max_abs_error": float((pa - pb).abs().max()),
        "phi_gradient_max_abs_error": float((phi_a.grad - phi_b.grad).abs().max()),
        "impulse_gradient_max_abs_error": float((q_a.grad - q_b.grad).abs().max()),
    }


def _configured_model(device: torch.device) -> DualStreamEventCTSSM:
    model = DualStreamEventCTSSM(
        3,
        4,
        taus_seconds=TAUS,
        burden_channels_per_tau=2,
        grammar_channels_per_tau=3,
        input_scale=0.02,
    ).to(device)
    return model


def _physical_memory(device: torch.device) -> dict:
    # DESIGN-CONFORMANCE CHECK, NOT A MEASUREMENT.
    #
    # The time constants are fixed buffers (``learnable_tau`` defaults to
    # False), so ``slowest_expected_retention`` below is exp(-dt / max tau) by
    # construction: at 32 h it is exp(-1) = 0.368 whatever the data contain.
    # This check only proves the implementation follows its own registered
    # bank.  It says nothing about whether human recordings carry structure at
    # these lags, and must never be reported as evidence that the architecture
    # "retains 37% of its influence after 32 hours" in a patient.
    #
    # One old observation and one zero-weight terminal clock event.  The
    # derivative should follow the explicitly registered physical-time bank.
    model = _configured_model(device)
    rows = []
    for hours in (0.0, 0.5, 2.0, 6.0, 8.0, 16.0, 32.0):
        burden = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]], device=device, requires_grad=True)
        grammar = torch.tensor(
            [[0.1, 0.2, -0.1, 0.8], [0.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True
        )
        out = model(
            torch.tensor([0.0, hours * 3600.0], device=device),
            burden,
            grammar,
            event_weight=torch.tensor([1.0, 0.0], device=device),
        )
        loss = out.post_burden[-1].sum()
        grad = torch.autograd.grad(loss, burden)[0][0].norm()
        rows.append({
            "lag_hours": hours,
            "burden_gradient_norm": float(grad),
            "slowest_expected_retention": float(np.exp(-(hours * 3600.0) / max(TAUS))),
        })
    observed = {row["lag_hours"]: row["burden_gradient_norm"] for row in rows}
    reference = observed.get(0.0, 0.0)
    return {
        "rows": rows,
        "maximum_registered_tau_hours": max(TAUS) / 3600.0,
        "check_semantics": "design_conformance_not_human_measurement",
        "expected_retention_is_definitional": True,
        "observed_gradient_ratio_at_max_tau": (
            None if reference == 0.0 else observed.get(max(TAUS) / 3600.0, 0.0) / reference
        ),
        "note": (
            "time constants are fixed buffers, so the expected-retention column is "
            "exp(-dt/tau) by construction; the observed ratio is a mixture over the "
            "whole bank and neither column is evidence about human data"
        ),
    }


def _density_and_null_invariance(device: torch.device) -> dict:
    torch.manual_seed(3702)
    model = _configured_model(device)
    base = model(
        torch.tensor([0.0, 7200.0], device=device),
        torch.tensor([[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]], device=device),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]], device=device),
        event_weight=torch.tensor([1.0, 0.0], device=device),
    )
    dense_t = torch.linspace(0.0, 7200.0, 502, device=device)
    dense_b = torch.zeros(502, 3, device=device); dense_b[0] = torch.tensor([1.0, 0.5, 0.25], device=device)
    dense_g = torch.zeros(502, 4, device=device); dense_g[0] = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
    weight = torch.zeros(502, device=device); weight[0] = 1.0
    inserted = model(dense_t, dense_b, dense_g, event_weight=weight)

    one = model(
        torch.tensor([0.0], device=device),
        torch.ones(1, 3, device=device),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device),
    )
    many = model(
        torch.zeros(11, device=device),
        torch.ones(11, 3, device=device),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device).repeat(11, 1),
    )
    return {
        "null_event_terminal_feature_max_abs_error": float(
            (base.post_features[-1] - inserted.post_features[-1]).abs().max()
        ),
        "repeated_identical_mark_grammar_max_abs_error": float(
            (one.post_grammar_composition[-1] - many.post_grammar_composition[-1]).abs().max()
        ),
        "repeated_identical_mark_burden_ratio": float(
            many.post_burden[-1].norm() / one.post_burden[-1].norm()
        ),
        "repeated_identical_mark_mass_ratio": float(
            many.post_grammar_mass[-1].mean() / one.post_grammar_mass[-1].mean()
        ),
    }


def _stability_72h(device: torch.device) -> dict:
    torch.manual_seed(3703)
    model = _configured_model(device)
    # About one event every 20 seconds for 72 hours.
    n = 12961
    times = torch.linspace(0.0, 72.0 * 3600.0, n, device=device)
    burden = torch.ones(n, 3, device=device)
    grammar = torch.tensor([[0.2, -0.1, 0.4, 0.5]], device=device).repeat(n, 1)
    with torch.no_grad():
        output = model(times, burden, grammar)
    grammar_abs_max = float(output.post_grammar_composition.abs().max())
    burden_abs_max = float(output.post_burden.abs().max())
    finite = bool(torch.isfinite(output.post_features).all())
    # Composition must be unchanged by a constant repeated mark even though
    # burden and reliability mass grow to their finite shot-noise equilibrium.
    grammar_drift = float(
        (output.post_grammar_composition[-1] - output.post_grammar_composition[n // 2]).abs().max()
    )
    return {
        "n_events": n,
        "duration_hours": 72.0,
        "all_finite": finite,
        "burden_abs_max": burden_abs_max,
        "grammar_abs_max": grammar_abs_max,
        "grammar_composition_half_to_end_max_abs_drift": grammar_drift,
        "post_feature_norm": float(output.post_features[-1].norm()),
    }


def _benchmark(device: torch.device) -> dict:
    torch.manual_seed(3704)
    model = _configured_model(device)
    n = 8192
    times = torch.cumsum(torch.rand(n, device=device) * 40.0 + 1.0, 0)
    burden = torch.randn(n, 3, device=device, requires_grad=True)
    grammar = torch.randn(n, 4, device=device, requires_grad=True)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = model(times, burden, grammar)
    loss = output.post_features[::64].square().mean()
    loss.backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "n_events": n,
        "forward_backward_seconds": elapsed,
        "events_per_second": n / elapsed,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "first_event_gradient_norm": float(burden.grad[0].norm()),
        "last_event_gradient_norm": float(burden.grad[-1].norm()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "/data/hfosp_group_event_state_v0_3_7/instrument/ctssm_instrument_audit.json"
    ))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = {
        "format": "group_event_state_v0_3_7_ctssm_instrument_audit_v2",
        "architecture_version": "0.3.7",
        "state_semantics": "observer_not_physiological_jump",
        "device": str(device),
        "code_commit": _git_commit(),
        "registered_taus_seconds": list(TAUS),
        "scan_parity": _parity(device),
        "physical_memory": _physical_memory(device),
        "density_and_null_invariance": _density_and_null_invariance(device),
        "stability_72h": _stability_72h(device),
        "long_chain_benchmark": _benchmark(device),
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    p = payload["scan_parity"]
    d = payload["density_and_null_invariance"]
    s = payload["stability_72h"]
    payload["checks"] = {
        "scan_value_parity": p["value_max_abs_error"] < 1e-4,
        "scan_gradient_parity": max(
            p["phi_gradient_max_abs_error"], p["impulse_gradient_max_abs_error"]
        ) < 1e-4,
        "null_event_invariance": d["null_event_terminal_feature_max_abs_error"] < 1e-5,
        "grammar_density_invariance": d["repeated_identical_mark_grammar_max_abs_error"] < 1e-5,
        "stable_72h": s["all_finite"] and s["grammar_composition_half_to_end_max_abs_drift"] < 1e-4,
        "first_event_gradient_nonzero": payload["long_chain_benchmark"]["first_event_gradient_norm"] > 0.0,
    }
    payload["status"] = "PASS" if all(payload["checks"].values()) else "FAIL"
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
