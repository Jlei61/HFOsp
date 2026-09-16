#!/usr/bin/env python3
"""Isolated per-GPU Lyapunov forward-backward readiness check.

This is an engineering stress test.  It does not fit or score scientific data.
One process runs on each GPU, matching the safe overnight scheduler contract.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_module
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(device: int, slot: int, repeats: int, queue) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    from src.topic5_group_event_state.v0312.numerics import Dynamics, stationary_covariance

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(device)
    dyn = Dynamics(coupled=True, nonlinear=True).to(f"cuda:{device}")
    begin = time.time()
    maximum_residual = 0.0
    minimum_gradient_norm = float("inf")
    try:
        for step in range(repeats):
            dyn.zero_grad(set_to_none=True)
            # Deterministically move away from the trivial initialization while
            # retaining a stable symmetric part.
            with torch.no_grad():
                dyn.omega.add_(1e-5 * ((step % 7) - 3))
                dyn.log_sigma.add_(1e-6 * ((step % 5) - 2))
            A = dyn.A()
            sigma = dyn.sigma()
            P = stationary_covariance(A, sigma)
            weight = torch.linspace(0.5, 1.5, P.numel(), device=P.device).reshape_as(P)
            loss = (P * weight).mean()
            loss.backward()
            residual = A.detach() @ P.detach() + P.detach() @ A.detach().T + torch.diag(sigma.detach() ** 2)
            maximum_residual = max(maximum_residual, float(residual.abs().max()))
            grad_norm = sum(float(p.grad.double().square().sum()) for p in dyn.parameters() if p.grad is not None) ** 0.5
            minimum_gradient_norm = min(minimum_gradient_norm, grad_norm)
            if not torch.isfinite(P).all() or not all(
                torch.isfinite(p.grad).all() for p in dyn.parameters() if p.grad is not None
            ):
                raise FloatingPointError(f"non-finite value at repeat {step}")
        queue.put(dict(status="COMPLETE", device=device, slot=slot, repeats=repeats,
                       seconds=time.time() - begin, maximum_residual=maximum_residual,
                       minimum_gradient_norm=minimum_gradient_norm))
    except Exception as exc:
        queue.put(dict(status="FAILED", device=device, slot=slot, repeats=repeats,
                       seconds=time.time() - begin, error=repr(exc)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    mp.set_start_method("spawn")
    queue = mp.Queue()
    processes = [mp.Process(target=worker, args=(device, 0, args.repeats, queue))
                 for device in (0, 1)]
    begin = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    rows = []
    while True:
        try:
            rows.append(queue.get_nowait())
        except queue_module.Empty:
            break
    reported={(row["device"],row["slot"]) for row in rows}
    for index, process in enumerate(processes):
        key=(index,0)
        if key not in reported:
            rows.append(dict(status="FAILED",device=key[0],slot=key[1],repeats=args.repeats,
                             error=f"worker exited without result; exit_code={process.exitcode}"))
    result = dict(status="COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "FAILED",
                  kind="engineering_concurrent_lyapunov_forward_backward_stress",
                  process_layout="one process per RTX 3090", repeats_per_process=args.repeats,
                  wall_seconds=time.time() - begin, rows=sorted(rows, key=lambda row: (row["device"], row["slot"])),
                  exit_codes=[process.exitcode for process in processes])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if result["status"] != "COMPLETE" or any(code != 0 for code in result["exit_codes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
