#!/usr/bin/env python3
"""Group-Event State v0.3.3 -- Workstream B training laboratory CLI.

Sub-commands: validate-request, ingest, t0, smoke, sentinel, controller, worker, status.
Human input views (R0 / R1) run only when V0_3_3_EXECUTION_RELEASE.json exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.v033_training_lab.diagnostics import run_t0  # noqa: E402
from src.topic5_group_event_state.v033_training_lab.objective import TRAINABLE_REGISTRY  # noqa: E402
from src.topic5_group_event_state.v033_training_lab.paths import (  # noqa: E402
    AGENT_B_ROOT, SHARED_ROOT, atomic_write_json, current_commit, release_status, results_index,
)
from src.topic5_group_event_state.v033_training_lab.queue import (  # noqa: E402
    Controller, SearchDriver, Unit, execute_unit, ingest_requests, recipe_from_dict, worker_main,
)
from src.topic5_group_event_state.v033_training_lab.request import validate_request  # noqa: E402
from src.topic5_group_event_state.v033_training_lab.resources import run_sentinel, snapshot  # noqa: E402
from src.topic5_group_event_state.v033_training_lab.trainer import RecipeConfig, train_recipe  # noqa: E402
from src.topic5_group_event_state.v033_training_lab.views import view_for_request  # noqa: E402


def _release() -> bool:
    return bool(release_status()["present"])


def _request_from_args(args: argparse.Namespace) -> dict:
    if args.request:
        return json.loads(Path(args.request).read_text())
    view = {"kind": args.view, "seed": args.seed}
    if args.view == "synthetic":
        view["synthetic"] = {"beta": args.beta, "dispersion_r": 8.0, "generator_seed": 1, "noise_seed": 2}
        if args.subject:
            view["base_subject"] = args.subject
    elif args.view in ("R0", "R1"):
        view["subject"] = args.subject
        view["data_registry_key"] = f"local-dev-{args.subject}"
    return {
        "request_id": args.request_id, "schema_version": "v2", "sealed": False,
        "scientific_target": {"family": "S_N", "predictive_view": "S_N", "objective": "count_profile",
                                                             "bin_convention": "left_closed_right_open_[t+a,t+b)",
                                                             "bins_seconds": [[0, 300], [300, 900], [900, 1800]]},
        "input_view": view, "state_architecture": "fixed_leaky", "split_hash": "auto",
        "baseline_H": {"name": "H_mark", "hash": "0" * 64,
                       "source": "provisional_local" if args.view in ("toy", "synthetic") and not args.subject else "agent2_registry"},
        "endpoint_and_reduction": {"selection_phase": "inner_val", "metric": "nb_nll", "reduction": "mean_per_anchor"},
        "search_budget": {"n_configs": args.n_configs, "max_steps": args.max_steps, "rung_steps": args.rungs or [args.max_steps],
                          "eta": 2, "seeds_low": 1, "seeds_mid": 3, "seeds_final": args.seeds_final, "n_final": 2,
                          "validate_every": 10, "max_batches": args.max_batches, "t0_tiny_steps": args.tiny_steps,
                          "t0_probe_steps": args.probe_steps, "space_restrict": json.loads(args.restrict) if args.restrict else None},
        "seed_policy": {"base_seed": args.seed}, "resource_ceiling": {"max_workers": args.max_workers, "gpu_ids": [],
                                                                       "vram_gib": 0, "ram_gib": 16, "threads": 1},
        "science_code_commit": current_commit(), "input_hash": "auto", "requested_by": "agent_c",
    }


def cmd_validate(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.file).read_text())
    verdict = validate_request(payload, registered_objectives=tuple(TRAINABLE_REGISTRY), release_present=_release(),
                               head_commit=current_commit())
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


def cmd_ingest(args: argparse.Namespace) -> None:
    out = ingest_requests(Path(args.shared_root), Path(args.agent_root), registered=tuple(TRAINABLE_REGISTRY),
                          release_present=_release(), head_commit=current_commit())
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_t0(args: argparse.Namespace) -> None:
    request = _request_from_args(args)
    view, meta = view_for_request(request, release_present=_release())
    trainable = TRAINABLE_REGISTRY["count_profile"]()
    cfg = RecipeConfig().with_overrides(scaling=view.scaling)
    out_dir = Path(args.out) if args.out else Path(args.agent_root) / "t0" / meta.get("subject", args.view)
    report = run_t0(trainable, view, cfg, args.seed, device=torch.device(args.device), out_dir=out_dir,
                    tiny_steps=args.tiny_steps, probe_steps=args.probe_steps)
    atomic_write_json(out_dir / "view_meta.json", {**meta, "view_summary": view.summary()})
    print(json.dumps({"out_dir": str(out_dir), "gradient_path_ok": report["gradient_path_ok"],
                      "tiny_overfit": report["tiny_slice_overfit"]["gap_closed"],
                      "jacobian_pass": report["state_write_jacobian"]["pass"], "amp": report["amp_small_gradient"]["status"],
                      "evidence_label": report["evidence_label"]}, ensure_ascii=False, indent=2))


def cmd_smoke(args: argparse.Namespace) -> None:
    """Whole pipeline (t0 -> search -> card) in-process through the driver's units."""

    request = _request_from_args(args)
    agent_root = Path(args.agent_root)
    shared = Path(args.shared_root)
    (shared / "job_requests").mkdir(parents=True, exist_ok=True)
    atomic_write_json(shared / "job_requests" / f"{request['request_id']}.json", request)
    ingest_requests(shared, agent_root, registered=tuple(TRAINABLE_REGISTRY), release_present=_release(), head_commit=current_commit())
    stored = json.loads((agent_root / "requests" / request["request_id"] / "request.json").read_text())
    driver = SearchDriver(stored, agent_root, shared_root=shared, device_hint=args.device)
    n_units = 0
    while True:
        units = driver.next_units()
        if not units:
            break
        for unit in units:
            code = execute_unit(unit, device=args.device, release_present=_release())
            n_units += 1
            print(f"[smoke] {unit.unit_type} {unit.unit_id} exit={code}", flush=True)
    print(json.dumps({"phase": driver.state["phase"], "n_units": n_units, "stop_reason": driver.state["stop_reason"],
                      "card_path": driver.state["card_path"], "incumbent": driver.status()["incumbent"]}, ensure_ascii=False, indent=2))


def cmd_sentinel(args: argparse.Namespace) -> None:
    request = _request_from_args(args)
    view, meta = view_for_request(request, release_present=_release())
    trainable = TRAINABLE_REGISTRY["count_profile"]()
    recipe = json.loads(Path(args.recipe).read_text()) if args.recipe else RecipeConfig().as_dict()
    cfg = recipe_from_dict(recipe).with_overrides(scaling=view.scaling, max_steps=args.max_steps, min_steps=args.max_steps,
                                                  patience=10 ** 9)
    if args.state_family == "gated_exploratory":
        from dataclasses import replace
        cfg = cfg.with_overrides(arch=replace(cfg.arch, state_family="gated_exploratory"))
    out_dir = Path(args.agent_root) / "sentinels"
    dev = torch.device(args.device)

    def work() -> dict:
        result = train_recipe(trainable, view, cfg, args.seed, device=dev, out_dir=out_dir / f"{args.workload_class}_run", overwrite=True)
        return {"effective_batch": view.n("train"), "n_events": int(view.event_times.size), "steps": result.get("n_steps_run"),
                "elapsed_seconds": result.get("elapsed_seconds"), "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
                "subject": meta.get("subject"), "arch": cfg.arch.as_dict()}

    report = run_sentinel(args.workload_class, work, out_path=out_dir / f"{args.workload_class}.json", device=args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_controller(args: argparse.Namespace) -> None:
    ctl = Controller(Path(args.shared_root), Path(args.agent_root), results_index=results_index(),
                     poll_seconds=args.poll, device_hint=args.device)
    ctl.run(once=args.once, max_idle_cycles=args.max_idle_cycles)


def cmd_worker(args: argparse.Namespace) -> None:
    sys.exit(worker_main(Path(args.unit), device=args.device))


def cmd_status(args: argparse.Namespace) -> None:
    page = Path(args.agent_root) / "agent_b.status.json"
    print(json.dumps({"release": release_status(), "snapshot": snapshot(),
                      "agent_status": json.loads(page.read_text()) if page.exists() else None}, ensure_ascii=False, indent=2, default=str))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-root", default=str(AGENT_B_ROOT))
    p.add_argument("--shared-root", default=str(SHARED_ROOT))
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="command", required=True)

    def view_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--request", default=None, help="request JSON file (overrides the view flags)")
        q.add_argument("--request-id", default="req_local")
        q.add_argument("--view", choices=("toy", "synthetic", "R0", "R1"), default="toy")
        q.add_argument("--subject", default=None)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--beta", type=float, default=0.7)
        q.add_argument("--n-configs", type=int, default=4)
        q.add_argument("--max-steps", type=int, default=60)
        q.add_argument("--rungs", type=int, nargs="*", default=None)
        q.add_argument("--seeds-final", type=int, default=2)
        q.add_argument("--max-batches", type=int, default=1)
        q.add_argument("--tiny-steps", type=int, default=300)
        q.add_argument("--probe-steps", type=int, default=50)
        q.add_argument("--max-workers", type=int, default=2)
        q.add_argument("--restrict", default=None, help="JSON dict of search-space restrictions")
        q.add_argument("--requested-by", default="agent_c")

    q = sub.add_parser("validate-request"); q.add_argument("--file", required=True); q.set_defaults(func=cmd_validate)
    q = sub.add_parser("ingest"); q.set_defaults(func=cmd_ingest)
    q = sub.add_parser("t0"); view_args(q); q.add_argument("--out", default=None); q.set_defaults(func=cmd_t0)
    q = sub.add_parser("smoke"); view_args(q); q.set_defaults(func=cmd_smoke)
    q = sub.add_parser("sentinel"); view_args(q)
    q.add_argument("--workload-class", required=True)
    q.add_argument("--recipe", default=None)
    q.add_argument("--state-family", choices=("fixed_leaky", "gated_exploratory"), default="fixed_leaky")
    q.set_defaults(func=cmd_sentinel)
    q = sub.add_parser("controller"); q.add_argument("--once", action="store_true"); q.add_argument("--poll", type=float, default=30.0)
    q.add_argument("--max-idle-cycles", type=int, default=None); q.set_defaults(func=cmd_controller)
    q = sub.add_parser("worker"); q.add_argument("--unit", required=True); q.set_defaults(func=cmd_worker)
    q = sub.add_parser("status"); q.set_defaults(func=cmd_status)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
