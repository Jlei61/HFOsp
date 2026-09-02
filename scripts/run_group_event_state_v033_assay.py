#!/usr/bin/env python3
"""Run resumable v0.3.3 synthetic oracle/power assays on a real patient time axis.

This script never trains on human targets. It keeps the real anchor, coverage,
split and event-time scaffold, generates synthetic outcomes, and writes each
replicate atomically before producing power summaries.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v032_eval.contract import atomic_json, load_eval_config  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import canonical as C  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import power as P  # noqa: E402
from src.topic5_group_event_state.v033_evaluator.scaffold import load_real_scaffold  # noqa: E402


def _cells(preset: str, replicates: int) -> list[P.ReplicateSpec]:
    if preset == "sentinel":
        definitions = (("D0", 0.0, 0.0), ("D3", 0.7, 2.5))
        n_rep = 1
    elif preset == "smoke":
        definitions = (("D0", 0.0, 0.0), ("D1", 0.7, 0.0), ("D2", 0.0, 2.5),
                       ("D3", 0.7, 2.5), ("D4", 0.7, 2.5))
        n_rep = int(replicates)
    elif preset == "power":
        definitions = (
            ("D0", 0.0, 0.0),
            ("D1", 0.15, 0.0), ("D1", 0.35, 0.0), ("D1", 0.7, 0.0),
            ("D2", 0.0, 0.8), ("D2", 0.0, 1.6), ("D2", 0.0, 2.5),
            ("D3", 0.15, 0.8), ("D3", 0.35, 1.6), ("D3", 0.7, 2.5),
            ("D4", 0.15, 0.8), ("D4", 0.35, 1.6), ("D4", 0.7, 2.5),
        )
        n_rep = int(replicates)
    else:  # pragma: no cover - argparse controls this
        raise ValueError(preset)
    return [P.ReplicateSpec(kind=k, beta_count=bc, beta_grammar=bg, replicate=r)
            for k, bc, bg in definitions for r in range(n_rep)]


def _key(subject: str, spec: P.ReplicateSpec) -> str:
    return (f"{subject}_{spec.kind}_bc{spec.beta_count:g}_bg{spec.beta_grammar:g}"
            f"_rep{spec.replicate:03d}").replace(".", "p")


def _one(job: tuple[str, str, dict, float, tuple[str, ...], tuple[int, ...], int, dict]) -> dict:
    subject, config_path, spec_dict, horizon, views, levels, n_steps, contract = job
    cfg = load_eval_config(Path(config_path))
    scaffold = load_real_scaffold(subject, cfg, carry="session")
    spec = P.ReplicateSpec(**{k: spec_dict[k] for k in
                              ("kind", "beta_count", "beta_grammar", "replicate",
                               "generator_seed", "noise_seed", "estimator_seed")})
    result = P.run_replicate(scaffold, spec, horizon=horizon, views=views, levels=levels, n_steps=n_steps)
    result["run_contract"] = contract
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    ap.add_argument("--subjects", nargs="+", default=["epilepsiae_1146"])
    ap.add_argument("--preset", choices=("sentinel", "smoke", "power"), default="sentinel")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--views", nargs="+", choices=("count_profile", "grammar"),
                    default=["count_profile", "grammar"])
    ap.add_argument("--levels", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--output-root", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_3/agent_a/assay"))
    ap.add_argument("--summary", type=Path,
                    default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/d0_d4_power_curve.json")
    args = ap.parse_args()
    if args.replicates < 1 or args.workers < 1:
        ap.error("replicates and workers must be positive")

    cfg = load_eval_config(args.config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    specs = _cells(args.preset, args.replicates)
    run_contract = {
        "source_commit": commit, "config_sha256": cfg["_config_sha256"],
        "canonical_schema_version": C.SCHEMA_VERSION, "carry": "session",
        "horizon_seconds": float(args.horizon), "views": list(args.views),
        "levels": list(args.levels), "n_steps": int(args.n_steps),
    }
    replicate_dir = args.output_root / "replicates"
    replicate_dir.mkdir(parents=True, exist_ok=True)
    jobs, results = [], []
    for subject in args.subjects:
        for spec in specs:
            path = replicate_dir / f"{_key(subject, spec)}.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if existing.get("run_contract") == run_contract:
                    results.append(existing)
                    continue
            jobs.append((subject, str(args.config), spec.resolved(), float(args.horizon),
                         tuple(args.views), tuple(args.levels), int(args.n_steps), run_contract))

    max_workers = min(int(args.workers), max(1, len(jobs)))
    if jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {pool.submit(_one, job): job for job in jobs}
            for i, future in enumerate(as_completed(future_to_job), start=1):
                result = future.result()
                spec = P.ReplicateSpec(**{k: result["spec"][k] for k in
                                          ("kind", "beta_count", "beta_grammar", "replicate")})
                path = replicate_dir / f"{_key(result['subject'], spec)}.json"
                atomic_json(path, result)
                results.append(result)
                print(f"[{i}/{len(jobs)}] {path.name}", flush=True)

    curves = []
    for subject in args.subjects:
        selected = [r for r in results if r["subject"] == subject]
        for view in args.views:
            curve = P.power_curve(selected, view=view)
            beta_key = "beta_count" if view == "count_profile" else "beta_grammar"
            curve.update({"subject": subject, "horizon_seconds": float(args.horizon),
                          "effect_tiers": P.assign_effect_tiers(curve["cells"], beta_key=beta_key)})
            curves.append(curve)
    payload = {
        "format": "group_event_state_v0_3_3_synthetic_oracle_power_curve",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": commit,
        "config_sha256": cfg["_config_sha256"],
        "preset": args.preset,
        "cadence": {"requested_replicates": int(args.replicates), "n_completed": len(results)},
        "views": list(args.views), "levels": list(args.levels), "horizon_seconds": float(args.horizon),
        "curves": curves,
        "evidence_label": "DIAGNOSTIC_SYNTHETIC_ASSAY_ONLY",
        "human_targets_used": False,
        "sealed_partition_opened": False,
    }
    atomic_json(args.summary, payload)
    atomic_json(args.output_root / f"{args.preset}_summary.json", payload)
    print(json.dumps({"status": "complete", "n_replicates": len(results), "summary": str(args.summary)}, indent=2))


if __name__ == "__main__":
    main()
