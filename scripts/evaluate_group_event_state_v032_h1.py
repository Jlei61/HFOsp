#!/usr/bin/env python3
"""H1 paired evaluation for Group-Event State v0.3.2.

Without the frozen state registry it scores the control arms (H, controls,
intercept-only).  With the registry it adds H+S_correct / shifted / mean for every
complete (patient, seed).  Every run writes per-patient arrays + JSON under
``/data/hfosp_group_event_state_v0_3_2/evaluation/h1/<subject>/``.
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v032_eval.contract import (  # noqa: E402
    EvalPaths, atomic_json, load_eval_config, now_iso, read_json, sha256_file, source_commit,
)
from src.topic5_group_event_state.v032_eval.h1_eval import build_h1_design, evaluate_h1_patient  # noqa: E402
from src.topic5_group_event_state.v032_eval.state_registry import (  # noqa: E402
    complete_seed_entries, load_registry, load_state_bundle,
)
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline  # noqa: E402


def _worker(args: tuple[str, str, str, str | None, bool]) -> dict[str, Any]:
    subject, config_path, out_root, registry_path, force = args
    started = time.time()
    try:
        cfg = load_eval_config(Path(config_path))
        out_dir = Path(out_root) / subject
        tl = load_eval_timeline(subject, cfg)
        design = build_h1_design(tl, cfg)
        labels: list[str] = []
        control_json = out_dir / "h1_result_controls.json"
        if force or not control_json.exists():
            evaluate_h1_patient(tl, cfg, design, state=None, out_dir=out_dir, label="controls")
        labels.append("controls")
        state_labels: list[str] = []
        if registry_path and Path(registry_path).exists():
            registry = load_registry(Path(registry_path))
            for seed, spec in complete_seed_entries(registry, subject).items():
                label = f"seed_{seed}"
                marker = out_dir / f"h1_result_{label}.json"
                arrays_sha = sha256_file(Path(spec["arrays_path"])) if Path(spec["arrays_path"]).exists() else None
                if marker.exists() and not force:
                    previous = read_json(marker)
                    if previous.get("state", {}).get("provenance", {}).get("arrays_sha256") == arrays_sha:
                        state_labels.append(label)
                        continue
                bundle = load_state_bundle(
                    spec, subject=subject, seed=seed,
                    grid_times=tl.grid.t_anchor, grid_segment=tl.grid.segment_index,
                    event_times=tl.event_times, event_segment=tl.event_segment,
                )
                bundle.provenance["arrays_sha256"] = arrays_sha
                evaluate_h1_patient(tl, cfg, design, state=bundle, out_dir=out_dir, label=label)
                state_labels.append(label)
        return {"subject": subject, "status": "ok", "labels": labels + state_labels,
                "seconds": time.time() - started}
    except Exception as exc:
        return {"subject": subject, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--registry", type=Path, default=None,
                        help="frozen_state_registry.json (default: shared dir; skipped when absent)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    paths = EvalPaths.from_config(cfg)
    paths.ensure()
    out_root = paths.evaluation / "h1"
    out_root.mkdir(parents=True, exist_ok=True)
    registry = args.registry or (paths.shared / "frozen_state_registry.json")
    registry_path = str(registry) if Path(registry).exists() else None
    subjects = args.subjects or sorted(p.name for p in Path(cfg["dataset_root"]).iterdir() if (p / "index.json").exists())
    status_path = out_root / "STATUS.json"
    results: dict[str, dict[str, Any]] = {}
    atomic_json(status_path, {"stage": "h1", "status": "running", "started": now_iso(), "subjects": subjects,
                              "registry": registry_path})
    jobs = [(s, str(args.config), str(out_root), registry_path, bool(args.force)) for s in subjects]
    with get_context("spawn").Pool(processes=max(1, min(args.workers, len(jobs)))) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            results[res["subject"]] = res
            print(f"[{now_iso()}] {res['subject']}: {res['status']} {res.get('labels', '')} ({res['seconds']:.1f}s)", flush=True)
            if res["status"] != "ok":
                print(res["traceback"], flush=True)
            atomic_json(status_path, {"stage": "h1", "status": "running", "updated": now_iso(), "registry": registry_path,
                                      "completed": sorted(s for s, r in results.items() if r["status"] == "ok"),
                                      "failed": sorted(s for s, r in results.items() if r["status"] != "ok")})
    failed = sorted(s for s, r in results.items() if r["status"] != "ok")
    atomic_json(status_path, {"stage": "h1", "status": "complete" if not failed else "complete_with_failures",
                              "finished": now_iso(), "registry": registry_path, "source_commit": source_commit(),
                              "labels": {s: r.get("labels") for s, r in results.items()}, "failed": failed})


if __name__ == "__main__":
    main()
