#!/usr/bin/env python3
"""Fit the shared explicit-history baseline H (H_rate / H_strong) and publish the registry."""
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
    EvalPaths, atomic_json, load_eval_config, now_iso, source_commit,
)
from src.topic5_group_event_state.v032_eval.h1_eval import build_h1_design  # noqa: E402
from src.topic5_group_event_state.v032_eval.history_registry import REGISTRY_FORMAT, fit_history_for_patient  # noqa: E402
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline  # noqa: E402


def _worker(args: tuple[str, str, str]) -> dict[str, Any]:
    subject, config_path, out_dir = args
    started = time.time()
    try:
        cfg = load_eval_config(Path(config_path))
        tl = load_eval_timeline(subject, cfg)
        design = build_h1_design(tl, cfg)
        entry = fit_history_for_patient(tl, cfg, design, Path(out_dir) / subject)
        return {"subject": subject, "status": "ok", "entry": entry, "seconds": time.time() - started}
    except Exception as exc:
        return {"subject": subject, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    paths = EvalPaths.from_config(cfg)
    paths.ensure()
    out_dir = paths.evaluation / "history_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or sorted(p.name for p in Path(cfg["dataset_root"]).iterdir() if (p / "index.json").exists())
    status_path = out_dir / "STATUS.json"
    results: dict[str, dict[str, Any]] = {}
    atomic_json(status_path, {"stage": "history_baseline", "status": "running", "started": now_iso(), "subjects": subjects})
    with get_context("spawn").Pool(processes=max(1, min(args.workers, len(subjects)))) as pool:
        for res in pool.imap_unordered(_worker, [(s, str(args.config), str(out_dir)) for s in subjects]):
            results[res["subject"]] = res
            print(f"[{now_iso()}] {res['subject']}: {res['status']} ({res['seconds']:.1f}s)", flush=True)
            if res["status"] != "ok":
                print(res["traceback"], flush=True)
            atomic_json(status_path, {"stage": "history_baseline", "status": "running", "updated": now_iso(),
                                      "completed": sorted(s for s, r in results.items() if r["status"] == "ok"),
                                      "failed": sorted(s for s, r in results.items() if r["status"] != "ok")})
    ok = {s: r["entry"] for s, r in results.items() if r["status"] == "ok"}
    failed = {s: {"error": r["error"], "traceback": r["traceback"]} for s, r in results.items() if r["status"] != "ok"}
    registry = {
        "format": REGISTRY_FORMAT,
        "generated": now_iso(),
        "source_commit": source_commit(),
        "config_sha256": cfg["_config_sha256"],
        "primary_variant": cfg["primary_history"],
        "history_variants": list(cfg["history_variants"]),
        "nb_family": "NB2, Var = mu + alpha mu^2; nb_log_dispersion = log(1/alpha) = log r",
        "test_time_fit": False,
        "dev_test_fitted": False,
        "partition": cfg["partition"],
        "subjects": ok,
        "patients": ok,
        "failed": failed,
    }
    atomic_json(paths.shared / "history_baseline_registry.json", registry)
    atomic_json(out_dir / "history_baseline_registry.json", registry)
    atomic_json(status_path, {"stage": "history_baseline", "status": "complete" if not failed else "complete_with_failures",
                              "finished": now_iso(), "n_ok": len(ok), "failed": sorted(failed)})
    summary = {s: {h: {"ridge": v.get("selected_ridge"), "edge": v.get("ridge_at_edge"), "alpha": v.get("nb_alpha"),
                       "sel": v.get("selection"),
                       "dev_test_nll": (v.get("scores") or {}).get("dev_test", {}).get("mean_nb_nll")}
                   for h, v in e["horizons"].items()} for s, e in ok.items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
