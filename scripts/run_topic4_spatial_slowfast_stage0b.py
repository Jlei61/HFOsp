#!/usr/bin/env python3
"""Run the cheap homogeneous Stage-0B E/I fast-topology screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
from pathlib import Path

# The screen is intentionally single-process.  Set BLAS limits before importing numpy/scipy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_spatial_slowfast_stage0b import (  # noqa: E402
    ForkClassifierThresholds,
    build_state_forks,
    classify_fork_batch,
    continuation_root_scan,
    exact_siegert_root_audit,
    summarize_exact_siegert_audit,
    root_boundary_summary,
    select_confirm_candidates,
    simulate_forks,
    summarize_stage0b,
)


DEFAULT_CONFIG = ROOT / "config" / "topic4_spatial_slowfast_stage0b.yaml"


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, payload) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)
        stream.write("\n")
    temp.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _classifier(cfg: dict) -> ForkClassifierThresholds:
    return ForkClassifierThresholds(**cfg["classifier"]).validate()


def run(config_path: Path) -> tuple[dict, Path]:
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    axes = cfg["axes"]
    w_values = [float(x) for x in axes["w_ee_mult"]]
    q_values = [float(x) for x in axes["q"]]
    ratio = float(axes["ratio"])
    if w_values != [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        raise ValueError("Stage0B w_ee_mult axis drifted from the locked 1.0:0.1:1.5 grid")
    if not np.allclose(q_values, np.round(np.arange(1.0, 0.79, -0.01), 2)):
        raise ValueError("Stage0B q axis drifted from the locked 1.00:-0.01:0.80 grid")
    if ratio != 1.0 or any(bool(cfg["scope"][key]) for key in ("noise", "slow_variables", "spatial_coupling", "dynamic_phi")):
        raise ValueError("Stage0B must remain ratio=1 with noise/slow/spatial/phi disabled")

    root_rows = continuation_root_scan(w_values, q_values, ratio=ratio)
    exact_root_audit = exact_siegert_root_audit(root_rows)
    metadata, initial_states, params = build_state_forks(root_rows)
    screen_cfg = cfg["screen"]
    screen_sim = simulate_forks(
        initial_states,
        params,
        dt_ms=float(screen_cfg["dt_ms"]),
        duration_ms=float(screen_cfg["duration_ms"]),
        save_stride=int(screen_cfg["save_stride"]),
    )
    thresholds = _classifier(cfg)
    screen_rows = classify_fork_batch(metadata, screen_sim, thresholds)

    candidate_indices = select_confirm_candidates(screen_rows)
    confirm_rows: list[dict] = []
    if candidate_indices:
        confirm_cfg = cfg["confirm"]
        confirm_sim = simulate_forks(
            initial_states[candidate_indices],
            [params[index] for index in candidate_indices],
            dt_ms=float(confirm_cfg["dt_ms"]),
            duration_ms=float(confirm_cfg["duration_ms"]),
            save_stride=int(confirm_cfg["save_stride"]),
        )
        confirm_classified = classify_fork_batch(
            [metadata[index] for index in candidate_indices], confirm_sim, thresholds
        )
        for local_index, screen_index in enumerate(candidate_indices):
            confirm_rows.append(
                {
                    **confirm_classified[local_index],
                    "screen_classification": screen_rows[screen_index]["classification"],
                    "confirm_dt_ms": float(confirm_cfg["dt_ms"]),
                    "confirm_duration_ms": float(confirm_cfg["duration_ms"]),
                }
            )

    summary = summarize_stage0b(root_rows, screen_rows, confirm_rows)
    exact_summary = summarize_exact_siegert_audit(exact_root_audit)
    summary["exact_siegert_root_audit"] = exact_summary
    if not exact_summary["supports_lut_no_go"]:
        summary["verdict"] = "INCONCLUSIVE_EXACT_SIEGERT_ROOT_AUDIT"
        summary["stage0b_pass"] = False
        summary["stage1_to_3_open"] = False
        summary["stop_rule_triggered"] = False
        summary["reason_cn"] = (
            "exact Siegert局部root复核未完整保持LUT拓扑；需先解决root审计失败再判Stage0B。"
        )
    max_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    summary.update(
        {
            "schema_version": "topic4_spatial_slowfast_stage0b.v1",
            "config": str(config_path.resolve()),
            "axes": axes,
            "screen_contract": screen_cfg,
            "confirm_contract": cfg["confirm"],
            "n_screen_forks": len(screen_rows),
            "n_confirm_forks": len(confirm_rows),
            "root_boundaries": root_boundary_summary(root_rows),
            "resource_usage": {
                "max_rss_gib": max_rss_gib,
                "max_memory_gib_contract": float(cfg["resource_contract"]["max_memory_gib"]),
                "within_memory_contract": max_rss_gib < float(cfg["resource_contract"]["max_memory_gib"]),
                "execution": "single_process_blas_threads_1",
            },
            "scientific_boundary_cn": (
                "Stage0B只检验冻结、均匀、无噪声E/I快子系统是否存在有限高态对象；"
                "它不证明发作、自发转换、恢复或空间传播。"
            ),
        }
    )
    # Memory overflow is a hard, fail-closed engineering failure.
    if not summary["resource_usage"]["within_memory_contract"]:
        summary["stage0b_pass"] = False
        summary["stage1_to_3_open"] = False
        summary["verdict"] = "ENGINEERING_FAIL_MEMORY_CONTRACT"
        summary["stop_rule_triggered"] = False
        summary["reason_cn"] = "峰值内存超过4 GiB合同，结果不可验收。"

    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "stage0b_summary.json", summary)
    _atomic_json(output / "root_continuation.json", root_rows)
    _atomic_json(output / "exact_siegert_root_audit.json", exact_root_audit)
    _atomic_json(output / "state_fork_screen.json", screen_rows)
    _atomic_json(output / "state_fork_confirm.json", confirm_rows)
    flat_roots = [
        {"w_ee_mult": point["w_ee_mult"], "q": point["q"], **root}
        for point in root_rows
        for root in point["roots"]
    ]
    _write_csv(output / "root_table.csv", flat_roots)
    _write_csv(output / "exact_siegert_root_audit.csv", exact_root_audit)
    _write_csv(output / "state_fork_screen.csv", screen_rows)
    _write_csv(output / "state_fork_confirm.csv", confirm_rows)
    return summary, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to execute the locked Stage0B screen")
    summary, output = run(args.config)
    print(json.dumps({"output": str(output), "verdict": summary["verdict"], "stage0b_pass": summary["stage0b_pass"], "max_rss_gib": summary["resource_usage"]["max_rss_gib"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
