#!/usr/bin/env python
"""Fail-closed dependency check for Topic 5 V2 Phase 2.

The check is intentionally read-only. It reports the Phase-1 foundation needed
before real Phase-2 scripts can run and exits non-zero if any required item is
missing or unusable.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PHASE1_ROOT = _ROOT / "results" / "topic5_ictal_recruitment" / "v2_band_scan"
_DEFAULT_OUT = _ROOT / "results" / "topic5_ictal_recruitment" / "v2_criticality"


def _item(present: bool, path: str | None, status: str, detail: str = "") -> dict:
    return {"present": bool(present), "path": path, "status": status, "detail": detail}


def _file_item(path: Path) -> dict:
    present = path.is_file()
    return _item(present, str(path), "ok" if present else "missing")


def _module_item(module_name: str, required_attrs: list[str]) -> dict:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - message depends on environment
        return _item(False, module_name, "error", f"{type(exc).__name__}: {exc}")
    missing = [name for name in required_attrs if not hasattr(mod, name)]
    if missing:
        return _item(False, module_name, "missing", "missing attrs: " + ", ".join(missing))
    return _item(True, module_name, "ok", "attrs: " + ", ".join(required_attrs))


def _cache_item(substrate: str) -> dict:
    candidates = [
        _PHASE1_ROOT / "cache",
        _PHASE1_ROOT / substrate / "cache",
        _PHASE1_ROOT / f"{substrate}_cache",
    ]
    for path in candidates:
        if path.is_dir():
            n_npz = sum(1 for _ in path.glob("*.npz"))
            if n_npz > 0:
                return _item(True, str(path), "ok", f"{n_npz} npz files")
            return _item(False, str(path), "missing", "cache directory exists but has no npz files")
    return _item(False, str(candidates[0]), "missing", "no candidate cache directory found")


def _manifest_item() -> dict:
    csv_path = _PHASE1_ROOT / "phase1_cohort_manifest.csv"
    json_path = _PHASE1_ROOT / "phase1_cohort_manifest.json"
    missing = [str(p) for p in (csv_path, json_path) if not p.is_file()]
    path = f"{csv_path};{json_path}"
    if missing:
        return _item(False, path, "missing", "missing: " + ", ".join(missing))
    return _item(True, path, "ok")


def _order_null_depcheck_item(substrate: str) -> dict:
    candidates = [
        _PHASE1_ROOT / substrate / "phase1_order_null_depcheck.csv",
        _PHASE1_ROOT / "phase1_order_null_depcheck.csv",
        _PHASE1_ROOT / f"phase1_order_null_depcheck_{substrate}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return _item(True, str(path), "ok")
    return _item(False, str(candidates[0]), "missing", "no candidate order-null depcheck CSV found")


def build_report(substrate: str, outdir: Path) -> dict:
    deps = {
        "phase2_config": _file_item(_ROOT / "config" / "topic5_v2_phase2.yaml"),
        "phase1_config": _file_item(_ROOT / "config" / "topic5_v2_phase1.yaml"),
        "phase1_band_scan_source": _file_item(_ROOT / "src" / "topic5_v2_band_scan.py"),
        "phase1_band_scan_import": _module_item(
            "src.topic5_v2_band_scan",
            [
                "load_phase1_config",
                "contact_alignment",
                "spatial_constrained_permute",
                "rebuild_typical_rank",
                "order_null_rank_pair",
            ],
        ),
        "phase1_band_cache": _cache_item(substrate),
        "phase1_cohort_manifest": _manifest_item(),
        "phase1_order_null_depcheck_csv": _order_null_depcheck_item(substrate),
        "ictal_field_load_context": _module_item(
            "scripts.run_topic5_ictal_field_dynamics",
            ["load_context"],
        ),
    }
    ok = all(v["status"] == "ok" and v["present"] for v in deps.values())
    return {
        "schema_version": "topic5_v2_phase2_dependency_report_v1",
        "status": "ok" if ok else "blocked",
        "fail_closed": True,
        "substrate": substrate,
        "outdir": str(outdir),
        "dependencies": deps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", choices=["broad", "narrow"], default="broad")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args(argv)

    outdir = args.outdir or (_DEFAULT_OUT / args.substrate)
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.substrate, outdir)
    report_path = outdir / "phase2_dependency_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
