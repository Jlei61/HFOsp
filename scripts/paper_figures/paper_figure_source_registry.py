#!/usr/bin/env python3
"""Resolve and validate the active paper-figure data contract.

The registry is intentionally tracked while numerical artifacts under
``results/`` are not.  A missing active artifact is therefore an error; this
module never falls back to an older figure source just because it exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "config/paper_figure_source_registry.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "paper_figure_source_registry_v1":
        raise ValueError(f"unsupported paper-figure registry: {path}")
    if payload.get("policy", {}).get("fallback_to_legacy_sources") is not False:
        raise ValueError("paper-figure registry must forbid implicit legacy fallback")
    active_id = payload.get("active_contract")
    contract = payload.get("contracts", {}).get(active_id)
    if not isinstance(contract, dict) or contract.get("status") != "active":
        raise ValueError(f"active paper-figure contract is invalid: {active_id!r}")
    return payload


def active_contract(path: Path = REGISTRY_PATH) -> tuple[str, dict[str, Any]]:
    payload = load_registry(path)
    contract_id = str(payload["active_contract"])
    return contract_id, payload["contracts"][contract_id]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def panel_source(figure: str, panel: str) -> dict[str, Any]:
    _, contract = active_contract()
    try:
        source = contract[figure][panel]
    except KeyError as exc:
        raise KeyError(f"unregistered paper panel: {figure}.{panel}") from exc
    if not isinstance(source, dict):
        raise TypeError(f"invalid source record for {figure}.{panel}")
    return source


def registered_path(figure: str, panel: str, key: str) -> Path:
    source = panel_source(figure, panel)
    try:
        value = source[key]
    except KeyError as exc:
        raise KeyError(f"unregistered path: {figure}.{panel}.{key}") from exc
    if not isinstance(value, str):
        raise TypeError(f"registered value is not a path: {figure}.{panel}.{key}")
    return resolve_repo_path(value)


def _registered_files(
    contract: dict[str, Any], figures: tuple[str, ...],
) -> Iterator[tuple[str, Path, str | None]]:
    for figure in figures:
        figure_record = contract[figure]
        for panel in figure_record["updated_panels"]:
            record = figure_record[panel]
            producer = resolve_repo_path(record["producer"])
            yield f"{figure}.{panel}.producer", producer, None
            if "analysis_producer" in record:
                yield (
                    f"{figure}.{panel}.analysis_producer",
                    resolve_repo_path(record["analysis_producer"]),
                    None,
                )
            for name, item in record.get("required_inputs", {}).items():
                yield f"{figure}.{panel}.{name}", resolve_repo_path(item["path"]), item["sha256"]
            if "source_pdf" in record:
                yield (
                    f"{figure}.{panel}.source_pdf",
                    resolve_repo_path(record["source_pdf"]),
                    record.get("source_sha256"),
                )


def validate_active_sources(
    *, check_hashes: bool = True, figures: tuple[str, ...] = ("fig2", "fig3"),
) -> dict[str, Any]:
    contract_id, contract = active_contract()
    unknown = sorted(set(figures) - {"fig2", "fig3"})
    if unknown:
        raise ValueError(f"unknown paper figures: {unknown}")
    checked = []
    missing = []
    hash_mismatch = []
    for name, path, expected_hash in _registered_files(contract, figures):
        if not path.exists():
            missing.append({"name": name, "path": str(path)})
            continue
        observed_hash = None
        if check_hashes and expected_hash is not None:
            observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed_hash != expected_hash:
                hash_mismatch.append({
                    "name": name,
                    "path": str(path),
                    "expected": expected_hash,
                    "observed": observed_hash,
                })
        checked.append({"name": name, "path": str(path), "sha256": observed_hash})
    result = {
        "contract_id": contract_id,
        "hard_event_qc_used": contract["hard_event_qc_used"],
        "fallback_to_legacy_sources": False,
        "figures": list(figures),
        "checked": checked,
        "missing": missing,
        "hash_mismatch": hash_mismatch,
        "ok": not missing and not hash_mismatch,
    }
    if not result["ok"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hash", action="store_true", help="Only check path existence")
    parser.add_argument("--figure", choices=("2", "3", "all"), default="all")
    args = parser.parse_args()
    figures = ("fig2", "fig3") if args.figure == "all" else (f"fig{args.figure}",)
    print(json.dumps(
        validate_active_sources(check_hashes=not args.no_hash, figures=figures),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
