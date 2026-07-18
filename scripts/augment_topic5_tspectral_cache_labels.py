#!/usr/bin/env python3
"""Attach versioned per-seizure spectral labels to T_spectral ictal caches.

This is a metadata-only augmentation.  It updates the JSON sidecars and the
existing cache summary, but never rewrites NPZ arrays or changes ``seizure_idxs``.
Only seizures already present in a T_spectral-aligned cache receive an event
record.  The frequency label is read from the committed classifier table even
when that table predates the event's later time-only T_spectral acceptance;
``has_accepted_tspectral`` is provenance, not a second classification gate.
Cache events absent from the 1--150 Hz classifier remain explicit as
``not_classified`` rather than being folded into ``other``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHENOTYPE_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/early_spectral_phenotype"
)
PHENOTYPE_CSV = PHENOTYPE_ROOT / "per_seizure_spectral_overlap_state.csv"
PHENOTYPE_EXCLUSIONS_CSV = PHENOTYPE_ROOT / "classification_exclusions.csv"
CACHE_ROOTS = (
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_common_1_80hz",
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_yuquan_common_1_80hz",
)
SIDECAR_SCHEMA_VERSION = "topic5_tspectral_cache_sidecar_v1p3"
LABEL_BLOCK = "early_spectral_phenotype"
LABEL_BANDS = (
    "delta_HYP_slow",
    "theta_preictal_PAC",
    "alpha_sharp_leq13",
    "beta_LVFA_low",
    "gamma_LVFA",
    "hg_low_ripple",
)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _optional_int(value: Any) -> int | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return int(out) if np.isfinite(out) else None


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    out = str(value).strip()
    return out or None


def build_label_payload(row: pd.Series) -> dict[str, Any]:
    """Convert one classification row to the stable cache-sidecar contract."""
    return {
        "label_status": "classified",
        "label_version": _text(row.get("analysis_version")),
        "source_has_accepted_tspectral": _truth(
            row.get("has_accepted_tspectral")
        ),
        "phenotype": _text(row.get("phenotype")),
        "phenotype_label": _text(row.get("phenotype_label")),
        "simple_phenotype": _text(row.get("simple_phenotype")),
        "simple_phenotype_label": _text(row.get("simple_phenotype_label")),
        "detection_gate_category": _text(row.get("detection_gate_category")),
        "classification_reason": _text(row.get("classification_reason")),
        "anchor_rel_eeg_sec": _optional_float(row.get("anchor_rel_eeg_sec")),
        "anchor_source": _text(row.get("anchor_source")),
        "accepted_tspectral_in_early_window": _truth(
            row.get("accepted_tspectral_in_early_window")
        ),
        "n_analysis_contacts": _optional_int(row.get("n_analysis_contacts")),
        "n_low_band_hits": _optional_int(row.get("n_low_band_hits")),
        "n_fast_band_hits": _optional_int(row.get("n_fast_band_hits")),
        "n_total_band_hits": _optional_int(row.get("n_total_band_hits")),
        "flags": {
            "strict_broadband_5of6": _truth(row.get("strict_broadband_5of6")),
            "gamma_band_30_80_support": _truth(
                row.get("gamma_band_30_80_support")
            ),
            "low_frequency_1_13_support": _truth(
                row.get("low_frequency_1_13_support")
            ),
        },
        "band_hits": {
            band: _truth(row.get(f"{band}__hit")) for band in LABEL_BANDS
        },
    }


def load_labels(
    phenotype_csv: Path,
    exclusions_csv: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], str], str]:
    events = pd.read_csv(phenotype_csv)
    required = {
        "analysis_version",
        "subject",
        "seizure_idx",
        "has_accepted_tspectral",
        "phenotype",
        "simple_phenotype",
        "strict_broadband_5of6",
        "gamma_band_30_80_support",
        "low_frequency_1_13_support",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"phenotype table missing columns: {missing}")
    if events.duplicated(["subject", "seizure_idx"]).any():
        dup = events.loc[
            events.duplicated(["subject", "seizure_idx"], keep=False),
            ["subject", "seizure_idx"],
        ]
        raise ValueError(f"duplicate phenotype keys: {dup.to_dict('records')}")
    versions = sorted(events["analysis_version"].dropna().astype(str).unique())
    if len(versions) != 1:
        raise ValueError(f"expected one phenotype version, found {versions}")

    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in events.iterrows():
        key = (str(row["subject"]), int(row["seizure_idx"]))
        labels[key] = build_label_payload(row)

    exclusions: dict[tuple[str, int], str] = {}
    if exclusions_csv.exists() and exclusions_csv.stat().st_size:
        excluded = pd.read_csv(exclusions_csv)
        for _, row in excluded.iterrows():
            idx = _optional_int(row.get("seizure_idx"))
            subject = _text(row.get("subject"))
            if subject is not None and idx is not None:
                exclusions[(subject, idx)] = str(row.get("reason", "not_classified"))
    return labels, exclusions, versions[0]


def _selectors_template() -> dict[str, Any]:
    return {
        "accepted_tspectral_labeled_idxs": [],
        "accepted_tspectral_strict_broadband_idxs": [],
        "accepted_tspectral_gamma_support_idxs": [],
        "accepted_tspectral_low_frequency_support_idxs": [],
        "accepted_tspectral_simple_phenotype_idxs": {},
    }


def augment_cache_root(
    cache_root: Path,
    *,
    labels: dict[tuple[str, int], dict[str, Any]],
    exclusions: dict[tuple[str, int], str],
    label_version: str,
    phenotype_csv: Path,
    exclusions_csv: Path,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    phenotype_counts: Counter[str] = Counter()
    selector_counts: Counter[str] = Counter()
    n_subjects = 0
    n_events = 0

    for path in sorted(cache_root.glob("*.json")):
        if path.name.startswith("cache_"):
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        subject = str(meta["subject"])
        seizure_idxs = [int(value) for value in meta.get("seizure_idxs", [])]
        per_event = meta.get("seizure", {})
        selectors = _selectors_template()

        for idx in seizure_idxs:
            event = per_event.get(str(idx))
            if event is None:
                raise ValueError(f"{path}: seizure {idx} missing per-event metadata")
            key = (subject, idx)
            payload = labels.get(key)
            if payload is None:
                payload = {
                    "label_status": "not_classified",
                    "label_version": label_version,
                    "reason": exclusions.get(
                        key, "not_in_committed_1_150hz_phenotype_denominator"
                    ),
                }
            else:
                payload = dict(payload)
                payload["current_t_spectral_rel_eeg_sec"] = _optional_float(
                    event.get("t_spectral_rel_eeg_sec")
                )
                payload["current_t_spectral_status"] = _text(
                    event.get("alignment_status")
                )
                selectors["accepted_tspectral_labeled_idxs"].append(idx)
                selector_counts["accepted_tspectral_labeled_idxs"] += 1
                flags = payload["flags"]
                if flags["strict_broadband_5of6"]:
                    selectors["accepted_tspectral_strict_broadband_idxs"].append(idx)
                    selector_counts[
                        "accepted_tspectral_strict_broadband_idxs"
                    ] += 1
                if flags["gamma_band_30_80_support"]:
                    selectors["accepted_tspectral_gamma_support_idxs"].append(idx)
                    selector_counts[
                        "accepted_tspectral_gamma_support_idxs"
                    ] += 1
                if flags["low_frequency_1_13_support"]:
                    selectors[
                        "accepted_tspectral_low_frequency_support_idxs"
                    ].append(idx)
                    selector_counts[
                        "accepted_tspectral_low_frequency_support_idxs"
                    ] += 1
                simple = str(payload["simple_phenotype"])
                selectors["accepted_tspectral_simple_phenotype_idxs"].setdefault(
                    simple, []
                ).append(idx)
                phenotype_counts[simple] += 1
            event[LABEL_BLOCK] = payload
            status_counts[str(payload["label_status"])] += 1
            n_events += 1

        for key in selectors:
            if key == "accepted_tspectral_simple_phenotype_idxs":
                selectors[key] = {
                    name: sorted(values)
                    for name, values in sorted(selectors[key].items())
                }
            else:
                selectors[key] = sorted(selectors[key])

        meta["metadata_schema_version"] = SIDECAR_SCHEMA_VERSION
        meta["early_spectral_phenotype_contract"] = {
            "label_version": label_version,
            "source_table": _display_path(phenotype_csv),
            "exclusion_table": _display_path(exclusions_csv),
            "join_key": ["subject", "seizure_idx"],
            "event_field": f"seizure.<idx>.{LABEL_BLOCK}",
            "selector_scope": (
                "accepted T_spectral events with a committed 1-150 Hz phenotype "
                "row; source has_accepted_tspectral is provenance only"
            ),
            "missing_label_policy": (
                "retain the accepted cache event with label_status=not_classified; "
                "never coerce it to other and never include it in phenotype selectors"
            ),
            "claim_boundary": (
                "The strict-broadband selector defines a phenotype-restricted spatial "
                "gradient analysis; it cannot independently establish that broadband "
                "energy enhancement exists."
            ),
        }
        meta["early_spectral_phenotype_selectors"] = selectors
        path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        n_subjects += 1

    summary_path = cache_root / "cache_alignment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metadata_schema_version"] = SIDECAR_SCHEMA_VERSION
    summary["early_spectral_phenotype_labels"] = {
        "label_version": label_version,
        "source_table": _display_path(phenotype_csv),
        "n_cache_events": n_events,
        "label_status_counts": dict(sorted(status_counts.items())),
        "simple_phenotype_counts": dict(sorted(phenotype_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
        "npz_arrays_modified": False,
        "existing_seizure_idxs_modified": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "cache_root": _display_path(cache_root),
        "n_subjects": n_subjects,
        "n_cache_events": n_events,
        "label_status_counts": dict(sorted(status_counts.items())),
        "simple_phenotype_counts": dict(sorted(phenotype_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
    }


def _update_readme(cache_root: Path, result: dict[str, Any]) -> None:
    path = cache_root / "README.md"
    text = path.read_text(encoding="utf-8").rstrip()
    marker = "## Per-seizure early spectral phenotype labels"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    counts = result["label_status_counts"]
    section = (
        f"\n\n{marker}\n\n"
        "每个 aligned event 的 JSON `seizure.<idx>.early_spectral_phenotype` "
        "保存版本化的逐发作频谱标签；可复用 selector 位于 "
        "`early_spectral_phenotype_selectors`。原 `seizure_idxs` 和 NPZ 数组均不改变。\n\n"
        f"本 cache 共 {result['n_cache_events']} 个 accepted `T_spectral` event："
        f"{counts.get('classified', 0)} 个具有正式 1–150 Hz 标签，"
        f"{counts.get('not_classified', 0)} 个明确标记为 `not_classified`。"
        "后者不并入 `other`，也不进入任何 phenotype selector。重建 aligned "
        "cache 后运行 `python scripts/augment_topic5_tspectral_cache_labels.py` "
        "即可恢复该 metadata 合同。\n"
    )
    path.write_text(text + section, encoding="utf-8")


def run(
    cache_roots: list[Path],
    phenotype_csv: Path,
    exclusions_csv: Path,
) -> list[dict[str, Any]]:
    labels, exclusions, label_version = load_labels(phenotype_csv, exclusions_csv)
    results = []
    for cache_root in cache_roots:
        result = augment_cache_root(
            cache_root,
            labels=labels,
            exclusions=exclusions,
            label_version=label_version,
            phenotype_csv=phenotype_csv,
            exclusions_csv=exclusions_csv,
        )
        _update_readme(cache_root, result)
        results.append(result)
    combined_status: Counter[str] = Counter()
    combined_phenotype: Counter[str] = Counter()
    combined_selectors: Counter[str] = Counter()
    for result in results:
        combined_status.update(result["label_status_counts"])
        combined_phenotype.update(result["simple_phenotype_counts"])
        combined_selectors.update(result["selector_counts"])
    combined = {
        "analysis_version": SIDECAR_SCHEMA_VERSION,
        "scientific_scope": (
            "existing frequency labels joined to accepted T_spectral cache events; "
            "no seizure type is recomputed"
        ),
        "label_version": label_version,
        "n_cache_roots": len(results),
        "n_subjects": int(sum(result["n_subjects"] for result in results)),
        "n_cache_events": int(sum(result["n_cache_events"] for result in results)),
        "label_status_counts": dict(sorted(combined_status.items())),
        "simple_phenotype_counts": dict(sorted(combined_phenotype.items())),
        "selector_counts": dict(sorted(combined_selectors.items())),
        "cache_roots": results,
    }
    (phenotype_csv.parent / "tspectral_aligned_cache_label_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", action="append", type=Path, default=[])
    parser.add_argument("--phenotype-csv", type=Path, default=PHENOTYPE_CSV)
    parser.add_argument(
        "--phenotype-exclusions-csv",
        type=Path,
        default=PHENOTYPE_EXCLUSIONS_CSV,
    )
    args = parser.parse_args()
    roots = [path.resolve() for path in args.cache_root] or list(CACHE_ROOTS)
    results = run(
        roots,
        args.phenotype_csv.resolve(),
        args.phenotype_exclusions_csv.resolve(),
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
