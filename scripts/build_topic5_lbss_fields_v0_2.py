#!/usr/bin/env python3
"""Freeze intact LBSS A/B/common/contrast fields before target access."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_rnn_motif_fields_v0_4 import (  # noqa: E402
    aggregate_records,
    derive_common_contrast,
    empirical_score,
    safe_corr,
    split_half_stability,
    template_for_mode,
)


ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
OLD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-rnn-motif-cross-state-v0-4/"
    "results/topic5_rnn_motif_cross_state_benchmark_v0_4"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_stability(arrays: list[dict], endpoint: str) -> float:
    values = [safe_corr(left[endpoint], right[endpoint]) for left, right in combinations(arrays, 2)]
    return float(np.nanmedian(values)) if values else float("nan")


def build_seed_fields(out: Path, field_root: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    fields = {}
    for metrics_path in sorted((out / "per_fit").glob("*/*/seed*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        fit_id, arm, seed = metrics["fit_id"], metrics["arm"], int(metrics["seed"])
        provenance = json.loads((out / "cache" / fit_id / "provenance.json").read_text())
        contacts = np.asarray(provenance["contacts"], dtype="U64")
        with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as stream:
            records = json.load(stream)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            template = template_for_mode(provenance, record["mode"])
            if template in ("A", "B"):
                grouped[template].append(record)
        empirical = json.loads((field_root / f"{metrics['subject']}.json").read_text())["interictal_field"]
        empirical_order = [str(value) for value in empirical["contact_order"]]
        take = np.asarray([empirical_order.index(str(contact)) for contact in contacts], dtype=int)
        empirical_by_template = {
            "A": empirical_score(np.asarray(empirical["rank_a"], float)[take]),
            "B": empirical_score(np.asarray(empirical["rank_b"], float)[take]),
        }
        for template, selected in grouped.items():
            aggregate = aggregate_records(selected, len(contacts))
            split = split_half_stability(selected, len(contacts))
            aggregate["canonical_full_split_half_stability"] = np.asarray([split["canonical_full"]])
            aggregate["seed_removed_split_half_stability"] = np.asarray([split["seed_removed"]])
            fields[(fit_id, arm, seed, template)] = {"contacts": contacts, **aggregate}
            destination = out / "model_fields" / "intact" / "per_fit_seed" / fit_id / arm / f"seed{seed}_{template}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, contacts=contacts, **aggregate)
            rows.append({
                "subject": metrics["subject"], "fit_id": fit_id, "scope": metrics["scope"],
                "arm": arm, "seed": seed, "template": template, "n_events": len(selected),
                "canonical_empirical_r": safe_corr(aggregate["canonical_full"], empirical_by_template[template]),
                "seed_removed_empirical_r": safe_corr(aggregate["seed_removed"], empirical_by_template[template]),
                "canonical_split_half_stability": split["canonical_full"],
                "seed_removed_split_half_stability": split["seed_removed"],
                "field_sha256": sha256(destination),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "model_field_fit_seed_metrics.csv", index=False)
    return frame, fields


def aggregate_fit_fields(frame: pd.DataFrame, fields: dict) -> dict:
    fit_fields = {}
    for key, group in frame.groupby(["subject", "fit_id", "scope", "arm", "template"], sort=False):
        subject, fit_id, scope, arm, template = key
        arrays = [fields[(fit_id, arm, int(seed), template)] for seed in group.seed]
        contacts = arrays[0]["contacts"]
        aggregate = {"contacts": contacts}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            aggregate[endpoint] = np.nanmedian(np.stack([item[endpoint] for item in arrays]), axis=0)
            aggregate[f"{endpoint}_seed_stability"] = pair_stability(arrays, endpoint)
        aggregate["seed_removed_denominator"] = np.sum(
            np.stack([item["seed_removed_denominator"] for item in arrays]), axis=0
        )
        aggregate["canonical_full_split_half_stability"] = float(np.nanmedian(
            [item["canonical_full_split_half_stability"][0] for item in arrays]
        ))
        aggregate["seed_removed_split_half_stability"] = float(np.nanmedian(
            [item["seed_removed_split_half_stability"][0] for item in arrays]
        ))
        fit_fields[(subject, fit_id, scope, arm, template)] = aggregate
    return fit_fields


def aggregate_patient_fields(out: Path, field_root: Path, fit_fields: dict) -> pd.DataFrame:
    rows = []
    manifest_rows = []
    subjects_arms = sorted({(key[0], key[3]) for key in fit_fields})
    for subject, arm in subjects_arms:
        candidates, producer = {}, {}
        for template in ("A", "B"):
            matches = [(key, value) for key, value in fit_fields.items()
                       if key[0] == subject and key[3] == arm and key[4] == template]
            if len(matches) != 1:
                continue
            key, value = matches[0]
            candidates[template] = value
            producer[template] = key[1]
        if set(candidates) != {"A", "B"}:
            continue
        if not np.array_equal(candidates["A"]["contacts"], candidates["B"]["contacts"]):
            raise RuntimeError(f"A/B contact support mismatch: {subject} {arm}")
        payload = {"contacts": candidates["A"]["contacts"]}
        for template in ("A", "B"):
            for endpoint in ("canonical_full", "seed_removed", "participation", "seed_removed_denominator"):
                payload[f"{template}_{endpoint}"] = candidates[template][endpoint]
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            common, contrast = derive_common_contrast(candidates["A"][endpoint], candidates["B"][endpoint])
            payload[f"{endpoint}_common"] = common
            payload[f"{endpoint}_contrast"] = contrast
        destination = out / "model_fields" / "intact" / "per_patient" / subject / f"{arm}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)

        empirical = json.loads((field_root / f"{subject}.json").read_text())["interictal_field"]
        order = [str(value) for value in empirical["contact_order"]]
        take = np.asarray([order.index(str(value)) for value in payload["contacts"]], dtype=int)
        empirical_a = empirical_score(np.asarray(empirical["rank_a"], float)[take])
        empirical_b = empirical_score(np.asarray(empirical["rank_b"], float)[take])
        canonical_matched = np.nanmean([
            safe_corr(payload["A_canonical_full"], empirical_a),
            safe_corr(payload["B_canonical_full"], empirical_b),
        ])
        seed_removed_matched = np.nanmean([
            safe_corr(payload["A_seed_removed"], empirical_a),
            safe_corr(payload["B_seed_removed"], empirical_b),
        ])
        rows.append({
            "subject": subject, "arm": arm,
            "aggregation": "shared_single_fit" if producer["A"] == producer["B"] else "own_a_own_b_separate",
            "producer_A": producer["A"], "producer_B": producer["B"],
            "canonical_empirical_r": float(canonical_matched),
            "seed_removed_empirical_r": float(seed_removed_matched),
            "canonical_contrast_empirical_r": safe_corr(
                payload["canonical_full_contrast"], empirical_a - empirical_b
            ),
            "seed_removed_contrast_empirical_r": safe_corr(
                payload["seed_removed_contrast"], empirical_a - empirical_b
            ),
            "canonical_seed_stability": float(np.nanmean([
                candidates["A"]["canonical_full_seed_stability"],
                candidates["B"]["canonical_full_seed_stability"],
            ])),
            "seed_removed_seed_stability": float(np.nanmean([
                candidates["A"]["seed_removed_seed_stability"],
                candidates["B"]["seed_removed_seed_stability"],
            ])),
            "field_sha256": sha256(destination),
        })
        for endpoint in (
            "A_canonical_full", "B_canonical_full", "canonical_full_common", "canonical_full_contrast",
            "A_seed_removed", "B_seed_removed", "seed_removed_common", "seed_removed_contrast",
        ):
            vector = np.asarray(payload[endpoint])
            manifest_rows.append({
                "subject": subject, "arm": arm, "endpoint": endpoint,
                "path": str(destination), "file_sha256": sha256(destination),
                "vector_sha256": hashlib.sha256(np.ascontiguousarray(vector).view(np.uint8)).hexdigest(),
                "n_contacts": len(vector), "target_values_read": False,
            })
    result = pd.DataFrame(rows)
    result.to_csv(out / "model_field_patient_metrics.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out / "MODEL_FIELD_MANIFEST.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    args = parser.parse_args()
    out = args.out_root.resolve()
    if not (out / "INTERICTAL_ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("interictal analysis must pass before field freeze")
    old_manifest = json.loads((OLD_ROOT / "INPUT_MANIFEST.json").read_text())
    field_root = Path(old_manifest["input_roots"]["field"])
    frame, fields = build_seed_fields(out, field_root)
    # 11 shared fits contribute A and B; 20 own-mode fits contribute one each.
    expected_fit_seed_templates = (11 * 2 + 20) * len(ARMS) * 3
    if len(frame) != expected_fit_seed_templates:
        raise RuntimeError(
            f"expected {expected_fit_seed_templates} fit-seed-template fields, observed {len(frame)}"
        )
    fit_fields = aggregate_fit_fields(frame, fields)
    patient = aggregate_patient_fields(out, field_root, fit_fields)
    expected = 21 * len(ARMS)
    if len(patient) != expected:
        raise RuntimeError(f"expected {expected} patient-arm fields, observed {len(patient)}")
    manifest = out / "MODEL_FIELD_MANIFEST.csv"
    (out / "MODEL_FIELDS_FROZEN.json").write_text(json.dumps({
        "status": "FROZEN",
        "n_fit_seed_template_fields": len(frame),
        "n_patient_arm_fields": len(patient),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
