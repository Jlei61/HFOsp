#!/usr/bin/env python3
"""Freeze v0.5 RNN and train-only template fields before target access."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_rnn_motif_fields_v0_4 import (  # noqa: E402
    aggregate_records, derive_common_contrast, empirical_score, safe_corr,
    split_half_stability,
)


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)
ARMS = (
    "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def unit_metrics_path(out: Path, old: Path, fit_id: str, arm: str, seed: int, reused: set[str]) -> Path:
    if fit_id in reused and arm in {
        "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", "L3_LOCAL_PLUS_LEARNED_LR"
    }:
        return old / "per_fit" / fit_id / arm / f"seed{seed}" / "metrics.json"
    return out / "formal_units" / fit_id / arm / f"seed{seed}" / "metrics.json"


def empirical_candidates(field_root: Path, subject: str, contacts: np.ndarray) -> dict[str, np.ndarray]:
    field = json.loads((field_root / f"{subject}.json").read_text())["interictal_field"]
    order = [str(value) for value in field["contact_order"]]
    take = np.asarray([order.index(str(contact)) for contact in contacts], dtype=int)
    return {
        "A": empirical_score(np.asarray(field["rank_a"], float)[take]),
        "B": empirical_score(np.asarray(field["rank_b"], float)[take]),
    }


def train_mode_to_ab(cache: Path, subject: str, contacts: np.ndarray, field_root: Path) -> dict[int, str]:
    modes = np.load(cache / "train_only_modes.npz", allow_pickle=False)
    template_earlyness = np.stack([empirical_score(row) for row in modes["templates"]])
    empirical = empirical_candidates(field_root, subject, contacts)
    direct = float(np.nansum([
        safe_corr(template_earlyness[0], empirical["A"]),
        safe_corr(template_earlyness[1], empirical["B"]),
    ]))
    reverse = float(np.nansum([
        safe_corr(template_earlyness[0], empirical["B"]),
        safe_corr(template_earlyness[1], empirical["A"]),
    ]))
    return {0: "A", 1: "B"} if direct >= reverse else {0: "B", 1: "A"}


def ab_prevalence(train_counts: np.ndarray, mapping: dict[int, str]) -> dict[str, float]:
    """Map train-only mode counts onto canonical A/B labels."""
    counts = np.asarray(train_counts, dtype=float)
    if counts.shape != (2,) or not np.isfinite(counts).all() or counts.sum() <= 0:
        raise ValueError("train-only mode prevalence requires two positive finite counts")
    result = {"A": 0.0, "B": 0.0}
    for mode in (0, 1):
        result[mapping[mode]] += float(counts[mode])
    total = result["A"] + result["B"]
    return {key: value / total for key, value in result.items()}


def remap_record_modes(records: list[dict], cache: Path) -> list[dict]:
    """Replace any legacy full-data labels by frozen train-only prefix modes."""
    events = np.load(cache / "events.npz", allow_pickle=False)
    mapping: dict[int, int] = {}
    for source, mode in zip(events["event_source_index"], events["mode"]):
        source, mode = int(source), int(mode)
        if source in mapping and mapping[source] != mode:
            raise RuntimeError(f"event_source_index {source} maps to two prefix modes")
        mapping[source] = mode
    output = []
    for record in records:
        source = int(record["event_source_index"])
        if source not in mapping:
            raise RuntimeError(f"rollout event_source_index absent from v0.5 cache: {source}")
        row = dict(record)
        row["mode"] = mapping[source]
        output.append(row)
    return output


def pair_stability(arrays: list[dict], endpoint: str) -> float:
    values = [safe_corr(left[endpoint], right[endpoint]) for left, right in combinations(arrays, 2)]
    return float(np.nanmedian(values)) if values else float("nan")


def build_seed_fields(out: Path, old: Path, field_root: Path) -> tuple[pd.DataFrame, dict]:
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    rows, fields = [], {}
    mapping_manifest = []
    for fit in census.itertuples():
        cache = out / "cache" / fit.fit_id
        provenance = json.loads((cache / "provenance.json").read_text())
        contacts = np.asarray(provenance["joint_contacts"], dtype="U64")
        if fit.scope == "shared":
            mode_to_ab = train_mode_to_ab(cache, fit.subject, contacts, field_root)
        else:
            mode_to_ab = {0: "A" if fit.scope == "own_a" else "B",
                          1: "A" if fit.scope == "own_a" else "B"}
        mapping_manifest.append({
            "subject": fit.subject, "fit_id": fit.fit_id, "scope": fit.scope,
            "mode0": mode_to_ab[0], "mode1": mode_to_ab[1],
            "mapping_source": "TRAIN_TEMPLATE_TO_EMPIRICAL_AB" if fit.scope == "shared" else "FROZEN_FIT_SCOPE",
        })
        empirical = empirical_candidates(field_root, fit.subject, contacts)
        for arm in ARMS:
            for seed in range(3):
                metrics_path = unit_metrics_path(out, old, fit.fit_id, arm, seed, reused)
                metrics = json.loads(metrics_path.read_text())
                if metrics.get("target_values_read") is not False:
                    raise RuntimeError(f"target marker is not false: {metrics_path}")
                with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as stream:
                    records = remap_record_modes(json.load(stream), cache)
                grouped: dict[str, list[dict]] = defaultdict(list)
                for record in records:
                    grouped[mode_to_ab[int(record["mode"])]].append(record)
                for template, selected in grouped.items():
                    aggregate = aggregate_records(selected, len(contacts))
                    split = split_half_stability(selected, len(contacts))
                    aggregate["canonical_full_split_half_stability"] = np.asarray([split["canonical_full"]])
                    aggregate["seed_removed_split_half_stability"] = np.asarray([split["seed_removed"]])
                    fields[(fit.fit_id, arm, seed, template)] = {"contacts": contacts, **aggregate}
                    destination = out / "model_fields/intact/per_fit_seed" / fit.fit_id / arm / f"seed{seed}_{template}.npz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination, contacts=contacts, **aggregate)
                    rows.append({
                        "subject": fit.subject, "fit_id": fit.fit_id, "scope": fit.scope,
                        "arm": arm, "seed": seed, "template": template,
                        "n_events": len(selected),
                        "canonical_empirical_r": safe_corr(aggregate["canonical_full"], empirical[template]),
                        "seed_removed_empirical_r": safe_corr(aggregate["seed_removed"], empirical[template]),
                        "canonical_split_half_stability": split["canonical_full"],
                        "seed_removed_split_half_stability": split["seed_removed"],
                        "field_sha256": sha256_file(destination),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "MODEL_FIELD_FIT_SEED_METRICS.csv", index=False)
    pd.DataFrame(mapping_manifest).to_csv(out / "TRAIN_MODE_TO_AB_MAPPING.csv", index=False)
    return frame, fields


def aggregate_fit_fields(frame: pd.DataFrame, fields: dict) -> dict:
    output = {}
    for key, group in frame.groupby(["subject", "fit_id", "scope", "arm", "template"], sort=False):
        subject, fit_id, scope, arm, template = key
        arrays = [fields[(fit_id, arm, int(seed), template)] for seed in group.seed]
        payload = {"contacts": arrays[0]["contacts"]}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            payload[endpoint] = np.nanmedian(np.stack([item[endpoint] for item in arrays]), axis=0)
            payload[f"{endpoint}_seed_stability"] = pair_stability(arrays, endpoint)
        payload["seed_removed_denominator"] = np.sum(
            np.stack([item["seed_removed_denominator"] for item in arrays]), axis=0
        )
        output[(subject, fit_id, scope, arm, template)] = payload
    return output


def aggregate_patient_fields(out: Path, field_root: Path, fit_fields: dict) -> pd.DataFrame:
    rows, manifest = [], []
    for subject, arm in sorted({(key[0], key[3]) for key in fit_fields}):
        candidates, producers = {}, {}
        for template in ("A", "B"):
            matches = [(key, value) for key, value in fit_fields.items()
                       if key[0] == subject and key[3] == arm and key[4] == template]
            if len(matches) != 1:
                raise RuntimeError(f"expected one {template} producer for {subject} {arm}, found {len(matches)}")
            candidates[template], producers[template] = matches[0][1], matches[0][0][1]
        contacts = candidates["A"]["contacts"]
        if not np.array_equal(contacts, candidates["B"]["contacts"]):
            raise RuntimeError(f"A/B contact mismatch: {subject} {arm}")
        prevalence_fit = sorted({producers["A"], producers["B"]})[0]
        prevalence_cache = out / "cache" / prevalence_fit
        prevalence_modes = np.load(prevalence_cache / "train_only_modes.npz", allow_pickle=False)
        prevalence_mapping = train_mode_to_ab(
            prevalence_cache, subject, contacts, field_root
        )
        prevalence = ab_prevalence(prevalence_modes["train_counts"], prevalence_mapping)
        payload = {"contacts": contacts}
        for template in ("A", "B"):
            for endpoint in ("canonical_full", "seed_removed", "participation", "seed_removed_denominator"):
                payload[f"{template}_{endpoint}"] = candidates[template][endpoint]
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            common, contrast = derive_common_contrast(candidates["A"][endpoint], candidates["B"][endpoint])
            payload[f"{endpoint}_common"] = common
            payload[f"{endpoint}_contrast"] = contrast
            left = np.asarray(candidates["A"][endpoint], float)
            right = np.asarray(candidates["B"][endpoint], float)
            if endpoint == "seed_removed":
                left = np.nan_to_num(left, nan=0.0)
                right = np.nan_to_num(right, nan=0.0)
            payload[f"{endpoint}_train_prevalence_mixture"] = (
                prevalence["A"] * left + prevalence["B"] * right
            )
        payload["train_prevalence_A"] = np.asarray(prevalence["A"], dtype=np.float32)
        payload["train_prevalence_B"] = np.asarray(prevalence["B"], dtype=np.float32)
        destination = out / "model_fields/intact/per_patient" / subject / f"{arm}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
        empirical = empirical_candidates(field_root, subject, contacts)
        rows.append({
            "subject": subject, "arm": arm,
            "aggregation": "shared_single_fit" if producers["A"] == producers["B"] else "own_a_own_b_separate",
            "producer_A": producers["A"], "producer_B": producers["B"],
            "train_prevalence_A": prevalence["A"],
            "train_prevalence_B": prevalence["B"],
            "canonical_empirical_r": np.nanmean([
                safe_corr(payload["A_canonical_full"], empirical["A"]),
                safe_corr(payload["B_canonical_full"], empirical["B"]),
            ]),
            "seed_removed_empirical_r": np.nanmean([
                safe_corr(payload["A_seed_removed"], empirical["A"]),
                safe_corr(payload["B_seed_removed"], empirical["B"]),
            ]),
            "field_sha256": sha256_file(destination),
        })
        for endpoint in (
            "A_canonical_full", "B_canonical_full", "canonical_full_common", "canonical_full_contrast",
            "A_seed_removed", "B_seed_removed", "seed_removed_common", "seed_removed_contrast",
            "canonical_full_train_prevalence_mixture",
            "seed_removed_train_prevalence_mixture",
        ):
            vector = np.asarray(payload[endpoint])
            manifest.append({
                "field_family": "RNN", "subject": subject, "arm": arm, "endpoint": endpoint,
                "path": str(destination), "file_sha256": sha256_file(destination),
                "vector_sha256": vector_sha256(vector), "n_contacts": len(vector),
                "target_values_read": False,
            })
    result = pd.DataFrame(rows)
    result.to_csv(out / "MODEL_FIELD_PATIENT_METRICS.csv", index=False)
    pd.DataFrame(manifest).to_csv(out / "MODEL_FIELD_MANIFEST.csv", index=False)
    return result


def freeze_template_fields(out: Path, field_root: Path) -> pd.DataFrame:
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    rows = []
    for subject, group in census.groupby("subject", sort=False):
        # Modes are an event-space baseline, independent of which retrospective
        # A/B geometry fit is used by the RNN.  All split-plane fits contain the
        # identical unfiltered rank task, so use one deterministic cache and
        # map its two train-only templates to canonical empirical A/B without
        # using held-out suffixes or any ictal value.
        fit = sorted(group.itertuples(), key=lambda row: row.fit_id)[0]
        cache = out / "cache" / fit.fit_id
        provenance = json.loads((cache / "provenance.json").read_text())
        contacts = np.asarray(provenance["joint_contacts"], dtype="U64")
        modes = np.load(cache / "train_only_modes.npz", allow_pickle=False)
        templates = np.stack([empirical_score(row) for row in modes["templates"]])
        mapping = train_mode_to_ab(cache, subject, contacts, field_root)
        candidates = {mapping[mode]: templates[mode] for mode in (0, 1)}
        prevalence = modes["train_counts"].astype(float)
        prevalence /= prevalence.sum()
        mixture = prevalence[0] * templates[0] + prevalence[1] * templates[1]
        common, contrast = derive_common_contrast(candidates["A"], candidates["B"])
        destination = out / "model_fields/templates/per_patient" / subject / "TRAIN_ONLY_TEMPLATE_FIELDS.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, contacts=contacts, A=candidates["A"], B=candidates["B"],
                            common=common, contrast=contrast, train_prevalence_mixture=mixture)
        for endpoint in ("A", "B", "common", "contrast", "train_prevalence_mixture"):
            vector = np.load(destination, allow_pickle=False)[endpoint]
            rows.append({
                "field_family": "TRAIN_ONLY_TEMPLATE", "subject": subject,
                "arm": "PREFIX_TEMPLATE", "endpoint": endpoint,
                "path": str(destination), "file_sha256": sha256_file(destination),
                "vector_sha256": vector_sha256(vector), "n_contacts": len(vector),
                "target_values_read": False,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "TEMPLATE_FIELD_MANIFEST.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--field-root", type=Path, default=FIELD_ROOT)
    args = parser.parse_args()
    out, old, field_root = args.out_root.resolve(), args.old_root.resolve(), args.field_root.resolve()
    if not (out / "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("Stage E patient-first analysis must pass before field construction")
    frame, fields = build_seed_fields(out, old, field_root)
    if len(frame) != 840:
        raise RuntimeError(f"expected 840 fit-seed-template fields, found {len(frame)}")
    patient = aggregate_patient_fields(out, field_root, aggregate_fit_fields(frame, fields))
    if len(patient) != 140:
        raise RuntimeError(f"expected 140 patient-arm fields, found {len(patient)}")
    templates = freeze_template_fields(out, field_root)
    manifest = out / "MODEL_FIELD_MANIFEST.csv"
    template_manifest = out / "TEMPLATE_FIELD_MANIFEST.csv"
    write_json(out / "MODEL_FIELDS_FROZEN.json", {
        "status": "FROZEN_TARGET_FREE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False, "fit_seed_template_fields": len(frame),
        "patient_arm_fields": len(patient), "template_field_vectors": len(templates),
        "model_manifest_sha256": sha256_file(manifest),
        "template_manifest_sha256": sha256_file(template_manifest),
    })


if __name__ == "__main__":
    main()
