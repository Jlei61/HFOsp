#!/usr/bin/env python3
"""Score frozen full-tissue LBSS fields on the canonical Figure 3 target.

The scorer never trains a model or constructs a new model field.  It reads the
141-seizure exact spatial-model intersection only after every intact and
attenuated field manifest has been frozen and hashed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_topic5_rnn_full_cohort_field_transfer_v0_1 import _load_activation  # noqa: E402
from score_topic5_lbss_early_ictal_v0_2 import (  # noqa: E402
    ENDPOINTS,
    build_scorer,
    load_candidates,
    permutation_indices,
    permutation_support,
    plot_stage,
    score_one,
    summarize_claims,
)


CANONICAL_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
N_PERM = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def exact_align(names: list[str], values: np.ndarray, order: list[str]) -> np.ndarray:
    lookup = {str(name): float(value) for name, value in zip(names, values)}
    return np.asarray([lookup.get(str(name), np.nan) for name in order], float)


def aggregate_subject(
    seizure: pd.DataFrame,
    nulls: dict[str, np.ndarray],
    *,
    group_label: str,
) -> list[dict]:
    rows = []
    for (_, condition, endpoint), group in seizure.groupby(
        ["subject", "condition", "endpoint"], sort=False
    ):
        all_null = np.nanmedian(
            np.stack([nulls[value] for value in group.null_key_all]), axis=0
        )
        shaft_null = np.nanmedian(
            np.stack([nulls[value] for value in group.null_key_shaft]), axis=0
        )
        common_null = np.nanmedian(
            np.stack([nulls[value] for value in group.null_key_common]), axis=0
        )
        observed = float(np.nanmedian(group.observed))
        common_observed = float(np.nanmedian(group.common_observed))
        first = group.iloc[0]
        rows.append({
            "subject": first.subject,
            "primary": True,
            "supportive": False,
            "analysis_group": group_label,
            "condition": condition,
            "family": first.family,
            "arm": first.arm,
            "target": first.target,
            "alpha": float(first.alpha),
            "endpoint": endpoint,
            "n_seizures": int(len(group)),
            "n_contacts_min": int(group.n_contacts.min()),
            "observed": observed,
            "all_contact_null_median": float(np.nanmedian(all_null)),
            "all_contact_margin": observed - float(np.nanmedian(all_null)),
            "all_contact_p": float(
                (1 + np.sum(all_null >= observed - 1e-15))
                / (1 + np.isfinite(all_null).sum())
            ),
            "within_shaft_null_median": float(np.nanmedian(shaft_null)),
            "within_shaft_margin": observed - float(np.nanmedian(shaft_null)),
            "common_observed": common_observed,
            "common_all_contact_null_median": float(np.nanmedian(common_null)),
            "common_all_contact_margin": common_observed - float(np.nanmedian(common_null)),
            "within_shaft_permutable_contacts_min": int(
                group.within_shaft_permutable_contacts.min()
            ),
        })
    return rows


def verify_scorer_authorization(out: Path, authorization: dict) -> None:
    """Accept the original scorer or one explicitly audited repair revision."""
    current = sha256(Path(__file__).resolve())
    if current == authorization.get("scorer_sha256"):
        return
    repair_path = out / "TARGET_SCORER_REPAIR_AUTHORIZATION.json"
    if not repair_path.exists():
        raise RuntimeError("authorized scorer hash does not match this source")
    repair = json.loads(repair_path.read_text())
    helper = Path(__file__).resolve().parent / "score_topic5_lbss_early_ictal_v0_2.py"
    checks = (
        repair.get("status") == "AUTHORIZED_ENGINEERING_REPAIR",
        repair.get("original_scorer_sha256") == authorization.get("scorer_sha256"),
        repair.get("repaired_scorer_sha256") == current,
        repair.get("repaired_helper_sha256") == sha256(helper),
        repair.get("model_or_field_values_changed") is False,
        repair.get("target_values_read_before_repair") is True,
    )
    if not all(checks):
        raise RuntimeError("target scorer repair authorization is incomplete or stale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()
    if args.n_perm < N_PERM:
        raise ValueError(f"n_perm must be at least {N_PERM}")
    out = args.out_root.resolve()
    canonical = args.canonical_root.resolve()

    authorization_path = out / "TARGET_UNSEAL_AUTHORIZATION.json"
    authorization = json.loads(authorization_path.read_text())
    if not authorization.get("authorized"):
        raise RuntimeError("target access is not authorized")
    verify_scorer_authorization(out, authorization)
    for name, digest in authorization["frozen_hashes"].items():
        if sha256(out / name) != digest:
            raise RuntimeError(f"frozen artifact changed after authorization: {name}")
    if sha256(out / "MODEL_FIELD_MANIFEST.csv") != authorization["intact_field_manifest_sha256"]:
        raise RuntimeError("intact field manifest changed after authorization")
    if sha256(out / "ATTENUATED_FIELD_MANIFEST.csv") != authorization["attenuated_field_manifest_sha256"]:
        raise RuntimeError("attenuated field manifest changed after authorization")

    metadata = json.loads((out / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    event_inventory = pd.read_csv(out / "EARLY_ICTAL_METADATA_INVENTORY.csv")
    if event_inventory.subject.nunique() != 12 or len(event_inventory) != 141:
        raise RuntimeError("frozen spatial/Figure 3 intersection changed")
    field_root = canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    # Preflight every target-free model-field inventory before reading any
    # early-ictal value. This catches incomplete required fields early while
    # allowing the prespecified matched-local control to be absent when its
    # target-free caliper search found no legal counterfactual subset.
    for subject in metadata["actual_spatial_join"]:
        record = json.loads((field_root / f"{subject}.json").read_text())
        order = [str(value) for value in record["interictal_field"]["contact_order"]]
        for endpoint in ENDPOINTS:
            load_candidates(out, subject, endpoint, order)
    early = out / "early_ictal"
    early.mkdir(exist_ok=True)
    unlock_path = early / "TARGET_UNLOCK_RECORD.json"
    if unlock_path.exists():
        write_json(early / "TARGET_SCORING_RESTART_RECORD.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorization_sha256": sha256(authorization_path),
            "repair_authorization_sha256": sha256(
                out / "TARGET_SCORER_REPAIR_AUTHORIZATION.json"
            ),
            "target_values_read_before_restart": True,
            "reason": "optional matched-local inventory was incorrectly required",
            "model_or_field_values_changed": False,
            "n_permutations": int(args.n_perm),
        })
    else:
        write_json(unlock_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorization_sha256": sha256(authorization_path),
            "target_values_read_before_this_record": False,
            "target_values_unlocked_after_all_field_freezes": True,
            "figure3_parent": {"patients": 17, "seizures": 167},
            "spatial_model_exact_join": {"patients": 12, "seizures": 141},
            "window": "clinical onset 0-10 s",
            "readouts": "Figure 3 phenotype matched; strict broadband uses 1-150 Hz",
            "n_permutations": int(args.n_perm),
        })

    all_seizure_rows: list[dict] = []
    all_patient_rows: list[dict] = []
    strict_patient_rows: list[dict] = []
    e1146_broadband: list[np.ndarray] = []
    e1146_order: list[str] | None = None

    for subject in metadata["actual_spatial_join"]:
        record = json.loads((field_root / f"{subject}.json").read_text())
        field = record["interictal_field"]
        order = [str(value) for value in field["contact_order"]]
        shafts = [str(value) for value in field["shafts"]]
        subject_inventory = event_inventory[event_inventory.subject.eq(subject)]
        subject_rows: list[dict] = []
        subject_nulls: dict[str, np.ndarray] = {}

        for event in subject_inventory.itertuples():
            names, activation = _load_activation(
                canonical, subject, int(event.seizure_idx), str(event.phenotype)
            )
            target = exact_align(names, np.asarray(activation, float), order)
            finite = np.isfinite(target)
            if int(finite.sum()) != int(event.n_exact_join_contacts) or int(finite.sum()) < 6:
                raise RuntimeError(
                    f"exact target support changed: {subject} seizure {event.seizure_idx}"
                )
            eligible = np.flatnonzero(finite)
            perm_all = permutation_indices(
                len(order), eligible, shafts, args.n_perm,
                int(event.permutation_seed), False,
            )
            perm_shaft = permutation_indices(
                len(order), eligible, shafts, args.n_perm,
                int(event.permutation_seed), True,
            )
            support = permutation_support(eligible, shafts)
            if subject == "epilepsiae_1146" and event.phenotype == "strict_broadband":
                e1146_broadband.append(target.copy())
                e1146_order = order

            for endpoint in ENDPOINTS:
                candidates = load_candidates(out, subject, endpoint, order)
                candidates["EMPIRICAL_REFERENCE"] = {
                    "family": "reference",
                    "arm": "EMPIRICAL_REFERENCE",
                    "target": "",
                    "alpha": 0.0,
                    "a": np.asarray(field["earliness_a"], float),
                    "b": np.asarray(field["earliness_b"], float),
                }
                for condition, candidate in candidates.items():
                    scorer = build_scorer(record, candidate["a"], candidate["b"], finite)
                    all_score = score_one(scorer, target, perm_all)
                    shaft_score = score_one(scorer, target, perm_shaft)
                    common = 0.5 * (candidate["a"] + candidate["b"])
                    common_score = score_one(
                        build_scorer(record, common, common, finite), target, perm_all
                    )
                    prefix = f"{subject}|{int(event.seizure_idx)}|{condition}|{endpoint}"
                    keys = (prefix + "|all", prefix + "|shaft", prefix + "|common")
                    subject_nulls[keys[0]] = np.asarray(all_score["null"], np.float32)
                    subject_nulls[keys[1]] = np.asarray(shaft_score["null"], np.float32)
                    subject_nulls[keys[2]] = np.asarray(common_score["null"], np.float32)
                    subject_rows.append({
                        "dataset": event.dataset,
                        "subject": subject,
                        "seizure_idx": int(event.seizure_idx),
                        "phenotype": event.phenotype,
                        "band": event.band,
                        "condition": condition,
                        "family": candidate["family"],
                        "arm": candidate["arm"],
                        "target": candidate["target"],
                        "alpha": candidate["alpha"],
                        "endpoint": endpoint,
                        "n_contacts": int(finite.sum()),
                        "observed": all_score["observed"],
                        "all_contact_margin": all_score["margin"],
                        "within_shaft_margin": shaft_score["margin"],
                        "common_observed": common_score["observed"],
                        "within_shaft_permutable_contacts": support[
                            "n_within_shaft_permutable_contacts"
                        ],
                        "null_key_all": keys[0],
                        "null_key_shaft": keys[1],
                        "null_key_common": keys[2],
                    })

        subject_frame = pd.DataFrame(subject_rows)
        all_seizure_rows.extend(subject_rows)
        all_patient_rows.extend(
            aggregate_subject(subject_frame, subject_nulls, group_label="all_phenotype_matched")
        )
        strict = subject_frame[subject_frame.phenotype.eq("strict_broadband")]
        if not strict.empty:
            strict_patient_rows.extend(
                aggregate_subject(strict, subject_nulls, group_label="strict_broadband")
            )

    seizure = pd.DataFrame(all_seizure_rows)
    patient = pd.DataFrame(all_patient_rows)
    strict_patient = pd.DataFrame(strict_patient_rows)
    if patient.subject.nunique() != 12 or patient.n_seizures.groupby(patient.subject).max().sum() != 141:
        raise RuntimeError("patient-first all-phenotype aggregation changed denominator")
    if strict_patient.subject.nunique() != 11 or strict_patient.n_seizures.groupby(strict_patient.subject).max().sum() != 92:
        raise RuntimeError("strict broadband sensitivity denominator changed")
    seizure.to_csv(early / "early_ictal_per_seizure.csv", index=False)
    patient.to_csv(early / "early_ictal_per_patient_condition.csv", index=False)
    strict_patient.to_csv(
        early / "early_ictal_strict_broadband_per_patient_condition.csv", index=False
    )

    fidelity = pd.read_csv(out / "model_field_patient_metrics.csv")
    summary = {
        "contract": "topic5_lbss_full_tissue_early_ictal_v0_3",
        "figure3_parent": {"patients": 17, "seizures": 167},
        "spatial_exact_join": {"patients": 12, "seizures": 141},
        "strict_broadband_sensitivity": {"patients": 11, "seizures": 92},
        "all_phenotype_matched": summarize_claims(patient, fidelity),
        "strict_broadband": summarize_claims(strict_patient, fidelity),
        "target_used_for_training_or_model_selection": False,
        "target_values_read": True,
    }
    write_json(early / "EARLY_ICTAL_SUMMARY.json", summary)
    plot_stage(patient, summary["all_phenotype_matched"], out)

    if len(e1146_broadband) != 15 or e1146_order is None:
        raise RuntimeError(f"expected 15 strict-broadband E1146 seizures, found {len(e1146_broadband)}")
    np.savez_compressed(
        early / "e1146_early_ictal_broadband_1_150.npz",
        contact_order=np.asarray(e1146_order, str),
        activation=np.nanmedian(np.stack(e1146_broadband), axis=0),
        n_seizures=np.asarray(len(e1146_broadband)),
        window=np.asarray(["clinical onset 0-10 s"]),
        band=np.asarray(["1-150 Hz broadband energy"]),
    )
    write_json(out / "TARGET_ACCESS_AUDIT.json", {
        "target_values_read": True,
        "training_or_model_selection_after_unseal": False,
        "figure3_parent_patients": 17,
        "figure3_parent_seizures": 167,
        "spatial_model_patients": 12,
        "spatial_model_seizures": 141,
        "strict_broadband_patients": 11,
        "strict_broadband_seizures": 92,
        "primary_null": "synchronized all-contact label shuffle",
        "within_shaft_sensitivity": True,
    })
    write_json(out / "EARLY_ICTAL_SCORING_COMPLETE.json", {
        "status": "PASS",
        "n_spatial_patients": 12,
        "n_spatial_seizures": 141,
        "target_values_read": True,
    })


if __name__ == "__main__":
    main()
