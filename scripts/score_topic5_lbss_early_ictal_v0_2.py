#!/usr/bin/env python3
"""Score frozen intact and attenuated LBSS fields against the early-ictal endpoint."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_topic5_rnn_motif_early_ictal_v0_4 import (  # noqa: E402
    build_scorer,
    holm,
    paired_summary,
    permutation_indices,
    permutation_support,
    score_one,
    stable_seed,
)


ARMS = (
    "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
ENDPOINTS = ("canonical_full", "seed_removed")
TARGET_BASE = {
    "L1_ADDED": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_ADDED": "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_ADDED": "L3_LOCAL_PLUS_LEARNED_LR",
    "L3_MATCHED_LOCAL": "L3_LOCAL_PLUS_LEARNED_LR",
}
REQUIRED_ATTENUATION_TARGETS = ("L1_ADDED", "L2_ADDED", "L3_ADDED")
OPTIONAL_ATTENUATION_TARGETS = ("L3_MATCHED_LOCAL",)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def align(values: np.ndarray, names: np.ndarray, order: list[str], endpoint: str) -> np.ndarray:
    lookup = {str(name): float(value) for name, value in zip(names, values)}
    result = np.asarray([lookup.get(name, np.nan) for name in order], float)
    # Evaluation support is fixed independently of model participation.  A
    # never-generated nonseed contact has zero recurrence score, not exclusion.
    if endpoint == "seed_removed":
        result = np.nan_to_num(result, nan=0.0)
    return result


def load_candidates(out: Path, subject: str, endpoint: str, order: list[str]) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    for arm in ARMS:
        path = out / "model_fields" / "intact" / "per_patient" / subject / f"{arm}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            names = data["contacts"]
            candidates[f"INTACT|{arm}"] = {
                "family": "intact", "arm": arm, "target": "", "alpha": 0.0,
                "a": align(data[f"A_{endpoint}"], names, order, endpoint),
                "b": align(data[f"B_{endpoint}"], names, order, endpoint),
            }
    for target, arm in TARGET_BASE.items():
        for alpha in (0.25, 0.50, 0.75, 1.00):
            path = out / "attenuation" / "fields" / "per_patient" / subject / target / f"alpha{alpha:.2f}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                names = data["contacts"]
                candidates[f"ATTEN|{target}|{alpha:.2f}"] = {
                    "family": "attenuated", "arm": arm, "target": target, "alpha": alpha,
                    "a": align(data[f"A_{endpoint}"], names, order, endpoint),
                    "b": align(data[f"B_{endpoint}"], names, order, endpoint),
                }
    intact = {f"INTACT|{arm}" for arm in ARMS}
    required_attenuation = {
        f"ATTEN|{target}|{alpha:.2f}"
        for target in REQUIRED_ATTENUATION_TARGETS
        for alpha in (0.25, 0.50, 0.75, 1.00)
    }
    missing_required = sorted((intact | required_attenuation) - set(candidates))
    if missing_required:
        raise RuntimeError(
            f"{subject} {endpoint}: missing required frozen candidates {missing_required}"
        )
    # Matched-local fields exist only when the target-free caliper search found
    # a legal active local subset.  A failed search is explicitly descriptive-
    # only and must not be repaired by inventing a counterfactual field.  The
    # optional control is therefore either complete at all four doses or absent.
    for target in OPTIONAL_ATTENUATION_TARGETS:
        optional = {
            f"ATTEN|{target}|{alpha:.2f}"
            for alpha in (0.25, 0.50, 0.75, 1.00)
        }
        present = optional & set(candidates)
        if present and present != optional:
            raise RuntimeError(
                f"{subject} {endpoint}: optional {target} fields are only partially frozen"
            )
    expected = len(intact) + len(required_attenuation)
    optional_count = sum(
        int(f"ATTEN|{target}|0.25" in candidates) * 4
        for target in OPTIONAL_ATTENUATION_TARGETS
    )
    if len(candidates) != expected + optional_count:
        raise RuntimeError(
            f"{subject} {endpoint}: unexpected frozen candidate inventory "
            f"({len(candidates)} rows)"
        )
    return candidates


def aggregate_patients(seizure: pd.DataFrame, nulls: dict[str, np.ndarray], supportive: str) -> pd.DataFrame:
    rows = []
    for key, group in seizure.groupby(["subject", "condition", "endpoint"], sort=False):
        subject, condition, endpoint = key
        all_null = np.nanmedian(np.stack([nulls[value] for value in group.null_key_all]), axis=0)
        shaft_null = np.nanmedian(np.stack([nulls[value] for value in group.null_key_shaft]), axis=0)
        common_null = np.nanmedian(np.stack([nulls[value] for value in group.null_key_common]), axis=0)
        observed = float(np.nanmedian(group.observed))
        common_observed = float(np.nanmedian(group.common_observed))
        first = group.iloc[0]
        rows.append({
            "subject": subject, "primary": subject != supportive, "supportive": subject == supportive,
            "condition": condition, "family": first.family, "arm": first.arm,
            "target": first.target, "alpha": float(first.alpha), "endpoint": endpoint,
            "n_seizures": len(group), "n_contacts_min": int(group.n_contacts.min()),
            "observed": observed, "all_contact_null_median": float(np.nanmedian(all_null)),
            "all_contact_margin": observed - float(np.nanmedian(all_null)),
            "all_contact_p": float((1 + np.sum(all_null >= observed - 1e-15)) / (1 + np.isfinite(all_null).sum())),
            "within_shaft_null_median": float(np.nanmedian(shaft_null)),
            "within_shaft_margin": observed - float(np.nanmedian(shaft_null)),
            "common_observed": common_observed,
            "common_all_contact_null_median": float(np.nanmedian(common_null)),
            "common_all_contact_margin": common_observed - float(np.nanmedian(common_null)),
            "within_shaft_permutable_contacts_min": int(group.within_shaft_permutable_contacts.min()),
        })
    return pd.DataFrame(rows)


def fixed_effect_fit(rows: list[dict], arms: tuple[str, ...], outcome: str = "outcome") -> dict:
    patients = sorted({row["patient_block"] for row in rows})
    reference = "L0_LOCAL_ONLY"
    nonreference = [arm for arm in arms if arm != reference]
    pindex = {patient: index for index, patient in enumerate(patients)}
    aindex = {arm: index for index, arm in enumerate(nonreference)}
    x = np.zeros((len(rows), len(patients) + 1 + len(nonreference)))
    y = np.asarray([row[outcome] for row in rows], float)
    for index, row in enumerate(rows):
        x[index, pindex[row["patient_block"]]] = 1.0
        x[index, len(patients)] = row["fidelity"]
        if row["arm"] != reference:
            x[index, len(patients) + 1 + aindex[row["arm"]]] = 1.0
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    gamma = {reference: 0.0, **{
        arm: float(beta[len(patients) + 1 + aindex[arm]]) for arm in nonreference
    }}
    return {"fidelity_slope": float(beta[len(patients)]), "gamma": gamma}


def conditional_effects(patient: pd.DataFrame, fidelity: pd.DataFrame, endpoint: str, draws: int = 5000) -> dict:
    arms = ARMS[:4]
    selected = patient[(patient.primary) & (patient.family == "intact") & (patient.endpoint == endpoint)
                       & patient.arm.isin(arms)].copy()
    column = "canonical_empirical_r" if endpoint == "canonical_full" else "seed_removed_empirical_r"
    selected = selected.merge(fidelity[["subject", "arm", column]], on=["subject", "arm"], how="inner")
    complete = sorted(subject for subject, group in selected.groupby("subject")
                      if set(group.arm) == set(arms))
    selected = selected[selected.subject.isin(complete)]
    rows = [{
        "subject": row.subject, "patient_block": row.subject, "arm": row.arm,
        "fidelity": float(getattr(row, column)), "outcome": float(row.all_contact_margin),
    } for row in selected.itertuples()]
    observed = fixed_effect_fit(rows, arms)
    rng = np.random.default_rng(stable_seed("lbss", endpoint, "conditional"))
    contrasts = {f"L3_vs_{arm}": [] for arm in arms[:3]}
    for _ in range(draws):
        sampled = rng.choice(complete, size=len(complete), replace=True)
        boot = []
        for replicate, subject in enumerate(sampled):
            for row in rows:
                if row["subject"] == subject:
                    boot.append({**row, "patient_block": f"{subject}|{replicate}"})
        fit = fixed_effect_fit(boot, arms)
        for arm in arms[:3]:
            contrasts[f"L3_vs_{arm}"].append(fit["gamma"][arms[3]] - fit["gamma"][arm])
    output = {}
    for arm in arms[:3]:
        name = f"L3_vs_{arm}"
        estimate = observed["gamma"][arms[3]] - observed["gamma"][arm]
        output[name] = {
            "estimate": float(estimate),
            "patient_cluster_bootstrap_95ci": np.quantile(contrasts[name], [0.025, 0.975]).tolist(),
        }
    return {
        "endpoint": endpoint, "n_complete_patients": len(complete), "complete_patients": complete,
        "model": "margin ~ patient_fixed_effect + interictal_fidelity + arm",
        "fidelity_slope": observed["fidelity_slope"], "contrasts": output,
    }


def summarize_claims(patient: pd.DataFrame, fidelity: pd.DataFrame) -> dict:
    primary = patient[patient.primary]
    lookup = {(row.subject, row.condition, row.endpoint): row for row in primary.itertuples()}
    output = {"D1_D2": {}, "attenuation": {}, "conditional": {}}
    for endpoint in ENDPOINTS:
        contrasts = {}
        l3 = f"INTACT|L3_LOCAL_PLUS_LEARNED_LR"
        values = [lookup[(subject, l3, endpoint)].all_contact_margin
                  for subject in primary.subject.unique() if (subject, l3, endpoint) in lookup]
        contrasts["L3_margin_gt_zero"] = paired_summary(values, seed=stable_seed(endpoint, "l3-zero"))
        for arm in ARMS[:3]:
            ref = f"INTACT|{arm}"
            values = [lookup[(subject, l3, endpoint)].all_contact_margin
                      - lookup[(subject, ref, endpoint)].all_contact_margin
                      for subject in primary.subject.unique()
                      if (subject, l3, endpoint) in lookup and (subject, ref, endpoint) in lookup]
            contrasts[f"L3_vs_{arm}"] = paired_summary(values, seed=stable_seed(endpoint, arm))
        p = {name: value["wilcoxon_p"] for name, value in contrasts.items() if name.startswith("L3_vs_")}
        q = holm(p)
        for name, value in q.items():
            contrasts[name]["holm_q_claim_D_family"] = value
        output["D1_D2"][endpoint] = contrasts
        output["conditional"][endpoint] = conditional_effects(patient, fidelity, endpoint)

        auc_rows = []
        for subject in primary.subject.unique():
            for target, arm in TARGET_BASE.items():
                base = lookup.get((subject, f"INTACT|{arm}", endpoint))
                if base is None:
                    continue
                x, y = [0.0], [0.0]
                for alpha in (0.25, 0.50, 0.75, 1.00):
                    item = lookup.get((subject, f"ATTEN|{target}|{alpha:.2f}", endpoint))
                    if item is not None:
                        x.append(alpha); y.append(base.all_contact_margin - item.all_contact_margin)
                if len(x) == 5:
                    auc_rows.append({"subject": subject, "target": target,
                                     "concordance_damage_auc": float(np.trapz(y, x))})
        auc = pd.DataFrame(auc_rows)
        for target in TARGET_BASE:
            values = auc[auc.target == target].concordance_damage_auc
            output["attenuation"].setdefault(endpoint, {})[f"{target}_damage_auc_gt_zero"] = paired_summary(
                values, seed=stable_seed(endpoint, target, "auc")
            )
        wide = auc.pivot(index="subject", columns="target", values="concordance_damage_auc")
        for other in ("L1_ADDED", "L2_ADDED", "L3_MATCHED_LOCAL"):
            values = wide["L3_ADDED"] - wide[other]
            output["attenuation"][endpoint][f"L3_ADDED_vs_{other}_damage_auc"] = paired_summary(
                values, seed=stable_seed(endpoint, other, "auc-diff")
            )
    return output


def plot_stage(patient: pd.DataFrame, summary: dict, out: Path) -> None:
    primary = patient[(patient.primary) & (patient.endpoint == "canonical_full")]
    intact = primary[primary.family == "intact"]
    pivot = intact.pivot(index="subject", columns="arm", values="all_contact_margin")
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.0))
    shown = ARMS[:4]
    for _, row in pivot.iterrows():
        axes[0].plot(range(4), row[list(shown)], color="#bfc5ca", lw=0.7, alpha=0.7)
    axes[0].scatter(range(4), [pivot[arm].median() for arm in shown], s=42,
                    color=["#8395a7", "#8aa85b", "#a970b5", "#c83e32"], zorder=3)
    axes[0].set_xticks(range(4), ["Local", "+extra", "+random", "+selected"], rotation=25, ha="right")
    axes[0].set_ylabel("Early-ictal margin")

    l3 = patient[(patient.primary) & (patient.family == "intact") &
                 (patient.arm == "L3_LOCAL_PLUS_LEARNED_LR")]
    for subject, group in l3.groupby("subject"):
        values = group.set_index("endpoint").all_contact_margin
        if set(ENDPOINTS) <= set(values.index):
            axes[1].plot([0, 1], [values["canonical_full"], values["seed_removed"]],
                         color="#bfc5ca", lw=0.7)
    axes[1].scatter([0, 1], [l3[l3.endpoint == endpoint].all_contact_margin.median()
                             for endpoint in ENDPOINTS], s=44, color=["#243f67", "#6c9ac4"])
    axes[1].set_xticks([0, 1], ["Full", "Start removed"])
    axes[1].set_ylabel("L3 early-ictal margin")

    attenuation = summary["attenuation"]["seed_removed"]
    names = ("L1_ADDED", "L2_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL")
    values = [attenuation[f"{name}_damage_auc_gt_zero"]["median"] for name in names]
    axes[2].bar(range(4), values, color=["#8395a7", "#a970b5", "#c83e32", "#2f6fa3"])
    axes[2].set_xticks(range(4), ["Extra", "Random", "Selected", "Local"], rotation=25, ha="right")
    axes[2].set_ylabel("Concordance damage AUC")
    for label, ax in zip("ABC", axes):
        ax.axhline(0, color="#777777", lw=0.6, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"stage_g_frozen_lbss_early_ictal.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    with (figures / "README.md").open("a") as stream:
        stream.write(
            "\n### stage_g_frozen_lbss_early_ictal.png\n\n"
            "A 比较四种冻结间期模型场的 early-ictal all-contact null-relative margin；B 分解完整场与去起点场；"
            "C 显示削弱各类边后 seed-removed 跨状态一致性的损害 AUC。\n\n"
            "**关注点**：只有 L3 超过匹配 arms，或削弱其 selected nonlocal edges 特异降低一致性，才支持 shortcut-specific 跨状态贡献。\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()
    out = args.out_root.resolve()
    authorization_path = out / "TARGET_UNSEAL_AUTHORIZATION.json"
    authorization = json.loads(authorization_path.read_text())
    if not authorization.get("authorized") or authorization["scorer_sha256"] != sha256(Path(__file__).resolve()):
        raise RuntimeError("target scorer is not the frozen authorized implementation")
    for name, digest in authorization["frozen_hashes"].items():
        if sha256(out / name) != digest:
            raise RuntimeError(f"frozen manifest changed after authorization: {name}")
    metadata = json.loads((out / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    target_root = Path(metadata["target_cache_root"])
    old_manifest = json.loads((OLD_ROOT / "MODEL_FIELD_MANIFEST.json").read_text())
    primary = list(metadata["actual_primary_join"])
    supportive = str(metadata["supportive_subject"])
    subjects = primary + ([supportive] if metadata.get("supportive_available") else [])
    early = out / "early_ictal"; early.mkdir(exist_ok=True)
    (early / "TARGET_UNLOCK_RECORD.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": sha256(authorization_path),
        "target_values_read_before_this_record": False,
        "target_values_unlocked_after_all_field_freezes": True,
        "target_key": "target_1_150", "window": "clinical onset 0-10 s",
        "band": "1-150 Hz broadband energy", "n_permutations": args.n_perm,
    }, indent=2) + "\n")

    seizure_rows, nulls = [], {}
    for subject in subjects:
        record = json.loads(Path(old_manifest["patient_geometry"][subject]["empirical_record"]).read_text())
        field = record["interictal_field"]
        order = [str(value) for value in field["contact_order"]]
        shafts = [str(value) for value in field["shafts"]]
        target_files = sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))
        for target_path in target_files:
            with np.load(target_path, allow_pickle=False) as data:  # only target-value access point
                names = data["contact_names"].astype(str).tolist()
                values = np.asarray(data["target_1_150"], float)
            target_lookup = dict(zip(names, values))
            target = np.asarray([target_lookup.get(name, np.nan) for name in order], float)
            seizure_id = target_path.stem.split("__", 1)[-1]
            for endpoint in ENDPOINTS:
                candidates = load_candidates(out, subject, endpoint, order)
                candidates["EMPIRICAL_REFERENCE"] = {
                    "family": "reference", "arm": "EMPIRICAL_REFERENCE", "target": "", "alpha": 0.0,
                    "a": np.asarray(field["earliness_a"], float),
                    "b": np.asarray(field["earliness_b"], float),
                }
                finite = np.isfinite(target) & np.isfinite(field["earliness_a"]) & np.isfinite(field["earliness_b"])
                eligible = np.flatnonzero(finite)
                if len(eligible) < 6:
                    raise RuntimeError(f"{subject} {seizure_id}: frozen support below 6")
                perm_all = permutation_indices(len(order), eligible, shafts, args.n_perm,
                    stable_seed(subject, seizure_id, endpoint, "all"), False)
                perm_shaft = permutation_indices(len(order), eligible, shafts, args.n_perm,
                    stable_seed(subject, seizure_id, endpoint, "shaft"), True)
                support = permutation_support(eligible, shafts)
                for condition, candidate in candidates.items():
                    scorer = build_scorer(record, candidate["a"], candidate["b"], finite)
                    all_score = score_one(scorer, target, perm_all)
                    shaft_score = score_one(scorer, target, perm_shaft)
                    common = 0.5 * (candidate["a"] + candidate["b"])
                    common_score = score_one(build_scorer(record, common, common, finite), target, perm_all)
                    prefix = f"{subject}|{seizure_id}|{condition}|{endpoint}"
                    keys = (prefix + "|all", prefix + "|shaft", prefix + "|common")
                    nulls[keys[0]] = all_score["null"]; nulls[keys[1]] = shaft_score["null"]
                    nulls[keys[2]] = common_score["null"]
                    seizure_rows.append({
                        "subject": subject, "seizure_id": seizure_id, "condition": condition,
                        "family": candidate["family"], "arm": candidate["arm"],
                        "target": candidate["target"], "alpha": candidate["alpha"], "endpoint": endpoint,
                        "n_contacts": len(eligible), "observed": all_score["observed"],
                        "all_contact_margin": all_score["margin"], "within_shaft_margin": shaft_score["margin"],
                        "common_observed": common_score["observed"],
                        "within_shaft_permutable_contacts": support["n_within_shaft_permutable_contacts"],
                        "null_key_all": keys[0], "null_key_shaft": keys[1], "null_key_common": keys[2],
                    })
    seizure = pd.DataFrame(seizure_rows)
    seizure.to_csv(early / "early_ictal_per_seizure.csv", index=False)
    patient = aggregate_patients(seizure, nulls, supportive)
    patient.to_csv(early / "early_ictal_per_patient_condition.csv", index=False)
    fidelity = pd.read_csv(out / "model_field_patient_metrics.csv")
    summary = summarize_claims(patient, fidelity)
    (early / "EARLY_ICTAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_stage(patient, summary, out)
    (out / "TARGET_ACCESS_AUDIT.json").write_text(json.dumps({
        "target_values_read": True, "training_or_model_selection_after_unseal": False,
        "n_primary_patients": len(primary), "n_primary_seizures": metadata["n_primary_seizures"],
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "primary_null": "synchronized all-contact label shuffle",
        "within_shaft_sensitivity": True,
        "support_rule": "fixed exact-joined contacts; missing seed-removed model scores assigned zero",
    }, indent=2) + "\n")
    (out / "EARLY_ICTAL_SCORING_COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "target_values_read": True,
        "n_seizure_rows": len(seizure), "n_patient_rows": len(patient),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
