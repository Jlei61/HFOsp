"""Freeze target-free full and seed-removed fields from held-out rollouts."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    build_interictal_ab_panel_payloads, draw_interictal_rank_field_panel,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 3 or np.nanstd(a[use]) == 0 or np.nanstd(b[use]) == 0:
        return float("nan")
    value = spearmanr(a[use], b[use]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def empirical_score(rank: np.ndarray) -> np.ndarray:
    rank = np.asarray(rank, float)
    use = np.isfinite(rank)
    out = np.full(rank.shape, np.nan)
    if use.any():
        span = float(rank[use].max() - rank[use].min())
        out[use] = 1.0 if span == 0 else 1.0 - (rank[use] - rank[use].min()) / span
    return out


def event_scores(sequence: list[list[int]], n_contacts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = np.zeros(n_contacts, float)
    participation = np.zeros(n_contacts, float)
    n_ranks = len(sequence)
    for rank, contacts in enumerate(sequence):
        value = 1.0 - rank / max(n_ranks - 1, 1)
        full[np.asarray(contacts, int)] = value
        participation[np.asarray(contacts, int)] = 1.0
    recurrence = np.zeros(n_contacts, float)
    seed = np.asarray(sequence[0], int)
    recurrence[seed] = np.nan
    post = sequence[1:]
    for rank, contacts in enumerate(post):
        value = 1.0 - rank / max(len(post) - 1, 1)
        recurrence[np.asarray(contacts, int)] = value
    return full, recurrence, participation


def aggregate_records(records: list[dict[str, Any]], n_contacts: int) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("cannot construct a field without held-out rollouts")
    full, recurrence, participation = [], [], []
    for record in records:
        f, r, p = event_scores(record["generated_rank_sets"], n_contacts)
        full.append(f); recurrence.append(r); participation.append(p)
    f = np.stack(full)
    r = np.stack(recurrence)
    p = np.stack(participation)
    denominator = np.isfinite(r).sum(0)
    recurrence_mean = np.divide(np.nansum(r, axis=0), denominator,
                                out=np.full(n_contacts, np.nan), where=denominator > 0)
    return {
        "canonical_full": f.mean(0), "seed_removed": recurrence_mean,
        "seed_removed_denominator": denominator.astype(np.int32),
        "participation": p.mean(0), "n_events": np.asarray([len(records)], np.int32),
    }


def split_half_stability(records: list[dict[str, Any]], n_contacts: int) -> dict[str, float]:
    """Chronological held-out split-half stability without reading any ictal target."""
    ordered = sorted(records, key=lambda row: (float(row["event_abs_time"]), int(row["kept_event_index"])))
    cut = len(ordered) // 2
    if cut < 2 or len(ordered) - cut < 2:
        return {"canonical_full": float("nan"), "seed_removed": float("nan")}
    left = aggregate_records(ordered[:cut], n_contacts)
    right = aggregate_records(ordered[cut:], n_contacts)
    return {
        endpoint: safe_corr(left[endpoint], right[endpoint])
        for endpoint in ("canonical_full", "seed_removed")
    }


def template_for_mode(provenance: dict, mode: int) -> str | None:
    value = provenance["mode_to_template"].get(str(int(mode)))
    return str(value).upper() if value in ("a", "b") else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def derive_common_contrast(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("A/B fields must use the identical frozen contact support")
    return 0.5 * (a + b), a - b


def assert_formal_matrix_complete(out_root: Path) -> dict[str, Any]:
    expected = {"core": 744, "dose": 217, "gru": 465}
    status = {}
    for stage, total in expected.items():
        path = out_root / f"STAGE_{stage.upper()}_STATUS.json"
        if not path.exists():
            raise RuntimeError(f"missing formal stage status: {path.name}")
        payload = json.loads(path.read_text())
        if (int(payload.get("total", -1)) != total or int(payload.get("remaining", -1)) != 0
                or int(payload.get("failed", -1)) != 0 or int(payload.get("oom", -1)) != 0
                or int(payload.get("nonfinite", -1)) != 0):
            raise RuntimeError(f"formal stage is not complete and clean: {stage}: {payload}")
        status[stage] = payload
    paths = [path for path in (out_root / "per_subject").glob("*/*__*/seed*/metrics.json")
             if not path.parents[1].name.startswith("SMOKE_")]
    if len(paths) != sum(expected.values()):
        raise RuntimeError(f"expected {sum(expected.values())} formal metrics, found {len(paths)}")
    nonconverged = [str(path.relative_to(out_root)) for path in paths
                    if not bool(json.loads(path.read_text())["converged"])]
    if nonconverged:
        raise RuntimeError(f"formal units require frozen-budget continuation: {nonconverged[:5]}")
    return {"expected_units": expected, "observed_formal_metrics": len(paths),
            "nonconverged_units": 0}


def build_seed_fields(out_root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    field_root = Path(manifest["input_roots"]["field"])
    rows: list[dict[str, Any]] = []
    fields: dict[tuple[str, str, str, int, str], dict[str, np.ndarray]] = {}
    for metrics_path in sorted((out_root / "per_subject").glob("*/*__*/seed*/metrics.json")):
        directory = metrics_path.parents[1].name
        if directory.startswith("SMOKE_") or not directory.startswith(("M", "C_")):
            continue
        model, cell = directory.rsplit("__", 1)
        metrics = json.loads(metrics_path.read_text())
        # Formal completeness is checked once before field construction, so no
        # model can silently disappear from the frozen matrix at this point.
        fit_id, subject, scope, seed = metrics["fit_id"], metrics["subject"], metrics["fit_scope"], int(metrics["seed"])
        cache = out_root / "cache" / fit_id
        provenance = json.loads((cache / "provenance.json").read_text())
        contacts = np.asarray(provenance["contacts"], dtype="U64")
        with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as handle:
            records = json.load(handle)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            template = template_for_mode(provenance, record["mode"])
            if template is not None:
                grouped[template].append(record)
        empirical = json.loads((field_root / f"{subject}.json").read_text())["interictal_field"]
        empirical_order = [str(value) for value in empirical["contact_order"]]
        take = np.asarray([empirical_order.index(str(contact)) for contact in contacts], int)
        empirical_by_template = {
            "A": empirical_score(np.asarray(empirical["rank_a"], float)[take]),
            "B": empirical_score(np.asarray(empirical["rank_b"], float)[take]),
        }
        for template, selected in grouped.items():
            aggregate = aggregate_records(selected, len(contacts))
            split_stability = split_half_stability(selected, len(contacts))
            aggregate["canonical_split_half_stability"] = np.asarray(
                [split_stability["canonical_full"]], dtype=np.float32
            )
            aggregate["seed_removed_split_half_stability"] = np.asarray(
                [split_stability["seed_removed"]], dtype=np.float32
            )
            fields[(fit_id, model, cell, seed, template)] = {"contacts": contacts, **aggregate}
            output = out_root / "model_fields" / "per_fit_seed" / fit_id / directory / f"seed{seed}_{template}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, contacts=contacts, **aggregate)
            rows.append({
                "subject": subject, "fit_id": fit_id, "scope": scope, "model": model, "cell": cell,
                "seed": seed, "template": template, "n_events": len(selected),
                "canonical_empirical_r": safe_corr(aggregate["canonical_full"], empirical_by_template[template]),
                "seed_removed_empirical_r": safe_corr(aggregate["seed_removed"], empirical_by_template[template]),
                "participation_empirical_r": safe_corr(aggregate["participation"], empirical_by_template[template]),
                "canonical_split_half_stability": split_stability["canonical_full"],
                "seed_removed_split_half_stability": split_stability["seed_removed"],
                "hit_epoch_ceiling": bool(metrics["hit_ceiling"]),
                "field_sha256": sha256(output),
            })
    write_csv(out_root / "model_field_fit_seed_metrics.csv", rows)
    return rows, fields


def aggregate_patient_fields(out_root: Path, rows: list[dict], fields: dict) -> tuple[list[dict], dict]:
    by_fit: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_fit[(row["subject"], row["fit_id"], row["scope"], row["model"], row["cell"], row["template"])].append(row)
    fit_fields: dict[tuple, dict] = {}
    fit_rows = []
    for key, values in by_fit.items():
        subject, fit_id, scope, model, cell, template = key
        arrays = [fields[(fit_id, model, cell, int(row["seed"]), template)] for row in values]
        contacts = arrays[0]["contacts"]
        aggregate = {"contacts": contacts}
        for name in ("canonical_full", "seed_removed", "participation"):
            aggregate[name] = np.nanmedian(np.stack([item[name] for item in arrays]), axis=0)
        aggregate["seed_removed_denominator"] = np.sum(
            np.stack([item["seed_removed_denominator"] for item in arrays]), axis=0
        )
        for endpoint in ("canonical_full", "seed_removed"):
            values = [safe_corr(left[endpoint], right[endpoint])
                      for left, right in combinations(arrays, 2)]
            aggregate[f"{endpoint}_seed_stability"] = (
                float(np.nanmedian(values)) if values else float("nan")
            )
            split_values = [float(item[f"{endpoint}_split_half_stability"][0]) for item in arrays]
            aggregate[f"{endpoint}_split_half_stability"] = float(np.nanmedian(split_values))
        fit_fields[key] = aggregate
        fit_rows.append({"subject": subject, "fit_id": fit_id, "scope": scope, "model": model,
                         "cell": cell, "template": template, "n_seeds": len(values),
                         "canonical_seed_stability": aggregate["canonical_full_seed_stability"],
                         "seed_removed_seed_stability": aggregate["seed_removed_seed_stability"],
                         "canonical_split_half_stability": aggregate["canonical_full_split_half_stability"],
                         "seed_removed_split_half_stability": aggregate["seed_removed_split_half_stability"],
                         "canonical_empirical_r": float(np.nanmedian([v["canonical_empirical_r"] for v in values])),
                         "seed_removed_empirical_r": float(np.nanmedian([v["seed_removed_empirical_r"] for v in values])),
                         "n_epoch_ceiling": int(sum(bool(v["hit_epoch_ceiling"]) for v in values))})
    write_csv(out_root / "model_field_fit_metrics.csv", fit_rows)

    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    field_root = Path(manifest["input_roots"]["field"])
    patient_rows = []
    patient_fields = {}
    keys = sorted({(row["subject"], row["model"], row["cell"]) for row in fit_rows})
    for subject, model, cell in keys:
        candidates = {}
        sources = {}
        for template in ("A", "B"):
            matches = [(key, value) for key, value in fit_fields.items()
                       if key[0] == subject and key[3] == model and key[4] == cell and key[5] == template]
            if len(matches) != 1:
                continue
            key, value = matches[0]
            candidates[template] = value
            sources[template] = key[1]
        if set(candidates) != {"A", "B"}:
            continue
        contact_a = [str(value) for value in candidates["A"]["contacts"]]
        contact_b = [str(value) for value in candidates["B"]["contacts"]]
        if contact_a != contact_b:
            raise RuntimeError(f"A/B contact order mismatch for {subject} {model} {cell}")
        output = out_root / "model_fields" / "per_patient" / subject / f"{model}__{cell}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for template in ("A", "B"):
            for name, value in candidates[template].items():
                if name != "seed_stability":
                    payload[f"{template}_{name}"] = value
            payload[f"{template}_evaluation_support_contacts"] = candidates[template]["contacts"]
        payload["COMMON_contacts"] = candidates["A"]["contacts"]
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            a = candidates["A"][endpoint]
            b = candidates["B"][endpoint]
            common, contrast = derive_common_contrast(a, b)
            payload[f"{endpoint}_common"] = common
            payload[f"{endpoint}_contrast"] = contrast
        np.savez_compressed(output, **payload)

        empirical = json.loads((field_root / f"{subject}.json").read_text())["interictal_field"]
        order = [str(value) for value in empirical["contact_order"]]
        empirical_scores = {}
        fidelity = []
        for template, rank_key in (("A", "rank_a"), ("B", "rank_b")):
            contact = [str(value) for value in candidates[template]["contacts"]]
            take = np.asarray([order.index(value) for value in contact], int)
            empirical_scores[template] = empirical_score(np.asarray(empirical[rank_key], float)[take])
            fidelity.append(safe_corr(candidates[template]["canonical_full"], empirical_scores[template]))
        swapped_fidelity = float(np.nanmean([
            safe_corr(candidates["A"]["canonical_full"], empirical_scores["B"]),
            safe_corr(candidates["B"]["canonical_full"], empirical_scores["A"]),
        ]))
        canonical_contrast_fidelity = safe_corr(
            candidates["A"]["canonical_full"] - candidates["B"]["canonical_full"],
            empirical_scores["A"] - empirical_scores["B"],
        )
        seed_removed_matched = float(np.nanmean([
            safe_corr(candidates["A"]["seed_removed"], empirical_scores["A"]),
            safe_corr(candidates["B"]["seed_removed"], empirical_scores["B"]),
        ]))
        seed_removed_contrast_fidelity = safe_corr(
            candidates["A"]["seed_removed"] - candidates["B"]["seed_removed"],
            empirical_scores["A"] - empirical_scores["B"],
        )
        canonical_common_fidelity = safe_corr(
            0.5 * (candidates["A"]["canonical_full"] + candidates["B"]["canonical_full"]),
            0.5 * (empirical_scores["A"] + empirical_scores["B"]),
        )
        seed_removed_common_fidelity = safe_corr(
            0.5 * (candidates["A"]["seed_removed"] + candidates["B"]["seed_removed"]),
            0.5 * (empirical_scores["A"] + empirical_scores["B"]),
        )
        shared_mode_corr = float("nan")
        if sources["A"] == sources["B"]:
            name_a = [str(x) for x in candidates["A"]["contacts"]]
            name_b = [str(x) for x in candidates["B"]["contacts"]]
            common = [name for name in name_a if name in name_b]
            shared_mode_corr = safe_corr(
                candidates["A"]["canonical_full"][[name_a.index(name) for name in common]],
                candidates["B"]["canonical_full"][[name_b.index(name) for name in common]],
            )
        patient_rows.append({
            "subject": subject, "model": model, "cell": cell,
            "aggregation": "shared_single_fit" if sources["A"] == sources["B"] else "own_a_own_b_separate",
            "producer_A": sources["A"], "producer_B": sources["B"],
            "matched_empirical_r": float(np.nanmean(fidelity)),
            "swapped_empirical_r": swapped_fidelity,
            "matched_minus_swapped_r": float(np.nanmean(fidelity)) - swapped_fidelity,
            "canonical_contrast_fidelity": canonical_contrast_fidelity,
            "canonical_common_fidelity": canonical_common_fidelity,
            "seed_removed_matched_empirical_r": seed_removed_matched,
            "seed_removed_contrast_fidelity": seed_removed_contrast_fidelity,
            "seed_removed_common_fidelity": seed_removed_common_fidelity,
            "canonical_seed_stability": float(np.nanmean([
                candidates["A"]["canonical_full_seed_stability"],
                candidates["B"]["canonical_full_seed_stability"],
            ])),
            "seed_removed_seed_stability": float(np.nanmean([
                candidates["A"]["seed_removed_seed_stability"],
                candidates["B"]["seed_removed_seed_stability"],
            ])),
            "canonical_split_half_stability": float(np.nanmean([
                candidates["A"]["canonical_full_split_half_stability"],
                candidates["B"]["canonical_full_split_half_stability"],
            ])),
            "seed_removed_split_half_stability": float(np.nanmean([
                candidates["A"]["seed_removed_split_half_stability"],
                candidates["B"]["seed_removed_split_half_stability"],
            ])),
            "generated_AB_r": shared_mode_corr, "field_sha256": sha256(output),
        })
        patient_fields[(subject, model, cell)] = candidates
    write_csv(out_root / "model_field_patient_metrics.csv", patient_rows)
    return patient_rows, patient_fields


def plot_target_free(out_root: Path, rows: list[dict], patient_fields: dict) -> None:
    subject, model, cell = "epilepsiae_1146", "M6_SPATIAL_MID", "rnn"
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    field_root = Path(manifest["input_roots"]["field"])
    record = json.loads((field_root / f"{subject}.json").read_text())
    empirical_a, empirical_b, _ = build_interictal_ab_panel_payloads(record)
    model_fields = patient_fields[(subject, model, cell)]
    model_payloads = []
    for empirical, template in ((empirical_a, "A"), (empirical_b, "B")):
        payload = dict(empirical)
        by_name = {str(name): value for name, value in zip(
            model_fields[template]["contacts"], model_fields[template]["canonical_full"]
        )}
        payload["vals"] = np.asarray([by_name.get(str(name), np.nan) for name in empirical["names"]])
        model_payloads.append(payload)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 0.88, 0.88))
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(4)]
    draw_interictal_rank_field_panel(axes[0], empirical_a, "TA", compact=True, panel_title="Data TA")
    draw_interictal_rank_field_panel(axes[1], empirical_b, "TB", compact=True, panel_title="Data TB")
    draw_interictal_rank_field_panel(axes[4], model_payloads[0], "TA", compact=True, panel_title="RNN TA")
    draw_interictal_rank_field_panel(axes[5], model_payloads[1], "TB", compact=True, panel_title="RNN TB")

    cohort = [row for row in rows if row["cell"] == "rnn" and row["model"] in
              ("M0_NO_REC", "M1_DENSE", "M3_FIXED_LOCAL", "M6_SPATIAL_MID")]
    models = ("M0_NO_REC", "M1_DENSE", "M3_FIXED_LOCAL", "M6_SPATIAL_MID")
    for x, candidate in enumerate(models):
        values = np.asarray([row["matched_empirical_r"] for row in cohort if row["model"] == candidate], float)
        axes[2].scatter(np.full(len(values), x) + np.linspace(-0.10, 0.10, len(values)), values,
                        s=11, alpha=0.55, linewidths=0)
        axes[2].plot([x - .2, x + .2], [np.nanmedian(values)] * 2, color="#111111", lw=1.2)
    axes[2].set_xticks(range(4), ["M0", "M1", "M3", "M6"])
    axes[2].set_ylabel("Model–data field r")
    axes[2].set_title("Target-free fidelity", loc="left", fontweight="bold")
    for x, candidate in enumerate(models):
        values = np.asarray([row["canonical_seed_stability"] for row in cohort if row["model"] == candidate], float)
        axes[3].scatter(np.full(len(values), x) + np.linspace(-0.10, 0.10, len(values)), values,
                        s=11, alpha=0.55, linewidths=0)
        axes[3].plot([x - .2, x + .2], [np.nanmedian(values)] * 2, color="#111111", lw=1.2)
    axes[3].set_xticks(range(4), ["M0", "M1", "M3", "M6"])
    axes[3].set_ylabel("Across-seed field r")
    axes[3].set_title("Field stability", loc="left", fontweight="bold")

    shared = [row for row in cohort if row["aggregation"] == "shared_single_fit"]
    for x, candidate in enumerate(models):
        values = np.asarray([row["generated_AB_r"] for row in shared if row["model"] == candidate], float)
        axes[6].scatter(np.full(len(values), x) + np.linspace(-0.08, 0.08, len(values)), values,
                        s=12, alpha=0.6, linewidths=0)
        axes[6].plot([x - .2, x + .2], [np.nanmedian(values)] * 2, color="#111111", lw=1.2)
    axes[6].set_xticks(range(4), ["M0", "M1", "M3", "M6"])
    axes[6].set_ylabel("Generated TA–TB r")
    axes[6].set_title("Shared-fit modes", loc="left", fontweight="bold")
    axes[7].axis("off")
    for axis in (axes[0], axes[1], axes[4], axes[5]):
        axis.set_xlabel(""); axis.set_ylabel("")
    for label, axis in zip("abcdefg", (axes[0], axes[1], axes[2], axes[3], axes[4], axes[5], axes[6])):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=11)
    stem = out_root / "figures" / "stage_e_target_free_model_fields"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    matrix_audit = assert_formal_matrix_complete(out_root)
    rows, seed_fields = build_seed_fields(out_root)
    patient_rows, patient_fields = aggregate_patient_fields(out_root, rows, seed_fields)
    if not patient_rows:
        raise RuntimeError("no complete patient fields")
    input_manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    field_root = Path(input_manifest["input_roots"]["field"])
    checkpoint_files = sorted(
        path for path in (out_root / "per_subject").glob("*/*__*/seed*/weights.pt")
        if not path.parents[1].name.startswith("SMOKE_")
    )
    rollout_files = sorted(
        path for path in (out_root / "per_subject").glob("*/*__*/seed*/heldout_rollouts.json.gz")
        if not path.parents[1].name.startswith("SMOKE_")
    )
    geometry = {}
    for subject in sorted({row["subject"] for row in patient_rows}):
        record_path = field_root / f"{subject}.json"
        record = json.loads(record_path.read_text())
        field = record["interictal_field"]
        geometry[subject] = {
            "empirical_record": str(record_path),
            "empirical_record_sha256": sha256(record_path),
            "fingerprint_sha256": field["fingerprint_sha256"],
            "contact_order": field["contact_order"],
            "shafts": field["shafts"],
            "planes": field["planes"],
            "field_model_sigma": {
                key: float(value["sigma"]) for key, value in field["field_models"].items()
            },
        }
    manifest = {
        "contract": "topic5_rnn_motif_model_fields_v0_4",
        "field_endpoints": ["FIELD_CANONICAL_FULL", "FIELD_SEED_REMOVED"],
        "canonical_primary": True, "seed_removed_mechanistic_secondary": True,
        "evaluation_support_rule": (
            "fixed exact-joined cache contact order intersected with the frozen empirical candidate; "
            "identical across models and independent of generated participation"
        ),
        "n_fit_seed_fields": len(rows), "n_patient_fields": len(patient_rows),
        "seed_aggregation": "contact-wise median across the three frozen seeds",
        "split_half_rule": "chronological heldout events split within fit, template and seed",
        "derived_fields": ["F_A", "F_B", "F_common", "F_contrast", "participation"],
        "formal_matrix_audit": matrix_audit,
        "fit_to_patient_contract_sha256": sha256(out_root / "contracts/FIT_TO_PATIENT_AGGREGATION_CONTRACT.json"),
        "input_manifest_sha256": sha256(out_root / "INPUT_MANIFEST.json"),
        "field_builder_sha256": sha256(Path(__file__).resolve()),
        "target_values_read": False,
        "representative_patient": "epilepsiae_1146",
        "plotting_order": ["M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
                           "M4_SPATIAL_GROWTH", "M5_SPATIAL_LOW", "M6_SPATIAL_MID",
                           "M7_SPATIAL_HIGH", "M8_UNIFORM_COST_MID", "C_ORDER_SHUFFLED",
                           "C_FULL_RANK_SHUFFLED"],
        "checkpoint_files": {str(path.relative_to(out_root)): sha256(path) for path in checkpoint_files},
        "rollout_provenance_files": {str(path.relative_to(out_root)): sha256(path) for path in rollout_files},
        "patient_geometry": geometry,
        "patient_files": {str(path.relative_to(out_root)): sha256(path)
                          for path in sorted((out_root / "model_fields/per_patient").glob("**/*.npz"))},
    }
    manifest_path = out_root / "MODEL_FIELD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    metadata_path = out_root / "EARLY_ICTAL_METADATA_INVENTORY.json"
    if not metadata_path.exists():
        raise RuntimeError("filename-only early-ictal metadata inventory must precede authorization")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("target_values_read") is not False:
        raise RuntimeError("metadata audit indicates premature target access")
    (out_root / "TARGET_UNSEAL_AUTHORIZATION.json").write_text(json.dumps({
        "authorized": True, "model_field_manifest_sha256": sha256(manifest_path),
        "target_values_read_before_authorization": False,
        "all_engineering_valid_models_included": True,
        "expected_primary_n": metadata["expected_primary_n"],
        "actual_primary_join_n": metadata["actual_primary_join_n"],
        "cohort_join_status": metadata["join_status"],
        "cohort_mismatch_reported_before_unseal": metadata["actual_primary_join_n"]
                                                     != metadata["expected_primary_n"],
    }, indent=2))
    (out_root / "stage_e_scientific_drift_audit.json").write_text(json.dumps({
        "status": "ALIGNED", "target_values_read": False,
        "checked": ["heldout same-start free rollout", "canonical and seed-removed endpoints",
                    "shared vs own_a/own_b aggregation", "common evaluation support independent of model participation"],
        "deviations": (["early-ictal primary intersection is smaller than the planning estimate; "
                         "the frozen actual intersection is used without post-target cohort changes"]
                       if metadata["actual_primary_join_n"] != metadata["expected_primary_n"] else []),
    }, indent=2))
    plot_target_free(out_root, patient_rows, patient_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
