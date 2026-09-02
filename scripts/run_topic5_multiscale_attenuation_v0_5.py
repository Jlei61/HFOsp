#!/usr/bin/env python3
"""Target-free arm-specific attenuation and field freeze for v0.5."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_rnn_motif_fields_v0_4 import aggregate_records, derive_common_contrast  # noqa: E402
from build_topic5_multiscale_fields_v0_5 import (  # noqa: E402
    remap_record_modes, sha256_file, train_mode_to_ab, vector_sha256,
)
from run_topic5_lbss_attenuation_v0_2 import evaluate_variant  # noqa: E402
from src.topic5_lbss_analysis_v0_2 import (  # noqa: E402
    attenuate_mask, mask_sha256, match_local_control_subsets,
)
from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract  # noqa: E402
from src.topic5_rnn_motif_v0_4 import RolloutSizeHead  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
TARGETS = {
    "L1_ADDED": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2M_ADDED": "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3_ADDED": "L3_LOCAL_PLUS_LEARNED_LR",
    "L3_MATCHED_LOCAL": "L3_LOCAL_PLUS_LEARNED_LR",
}


def strict_seed_removed_rollout(ranks: np.ndarray, records: list[dict]) -> tuple[float, int]:
    """Accepted post-seed rollout correlation; rank 0 is never credited."""
    values = []
    for record in records:
        event_index = int(record["kept_event_index"])
        observed = np.asarray(ranks[event_index], dtype=int)
        generated_order = {
            int(contact): rank
            for rank, rank_set in enumerate(record["generated_rank_sets"][1:])
            for contact in rank_set
        }
        shared = [
            int(contact) for contact in np.flatnonzero(observed > 0)
            if int(contact) in generated_order
        ]
        if len(shared) < 3:
            continue
        observed_rank = np.asarray([observed[contact] - 1 for contact in shared], float)
        generated_rank = np.asarray([generated_order[contact] for contact in shared], float)
        if np.unique(observed_rank).size < 2 or np.unique(generated_rank).size < 2:
            continue
        value = spearmanr(observed_rank, generated_rank).statistic
        if np.isfinite(value):
            values.append(float(value))
    return (float(np.median(values)) if values else float("nan"), len(values))


def metrics_paths(out: Path, old: Path) -> list[tuple[Path, str]]:
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    paths = []
    for fit_id in census.fit_id.astype(str):
        for target, arm in TARGETS.items():
            for seed in range(3):
                if fit_id in reused and arm in {
                    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", "L3_LOCAL_PLUS_LEARNED_LR"
                }:
                    path = old / "per_fit" / fit_id / arm / f"seed{seed}" / "metrics.json"
                else:
                    path = out / "formal_units" / fit_id / arm / f"seed{seed}" / "metrics.json"
                paths.append((path, target))
    if len(paths) != 504:
        raise RuntimeError(f"expected 504 attenuation unit-targets, found {len(paths)}")
    return paths


def instantiate(out: Path, metrics_path: Path, device: torch.device):
    metrics = json.loads(metrics_path.read_text())
    cache = out / "cache" / metrics["fit_id"]
    plane_npz = np.load(cache / "plane.npz", allow_pickle=False)
    events_npz = np.load(cache / "events.npz", allow_pickle=False)
    plane = {key: plane_npz[key] for key in plane_npz.files}
    events = {key: events_npz[key] for key in events_npz.files}
    provenance = json.loads((cache / "provenance.json").read_text())
    graph = np.load(metrics_path.parent / "graph.npz", allow_pickle=False)
    cfg = metrics["config"]
    pools = build_pool_contract(
        plane["D_mm"], cfg["density"], cfg["added_fraction"], cfg.get("r_local_multiplier", 2.0)
    )
    fixed = graph["added_mask"] if metrics["arm"] == "L2M_MACRO_MATCHED_RANDOM_LR" else None
    n_contacts = int(provenance["n_joint_contacts"])
    model = LBSSModel(LBSSConfig(
        arm=metrics["arm"], n_contacts=n_contacts, n_nodes=int(provenance["n_nodes"]),
        observation_operator=plane["H"], node_distance_mm=plane["D_mm"],
        local_mask=pools.local_mask, extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool, k_added=pools.k_added,
        seed=int(metrics["seed"]), state_dim=int(cfg["state_dim"]), fixed_added_mask=fixed,
    )).to(device)
    model.load_state_dict(torch.load(metrics_path.parent / "weights.pt", map_location=device, weights_only=True))
    model.freeze_mask(); model.eval()
    decoder = RolloutSizeHead(n_contacts).to(device)
    decoder.load_state_dict(torch.load(metrics_path.parent / "rollout_size_head.pt", map_location=device, weights_only=True))
    decoder.eval()
    provenance = dict(provenance)
    # v0.5's mechanistic estimand is local (<= r_local) versus nonlocal
    # (> r_local).  The inherited q50/q80 bins remain descriptive only.
    provenance["distance_thresholds_mm"] = (
        float(pools.r_local_mm), float(pools.r_local_mm),
    )
    provenance["r_local_mm"] = float(pools.r_local_mm)
    return model, decoder, metrics, plane, events, provenance, graph


def local_controls(out: Path, metrics: dict, plane: dict, graph) -> dict:
    seed = int.from_bytes(hashlib.sha256(
        f"{metrics['fit_id']}|{metrics['seed']}|v05-local-control".encode()
    ).digest()[:4], "little")
    matched = match_local_control_subsets(
        graph["local_mask"], graph["added_mask"], graph["strength"],
        plane["nodes_xy_mm"], plane["H"], seed=seed,
    )
    root = out / "attenuation/matched_local" / metrics["fit_id"] / f"seed{metrics['seed']}"
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "selected_masks.npz", masks=matched["selected_masks"],
                        scores=matched["selected_scores"])
    serializable = {key: value for key, value in matched.items()
                    if key not in {"selected_masks", "selected_scores"}}
    (root / "match.json").write_text(json.dumps(serializable, indent=2) + "\n")
    return matched


def aggregate_draw_fields(records_by_draw: list[list[dict]], cache: Path, provenance: dict,
                          n_contacts: int) -> dict[str, dict[str, np.ndarray]]:
    records_by_draw = [remap_record_modes(records, cache) for records in records_by_draw]
    if provenance["scope"] == "shared":
        mapping = train_mode_to_ab(cache, provenance["subject"],
                                   np.asarray(provenance["joint_contacts"]), FIELD_ROOT)
    else:
        template = "A" if provenance["scope"] == "own_a" else "B"
        mapping = {0: template, 1: template}
    output = {}
    for template in ("A", "B"):
        draw_fields = []
        for records in records_by_draw:
            selected = [row for row in records if mapping[int(row["mode"])] == template]
            if selected:
                draw_fields.append(aggregate_records(selected, n_contacts))
        if not draw_fields:
            continue
        output[template] = {}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            output[template][endpoint] = np.nanmedian(
                np.stack([item[endpoint] for item in draw_fields]), axis=0
            )
        output[template]["seed_removed_denominator"] = np.nanmedian(
            np.stack([item["seed_removed_denominator"] for item in draw_fields]), axis=0
        ).astype(np.int32)
    return output


def unit_cache(out: Path, metrics_path: Path, target: str) -> Path:
    metrics = json.loads(metrics_path.read_text())
    return out / "attenuation/unit_cache" / metrics["fit_id"] / target / f"seed{metrics['seed']}.json.gz"


def evaluate_unit(out: Path, metrics_path: Path, target: str, device: torch.device) -> tuple[list[dict], list[dict]]:
    cache_path = unit_cache(out, metrics_path, target)
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("metrics_sha256") == sha256_file(metrics_path):
            return payload["metric_rows"], payload["field_rows"]
    model, decoder, metrics, plane, events, provenance, graph = instantiate(out, metrics_path, device)
    base = model.recurrent.detach().clone()
    intact_decisions = json.loads((metrics_path.parent / "distance_decisions.json").read_text())
    r_local = float(provenance["r_local_mm"])
    intact_local = [row["contact_nll"] for row in intact_decisions
                    if np.isfinite(row["frontier_distance_mm"])
                    and row["frontier_distance_mm"] <= r_local]
    intact_distal = [row["contact_nll"] for row in intact_decisions
                     if np.isfinite(row["frontier_distance_mm"])
                     and row["frontier_distance_mm"] > r_local]
    keep = events["split"] >= 0
    ranks = events["ranks"][keep]
    with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as stream:
        intact_records = json.load(stream)
    intact_rollout, intact_rollout_n = strict_seed_removed_rollout(ranks, intact_records)
    intact = {
        "intact_contact_nll": float(metrics["test"]["contact_nll"]),
        "intact_local_nll": float(np.mean(intact_local)) if intact_local else float("nan"),
        "intact_distal_nll": float(np.mean(intact_distal)) if intact_distal else float("nan"),
        "intact_rollout_spearman": intact_rollout,
        "intact_rollout_spearman_n": int(intact_rollout_n),
        "intact_rollout_spearman_legacy_includes_seed": float(
            metrics["rollout"]["seed_removed_spearman_median"]
        ),
        "r_local_mm": r_local,
    }
    if target == "L3_MATCHED_LOCAL":
        matched = local_controls(out, metrics, plane, graph)
        target_masks = [mask.astype(bool) for mask in matched["selected_masks"]]
        target_hashes = list(matched["selected_hashes"])
        eligible = bool(matched["inferential_eligible"])
        valid_draws = int(matched["n_valid_matched_draws"])
    else:
        target_masks = [graph["added_mask"].astype(bool)]
        target_hashes = [mask_sha256(target_masks[0])]
        eligible, valid_draws = True, 1
    metric_rows, field_rows = [], []
    if not target_masks:
        for alpha in ALPHAS:
            metric_rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"],
                "scope": metrics["scope"], "arm": metrics["arm"], "seed": metrics["seed"],
                "target": target, "alpha": alpha, "draw": -1,
                **intact, "contact_nll": np.nan, "local_nll": np.nan, "distal_nll": np.nan,
                "rollout_spearman": np.nan, "distal_selectivity": np.nan,
                "n_valid_matched_draws": valid_draws, "inferential_eligible": False,
                "target_values_read": False,
            })
    for alpha in ALPHAS:
        records_by_draw = []
        for draw, mask in enumerate(target_masks):
            with torch.no_grad(): model.recurrent.copy_(base)
            attenuate_mask(model, mask, alpha)
            result, records = evaluate_variant(model, decoder, events, provenance, plane, device)
            result["rollout_spearman"], result["rollout_spearman_n"] = (
                strict_seed_removed_rollout(ranks, records)
            )
            records_by_draw.append(records)
            local_damage = result["local_nll"] - intact["intact_local_nll"]
            distal_damage = result["distal_nll"] - intact["intact_distal_nll"]
            metric_rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"],
                "scope": metrics["scope"], "arm": metrics["arm"], "seed": metrics["seed"],
                "target": target, "alpha": alpha, "draw": draw,
                **intact, **result, "local_damage": local_damage,
                "distal_damage": distal_damage,
                "distal_selectivity": distal_damage - local_damage,
                "target_mask_sha256": target_hashes[draw],
                "n_valid_matched_draws": valid_draws, "inferential_eligible": eligible,
                "target_values_read": False,
            })
        fields = aggregate_draw_fields(
            records_by_draw, out / "cache" / metrics["fit_id"], provenance,
            int(provenance["n_joint_contacts"]),
        )
        for template, payload in fields.items():
            destination = (out / "attenuation/fields/per_fit_seed" / metrics["fit_id"] /
                           target / f"seed{metrics['seed']}_alpha{alpha:.2f}_{template}.npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, contacts=np.asarray(provenance["joint_contacts"], dtype="U64"), **payload)
            field_rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"],
                "scope": metrics["scope"], "arm": metrics["arm"], "seed": metrics["seed"],
                "target": target, "alpha": alpha, "template": template,
                "path": str(destination), "field_sha256": sha256_file(destination),
                "n_control_draws": len(target_masks), "n_valid_matched_draws": valid_draws,
                "inferential_eligible": eligible, "target_values_read": False,
            })
    with torch.no_grad(): model.recurrent.copy_(base)
    payload = {
        "metrics_sha256": sha256_file(metrics_path), "metric_rows": metric_rows,
        "field_rows": field_rows, "target_values_read": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=True)
    os.replace(temporary, cache_path)
    return metric_rows, field_rows


def worker(item: tuple[str, str, str, str]):
    out, path, target, device = item
    torch.set_num_threads(2)
    return evaluate_unit(Path(out), Path(path), target, torch.device(device))


def aggregate_fields(out: Path, field_frame: pd.DataFrame) -> pd.DataFrame:
    fit_fields = {}
    for key, group in field_frame.groupby(
        ["subject", "fit_id", "scope", "arm", "target", "alpha", "template"], sort=False
    ):
        arrays = [np.load(path, allow_pickle=False) for path in group.path]
        payload = {"contacts": arrays[0]["contacts"]}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            payload[endpoint] = np.nanmedian(np.stack([item[endpoint] for item in arrays]), axis=0)
        payload["seed_removed_denominator"] = np.sum(
            np.stack([item["seed_removed_denominator"] for item in arrays]), axis=0
        )
        fit_fields[key] = payload
    rows = []
    for subject in sorted({key[0] for key in fit_fields}):
        for target, arm in TARGETS.items():
            for alpha in ALPHAS:
                candidates = {}
                for template in ("A", "B"):
                    matches = [(key, value) for key, value in fit_fields.items()
                               if key[0] == subject and key[3] == arm and key[4] == target
                               and np.isclose(key[5], alpha) and key[6] == template]
                    if len(matches) != 1:
                        continue
                    candidates[template] = matches[0][1]
                if set(candidates) != {"A", "B"}:
                    continue
                contacts = candidates["A"]["contacts"]
                payload = {"contacts": contacts}
                for template in ("A", "B"):
                    for endpoint in ("canonical_full", "seed_removed", "participation", "seed_removed_denominator"):
                        payload[f"{template}_{endpoint}"] = candidates[template][endpoint]
                for endpoint in ("canonical_full", "seed_removed", "participation"):
                    common, contrast = derive_common_contrast(
                        candidates["A"][endpoint], candidates["B"][endpoint]
                    )
                    payload[f"{endpoint}_common"] = common
                    payload[f"{endpoint}_contrast"] = contrast
                destination = out / "attenuation/fields/per_patient" / subject / target / f"alpha{alpha:.2f}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **payload)
                for endpoint in (
                    "A_canonical_full", "B_canonical_full", "canonical_full_common", "canonical_full_contrast",
                    "A_seed_removed", "B_seed_removed", "seed_removed_common", "seed_removed_contrast",
                ):
                    vector = np.asarray(payload[endpoint])
                    rows.append({
                        "subject": subject, "arm": arm, "target": target, "alpha": alpha,
                        "endpoint": endpoint, "path": str(destination),
                        "file_sha256": sha256_file(destination), "vector_sha256": vector_sha256(vector),
                        "n_contacts": len(vector), "target_values_read": False,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "ATTENUATED_FIELD_MANIFEST.csv", index=False)
    return frame


def aggregate_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["local_damage", "distal_damage", "distal_selectivity", "contact_nll",
               "rollout_spearman"]
    draw_keys = ["subject", "fit_id", "scope", "target", "alpha", "seed"]
    draw = frame.groupby(draw_keys, as_index=False).agg(
        **{name: (name, "median") for name in numeric},
        inferential_eligible=("inferential_eligible", "all"),
        n_valid_matched_draws=("n_valid_matched_draws", "min"),
    )
    seed_keys = ["subject", "fit_id", "scope", "target", "alpha"]
    seed = draw.groupby(seed_keys, as_index=False).agg(
        **{name: (name, "median") for name in numeric},
        inferential_eligible=("inferential_eligible", "all"),
        n_valid_matched_draws=("n_valid_matched_draws", "min"),
    )
    patient = seed.groupby(["subject", "target", "alpha"], as_index=False).agg(
        **{name: (name, "mean") for name in numeric},
        inferential_eligible=("inferential_eligible", "all"),
        n_valid_matched_draws=("n_valid_matched_draws", "min"),
    )
    auc_rows = []
    for (subject, target), group in patient.groupby(["subject", "target"]):
        group = group.sort_values("alpha")
        x = np.r_[0.0, group.alpha.to_numpy()]
        auc_rows.append({
            "subject": subject, "target": target,
            "inferential_eligible": bool(group.inferential_eligible.all()),
            "n_valid_matched_draws": int(group.n_valid_matched_draws.min()),
            **{f"auc_{name}": float(np.trapz(np.r_[0.0, group[name].to_numpy()], x=x))
               for name in ("local_damage", "distal_damage", "distal_selectivity")},
        })
    return patient, pd.DataFrame(auc_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if not (out / "MODEL_FIELDS_FROZEN.json").exists():
        raise RuntimeError("intact fields must be frozen before attenuation")
    jobs = [(str(out), str(path), target, args.device)
            for path, target in metrics_paths(out, old)]
    metric_rows, field_rows = [], []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 8)) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            metrics, fields = future.result()
            metric_rows.extend(metrics); field_rows.extend(fields)
            if index % 20 == 0:
                print(json.dumps({"completed": index, "total": len(jobs)}), flush=True)
    metrics = pd.DataFrame(metric_rows)
    fields = pd.DataFrame(field_rows)
    metrics.to_csv(out / "ATTENUATION_PER_DRAW.csv", index=False)
    fields.to_csv(out / "ATTENUATION_FIT_FIELD_INDEX.csv", index=False)
    patient, auc = aggregate_metrics(metrics)
    patient.to_csv(out / "ATTENUATION_PER_PATIENT_DOSE.csv", index=False)
    auc.to_csv(out / "ATTENUATION_PER_PATIENT_AUC.csv", index=False)
    manifest = aggregate_fields(out, fields)
    marker = {
        "status": "FROZEN_TARGET_FREE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False, "unit_targets": len(jobs),
        "field_vectors": len(manifest), "manifest_sha256": sha256_file(out / "ATTENUATED_FIELD_MANIFEST.csv"),
        "eligible_local_control_patients": int(metrics.loc[
            metrics.target == "L3_MATCHED_LOCAL"
        ].groupby("subject").inferential_eligible.all().sum()),
    }
    (out / "ATTENUATED_FIELDS_FROZEN.json").write_text(json.dumps(marker, indent=2) + "\n")


if __name__ == "__main__":
    main()
