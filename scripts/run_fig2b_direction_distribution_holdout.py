#!/usr/bin/env python3
"""Run the frozen Figure 2B marginal direction-density holdout supplement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_topic5_interictal_template_direction_rose import _subject_dir
from scripts.run_timing_plus_space_rank_readback import _alternating_block_folds
from src.interictal_direction_distribution_holdout import (
    METHODS, KAPPA_CAP, LOG_SPHERE_AREA, evaluate_models, fit_template_density,
    fit_template_model, mixture_nll, shuffle_training_directions, single_direction_density,
)
from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src.seeg_coord_loader import load_subject_coords
from src.topic5_interictal_direction_rose import fit_event_directions_3d

SOURCE = ROOT / "results/interictal_propagation_masked/spatial_information_gain_all_events"
OUT = ROOT / "results/topic1/direction_distribution_holdout_20261022"
SEED = 20261022


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n")


def run_patient(subject_id, out, n_shuffles):
    with threadpool_limits(limits=1):
        return _run_patient(subject_id, Path(out), n_shuffles)


def _run_patient(subject_id, out, n_shuffles):
    dataset, subject = subject_id.split("_", 1)
    source_dir = _subject_dir(dataset, subject)
    events = load_subject_propagation_events(source_dir)
    ranks, bools, blocks = events["ranks"], events["bools"], events["block_ids"]
    names = list(map(str, events["channel_names"]))
    coords = np.asarray(load_subject_coords(dataset, subject, names).coords_array_in_requested_order)
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    direction_fit = fit_event_directions_3d(masked, coords, min_contacts=3)
    directions = direction_fit["directions"]
    splits = _alternating_block_folds(blocks)
    unique = np.unique(blocks)
    chronological = (np.flatnonzero(np.isin(blocks, unique[:len(unique)//2])),
                     np.flatnonzero(np.isin(blocks, unique[len(unique)//2:])))
    old = json.loads((SOURCE / "per_subject" / f"{subject_id}.json").read_text())
    if len(blocks) != old["n_events_total"] or len(names) != old["n_channels"]:
        raise ValueError("source event/channel universe drifted from current Figure 2B")
    folder = out / "per_subject"
    folder.mkdir(parents=True, exist_ok=True)
    fold_results, arrays, controls, control_failures = [], {}, [], []
    rng = np.random.default_rng(SEED + zlib.crc32(subject_id.encode()))
    for f, (train, test) in enumerate(splits + [chronological]):
        if set(blocks[train]) & set(blocks[test]):
            raise AssertionError("training/test block leakage")
        models = {m: fit_template_model(ranks[:, train], bools[:, train], directions[train],
                                        coords, method=m) for m in METHODS}
        # Label matching only for plotting; no outcomes enter this operation.
        if np.mean(models[METHODS[0]]["labels"] != models[METHODS[1]]["labels"]) > .5:
            h = models[METHODS[1]]
            h["labels"] = 1 - h["labels"]
            for key in ("templates", "axes", "train_counts"):
                h[key] = h[key][::-1].copy()
        densities = {m: fit_template_density(models[m], directions[train]) for m in METHODS}
        evaluated = evaluate_models(models, densities, ranks[:, test], bools[:, test], directions[test])
        finite, common = evaluated["finite_mask"], evaluated["common_mask"]
        baseline = single_direction_density(directions[train])
        row = dict(fold=f, split="alternating" if f < 2 else "chronological",
                   n_train=int(len(train)), n_test=int(len(test)),
                   n_direction_test=int(finite.sum()), n_common_test=int(common.sum()),
                   train_blocks=np.unique(blocks[train]), test_blocks=np.unique(blocks[test]),
                   single_vmf_nll=float(mixture_nll(baseline, directions[test][finite]).mean()),
                   uniform_nll=LOG_SPHERE_AREA, metrics=evaluated["metrics"],
                   models={m: {k: v for k, v in models[m].items() if k != "labels"} for m in METHODS},
                   densities=densities)
        fold_results.append(row)
        for m in METHODS:
            arrays[f"f{f}_{m}_nll"] = evaluated["event_losses"][m]["marginal_nll"]
            arrays[f"f{f}_{m}_conditional_nll"] = evaluated["event_losses"][m]["conditional_nll"]
            arrays[f"f{f}_{m}_assignments"] = evaluated["assignments"][m]
        arrays[f"f{f}_test_indices"] = test
        arrays[f"f{f}_finite_mask"] = finite
        arrays[f"f{f}_common_mask"] = common
        arrays[f"f{f}_test_directions"] = directions[test]
        arrays[f"f{f}_test_blocks"] = blocks[test]
        if f < 2:
            null = np.full(n_shuffles, np.nan)
            for draw in range(n_shuffles):
                try:
                    shuffled = shuffle_training_directions(directions[train], blocks[train], rng)
                    model = fit_template_model(ranks[:, train], bools[:, train], shuffled,
                                               coords, method=METHODS[1])
                    density = fit_template_density(model, shuffled)
                    null[draw] = mixture_nll(density, directions[test][finite]).mean()
                except ValueError as exc:
                    control_failures.append(dict(fold=f, draw=draw, reason=str(exc)))
            controls.append(null)
    record = dict(subject_id=subject_id, dataset=dataset, status="ok", n_events=len(blocks),
                  n_direction_estimable=int(np.isfinite(directions).all(axis=1).sum()),
                  n_channels=len(names), n_blocks=len(unique), channel_names=names,
                  event_artifact_root=str(source_dir), source_block_inventory=events["block_boundaries"],
                  folds=fold_results, training_shuffle_failures=control_failures)
    for m, old_key in zip(METHODS, ("timing_only_score", "timing_plus_space_score")):
        for metric in ("nll", "conditional_nll", "angular_mae_deg", "legacy_direction_score"):
            record[f"{m}_{metric}"] = float(np.mean([r["metrics"][m][metric] for r in fold_results[:2]]))
        record[f"{m}_event_weighted_nll"] = float(np.average(
            [r["metrics"][m]["nll"] for r in fold_results[:2]],
            weights=[r["n_direction_test"] for r in fold_results[:2]]))
        record[f"{m}_chronological_nll"] = fold_results[2]["metrics"][m]["nll"]
        record[f"{m}_legacy_difference"] = record[f"{m}_legacy_direction_score"] - old[old_key]
        if abs(record[f"{m}_legacy_difference"]) > 1e-8:
            raise ValueError(f"legacy template score mismatch: {m}: {record[f'{m}_legacy_difference']}")
    record["single_vmf_nll"] = float(np.mean([r["single_vmf_nll"] for r in fold_results[:2]]))
    record["nll_improvement"] = record["rank_sequence_nll"] - record["rank_trajectory_nll"]
    record["kappa_cap_hits"] = sum(int(np.sum(r["densities"][m]["kappas"] >= KAPPA_CAP))
                                   for r in fold_results for m in METHODS)
    if n_shuffles:
        controls = np.asarray(controls)
        complete = np.isfinite(controls).all(axis=0)
        record["training_shuffle_complete_draws"] = int(complete.sum())
        record["training_shuffle_nll"] = controls[:, complete].mean(axis=0)
        record["training_shuffle_median_nll"] = float(np.median(record["training_shuffle_nll"])) if complete.any() else None
    np.savez_compressed(folder / f"{subject_id}_heldout.npz", **arrays)
    write_json(folder / f"{subject_id}.json", record)
    return record


def paired_summary(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = a - b
    rng = np.random.default_rng(SEED)
    draws = np.median(delta[rng.integers(0, len(delta), (10000, len(delta)))], axis=1)
    return dict(n=len(a), rank_sequence_median=float(np.median(a)),
                rank_trajectory_median=float(np.median(b)), median_improvement=float(np.median(delta)),
                improvement_ci95=np.quantile(draws, [.025, .975]), n_improved=int((delta > 0).sum()),
                wilcoxon_two_sided_p=float(wilcoxon(a, b, alternative="two-sided").pvalue)
                if np.any(delta != 0) else 1.)


def summarize(records, failures, out, n_requested, n_shuffles):
    metrics = ("nll", "conditional_nll", "angular_mae_deg", "event_weighted_nll", "chronological_nll")
    summary = dict(status="complete" if not failures else "partial", n_requested=n_requested,
                   n_complete=len(records), failures=failures, primary="nll", statistical_unit="patient",
                   n_shuffles=n_shuffles, uniform_nll=LOG_SPHERE_AREA,
                   n_events=sum(r["n_events"] for r in records),
                   n_direction_estimable=sum(r["n_direction_estimable"] for r in records),
                   kappa_cap_hits=sum(r["kappa_cap_hits"] for r in records),
                   legacy_max_absolute_difference=max(abs(r[f"{m}_legacy_difference"]) for r in records for m in METHODS),
                   metrics={key: paired_summary([r[f"rank_sequence_{key}"] for r in records],
                                                [r[f"rank_trajectory_{key}"] for r in records]) for key in metrics},
                   by_dataset={ds: paired_summary([r["rank_sequence_nll"] for r in records if r["dataset"] == ds],
                                                  [r["rank_trajectory_nll"] for r in records if r["dataset"] == ds])
                               for ds in sorted({r["dataset"] for r in records})},
                   single_vmf_median_nll=float(np.median([r["single_vmf_nll"] for r in records])),
                   trajectory_vs_single_vmf=paired_summary([r["single_vmf_nll"] for r in records],
                                                          [r["rank_trajectory_nll"] for r in records]),
                   claim_status="NEW_SUPPLEMENT_PENDING_AUTHOR_REVIEW")
    if n_shuffles:
        valid = [r for r in records if r.get("training_shuffle_median_nll") is not None]
        summary["training_shuffle"] = dict(
            **paired_summary([r["training_shuffle_median_nll"] for r in valid], [r["rank_trajectory_nll"] for r in valid]),
            comparison="shuffled-training trajectory versus real-training trajectory; positive favors real",
            failed_draws=sum(len(r["training_shuffle_failures"]) for r in records),
            minimum_complete_draws=min(r["training_shuffle_complete_draws"] for r in records))
    write_json(out / "summary.json", summary)
    keys = [k for k, v in records[0].items() if isinstance(v, (str, float, int)) or v is None]
    with (out / "patients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((SOURCE / "subject_spatial_information_gain.csv").open()))
    subjects = args.subjects or [r["subject_id"] for r in rows]
    sources = [Path(__file__), ROOT / "src/interictal_direction_distribution_holdout.py",
               ROOT / "scripts/run_timing_plus_space_rank_readback.py",
               ROOT / "src/interictal_propagation.py", ROOT / "src/lagpat_rank_audit.py",
               ROOT / "src/topic5_interictal_direction_rose.py", ROOT / "src/seeg_coord_loader.py",
               ROOT / "docs/fig2b_direction_distribution_holdout.md"]
    snap = args.output / "source_snapshot"
    snap.mkdir(exist_ok=True)
    manifest = {}
    for path in sources:
        data = path.read_bytes()
        (snap / path.name).write_bytes(data)
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    write_json(args.output / "run_manifest.json", dict(subjects=subjects, n_shuffles=args.n_shuffles,
                                                       seed=SEED, source_sha256=manifest))
    records, failures, pending = [], [], []
    for sid in subjects:
        path = args.output / "per_subject" / f"{sid}.json"
        if args.resume and path.exists():
            record = json.loads(path.read_text())
            if args.n_shuffles == 0 or record.get("training_shuffle_complete_draws", 0) == args.n_shuffles:
                records.append(record)
                continue
        pending.append(sid)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_patient, sid, str(args.output), args.n_shuffles): sid for sid in pending}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                r = future.result()
                records.append(r)
                print(f"OK {sid}: NLL {r['rank_sequence_nll']:.4f} -> {r['rank_trajectory_nll']:.4f}; legacy matched", flush=True)
            except Exception as exc:
                failures.append(dict(subject_id=sid, reason=repr(exc)))
                print(f"FAILED {sid}: {exc!r}", flush=True)
            write_json(args.output / "progress.json", dict(n_complete=len(records), failures=failures, n_requested=len(subjects)))
    records.sort(key=lambda r: subjects.index(r["subject_id"]))
    if records:
        summary = summarize(records, failures, args.output, len(subjects), args.n_shuffles)
        print(json.dumps(jsonable(summary), indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
