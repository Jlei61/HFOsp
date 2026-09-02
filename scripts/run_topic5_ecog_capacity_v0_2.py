#!/usr/bin/env python3
"""Phase G: ECoG construct-validity case series (E958, E1084).

Two patients, reported one at a time.  No pooled p-value across the pair, no
merged denominator with the 28-patient SEEG cohort, and "replicated" is only
allowed if both pre-specified directions and effects agree — and then only as
two-case consistency.

The task, the two unordered baselines, the exact subset law and the
direct/autonomous split are literally the same code as the SEEG side; what
differs is that the ordered state here is the full contact field evolved by a
few-parameter polynomial of the frozen grid, and that the swap is a genuine
runtime graph swap because contact identity, inputs and outputs never move.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_ecog_graph_capacity_v0_2 import (  # noqa: E402
    GRAPH_FAMILIES,
    NULL_GRAPH_INDICES,
    PRIMARY_MICROSTEPS,
    SENSITIVITY_MICROSTEPS,
    EcogConfig,
    EcogGraphMotif,
    graph_blocks,
    load_graph,
    readout_blocks,
    swap_graph,
)
from src.topic5_strict_history_data_v0_2 import load_sample_set  # noqa: E402
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    TrainConfig,
    checkpoint_objective,
    combine_logits,
    evaluate,
    fit,
    perturb_prefix_order,
    primary_field_kind,
    tensors_from_samples,
    training_loss,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
ECOG_ROOT = RESULT_ROOT / "ecog_construct_validity"
GRAPH_ROOT = ROOT / "results/topic5_ecog_physical_neighborhood_rnn_v0_1/graphs"
CACHE_ROOT = ROOT / "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
SUBJECTS = ("E958", "E1084")
DIRECT = "DIRECT_HORIZON_UPPER_BOUND"
AUTO = "AUTONOMOUS_SHARED_OPERATOR"


def build_manifest() -> pd.DataFrame:
    rows = []

    def add(subject, block, capacity, structure, graph_index, family, level, fraction, seed,
            microsteps):
        key = "|".join(map(str, [subject, block, capacity, structure, graph_index, family,
                                level, fraction, seed, microsteps]))
        rows.append({
            "subject": subject, "block": block, "capacity": capacity, "structure": structure,
            "graph_family": GRAPH_FAMILIES[structure] if structure in GRAPH_FAMILIES else "TRUE_GRID",
            "graph_index": graph_index, "family": family, "baseline_level": level,
            "data_fraction": fraction, "seed": seed, "microsteps": microsteps,
            "unit_id": key, "unit_hash": hashlib.sha256(key.encode()).hexdigest()[:16],
        })

    for subject in SUBJECTS:
        for level in ("U_FULL_SET", "U_MINIMAL"):
            for family in (AUTO, DIRECT):
                for seed in (0, 1, 2):
                    add(subject, "ECOG_CORE", "G2", "OBSERVED_GRID", -1, family, level, 100,
                        seed, PRIMARY_MICROSTEPS)
                for structure in ("IDENTITY_PERMUTED_GRID", "DEGREE_AND_DISTANCE_REWIRED_GRID"):
                    for index in NULL_GRAPH_INDICES:
                        add(subject, "ECOG_CORE", "G2", structure, index, family, level, 100, 0,
                            PRIMARY_MICROSTEPS)
                add(subject, "ECOG_CORE", "FREE_SAME_STATE_UPPER_BOUND", "OBSERVED_GRID", -1,
                    family, level, 100, 0, PRIMARY_MICROSTEPS)
        for capacity in ("G1", "G3"):
            add(subject, "ECOG_CAPACITY", capacity, "OBSERVED_GRID", -1, AUTO, "U_FULL_SET",
                100, 0, PRIMARY_MICROSTEPS)
            for structure in ("IDENTITY_PERMUTED_GRID", "DEGREE_AND_DISTANCE_REWIRED_GRID"):
                for index in NULL_GRAPH_INDICES[:4]:
                    add(subject, "ECOG_CAPACITY", capacity, structure, index, AUTO, "U_FULL_SET",
                        100, 0, PRIMARY_MICROSTEPS)
            add(subject, "ECOG_CAPACITY", "FREE_SAME_STATE_UPPER_BOUND", "OBSERVED_GRID", -1,
                AUTO, "U_FULL_SET", 100, 0, PRIMARY_MICROSTEPS)
        for fraction in (25, 50):
            add(subject, "ECOG_DATA", "G2", "OBSERVED_GRID", -1, AUTO, "U_FULL_SET", fraction, 0,
                PRIMARY_MICROSTEPS)
            for structure in ("IDENTITY_PERMUTED_GRID", "DEGREE_AND_DISTANCE_REWIRED_GRID"):
                for index in NULL_GRAPH_INDICES[:4]:
                    add(subject, "ECOG_DATA", "G2", structure, index, AUTO, "U_FULL_SET",
                        fraction, 0, PRIMARY_MICROSTEPS)
        add(subject, "ECOG_MICROSTEPS", "G2", "OBSERVED_GRID", -1, AUTO, "U_FULL_SET", 100, 0,
            SENSITIVITY_MICROSTEPS)
        for structure in ("IDENTITY_PERMUTED_GRID", "DEGREE_AND_DISTANCE_REWIRED_GRID"):
            for index in NULL_GRAPH_INDICES[:4]:
                add(subject, "ECOG_MICROSTEPS", "G2", structure, index, AUTO, "U_FULL_SET", 100,
                    0, SENSITIVITY_MICROSTEPS)
    table = pd.DataFrame(rows)
    table["output_dir"] = ["ecog_construct_validity/units/" + row["subject"] + "/" + row["unit_hash"]
                           for row in rows]
    return table


class EcogWorkspace:
    def __init__(self, subject: str, device: str) -> None:
        self.subject = subject
        self.device = device
        self.samples = load_sample_set(RESULT_ROOT / "sample_cache" / "prefix3" / f"{subject}.npz")
        self.observed = np.flatnonzero(self.samples.split >= 0)
        self.batch = tensors_from_samples(self.samples, self.observed, device=device)
        payload = np.load(CACHE_ROOT / subject[1:] / "events.npz", allow_pickle=False)
        names = [str(v) for v in payload["channel_names"]]
        rows = sorted({name[1] for name in names})
        coords = np.zeros((len(names), 2))
        for position, name in enumerate(names):
            coords[position] = ((int(name[2:]) - 1) * 10.0, rows.index(name[1]) * 10.0)
        self.contact_xy = torch.as_tensor(coords, dtype=torch.float32).to(device)
        self._baseline: dict[str, dict[str, torch.Tensor]] = {}
        self._graphs: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict]] = {}

    def baseline(self, level: str) -> dict[str, torch.Tensor]:
        if level not in self._baseline:
            path = RESULT_ROOT / "baseline" / level / "prefix3" / self.subject / "logits.npz"
            payload = np.load(path, allow_pickle=False)
            self._baseline[level] = {
                name: torch.as_tensor(payload[name][self.observed], dtype=torch.float32).to(self.device)
                for name in ("contact", "cardinality", "suffix")
            }
        return self._baseline[level]

    def graph(self, family: str, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
        key = (family, index)
        if key not in self._graphs:
            self._graphs[key] = load_graph(GRAPH_ROOT, self.subject[1:], family,
                                           None if index < 0 else index)
        return self._graphs[key]

    def rows(self, kind: str, fraction: int = 100) -> torch.Tensor:
        split = self.samples.split[self.observed]
        if kind == "train":
            mask = self.samples.fraction_mask(fraction)[self.observed]
        else:
            mask = split == {"calibration": 1, "development_test": 2}[kind]
        return torch.as_tensor(np.flatnonzero(mask)).to(self.device)


def build_model(workspace: EcogWorkspace, unit: dict) -> EcogGraphMotif:
    mask, coords, _ = workspace.graph(unit["graph_family"], int(unit["graph_index"]))
    config = EcogConfig(
        structure=unit["structure"], family=unit["family"], capacity=unit["capacity"],
        n_contacts=workspace.samples.n_contacts, n_horizons=workspace.batch.n_horizons,
        max_cardinality=workspace.samples.max_cardinality, microsteps=int(unit["microsteps"]),
    )
    return EcogGraphMotif(config, graph_blocks(mask, coords, unit["capacity"]),
                          readout_blocks(mask)).to(workspace.device)


def train_unit(workspace: EcogWorkspace, unit: dict, config: dict) -> dict:
    baseline = workspace.baseline(unit["baseline_level"])
    train_rows = workspace.rows("train", int(unit["data_fraction"]))
    valid_rows = workspace.rows("calibration")
    test_rows = workspace.rows("development_test")
    train_batch = workspace.batch.index(train_rows)
    valid_batch = workspace.batch.index(valid_rows)
    test_batch = workspace.batch.index(test_rows)
    torch.manual_seed(int(unit["seed"]) + 909)
    model = build_model(workspace, unit)
    field_kind = primary_field_kind(unit["family"])
    baseline_train = {key: value[train_rows] for key, value in baseline.items()}

    def forward(piece, rows):
        merged = combine_logits({k: v[rows] for k, v in baseline_train.items()}, model(piece))
        return training_loss(merged, piece, field_kind)

    def objective(_module):
        return checkpoint_objective(evaluate(
            model, {k: v[valid_rows] for k, v in baseline.items()}, valid_batch,
            workspace.contact_xy, chunk=config["chunk"]), unit["family"])

    history = fit(model, forward, train_batch, valid_batch, objective,
                  TrainConfig(seed=int(unit["seed"]), **config["train"]))

    metrics = {}
    for name, rows, piece in (("calibration", valid_rows, valid_batch),
                              ("development_test", test_rows, test_batch)):
        base = {k: v[rows] for k, v in baseline.items()}
        intact = evaluate(model, base, piece, workspace.contact_xy, chunk=config["chunk"])
        ablated = evaluate(model, base, piece, workspace.contact_xy, ordered_path=False,
                           chunk=config["chunk"])
        entry = {"per_horizon": intact.per_horizon, "scalars": intact.scalars,
                 "ordered_path_ablated": {"per_horizon": ablated.per_horizon,
                                          "scalars": ablated.scalars}}
        if name == "development_test":
            permuted = evaluate(model, base, perturb_prefix_order(piece, "swap_middle"),
                                workspace.contact_xy, chunk=config["chunk"])
            entry["prefix_order_perturbed"] = {"per_horizon": permuted.per_horizon,
                                               "scalars": permuted.scalars}
        metrics[name] = entry

    # runtime graph swap: contact identity, inputs and outputs unchanged
    swap = {}
    if unit["structure"] == "OBSERVED_GRID" and not model.free:
        digest_before = hashlib.sha256(
            b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        original = (model.transition_blocks.detach().cpu().numpy().copy(),
                    model.output_blocks.detach().cpu().numpy().copy())
        base = {k: v[test_rows] for k, v in baseline.items()}
        for structure, family in GRAPH_FAMILIES.items():
            if structure == "OBSERVED_GRID":
                continue
            scores = []
            for index in NULL_GRAPH_INDICES[:4]:
                mask, coords, _ = workspace.graph(family, index)
                swap_graph(model, graph_blocks(mask, coords, unit["capacity"]), readout_blocks(mask))
                scores.append(checkpoint_objective(evaluate(
                    model, base, test_batch, workspace.contact_xy, chunk=config["chunk"]),
                    unit["family"]))
            swap[structure] = float(np.median(scores))
        swap_graph(model, [block for block in original[0]], [block for block in original[1]])
        digest_after = hashlib.sha256(
            b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        swap["parameters_unchanged"] = digest_before == digest_after
        swap["intact"] = checkpoint_objective(evaluate(
            model, base, test_batch, workspace.contact_xy, chunk=config["chunk"]), unit["family"])

    return {
        "metrics": metrics,
        "training": {k: v for k, v in history.items() if k != "history"},
        "runtime_graph_swap": swap,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "diagnostics": {
            "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
            "graph_parameters": int(
                model.free_transition.numel() if model.free else model.alpha.numel()) + 1,
            "n_train": int(train_rows.numel()),
            "n_calibration": int(valid_rows.numel()),
            "n_development_test": int(test_rows.numel()),
        },
    }


def run_shard(payload: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    subject, units, config = payload["subject"], payload["units"], payload["config"]
    workspace = EcogWorkspace(subject, config["device"])
    done, skipped, failed = 0, 0, []
    for unit in units:
        directory = RESULT_ROOT / unit["output_dir"]
        status_path = directory / "status.json"
        if status_path.exists():
            try:
                if json.loads(status_path.read_text())["state"] == "complete":
                    skipped += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(unit, indent=2, default=str) + "\n")
        started = time.time()
        try:
            result = train_unit(workspace, unit, config)
        except Exception:
            (directory / "status.json").write_text(json.dumps({
                "state": "unresolved", "error": traceback.format_exc(),
                "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            failed.append(unit["unit_id"])
            continue
        torch.save({"state_dict": result.pop("state_dict")}, directory / "checkpoint.pt")
        (directory / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        (directory / "status.json").write_text(json.dumps({
            "state": "complete", "wall_seconds": time.time() - started,
            "checkpoint_sha256": hashlib.sha256((directory / "checkpoint.pt").read_bytes()).hexdigest(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "pid": os.getpid()}, indent=2) + "\n")
        done += 1
    return {"subject": subject, "done": done, "skipped": skipped, "failed": failed}


def summarise(manifest: pd.DataFrame) -> dict:
    rows = []
    for unit in manifest.to_dict("records"):
        path = RESULT_ROOT / unit["output_dir"] / "metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        test = payload["metrics"]["development_test"]
        kind = primary_field_kind(unit["family"])
        objective = sum(
            test["per_horizon"]["total_nll"][h - 1] / 3.0 for h in (1, 2, 3)
            if test["per_horizon"]["total_nll"][h - 1] is not None
            and not np.isnan(test["per_horizon"]["total_nll"][h - 1])
        ) + test["scalars"][f"{kind}_balanced_bce"]
        rows.append({
            **{key: unit[key] for key in ("subject", "block", "capacity", "structure", "family",
                                          "baseline_level", "data_fraction", "seed", "microsteps",
                                          "graph_index", "unit_id")},
            "primary_objective": objective,
            "suffix_balanced_bce": test["scalars"][f"{kind}_balanced_bce"],
            "total_nll_h1": test["per_horizon"]["total_nll"][0],
            "total_nll_h2": test["per_horizon"]["total_nll"][1],
            "total_nll_h3": test["per_horizon"]["total_nll"][2],
            "trainable_parameters": payload["diagnostics"]["trainable_parameters"],
            "graph_parameters": payload["diagnostics"]["graph_parameters"],
            "runtime_graph_swap": payload.get("runtime_graph_swap", {}),
        })
    table = pd.DataFrame(rows)
    table.drop(columns=["runtime_graph_swap"]).to_csv(
        ECOG_ROOT / "ECOG_PER_UNIT_SCORES.csv", index=False)

    matrix: dict = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_ecog_case_series",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_rules": [
            "each subject is reported on its own; no pooled p-value across the pair",
            "E958 positive does not make an ECoG cohort mechanism",
            "E1084 disagreeing does not negate the SEEG cohort",
            "the physical-grid advantage is an observed-grid inductive bias, "
            "never a cortical synaptic graph",
        ],
        "microsteps_primary": PRIMARY_MICROSTEPS,
        "microsteps_sensitivity": SENSITIVITY_MICROSTEPS,
        "null_graph_indices": list(NULL_GRAPH_INDICES),
        "subjects": {},
    }
    for subject in SUBJECTS:
        frame = table[table["subject"] == subject]
        if frame.empty:
            continue
        entry: dict = {"n_units": int(len(frame))}
        for block in sorted(frame["block"].unique()):
            block_frame = frame[frame["block"] == block]
            grouped = block_frame.groupby(
                ["capacity", "structure", "family", "baseline_level", "data_fraction", "microsteps"]
            )["primary_objective"].median()
            entry[block] = {"|".join(map(str, key)): float(value)
                            for key, value in grouped.items()}
        swaps = [row["runtime_graph_swap"] for row in rows
                 if row["subject"] == subject and row["runtime_graph_swap"]]
        entry["runtime_graph_swap"] = {
            "n_units": len(swaps),
            "parameters_unchanged": all(s.get("parameters_unchanged", True) for s in swaps),
            "median_cost_identity_permuted": float(np.median(
                [s["IDENTITY_PERMUTED_GRID"] - s["intact"] for s in swaps
                 if "IDENTITY_PERMUTED_GRID" in s])) if swaps else None,
            "median_cost_degree_rewired": float(np.median(
                [s["DEGREE_AND_DISTANCE_REWIRED_GRID"] - s["intact"] for s in swaps
                 if "DEGREE_AND_DISTANCE_REWIRED_GRID" in s])) if swaps else None,
        }
        matrix["subjects"][subject] = entry
    (ECOG_ROOT / "ECOG_CASE_SERIES_MATRIX.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (RESULT_ROOT / "ECOG_CASE_SERIES_MATRIX.json").write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--min-updates", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--subjects", default="")
    arguments = parser.parse_args()

    ECOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest.to_csv(ECOG_ROOT / "ECOG_UNIT_MANIFEST.csv", index=False)
    if arguments.subjects:
        manifest = manifest[manifest["subject"].isin(arguments.subjects.split(","))]
    print(f"ECoG manifest: {len(manifest)} units "
          f"({manifest.groupby('subject').size().to_dict()})", flush=True)

    if not arguments.summarise_only:
        config = {
            "train": {"max_epochs": arguments.max_epochs, "patience": arguments.patience,
                      "batch_size": arguments.batch_size, "max_seconds": arguments.max_seconds,
                  "min_updates_per_epoch": arguments.min_updates},
            "chunk": arguments.chunk, "device": arguments.device,
        }
        payloads = []
        for subject, group in manifest.groupby("subject"):
            units = group.to_dict("records")
            for start in range(0, len(units), arguments.shard_size):
                payloads.append({"subject": subject, "units": units[start:start + arguments.shard_size],
                                 "config": json.loads(json.dumps(config))})
        started = time.time()
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            for index, outcome in enumerate(pool.map(run_shard, payloads), start=1):
                print(f"[{index}/{len(payloads)}] {outcome['subject']} done={outcome['done']} "
                      f"skipped={outcome['skipped']} failed={len(outcome['failed'])} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)

    matrix = summarise(manifest)
    for subject, entry in matrix["subjects"].items():
        print(f"{subject}: {entry['n_units']} units, "
              f"runtime graph swap parameters unchanged="
              f"{entry['runtime_graph_swap']['parameters_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
