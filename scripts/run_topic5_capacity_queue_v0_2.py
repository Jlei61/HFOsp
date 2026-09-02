#!/usr/bin/env python3
"""Phase F: train every eligible ordered unit in the frozen manifest.

Workers claim units through atomic lock files, so several queue processes can
run at once (and a killed run can be resumed) without two workers ever training
the same unit.  Each unit writes its own directory atomically; nothing is
appended to a shared file, and the reducer runs separately.

``split == -1`` is dropped the moment a patient is loaded, so no training or
checkpoint-selection path can reach the model-unseen tier.  The compact
confirmation scorer is the only code allowed to read it.
"""
from __future__ import annotations

# One worker must not also fan out inside BLAS: these processes are run many at a
# time on a shared machine, and the default OpenMP thread count is the core count,
# which produced a load average of ~860 on an 80-core host before this was set.
import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, _os.environ.get("TOPIC5_TORCH_THREADS", "1"))

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

from src.topic5_strict_history_data_v0_2 import load_sample_set  # noqa: E402
from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    MotifConfig,
    OrderedMotif,
    TrainConfig,
    checkpoint_objective,
    combine_logits,
    evaluate,
    fit,
    primary_field_kind,
    tensors_from_samples,
    training_loss,
)
from src.topic5_structural_identifiability_v0_2 import load_basis_bundle  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
LOCK_ROOT = RESULT_ROOT / "locks"
TIME_BINS = 3
TIME_LOSS_WEIGHT = 1.0
SOURCE_FILES = (
    ROOT / "src/topic5_strict_history_motif_v0_2.py",
    ROOT / "src/topic5_strict_history_data_v0_2.py",
    ROOT / "src/topic5_structural_identifiability_v0_2.py",
)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_FILES:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


class TimeHead(torch.nn.Module):
    """Spectral-centroid latency proxy — never a conduction delay or a speed."""

    def __init__(self, rank: int, n_horizons: int) -> None:
        super().__init__()
        self.readout = torch.nn.Parameter(torch.randn(rank) / max(1.0, rank ** 0.5))
        self.bin_scale = torch.nn.Parameter(torch.zeros(TIME_BINS))
        self.bin_bias = torch.nn.Parameter(torch.zeros(n_horizons, TIME_BINS))
        self.continuous_scale = torch.nn.Parameter(torch.zeros(1))
        self.continuous_bias = torch.nn.Parameter(torch.zeros(n_horizons))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = torch.einsum("bhr,r->bh", states, self.readout)
        bins = scalar.unsqueeze(2) * self.bin_scale + self.bin_bias
        continuous = scalar * self.continuous_scale + self.continuous_bias
        return bins, continuous


def rolled_states(model: OrderedMotif, batch) -> torch.Tensor:
    state = model.prefix_state(batch)
    transition = model.transition()
    nonlinear = model.config.f_form == "LOW_DIMENSIONAL_TANH"
    states = []
    rolled = state
    for _ in range(batch.n_horizons):
        rolled = rolled @ transition.T
        if nonlinear:
            rolled = torch.tanh(rolled)
        states.append(rolled)
    return torch.stack(states, dim=1)


def time_loss(head: TimeHead, states: torch.Tensor, bins, continuous, valid) -> torch.Tensor:
    predicted_bins, predicted_continuous = head(states)
    horizons = min(3, states.shape[1])
    total = states.new_zeros(())
    for horizon in range(horizons):
        keep = valid[:, horizon]
        if not bool(keep.any()):
            continue
        logits = predicted_bins[keep, horizon]
        total = total + torch.nn.functional.cross_entropy(logits, bins[keep, horizon])
        total = total + torch.nn.functional.mse_loss(
            predicted_continuous[keep, horizon], continuous[keep, horizon]
        )
    return total / max(horizons, 1)


class PatientWorkspace:
    """Per-patient tensors, baseline logits and bases, loaded once per shard."""

    def __init__(self, patient: str, device: str = "cpu") -> None:
        self.patient = patient
        self.device = device
        self._samples: dict[int, object] = {}
        self._tensors: dict[int, object] = {}
        self._observed: dict[int, np.ndarray] = {}
        self._baseline: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self._bases, index = load_basis_bundle(RESULT_ROOT / "basis" / "per_patient" / f"{patient}.npz")
        self._basis_index = {entry["key"]: entry for entry in index}
        self.contact_xy = torch.as_tensor(
            np.asarray(np.load(FRAME_ROOT / patient / "plane.npz")["contacts_xy_mm"], dtype=np.float32)
        ).to(device)

    def samples(self, prefix_len: int):
        if prefix_len not in self._samples:
            self._samples[prefix_len] = load_sample_set(
                RESULT_ROOT / "sample_cache" / f"prefix{prefix_len}" / f"{self.patient}.npz"
            )
        return self._samples[prefix_len]

    def observed_rows(self, prefix_len: int) -> np.ndarray:
        """Everything the trainer may see: train, calibration, development test."""
        if prefix_len not in self._observed:
            split = self.samples(prefix_len).split
            self._observed[prefix_len] = np.flatnonzero(split >= 0)
        return self._observed[prefix_len]

    def tensors(self, prefix_len: int):
        if prefix_len not in self._tensors:
            self._tensors[prefix_len] = tensors_from_samples(
                self.samples(prefix_len), self.observed_rows(prefix_len), device=self.device
            )
        return self._tensors[prefix_len]

    def baseline(self, level: str, prefix_len: int) -> dict[str, torch.Tensor]:
        key = (level, prefix_len)
        if key not in self._baseline:
            path = RESULT_ROOT / "baseline" / level / f"prefix{prefix_len}" / self.patient / "logits.npz"
            payload = np.load(path, allow_pickle=False)
            rows = self.observed_rows(prefix_len)
            self._baseline[key] = {
                name: torch.as_tensor(payload[name][rows], dtype=torch.float32).to(self.device)
                for name in ("contact", "cardinality", "suffix")
            }
        return self._baseline[key]

    def split_mask(self, prefix_len: int, value: int) -> np.ndarray:
        return self.samples(prefix_len).split[self.observed_rows(prefix_len)] == value

    def fraction_mask(self, prefix_len: int, fraction: int) -> np.ndarray:
        samples = self.samples(prefix_len)
        return samples.fraction_mask(fraction)[self.observed_rows(prefix_len)]

    def basis(self, key: str) -> np.ndarray:
        return self._bases[key]

    def basis_meta(self, key: str) -> dict:
        return self._basis_index[key]


def train_unit(workspace: PatientWorkspace, unit: dict, config: dict) -> dict:
    prefix_len = int(unit["prefix_len"])
    samples = workspace.samples(prefix_len)
    batch = workspace.tensors(prefix_len)
    baseline = workspace.baseline(unit["baseline_level"], prefix_len)
    observed = workspace.observed_rows(prefix_len)
    if (samples.split[observed] == -1).any():
        raise RuntimeError("model-unseen events reached the trainer")

    device = workspace.device
    train_rows = torch.as_tensor(
        np.flatnonzero(workspace.fraction_mask(prefix_len, int(unit["data_fraction"])))).to(device)
    valid_rows = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 1))).to(device)
    test_rows = torch.as_tensor(np.flatnonzero(workspace.split_mask(prefix_len, 2))).to(device)
    train_batch, valid_batch, test_batch = (batch.index(rows) for rows in (train_rows, valid_rows, test_rows))

    structure = unit["structure"]
    free = structure == "H1_FREE_LOW_RANK"
    basis = None if free else workspace.basis(unit["basis_key"])
    motif = MotifConfig(
        structure=structure, family=unit["family"], rank=int(unit["rank"]),
        n_contacts=samples.n_contacts, n_horizons=batch.n_horizons,
        max_cardinality=samples.max_cardinality, f_form=unit["f_form"], free_basis=free,
    )
    torch.manual_seed(int(unit["seed"]) + 1000)
    model = OrderedMotif(motif, basis).to(device)
    field_kind = primary_field_kind(unit["family"])

    head = None
    time_pack = None
    if bool(unit["time_head"]):
        train_index = train_rows.cpu().numpy()
        train_proxy = np.clip(np.asarray(samples.latency_proxy)[observed][train_index], 0.0, None)
        finite = train_proxy[np.asarray(samples.latency_valid)[observed][train_index]]
        edges = np.quantile(finite, [1 / TIME_BINS, 2 / TIME_BINS]) if finite.size else np.array([0.0, 0.0])
        ceiling = float(np.quantile(np.log1p(finite), 0.99)) if finite.size else 0.0
        head = TimeHead(int(unit["rank"]), batch.n_horizons).to(device)
        packs = {}
        for name, rows in (("train", train_rows), ("valid", valid_rows), ("test", test_rows)):
            index = rows.cpu().numpy()
            proxy = np.clip(np.asarray(samples.latency_proxy)[observed][index], 0.0, None)
            valid_flag = np.asarray(samples.latency_valid)[observed][index]
            packs[name] = (
                torch.as_tensor(np.clip(np.digitize(proxy, edges), 0, TIME_BINS - 1),
                                dtype=torch.long).to(device),
                torch.as_tensor(np.minimum(np.log1p(proxy), ceiling),
                                dtype=torch.float32).to(device),
                torch.as_tensor(valid_flag, dtype=torch.bool).to(device),
            )
        time_pack = {"packs": packs, "edges": edges.tolist(), "ceiling": ceiling}

    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    container = torch.nn.ModuleList([model] + ([head] if head is not None else []))
    baseline_train = {key: value[train_rows] for key, value in baseline.items()}

    def forward(piece, rows):
        residual = model(piece)
        merged = combine_logits({key: value[rows] for key, value in baseline_train.items()}, residual)
        loss = training_loss(merged, piece, field_kind)
        if head is not None:
            bins, continuous, valid_flag = time_pack["packs"]["train"]
            loss = loss + TIME_LOSS_WEIGHT * time_loss(
                head, rolled_states(model, piece), bins[rows], continuous[rows], valid_flag[rows]
            )
        return loss

    def objective(_module):
        result = evaluate(
            model, {key: value[valid_rows] for key, value in baseline.items()},
            valid_batch, workspace.contact_xy,
        )
        return checkpoint_objective(result, unit["family"])

    history = fit(
        container, forward, train_batch, valid_batch, objective,
        TrainConfig(seed=int(unit["seed"]), **config["train"]),
    )

    metrics = {}
    for name, rows, piece in (("calibration", valid_rows, valid_batch), ("development_test", test_rows, test_batch)):
        result = evaluate(model, {key: value[rows] for key, value in baseline.items()},
                          piece, workspace.contact_xy)
        entry = {"per_horizon": result.per_horizon, "scalars": result.scalars}
        ablated = evaluate(model, {key: value[rows] for key, value in baseline.items()},
                           piece, workspace.contact_xy, ordered_path=False)
        entry["ordered_path_ablated"] = {"per_horizon": ablated.per_horizon, "scalars": ablated.scalars}
        if head is not None:
            bins, continuous, valid_flag = time_pack["packs"]["valid" if name == "calibration" else "test"]
            with torch.no_grad():
                entry["time_proxy_loss"] = float(
                    time_loss(head, rolled_states(model, piece), bins, continuous, valid_flag)
                )
        metrics[name] = entry

    with torch.no_grad():
        transition = (model.transition().cpu().numpy()
                      if unit["family"] != "ORDERLESS_BAG" else None)
    spectrum = (
        [float(v) for v in np.abs(np.linalg.eigvals(transition))] if transition is not None else []
    )
    return {
        "metrics": metrics,
        "training": {key: value for key, value in history.items() if key != "history"},
        "history": history["history"],
        "state_dict": {key: value.cpu() for key, value in container.state_dict().items()},
        "diagnostics": {
            "ordered_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in parameters)),
            "transition_spectral_radius": max(spectrum) if spectrum else float("nan"),
            "transition_eigenvalue_moduli": spectrum,
            "n_train": int(train_rows.numel()),
            "n_calibration": int(valid_rows.numel()),
            "n_development_test": int(test_rows.numel()),
            "basis_sha256": workspace.basis_meta(unit["basis_key"])["sha256"] if not free else "",
            "time_proxy": {k: v for k, v in (time_pack or {}).items() if k != "packs"},
        },
    }


def claim(unit_hash: str) -> bool:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOCK_ROOT / f"{unit_hash}.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(handle, f"{socket.gethostname()}:{os.getpid()}:{time.time()}\n".encode())
    os.close(handle)
    return True


def run_shard(payload: dict) -> dict:
    torch.set_num_threads(int(os.environ.get("TOPIC5_TORCH_THREADS", "2")))
    patient, units, config = payload["patient"], payload["units"], payload["config"]
    workspace = None
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
        if not claim(unit["unit_hash"]):
            skipped += 1
            continue
        if workspace is None:
            workspace = PatientWorkspace(patient, config.get("device", "cpu"))
        started = time.time()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(
            {**{k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                    int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in unit.items()},
             "train_config": config["train"], "source_hash": source_hash()},
            indent=2, default=str) + "\n")
        attempts = 0
        unit_config = json.loads(json.dumps(config))
        while True:
            attempts += 1
            try:
                result = train_unit(workspace, unit, unit_config)
                break
            except Exception:
                if attempts >= config["max_attempts"]:
                    (directory / "status.json").write_text(json.dumps({
                        "state": "unresolved", "attempts": attempts,
                        "error": traceback.format_exc(),
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                    }, indent=2) + "\n")
                    failed.append(unit["unit_id"])
                    result = None
                    break
                unit_config["train"]["batch_size"] = max(256, unit_config["train"]["batch_size"] // 2)
        if result is None:
            continue
        torch.save({"state_dict": result.pop("state_dict")}, directory / "checkpoint.pt")
        digest = hashlib.sha256((directory / "checkpoint.pt").read_bytes()).hexdigest()
        (directory / "metrics.json").write_text(json.dumps(
            {"metrics": result["metrics"], "training": result["training"],
             "diagnostics": result["diagnostics"], "history": result["history"]},
            indent=2, default=float) + "\n")
        (directory / "status.json").write_text(json.dumps({
            "state": "complete", "attempts": attempts,
            "checkpoint_sha256": digest,
            "wall_seconds": time.time() - started,
            "nonfinite_batches": result["training"]["nonfinite_batches"],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "pid": os.getpid(),
        }, indent=2) + "\n")
        done += 1
    return {"patient": patient, "done": done, "skipped": skipped, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--blocks", default="")
    parser.add_argument("--patients", default="")
    parser.add_argument("--shard-size", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-updates", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    table = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    table = table[table["eligible"]]
    if arguments.blocks:
        table = table[table["block"].isin(arguments.blocks.split(","))]
    if arguments.patients:
        table = table[table["patient"].isin(arguments.patients.split(","))]
    config = {
        "train": {"max_epochs": arguments.max_epochs, "patience": arguments.patience,
                  "batch_size": arguments.batch_size, "max_seconds": arguments.max_seconds,
                  "min_updates_per_epoch": arguments.min_updates},
        "max_attempts": arguments.max_attempts, "device": arguments.device,
    }

    payloads = []
    for patient, group in table.groupby("patient"):
        units = group.to_dict("records")
        for start in range(0, len(units), arguments.shard_size):
            payloads.append({"patient": patient, "units": units[start:start + arguments.shard_size],
                             "config": json.loads(json.dumps(config))})
    payloads.sort(key=lambda item: -len(item["units"]))
    print(f"queue: {len(table)} units in {len(payloads)} shards, {arguments.workers} workers", flush=True)

    started = time.time()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = []
        for index, outcome in enumerate(pool.map(run_shard, payloads), start=1):
            results.append(outcome)
            print(f"[{index}/{len(payloads)}] {outcome['patient']:22s} done={outcome['done']:3d} "
                  f"skipped={outcome['skipped']:3d} failed={len(outcome['failed'])} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    failures = [unit for outcome in results for unit in outcome["failed"]]
    print(f"total done={sum(o['done'] for o in results)} skipped={sum(o['skipped'] for o in results)} "
          f"unresolved={len(failures)}")
    for unit in failures[:20]:
        print(f"  unresolved: {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
