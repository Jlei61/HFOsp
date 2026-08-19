#!/usr/bin/env python3
"""Resource-aware unattended optimizer for the patient-specific Topic 4 cohort."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_cmaes import CMAES  # noqa: E402
from src.topic4_patient_specific_field_cohort import (  # noqa: E402
    array_sha256,
    atomic_json,
    candidate_from_vector,
    initial_vector,
    load_config,
    projected_field_basis,
    sha256,
    verify_inputs,
)


DEFAULT_CONFIG = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2.json"
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _resources(config: dict) -> dict:
    execution = config["execution"]
    available = _mem_available_gib()
    free_disk = shutil.disk_usage(Path(config["output_root"]).parent).free / 1024.0 ** 3
    load1 = os.getloadavg()[0]
    cpu = os.cpu_count() or 1
    memory_workers = math.floor(
        (available - float(execution["reserved_available_memory_gib"]))
        / float(execution["estimated_memory_gib_per_worker"])
    )
    workers = max(0, min(int(execution["max_workers"]), memory_workers))
    ready = (
        available >= float(execution["minimum_available_memory_gib"])
        and free_disk >= float(execution["minimum_output_free_disk_gib"])
        and load1 <= float(execution["maximum_load_fraction"]) * cpu
        and workers >= 1
    )
    return {
        "ready": bool(ready), "workers": int(workers),
        "mem_available_gib": available, "output_free_disk_gib": free_disk,
        "load1": load1, "logical_cpus": cpu,
    }


def paired_runtime_mode(mode: str) -> str:
    """The mechanism replay always pairs the fit against the other slow state."""
    pairs = {"paired_slow_off": "active_z_plus_m", "active_z_plus_m": "paired_slow_off"}
    if mode not in pairs:
        raise ValueError(f"unknown runtime mode: {mode}")
    return pairs[mode]


class WorkerAdmission:
    """Admit workers one at a time under a global cap and a live memory floor.

    A batch-level check cannot see memory move while its own workers start, and
    several subjects now run at once, so admission is per worker: hold a global
    slot, wait until the host has the floor plus one worker of headroom free,
    then stagger the launch so the new footprint appears before the next check.
    """

    def __init__(self, max_workers: int, floor_gib: float, poll_seconds: float, *,
                 headroom_gib: float = 0.0, stagger_seconds: float = 0.0,
                 memory_reader=None, sleeper=time.sleep):
        self.max_workers = int(max_workers)
        self.floor_gib = float(floor_gib)
        self.poll_seconds = float(poll_seconds)
        self.headroom_gib = float(headroom_gib)
        self.stagger_seconds = float(stagger_seconds)
        self._memory_reader = memory_reader or _mem_available_gib
        self._sleeper = sleeper
        self._slots = threading.Semaphore(self.max_workers)
        self._gate = threading.Lock()
        self.launched = 0

    @classmethod
    def from_config(cls, execution: dict, **kwargs) -> "WorkerAdmission":
        return cls(
            int(execution["max_workers"]),
            float(execution["worker_admission_memory_floor_gib"]),
            float(execution["worker_admission_poll_seconds"]),
            headroom_gib=float(execution["estimated_memory_gib_per_worker"]),
            stagger_seconds=float(execution["worker_admission_stagger_seconds"]),
            **kwargs,
        )

    @contextmanager
    def slot(self):
        self._slots.acquire()
        try:
            with self._gate:
                while self._memory_reader() < self.floor_gib + self.headroom_gib:
                    self._sleeper(self.poll_seconds)
                self.launched += 1
                if self.stagger_seconds:
                    self._sleeper(self.stagger_seconds)
            yield
        finally:
            self._slots.release()


class Supervisor:
    def __init__(self, config_path: Path, expected_commit: str):
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        verify_inputs(self.config, code_root=ROOT)
        self.expected_commit = subprocess.check_output(
            ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
        ).strip()
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(self.config_path.relative_to(ROOT))],
            cwd=ROOT, text=True,
        ).strip():
            raise RuntimeError("cannot launch with a dirty config")
        if subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip() != self.expected_commit:
            raise RuntimeError("launcher HEAD differs from expected commit")
        self.output = Path(self.config["output_root"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.output / "run_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output / "controller.status"
        self.basis = projected_field_basis(self.config)
        self.worker_script = ROOT / "scripts/run_topic4_patient_specific_field_worker_v2.py"
        self.admission = WorkerAdmission.from_config(self.config["execution"])
        self.runtime_mode = str(self.config["runtime"]["mode"])
        self.subject_concurrency = max(
            1, int(self.config["execution"].get("subject_concurrency", 1))
        )
        self._status_lock = threading.Lock()
        self._active: set[str] = set()

    def status(self, state: str, **details) -> None:
        with self._status_lock:
            tokens = [state] + [f"{key}={value}" for key, value in details.items()]
            if self._active:
                tokens.append("active=" + ",".join(sorted(self._active)))
            self.status_path.write_text(" ".join(tokens) + "\n")

    @contextmanager
    def _active_context(self, context: str):
        with self._status_lock:
            self._active.add(context)
        try:
            yield
        finally:
            with self._status_lock:
                self._active.discard(context)

    def wait_for_resources(self, context: str) -> int:
        wait = int(self.config["execution"]["wait_seconds"])
        while True:
            resource = _resources(self.config)
            if resource["ready"]:
                self.status("RUNNING", context=context, workers=resource["workers"],
                            mem_gib=f"{resource['mem_available_gib']:.1f}",
                            load=f"{resource['load1']:.1f}")
                return int(resource["workers"])
            self.status("WAITING_RESOURCES", context=context,
                        mem_gib=f"{resource['mem_available_gib']:.1f}",
                        disk_gib=f"{resource['output_free_disk_gib']:.1f}",
                        load=f"{resource['load1']:.1f}", retry_s=wait)
            time.sleep(wait)

    def _job_paths(self, subject_id: str, phase: str, candidate_id: str,
                   seed: int) -> tuple[Path, Path, Path]:
        root = self.output / "per_subject" / subject_id / "workers" / phase
        stem = f"{candidate_id}_seed_{int(seed)}"
        return root / f"{stem}.json", root / f"{stem}.npz", self.logs / f"{stem}.log"

    def _run_one(self, job: dict) -> dict:
        out_json, out_npz, log_path = self._job_paths(
            job["subject_id"], job["phase"], job["candidate_id"], job["seed"],
        )
        if out_json.exists() and out_npz.exists():
            payload = json.loads(out_json.read_text())
            if (payload.get("candidate_sha256") == sha256(job["candidate_json"])
                    and payload.get("seed") == int(job["seed"])
                    and payload.get("provenance", {}).get("expected_git_commit")
                    == self.expected_commit
                    and payload.get("output_npz_sha256") == sha256(out_npz)):
                return payload
        out_json.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(self.worker_script), "--config", str(self.config_path),
            "--subject-id", job["subject_id"], "--candidate-json", str(job["candidate_json"]),
            "--seed", str(job["seed"]), "--phase", job["phase"],
            "--runtime-mode", job.get("runtime_mode", self.runtime_mode),
            "--expected-commit", self.expected_commit,
            "--out-json", str(out_json), "--out-npz", str(out_npz),
        ]
        if job.get("store_envelope"):
            command.append("--store-envelope")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["NUMEXPR_NUM_THREADS"] = "1"
        environment["TOPIC4_PATIENT_SPECIFIC_SYSTEMD_UNIT"] = environment.get(
            "SYSTEMD_UNIT", "topic4-patient-specific-v2"
        )
        with self.admission.slot():
            with log_path.open("a") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, text=True,
                )
        if result.returncode != 0 or not out_json.exists():
            raise RuntimeError(
                f"worker failed for {job['subject_id']} {job['candidate_id']} "
                f"seed {job['seed']}; see {log_path}"
            )
        return json.loads(out_json.read_text())

    def run_jobs(self, jobs: list[dict], context: str) -> list[dict]:
        if not jobs:
            return []
        with self._active_context(context):
            self.wait_for_resources(context)
            outputs = [None] * len(jobs)
            # Per-worker admission, not the pool size, is what caps concurrency
            # now that several subjects share the host.
            pool_size = min(len(jobs), self.admission.max_workers)
            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                futures = {pool.submit(self._run_one, job): index
                           for index, job in enumerate(jobs)}
                for future in as_completed(futures):
                    outputs[futures[future]] = future.result()
        return outputs

    def _subject_root(self, subject_id: str) -> Path:
        path = self.output / "per_subject" / subject_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_candidate(self, subject_id: str, candidate: dict) -> Path:
        path = self._subject_root(subject_id) / "candidates" / f"{candidate['candidate_id']}.json"
        if path.exists():
            if json.loads(path.read_text()) != candidate:
                raise RuntimeError(f"candidate identity collision: {path}")
        else:
            atomic_json(candidate, path)
        return path

    def _load_optimizer(self, subject_id: str) -> tuple[CMAES, dict]:
        root = self._subject_root(subject_id)
        checkpoint = root / "optimizer_checkpoint.json"
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if (state.get("expected_git_commit") != self.expected_commit
                    or state.get("config_sha256") != sha256(self.config_path)):
                raise RuntimeError(f"stale optimizer checkpoint: {checkpoint}")
            return CMAES.from_state(state["optimizer"]), state
        restart = 0
        optimizer = CMAES(
            initial_vector(subject_id, self.config, restart=restart),
            float(self.config["search"]["sigma0"]),
            seed=20260819 + int.from_bytes(subject_id.encode("utf-8")[:4].ljust(4, b"0"), "little"),
            popsize=int(self.config["search"]["population_size"]),
        )
        return optimizer, {
            "subject_id": subject_id, "restart": restart, "history": [],
            "expected_git_commit": self.expected_commit,
            "config_sha256": sha256(self.config_path),
        }

    def ensure_fit(self, subject_id: str, generations: int) -> dict:
        root = self._subject_root(subject_id)
        checkpoint = root / "optimizer_checkpoint.json"
        optimizer, state = self._load_optimizer(subject_id)
        restart = int(state.get("restart", 0))
        history = list(state.get("history", []))
        while optimizer.generation < int(generations):
            generation = int(optimizer.generation)
            pending = root / f"pending_r{restart}_g{generation:02d}.json"
            if pending.exists():
                pending_payload = json.loads(pending.read_text())
                vectors = [np.asarray(value, float) for value in pending_payload["vectors"]]
                candidate_paths = [Path(path) for path in pending_payload["candidate_paths"]]
            else:
                vectors = optimizer.ask()
                candidate_paths = []
                for index, vector in enumerate(vectors):
                    candidate = candidate_from_vector(
                        subject_id, vector, self.config, self.basis,
                        generation=generation, candidate_index=index, restart=restart,
                    )
                    candidate_paths.append(self._save_candidate(subject_id, candidate))
                atomic_json({
                    "restart": restart, "generation": generation,
                    "vectors": [value.tolist() for value in vectors],
                    "candidate_paths": [str(path) for path in candidate_paths],
                }, pending)
            seed = int(self.config["search"]["fit_network_seeds"][generation])
            jobs = [{
                "subject_id": subject_id, "candidate_id": path.stem,
                "candidate_json": path, "seed": seed, "phase": "fit",
            } for path in candidate_paths]
            outputs = self.run_jobs(jobs, f"fit:{subject_id}:r{restart}:g{generation}")
            keys, rows = [], []
            for path, vector, output in zip(candidate_paths, vectors, outputs):
                objective = float(output["objective"]["objective"])
                keys.append(-objective)
                rows.append({
                    "candidate_id": path.stem, "candidate_json": str(path),
                    "restart": restart, "generation": generation, "seed": seed,
                    "objective": objective,
                    "score_status": output["objective"]["status"],
                    "n_returned_events": output["n_returned_events"],
                    "runaway": bool(output["runaway"]),
                    "optimizer_vector": np.asarray(vector, float).tolist(),
                })
            optimizer.tell(vectors, keys)
            history.extend(rows)
            pending.unlink(missing_ok=True)
            state = {
                "subject_id": subject_id, "restart": restart,
                "optimizer": optimizer.get_state(), "history": history,
                "expected_git_commit": self.expected_commit,
                "config_sha256": sha256(self.config_path),
            }
            atomic_json(state, checkpoint)
            if (optimizer.generation == int(self.config["search"]["restart_after_generations_without_evaluable"])
                    and restart < int(self.config["search"]["maximum_restarts"])
                    and not any(row["score_status"] == "EVALUABLE" for row in history)):
                restart += 1
                optimizer = CMAES(
                    initial_vector(subject_id, self.config, restart=restart),
                    float(self.config["search"]["sigma0"]),
                    seed=20261819 + restart,
                    popsize=int(self.config["search"]["population_size"]),
                )
                state.update(restart=restart, optimizer=optimizer.get_state())
                atomic_json(state, checkpoint)
        return state

    def canary(self, subjects: list[str]) -> None:
        summaries = []
        for subject_id in subjects:
            state = self.ensure_fit(subject_id, 1)
            best = min(state["history"], key=lambda row: row["objective"])
            output = self.run_jobs([{
                "subject_id": subject_id, "candidate_id": best["candidate_id"],
                "candidate_json": Path(best["candidate_json"]),
                "seed": int(self.config["search"]["selection_network_seeds"][0]),
                "phase": "canary",
            }], f"canary-replay:{subject_id}")[0]
            summaries.append({
                "subject_id": subject_id, "best_fit_objective": best["objective"],
                "replay_status": output["status"],
                "replay_score_status": output["objective"]["status"],
                "runaway": output["runaway"],
            })
        atomic_json({
            "status": "ENGINEERING_CANARY_COMPLETE",
            "subjects": summaries,
            "scientific_selection_changed": False,
        }, self.output / "CANARY.json")
        subprocess.run(["notify-send", "Topic 4 patient-specific cohort", "Engineering canary complete; full queue continuing"], check=False)

    def select(self, subject_id: str, state: dict) -> dict:
        root = self._subject_root(subject_id)
        path = root / "selection.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if (payload.get("expected_git_commit") != self.expected_commit
                    or payload.get("config_sha256") != sha256(self.config_path)):
                raise RuntimeError(f"stale selection artifact: {path}")
            return payload
        count = int(self.config["search"]["selection_candidate_count"])
        ranked = sorted(state["history"], key=lambda row: row["objective"])
        chosen, seen = [], set()
        for row in ranked:
            candidate_hash = sha256(row["candidate_json"])
            if candidate_hash in seen:
                continue
            seen.add(candidate_hash)
            chosen.append(row)
            if len(chosen) == count:
                break
        jobs = []
        for row in chosen:
            for seed in self.config["search"]["selection_network_seeds"]:
                jobs.append({
                    "subject_id": subject_id, "candidate_id": row["candidate_id"],
                    "candidate_json": Path(row["candidate_json"]),
                    "seed": int(seed), "phase": "selection",
                })
        outputs = self.run_jobs(jobs, f"selection:{subject_id}")
        aggregates = []
        offset = 0
        for row in chosen:
            chunk = outputs[offset:offset + len(self.config["search"]["selection_network_seeds"])]
            offset += len(chunk)
            aggregates.append({
                "candidate_id": row["candidate_id"],
                "candidate_json": row["candidate_json"],
                "fit_objective": row["objective"],
                "selection_objectives": [value["objective"]["objective"] for value in chunk],
                "mean_selection_objective": float(np.mean([
                    value["objective"]["objective"] for value in chunk
                ])),
                "score_statuses": [value["objective"]["status"] for value in chunk],
            })
        winner = min(aggregates, key=lambda row: row["mean_selection_objective"])
        payload = {
            "status": "PATIENT_WINNER_SELECTED_ON_TRAIN_ONLY",
            "subject_id": subject_id, "candidates": aggregates,
            "winner": winner,
            "heldout_read": False,
            "expected_git_commit": self.expected_commit,
            "config_sha256": sha256(self.config_path),
        }
        atomic_json(payload, path)
        return payload

    def confirm(self, subject_id: str, selection: dict) -> dict:
        root = self._subject_root(subject_id)
        path = root / "confirmation.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if (payload.get("expected_git_commit") != self.expected_commit
                    or payload.get("config_sha256") != sha256(self.config_path)):
                raise RuntimeError(f"stale confirmation artifact: {path}")
            return payload
        winner = selection["winner"]
        jobs = [{
            "subject_id": subject_id, "candidate_id": winner["candidate_id"],
            "candidate_json": Path(winner["candidate_json"]),
            "seed": int(seed), "phase": "confirmation", "store_envelope": True,
        } for seed in self.config["search"]["confirmation_network_seeds"]]
        outputs = self.run_jobs(jobs, f"confirmation:{subject_id}")
        rows = [value["confirmation"] for value in outputs]
        payload = {
            "status": "PATIENT_HELDOUT_CONFIRMATION_COMPLETE",
            "subject_id": subject_id,
            "winner": winner,
            "network_seeds": self.config["search"]["confirmation_network_seeds"],
            "per_network": rows,
            "mean_observed_weakest_mode_loss": float(np.mean([
                row["observed_weakest_mode_loss"] for row in rows
            ])),
            "mean_null_median": float(np.mean([row["null_median"] for row in rows])),
            "mean_null_advantage": float(np.mean([
                row["delta_null_median_minus_observed"] for row in rows
            ])),
            "same_network_k2_count": int(sum(
                bool(row["natural_kmeans"]["same_network_k2"]) for row in rows
            )),
            "mean_ood_fraction": float(np.mean([
                row["ood_fraction"] if row["ood_fraction"] is not None else 1.0
                for row in rows
            ])),
            "expected_git_commit": self.expected_commit,
            "config_sha256": sha256(self.config_path),
        }
        atomic_json(payload, path)
        return payload

    def mechanism(self, subject_id: str, selection: dict) -> dict:
        root = self._subject_root(subject_id)
        path = root / "mechanism_replay.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if (payload.get("expected_git_commit") != self.expected_commit
                    or payload.get("config_sha256") != sha256(self.config_path)):
                raise RuntimeError(f"stale mechanism artifact: {path}")
            return payload
        winner_path = Path(selection["winner"]["candidate_json"])
        winner = json.loads(winner_path.read_text())
        edge = np.asarray(winner["edge_coefficients"], float)
        arms = {
            "node_only": np.zeros_like(edge),
            "node_plus_ee": np.vstack([edge[0], np.zeros(6)]),
            "node_plus_etoi": np.vstack([np.zeros(6), edge[1]]),
            "node_plus_joint": edge,
        }
        variants = []
        for arm, values in arms.items():
            candidate = copy.deepcopy(winner)
            candidate["candidate_id"] = f"{winner['candidate_id']}_{arm}"
            candidate["edge_coefficients"] = values.tolist()
            candidate["edge_coefficients_sha256"] = array_sha256(values)
            variants.append((arm, self._save_candidate(subject_id, candidate), self.runtime_mode))
        paired = paired_runtime_mode(self.runtime_mode)
        variants.append((f"node_plus_joint_{paired}", winner_path, paired))
        jobs = []
        for arm, candidate_path, runtime_mode in variants:
            candidate_id = json.loads(candidate_path.read_text())["candidate_id"]
            for seed in self.config["search"]["mechanism_network_seeds"]:
                jobs.append({
                    "subject_id": subject_id, "candidate_id": f"{candidate_id}_{arm}",
                    "candidate_json": candidate_path, "seed": int(seed),
                    "phase": "mechanism", "runtime_mode": runtime_mode,
                })
        outputs = self.run_jobs(jobs, f"mechanism:{subject_id}")
        summary, offset = {}, 0
        for arm, _, runtime_mode in variants:
            chunk = outputs[offset:offset + len(self.config["search"]["mechanism_network_seeds"])]
            offset += len(chunk)
            summary[arm] = {
                "runtime_mode": runtime_mode,
                "runaway_early_stop_ms": [value.get("runaway_early_stop_ms") for value in chunk],
                "simulated_until_ms": [value.get("simulated_until_ms") for value in chunk],
                "objectives": [value["objective"]["objective"] for value in chunk],
                "score_statuses": [value["objective"]["status"] for value in chunk],
                "events": [value["n_returned_events"] for value in chunk],
                "ood_fraction": [value["score"].get("ood_fraction") for value in chunk],
                "weakest_mode_loss": [value["score"].get("weakest_mode_loss") for value in chunk],
                "supervised_mode_counts": [
                    value["score"].get("supervised_mode_counts") for value in chunk
                ],
                "natural_kmeans": [value["score"].get("natural_kmeans") for value in chunk],
            }
        payload = {
            "status": "FROZEN_WINNER_MECHANISM_REPLAY_COMPLETE",
            "subject_id": subject_id, "arms": summary,
            "expected_git_commit": self.expected_commit,
            "config_sha256": sha256(self.config_path),
        }
        atomic_json(payload, path)
        return payload

    def finalize_subject(self, subject_id: str) -> dict:
        state = self.ensure_fit(subject_id, int(self.config["search"]["generations"]))
        selection = self.select(subject_id, state)
        confirmation = self.confirm(subject_id, selection)
        mechanism = self.mechanism(subject_id, selection)
        payload = {
            "status": "PATIENT_PIPELINE_COMPLETE", "subject_id": subject_id,
            "n_fit_candidates": len(state["history"]),
            "selection": selection, "confirmation": confirmation,
            "mechanism": mechanism,
        }
        atomic_json(payload, self._subject_root(subject_id) / "FINAL.json")
        return payload

    def aggregate(self, subjects: list[str]) -> dict:
        rows = []
        development = self.config["cohort"]["development_source_subject"]
        for subject_id in subjects:
            final = json.loads((self._subject_root(subject_id) / "FINAL.json").read_text())
            confirmation = final["confirmation"]
            rows.append({
                "subject_id": subject_id,
                "development_source": subject_id == development,
                "heldout_null_advantage": confirmation["mean_null_advantage"],
                "heldout_weakest_mode_loss": confirmation["mean_observed_weakest_mode_loss"],
                "same_network_k2_count_of_4": confirmation["same_network_k2_count"],
                "mean_ood_fraction": confirmation["mean_ood_fraction"],
            })
        primary = [row for row in rows if not row["development_source"]]
        deltas = np.asarray([row["heldout_null_advantage"] for row in primary], float)
        finite = deltas[np.isfinite(deltas)]
        nonzero = finite[finite != 0.0]
        n_positive = int(np.sum(nonzero > 0.0))
        payload = {
            "status": "PATIENT_SPECIFIC_COHORT_SIMULATION_COMPLETE",
            "n_real_geometry_fitted": len(rows),
            "n_primary_nondevelopment": len(primary),
            "n_positive_null_advantage": int(np.sum(finite > 0.0)),
            "median_null_advantage": float(np.median(finite)),
            "null_advantage_sign_test_p": (
                float(binomtest(n_positive, len(nonzero), 0.5).pvalue)
                if len(nonzero) else None
            ),
            "null_advantage_wilcoxon_p": (
                float(wilcoxon(finite, zero_method="wilcox").pvalue)
                if len(nonzero) >= 6 else None
            ),
            "same_network_k2_majority": int(sum(
                row["same_network_k2_count_of_4"] >= 3 for row in primary
            )),
            "subjects": rows,
            "claim_boundary": self.config["claim_boundary"],
        }
        atomic_json(payload, self.output / "COHORT_RESULT.json")
        return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    supervisor = Supervisor(args.config, args.expected_commit)
    eligibility = json.loads((supervisor.output / "GEOMETRY_ELIGIBILITY.json").read_text())
    subjects = list(eligibility["eligible_subjects"])
    canaries = list(supervisor.config["cohort"]["engineering_canary_subjects"])
    try:
        supervisor.status("STARTING", phase="canary", commit=supervisor.expected_commit)
        supervisor.canary(canaries)
        if args.canary_only:
            supervisor.status("CANARY_COMPLETE", commit=supervisor.expected_commit)
            return
        ordered = canaries + [subject for subject in subjects if subject not in canaries]
        supervisor.status("RUNNING_COHORT", subjects=len(ordered),
                          subject_concurrency=supervisor.subject_concurrency,
                          max_workers=supervisor.admission.max_workers)
        with ThreadPoolExecutor(max_workers=supervisor.subject_concurrency) as pool:
            futures = {pool.submit(supervisor.finalize_subject, subject_id): subject_id
                       for subject_id in ordered}
            for future in as_completed(futures):
                future.result()
        result = supervisor.aggregate(ordered)
        finalizer = ROOT / "scripts/finalize_topic4_patient_specific_field_cohort_v2.py"
        subprocess.run([
            str(PYTHON), str(finalizer), "--config", str(supervisor.config_path),
            "--expected-commit", supervisor.expected_commit,
        ], cwd=ROOT, check=True)
        supervisor.status("DONE", fitted=result["n_real_geometry_fitted"],
                          commit=supervisor.expected_commit)
        subprocess.run(["notify-send", "Topic 4 patient-specific cohort",
                        f"Run and figures complete for {result['n_real_geometry_fitted']} patients"], check=False)
    except Exception as exc:
        supervisor.status("FAILED", error=repr(exc), commit=supervisor.expected_commit)
        subprocess.run(["notify-send", "Topic 4 patient-specific cohort failed", repr(exc)], check=False)
        raise


if __name__ == "__main__":
    main()
