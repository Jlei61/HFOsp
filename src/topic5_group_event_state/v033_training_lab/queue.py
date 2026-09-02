"""Training queue: request ingest, units, search driver, controller and worker (design §9).

One process (the controller) owns the queue.  Every unit of work (T0, one
recipe x seed x rung, the card) is an independent worker subprocess launched in
its own session (``setsid``), with its own log, PID, heartbeat and atomic
status; the driver that expands a request into rungs is a state machine over
files, so a restarted controller resumes instead of repeating.

Contract clauses (plan Task 9):
  [K1] ingest writes one atomic status per request (INVALID / HELD / PENDING), idempotent on the request hash;
  [K2] a unit whose job key already completed is SKIPPED_EXISTING;
  [K3] OOM -> OOM_RETRYABLE and the honest back-off ladder (chunk -> checkpointing -> smaller chunk; there is no
       gradient accumulation to fall back on in a full-batch closed-form scan), RESOURCE_UNRESOLVED after 3 retries;
  [K4] NaN -> NAN plus one derived diagnostic unit (LR x0.5, AMP off) that never counts as science;
  [K5] STALE only when the PID is gone AND the heartbeat is old; only unfinished units are resumed;
  [K6] never more workers than the concurrency plan / lease; no human unit without the release;
  [K7] the agent status page carries commit, sealed flag, heartbeat, counts, resources, leases and rationale;
  [K8] processes are managed only by their recorded PID / PGID.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .card import build_training_card
from .diagnostics import (
    blocked_inner_val_gain,
    random_reservoir_delta,
    run_t0,
    shift_null,
    state_output_modulation,
    state_variance_rank,
    synthetic_recovery,
)
from .models import ArchConfig
from .objective import TRAINABLE_REGISTRY
from .paths import (
    AGENT_B_ROOT,
    PYTHON,
    SHARED_ROOT,
    atomic_write_json,
    current_commit,
    payload_hash,
    release_status,
    repo_root,
    results_index,
    set_single_thread_env,
)
from .request import JobStatus, hash_mismatch_verdict, is_human_view, job_key, parse_request, validate_request
from .resources import DEFAULT_LEASE, plan_concurrency, read_supervisor_lease, snapshot, write_agent_lease
from .search import SearchBudget, SearchSpace, _unit_row, asha_promote, classify_failure
from .trainer import RecipeConfig, load_trained, train_recipe
from .views import ViewHeld, view_for_request

EXIT_OK, EXIT_FAILED, EXIT_OOM, EXIT_NAN, EXIT_HELD = 0, 1, 3, 4, 5
UNIT_TYPES = ("t0", "train", "card", "sentinel")
HEARTBEAT_SECONDS = 60.0
STALE_TIMEOUT_SECONDS = 900.0
MAX_OOM_RETRIES = 3
DEFAULT_SENTINEL = {"workload_class": "unmeasured", "uses_gpu": False, "peak_reserved_gib": 0.0, "rss_peak_gib": 4.0,
                    "threads": 1}
SCRIPT = repo_root() / "scripts" / "run_group_event_state_v033_training_lab.py"


# ------------------------------------------------------------------- units
@dataclass
class Unit:
    unit_id: str
    unit_type: str
    request_id: str
    job_key: str
    params: dict[str, Any]
    out_dir: str
    attempt: int = 0
    gpu: int | None = None
    derived_from: str | None = None
    diagnostic_rerun: bool = False
    workload_class: str = "cpu_train_fixed_leaky"
    backoff: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Unit":
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    @property
    def status_path(self) -> Path:
        return Path(self.out_dir) / "unit_status.json"

    @property
    def result_path(self) -> Path:
        return Path(self.out_dir) / "unit_result.json"


def read_json(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_unit_status(unit: Unit, status: str, **fields: Any) -> None:
    previous = read_json(unit.status_path) or {}
    atomic_write_json(unit.status_path, {**previous, "unit_id": unit.unit_id, "unit_type": unit.unit_type,
                                         "request_id": unit.request_id, "job_key": unit.job_key, "status": status,
                                         "attempt": unit.attempt, "updated_epoch": time.time(), **fields})


def pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def detect_stale(status: Mapping[str, Any], *, now: float, heartbeat_timeout: float = STALE_TIMEOUT_SECONDS) -> bool:
    """[K5] RUNNING with a dead PID and an old heartbeat -- both, never one."""

    if status.get("status") != JobStatus.RUNNING.value:
        return False
    if pid_alive(status.get("pid")):
        return False
    return (now - float(status.get("heartbeat_epoch", 0.0))) > float(heartbeat_timeout)


def classify_exit(exit_code: int, log_tail: str) -> str:
    """[K3] Exit code / traceback -> job status."""

    if int(exit_code) == EXIT_OK:
        return JobStatus.COMPLETE.value
    if int(exit_code) == EXIT_OOM or "out of memory" in (log_tail or "").lower():
        return JobStatus.OOM_RETRYABLE.value
    if int(exit_code) == EXIT_NAN:
        return JobStatus.NAN.value
    if int(exit_code) == EXIT_HELD:
        return JobStatus.HELD_NO_RELEASE.value
    return JobStatus.FAILED.value


def oom_backoff(params: Mapping[str, Any], attempt: int) -> dict[str, Any] | None:
    """[K3] chunk/2 -> encoder checkpointing -> chunk/4 more; None after MAX_OOM_RETRIES."""

    if int(attempt) > MAX_OOM_RETRIES:
        return None
    out = json.loads(json.dumps(dict(params)))
    recipe = out.setdefault("recipe", {})
    arch = recipe.setdefault("arch", {})
    chunk = float(arch.get("chunk_seconds", 3600.0))
    if int(attempt) == 1:
        arch["chunk_seconds"] = chunk / 2.0
    elif int(attempt) == 2:
        recipe["checkpointing"] = True
    else:
        recipe["checkpointing"] = True
        arch["chunk_seconds"] = chunk / 4.0
    out["checkpointing"] = bool(recipe.get("checkpointing", False))
    out["gradient_accumulation"] = "not_applicable_full_batch_closed_form_scan"
    out["backoff_attempt"] = int(attempt)
    return out


def derive_nan_diagnostic_unit(unit: Unit) -> Unit:
    """[K4] Same unit at half the learning rates with AMP off; labelled, never counted as science."""

    params = json.loads(json.dumps(unit.params))
    recipe = params.setdefault("recipe", {})
    recipe["lr"] = {k: float(v) * 0.5 for k, v in (recipe.get("lr") or {}).items()}
    recipe["amp_encoder"] = False
    return Unit(unit_id=f"{unit.unit_id}_nandiag", unit_type=unit.unit_type, request_id=unit.request_id,
                job_key=payload_hash({"base": unit.job_key, "nandiag": True}), params=params,
                out_dir=str(Path(unit.out_dir).parent / f"{Path(unit.out_dir).name}_nandiag"), attempt=0, gpu=unit.gpu,
                derived_from=unit.unit_id, diagnostic_rerun=True, workload_class=unit.workload_class)


# ------------------------------------------------------------------ ingest
def request_hash(payload: Mapping[str, Any]) -> str:
    return payload_hash(dict(payload))


def _status_path(shared_root: Path, request_id: str) -> Path:
    return Path(shared_root) / "job_status" / f"training_{request_id}.json"


def ingest_requests(shared_root: Path, agent_root: Path, *, registered: Sequence[str], release_present: bool,
                    head_commit: str) -> list[dict[str, Any]]:
    """[K1] Validate every request file; write ``job_status/training_<id>.json`` atomically; idempotent."""

    shared_root, agent_root = Path(shared_root), Path(agent_root)
    out: list[dict[str, Any]] = []
    for path in sorted((shared_root / "job_requests").glob("science_*.json")):
        if path.name.endswith(".SUPERVISOR_HOLD.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            payload = {"request_id": path.stem, "_unreadable": str(exc)}
        rid = str(payload.get("request_id") or path.stem)
        rh = request_hash(payload)
        status_path = _status_path(shared_root, rid)
        existing = read_json(status_path)
        reevaluate = existing is None or existing.get("request_hash") != rh \
            or existing.get("trainer_code_commit") != head_commit or existing.get("status") in (
            JobStatus.HELD_NO_RELEASE.value, JobStatus.HELD_CODE_COMMIT_MISMATCH.value)
        if not reevaluate:
            out.append({"request_id": rid, "status": existing["status"], "reasons": existing.get("reasons", [])})
            continue
        hold_path = path.with_name(f"{path.stem}.SUPERVISOR_HOLD.json")
        if hold_path.exists():
            hold = read_json(hold_path) or {}
            verdict = {"status": JobStatus.HELD_MISMATCH.value, "missing_fields": [],
                       "reasons": [f"supervisor hold: {hold.get('reason', hold_path.name)}"]}
        else:
            verdict = validate_request(payload, registered_objectives=registered, release_present=release_present,
                                       head_commit=head_commit)
        bound: dict[str, Any] = {}
        if verdict["status"] == JobStatus.PENDING.value:
            request, _ = parse_request(payload)
            auto = request.split_hash == "auto" or request.input_hash == "auto"
            if auto and is_human_view(request.input_view):
                verdict = {"status": JobStatus.INVALID_REQUEST.value, "missing_fields": [],
                           "reasons": ["split_hash/input_hash='auto' is not allowed for a human input view; "
                                       "Agent C must supply the hashes of the view it built"]}
            else:
                try:
                    view, meta = view_for_request(payload, release_present=release_present)
                    mismatch: list[str] = []
                    if request.split_hash != "auto" and request.split_hash != view.split_hash:
                        mismatch.append(f"split_hash request={request.split_hash} view={view.split_hash}")
                    if request.input_hash != "auto" and request.input_hash != view.input_hash:
                        mismatch.append(f"input_hash request={request.input_hash} view={view.input_hash}")
                    if view.missing_h_bins:
                        mismatch.append(f"baseline_H lacks required bins {view.missing_h_bins}")
                    verdict = ({"status": JobStatus.HELD_MISMATCH.value, "reasons": mismatch, "missing_fields": []}
                               if mismatch else {"status": JobStatus.PENDING.value, "reasons": [], "missing_fields": []})
                    bound = {"split_hash": view.split_hash, "input_hash": view.input_hash, "view_meta": meta,
                             "view_summary": view.summary()}
                except ViewHeld as exc:
                    verdict = {"status": JobStatus.HELD_NO_RELEASE.value, "reasons": [str(exc)], "missing_fields": []}
                except Exception as exc:  # noqa: BLE001 - surfaced, never hidden
                    verdict = {"status": JobStatus.FAILED.value, "reasons": [f"view build failed: {type(exc).__name__}: {exc}"],
                               "missing_fields": []}
        ingested_epoch = existing.get("ingested_epoch") if existing and existing.get("request_hash") == rh else time.time()
        record = {"request_id": rid, "owner": "agent_b", "status": verdict["status"], "reasons": verdict.get("reasons", []),
                  "missing_fields": verdict.get("missing_fields", []), "request_hash": rh, "request_file": str(path),
                  "ingested_epoch": ingested_epoch, "updated_epoch": time.time(), "release_present": bool(release_present),
                  "trainer_code_commit": head_commit, **bound}
        atomic_write_json(status_path, record)
        if verdict["status"] not in (JobStatus.INVALID_REQUEST.value,):
            merged = dict(payload)
            if bound:
                merged["split_hash"], merged["input_hash"] = bound["split_hash"], bound["input_hash"]
            atomic_write_json(agent_root / "requests" / rid / "request.json", merged)
        out.append({"request_id": rid, "status": verdict["status"], "reasons": verdict.get("reasons", [])})
    return out


def update_job_status(shared_root: Path, request_id: str, **fields: Any) -> None:
    path = _status_path(shared_root, request_id)
    previous = read_json(path) or {"request_id": request_id, "owner": "agent_b"}
    atomic_write_json(path, {**previous, **fields, "updated_epoch": time.time()})


# ------------------------------------------------------------------ recipes
def recipe_from_dict(payload: Mapping[str, Any]) -> RecipeConfig:
    data = dict(payload)
    arch = dict(data.pop("arch", {}))
    if "taus_seconds" in arch:
        arch["taus_seconds"] = tuple(float(v) for v in arch["taus_seconds"])
    known = {k: v for k, v in data.items() if k in RecipeConfig.__dataclass_fields__}
    return RecipeConfig(arch=ArchConfig(**arch), **known).validate()


# ---------------------------------------------------------------- execute
def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def execute_unit(unit: Unit, *, device: str = "cpu", release_present: bool | None = None) -> int:
    """Run one unit in-process; the worker and the tests share this path.  [K2] skip on same job key."""

    out_dir = Path(unit.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = read_json(unit.result_path)
    if previous and previous.get("job_key") == unit.job_key and previous.get("status") == JobStatus.COMPLETE.value:
        write_unit_status(unit, JobStatus.SKIPPED_EXISTING.value, finished_epoch=time.time())
        return EXIT_OK
    write_unit_status(unit, JobStatus.RUNNING.value, pid=os.getpid(), pgid=os.getpgid(0), started_epoch=time.time(),
                      heartbeat_epoch=time.time(), device=str(device))
    started = time.time()
    try:
        release = release_status()["present"] if release_present is None else bool(release_present)
        dev = _device(device)
        request = unit.params["request"]
        scaling = str(unit.params.get("scaling", "zscore"))
        view, view_meta = view_for_request(request, release_present=release, scaling=scaling)
        trainable = TRAINABLE_REGISTRY[str(request["scientific_target"]["objective"])]()
        seed = int(unit.params.get("seed", 0))
        payload: dict[str, Any] = {"job_key": unit.job_key, "unit_id": unit.unit_id, "unit_type": unit.unit_type,
                                   "request_id": unit.request_id, "view_meta": view_meta, "device": str(dev),
                                   "diagnostic_rerun": unit.diagnostic_rerun, "attempt": unit.attempt}
        if unit.unit_type == "t0":
            cfg = recipe_from_dict(unit.params.get("recipe") or RecipeConfig().as_dict()).with_overrides(scaling=view.scaling)
            true_state = None
            if view_meta.get("kind") in ("synthetic",):
                true_state = None  # the planted z is not exposed to the lab; oracle head is Agent A's Level 0
            report = run_t0(trainable, view, cfg, seed, device=dev, out_dir=out_dir / "t0",
                            tiny_steps=int(unit.params.get("tiny_steps", 300)),
                            probe_steps=int(unit.params.get("probe_steps", 50)), true_state=true_state)
            payload.update({"t0_path": str(out_dir / "t0" / "t0.json"), "gradient_path_ok": report["gradient_path_ok"],
                            "tiny_overfit_pass": report["tiny_slice_overfit"]["pass"]})
        elif unit.unit_type == "train":
            cfg = recipe_from_dict(unit.params["recipe"])
            if cfg.scaling != view.scaling:
                view, view_meta = view_for_request(request, release_present=release, scaling=cfg.scaling)
            result = train_recipe(trainable, view, cfg, seed, device=dev, out_dir=out_dir / "run",
                                  arm=str(unit.params.get("arm", "learned")),
                                  steps_budget=int(unit.params["steps_budget"]) if unit.params.get("steps_budget") else None)
            payload.update({"train_status": result["status"], "result_path": str(out_dir / "run" / "result.json"),
                            "selected_step": result.get("selected_step"), "best_validation": result.get("best_validation")})
            if result["status"] == "nan":
                payload["status"] = JobStatus.NAN.value
                atomic_write_json(unit.result_path, payload)
                write_unit_status(unit, JobStatus.NAN.value, finished_epoch=time.time(),
                                  nan_dump=str(out_dir / "run" / "nan_dump.json"))
                return EXIT_NAN
        elif unit.unit_type == "card":
            cfg = recipe_from_dict(unit.params["recipe"])
            if cfg.scaling != view.scaling:
                view, view_meta = view_for_request(request, release_present=release, scaling=cfg.scaling)
            learned_dir = Path(unit.params["learned_dir"])
            learned = read_json(learned_dir / "result.json") or {}
            seeds = [read_json(Path(p) / "result.json") or {} for p in unit.params.get("seed_dirs", [])]
            model = load_trained(learned_dir, trainable, view, dev)
            diagnostics = {
                "blocked_inner_val_gain": blocked_inner_val_gain(trainable, view, model, device=dev),
                "shift_null": shift_null(trainable, view, model, device=dev),
                "state_variance_rank": state_variance_rank(trainable, view, model, device=dev),
                "state_output_modulation": state_output_modulation(trainable, view, model, device=dev),
                "random_reservoir_delta": random_reservoir_delta(trainable, view, cfg, seed, device=dev, out_dir=out_dir,
                                                                 learned_dir=learned_dir),
                "synthetic_recovery": synthetic_recovery(trainable, view, cfg, seed, device=dev, out_dir=out_dir,
                                                         beta=float(unit.params.get("synthetic_beta", 0.7))),
            }
            t0 = read_json(Path(unit.params["t0_path"])) or {}
            card = build_training_card(request=request, recipe_result=learned, seed_results=seeds, t0=t0,
                                       diagnostics=diagnostics, search_summary=unit.params.get("search_summary"))
            card_path = out_dir / "training_card.json"
            atomic_write_json(card_path, card)
            payload.update({"card_path": str(card_path), "evidence_label": card["evidence_label"]})
        elif unit.unit_type == "sentinel":
            from .resources import run_sentinel

            cfg = recipe_from_dict(unit.params["recipe"])

            def work() -> dict[str, Any]:
                result = train_recipe(trainable, view, cfg, seed, device=dev, out_dir=out_dir / "run", overwrite=True)
                return {"effective_batch": view.n("train"), "n_events": int(view.event_times.size),
                        "steps": result.get("n_steps_run"), "elapsed_seconds": result.get("elapsed_seconds")}

            report = run_sentinel(unit.workload_class, work, out_path=out_dir / "sentinel.json", device=str(dev))
            payload.update({"sentinel_path": str(out_dir / "sentinel.json"), "sentinel": report})
        else:
            raise ValueError(f"unknown unit type {unit.unit_type!r}")
        payload["status"] = JobStatus.COMPLETE.value
        payload["elapsed_seconds"] = time.time() - started
        atomic_write_json(unit.result_path, payload)
        write_unit_status(unit, JobStatus.COMPLETE.value, finished_epoch=time.time())
        return EXIT_OK
    except ViewHeld as exc:
        write_unit_status(unit, JobStatus.HELD_NO_RELEASE.value, finished_epoch=time.time(), error=str(exc))
        return EXIT_HELD
    except BaseException as exc:  # noqa: BLE001 - classified and recorded, never swallowed
        tb = traceback.format_exc()
        if _is_oom(exc):
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            atomic_write_json(out_dir / "oom_dump.json", {"traceback": tb, "peak_allocated_bytes": peak,
                                                          "params": unit.params, "attempt": unit.attempt})
            write_unit_status(unit, JobStatus.OOM_RETRYABLE.value, finished_epoch=time.time(), error=str(exc)[:500])
            return EXIT_OOM
        write_unit_status(unit, JobStatus.FAILED.value, finished_epoch=time.time(), error=str(exc)[:500], traceback=tb[-4000:])
        return EXIT_FAILED


# ------------------------------------------------------------------- driver
class SearchDriver:
    """File-backed state machine: t0 -> ASHA rungs (seed policy) -> card -> done."""

    def __init__(self, request: Mapping[str, Any], agent_root: Path, *, shared_root: Path | None = None,
                 device_hint: str = "cpu") -> None:
        self.request = dict(request)
        self.rid = str(request["request_id"])
        self.agent_root = Path(agent_root)
        self.shared_root = Path(shared_root) if shared_root is not None else self.agent_root.parent / "shared"
        self.root = self.agent_root / "search" / self.rid
        self.root.mkdir(parents=True, exist_ok=True)
        sb = dict(request["search_budget"])
        self.budget = SearchBudget.from_request(sb)
        self.space = SearchSpace.for_family(str(request["state_architecture"]), gated_approved=bool(request.get("exploratory_approved")),
                                            restrict=sb.get("space_restrict"))
        self.max_batches = int(sb.get("max_batches", 4))
        self.tol = float(sb.get("tol", 1e-3))
        self.base_seed = int(request["seed_policy"].get("base_seed", 0))
        self.t0_tiny_steps = int(sb.get("t0_tiny_steps", 300))
        self.t0_probe_steps = int(sb.get("t0_probe_steps", 50))
        self.device_hint = device_hint
        # Agent C owns the immutable science/input commit in the request;
        # Agent B's code identity is a separate part of every unit key.
        self.trainer_commit = current_commit()
        self.state_path = self.root / "driver_state.json"
        self.state: dict[str, Any] = read_json(self.state_path) or {
            "phase": "t0", "batch_index": 0, "rung_index": 0, "configs": {}, "survivors": [], "rows": [],
            "decisions": [], "scores_by_rung": {}, "incumbent": None, "no_improve": 0, "stop_reason": None,
            "batches": [], "card_path": None, "t0_path": None, "grace_deferred": [],
        }
        self.save()

    def save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    # ------------------------------------------------------------ helpers
    def _unit(self, unit_type: str, key: str, params: dict[str, Any], out_dir: Path, *, seed: int,
              config_hash: str, workload_class: str) -> Unit:
        request, _ = parse_request(self.request)
        subject = str(self.request["input_view"].get("subject") or self.request["input_view"].get("base_subject") or "toy")
        return Unit(unit_id=f"{self.rid}__{key}", unit_type=unit_type, request_id=self.rid,
                    job_key=job_key(request, subject=subject, seed=seed,
                                    config_hash=payload_hash({"recipe": config_hash, "unit_key": key,
                                                              "unit_type": unit_type}),
                                    trainer_code_commit=self.trainer_commit),
                    params={**params, "request": self.request}, out_dir=str(out_dir), workload_class=workload_class)

    def _unit_busy(self, out_dir: Path) -> bool:
        status = read_json(out_dir / "unit_status.json")
        return bool(status and status.get("status") == JobStatus.RUNNING.value and pid_alive(status.get("pid")))

    def _unit_terminal_failure(self, out_dir: Path) -> bool:
        status = read_json(out_dir / "unit_status.json")
        return bool(status and status.get("status") in (JobStatus.FAILED.value, JobStatus.RESOURCE_UNRESOLVED.value,
                                                        JobStatus.HELD_NO_RELEASE.value, JobStatus.NAN.value))

    def _start_batch(self) -> None:
        st = self.state
        rng = np.random.default_rng(self.base_seed + 1000 * int(st["batch_index"]))
        st["configs"] = {}
        for i in range(int(self.budget.n_configs)):
            cfg = self.space.sample(rng, budget=self.budget)
            st["configs"][f"b{st['batch_index']}_c{i:03d}"] = cfg.as_dict()
        st["survivors"] = list(st["configs"])
        st["rung_index"] = 0
        st["scores_by_rung"] = {}
        st["decisions"] = []

    def _batch_dir(self) -> Path:
        return self.root / f"batch_{int(self.state['batch_index']):02d}"

    def _seed_for(self, seed_index: int) -> int:
        return self.base_seed + 100 * int(seed_index) + 7 * int(self.state["batch_index"])

    def _workload(self) -> str:
        family = "gated" if self.request["state_architecture"] == "gated_exploratory" else "fixed_leaky"
        kind = "gpu" if str(self.device_hint).startswith("cuda") else "cpu"
        return f"{kind}_train_{family}"

    # ------------------------------------------------------------- phases
    def next_units(self) -> list[Unit]:
        st = self.state
        if st["phase"] == "t0":
            t0_dir = self.root / "t0"
            result = read_json(t0_dir / "unit_result.json")
            if result and result.get("status") == JobStatus.COMPLETE.value:
                st["t0_path"] = result.get("t0_path")
                st["phase"] = "search"
                self._start_batch()
                self.save()
            elif self._unit_terminal_failure(t0_dir):
                st["phase"] = "failed"
                st["stop_reason"] = "t0_unit_failed"
                self.save()
                update_job_status(self.shared_root, self.rid, status=JobStatus.FAILED.value,
                                  reasons=["T0 unit failed; see unit_status.json"], t0_dir=str(t0_dir))
                return []
            elif self._unit_busy(t0_dir):
                return []
            else:
                cfg = RecipeConfig()
                return [self._unit("t0", "t0", {"recipe": cfg.as_dict(), "seed": self.base_seed, "tiny_steps": self.t0_tiny_steps,
                                                "probe_steps": self.t0_probe_steps}, t0_dir, seed=self.base_seed,
                                   config_hash=cfg.config_hash(), workload_class="cpu_t0")]
        if st["phase"] == "search":
            return self._search_units()
        if st["phase"] == "card":
            return self._card_units()
        return []

    def _search_units(self) -> list[Unit]:
        st = self.state
        rungs = self.budget.rung_steps
        rung_index = int(st["rung_index"])
        rung = int(rungs[rung_index])
        is_final = rung_index == len(rungs) - 1
        if is_final and len(rungs) > 1 and rung_index > 0 and not st.get("final_selected"):
            previous = st["scores_by_rung"][str(rung_index - 1)]
            ranked = sorted(st["survivors"], key=lambda c: float(previous[c]["score"]))
            st["survivors"] = ranked[: int(self.budget.n_final)]
            st["final_selected"] = True
            self.save()
        n_seeds = self.budget.seeds_for(rung_index)
        units: list[Unit] = []
        rows: dict[str, list[dict[str, Any]]] = {cid: [] for cid in st["survivors"]}
        for cid in st["survivors"]:
            recipe = st["configs"][cid]
            cfg = recipe_from_dict(recipe)
            for seed_index in range(n_seeds):
                seed = self._seed_for(seed_index)
                out_dir = self._batch_dir() / cid / f"seed_{seed_index}"
                result = read_json(out_dir / "run" / "result.json")
                done = bool(result and (result.get("status") in ("nan", "insufficient_anchors")
                                        or (result.get("status") == "complete"
                                            and (int(result.get("n_steps_run", 0)) >= rung or result.get("stopped_reason") == "patience"))))
                if done:
                    rows[cid].append(_unit_row(cid, cfg, seed, seed_index, rung_index, rung, result))
                elif self._unit_terminal_failure(out_dir):
                    rows[cid].append({"config_id": cid, "config_hash": cfg.config_hash(), "seed": seed, "seed_index": seed_index,
                                      "rung_index": rung_index, "steps_budget": rung, "status": "failed", "inner_val_nll": None,
                                      "gain_h_minus_model": None, "grace_ok": True, "plateau_reached": False,
                                      "selected_at_budget_edge": False})
                elif not self._unit_busy(out_dir):
                    units.append(self._unit("train", f"{cid}_s{seed_index}_r{rung_index}",
                                            {"recipe": recipe, "seed": seed, "steps_budget": rung, "arm": "learned",
                                             "scaling": recipe.get("scaling", "zscore")}, out_dir, seed=seed,
                                            config_hash=cfg.config_hash(), workload_class=self._workload()))
        if units or any(len(rows[cid]) < n_seeds for cid in st["survivors"]):
            return units
        scores: dict[str, dict[str, Any]] = {}
        for cid, seed_rows in rows.items():
            finite = [r["inner_val_nll"] for r in seed_rows if r.get("status") == "complete" and r.get("inner_val_nll") is not None]
            gains = [r["gain_h_minus_model"] for r in seed_rows if r.get("status") == "complete" and r.get("gain_h_minus_model") is not None]
            scores[cid] = {"config_id": cid, "config_hash": st["configs"][cid] and recipe_from_dict(st["configs"][cid]).config_hash(),
                           "score": float(np.median(finite)) if finite else float("inf"),
                           "gain_median": float(np.median(gains)) if gains else None, "n_seeds": len(seed_rows),
                           "grace_ok": all(bool(r.get("grace_ok", True)) for r in seed_rows),
                           "plateau_reached": any(bool(r.get("plateau_reached")) for r in seed_rows),
                           "budget_edge": any(bool(r.get("selected_at_budget_edge")) for r in seed_rows),
                           "rung_index": rung_index, "rung": rung}
            st["rows"].extend(seed_rows)
        st["scores_by_rung"][str(rung_index)] = scores
        if not is_final:
            decision = asha_promote(list(scores.values()), eta=self.budget.eta)
            st["decisions"].append({"rung_index": rung_index, "rung": rung, **decision})
            st["grace_deferred"].extend(decision["grace_deferred"])
            st["survivors"] = decision["promoted"]
            st["rung_index"] = rung_index + 1
            self.save()
            return self._search_units()
        best_id = min(scores, key=lambda c: scores[c]["score"])
        cand = dict(scores[best_id])
        cand["recipe"] = st["configs"][best_id]
        cand["batch_index"] = int(st["batch_index"])
        cand["run_dir"] = str(self._batch_dir() / best_id)
        cand["seed_dirs"] = [str(self._batch_dir() / best_id / f"seed_{i}" / "run") for i in range(n_seeds)]
        improved = st["incumbent"] is None or float(cand["score"]) < float(st["incumbent"]["score"]) - self.tol
        st["batches"].append({"batch_index": int(st["batch_index"]), "incumbent": cand, "improved": bool(improved)})
        if improved:
            st["incumbent"], st["no_improve"] = cand, 0
        else:
            st["no_improve"] = int(st["no_improve"]) + 1
        atomic_write_json(self._batch_dir() / "search_trace.json", {"batch_index": int(st["batch_index"]), "rows": st["rows"],
                                                                     "decisions": st["decisions"]})
        if int(st["no_improve"]) >= 2:
            st["stop_reason"] = "no_improvement_two_batches"
        elif int(st["no_improve"]) >= 1 and bool(st["incumbent"].get("plateau_reached")):
            st["stop_reason"] = "stable_plateau"
        elif int(st["batch_index"]) + 1 >= self.max_batches:
            st["stop_reason"] = "max_batches"
        if st["stop_reason"]:
            st["phase"] = "card"
            st["final_selected"] = False
            self.save()
            return self._card_units()
        st["batch_index"] = int(st["batch_index"]) + 1
        st["final_selected"] = False
        self._start_batch()
        self.save()
        return self._search_units()

    def _card_units(self) -> list[Unit]:
        st = self.state
        inc = st["incumbent"]
        card_dir = self.root / "card"
        result = read_json(card_dir / "unit_result.json")
        if result and result.get("status") == JobStatus.COMPLETE.value:
            st["card_path"] = result.get("card_path")
            st["phase"] = "done"
            self.save()
            self._finalize(result)
            return []
        if self._unit_terminal_failure(card_dir):
            st["phase"] = "failed"
            st["stop_reason"] = st.get("stop_reason") or "card_unit_failed"
            self.save()
            update_job_status(self.shared_root, self.rid, status=JobStatus.FAILED.value, reasons=["card unit failed"])
            return []
        if self._unit_busy(card_dir):
            return []
        cfg = recipe_from_dict(inc["recipe"])
        search_summary = {"incumbent": {k: v for k, v in inc.items() if k != "recipe"}, "stop_reason": st["stop_reason"],
                          "n_batches": len(st["batches"])}
        params = {"recipe": inc["recipe"], "seed": self._seed_for(0), "learned_dir": str(Path(inc["run_dir"]) / "seed_0" / "run"),
                  "seed_dirs": inc["seed_dirs"], "t0_path": st["t0_path"], "search_summary": search_summary,
                  "scaling": inc["recipe"].get("scaling", "zscore")}
        return [self._unit("card", "card", params, card_dir, seed=self._seed_for(0), config_hash=cfg.config_hash(),
                           workload_class=self._workload())]

    def _finalize(self, card_result: Mapping[str, Any]) -> None:
        st = self.state
        card = read_json(Path(card_result["card_path"])) or {}
        t0 = read_json(Path(st["t0_path"])) if st.get("t0_path") else {}
        gain = card.get("blocked_inner_val_gain") or {}
        rr = (card.get("random_reservoir_delta") or {}).get("learned_minus_random") or {}
        curves = card.get("curves") or {}
        obs = {
            "tiny_overfit_pass": (card.get("tiny_overfit") or {}).get("pass"),
            "all_groups_active": card.get("all_groups_active_before_selection"),
            "train_learned": (curves.get("train_nll_last") is not None and curves.get("inner_val_nll_h") is not None
                              and float(curves["train_nll_last"]) < float((t0 or {}).get("probe", {}).get("best_validation", {}).get("inner_val_nll_h", curves["inner_val_nll_h"]))),
            "inner_val_gain_ci_low": gain.get("ci_low"),
            "synthetic_recovery_pass": (card.get("synthetic_recovery") or {}).get("pass"),
            "random_reservoir_equivalent": rr.get("ci_high") is not None and float(rr["ci_high"]) >= 0.0,
            "selected_at_budget_edge": card.get("selected_at_budget_edge"),
            "search_no_improvement_batches": int(st.get("no_improve", 0)),
            "effective_windows": (gain.get("effective_independent_windows") or 0),
            "min_effective_windows": int(self.request.get("search_budget", {}).get("min_effective_windows", 0)),
        }
        classification = classify_failure(obs)
        st["failure_classification"] = classification
        self.save()
        update_job_status(self.shared_root, self.rid, status=JobStatus.COMPLETE.value, card_path=card_result["card_path"],
                          evidence_label=card.get("evidence_label"), failure_classification=classification,
                          stop_reason=st["stop_reason"], incumbent={k: v for k, v in (st["incumbent"] or {}).items() if k != "recipe"},
                          driver_state=str(self.state_path), training_adequacy_is_not_a_scientific_result=True)

    def status(self) -> dict[str, Any]:
        st = self.state
        return {"phase": st["phase"], "batch_index": st["batch_index"], "rung_index": st["rung_index"],
                "n_rows": len(st["rows"]), "stop_reason": st["stop_reason"],
                "incumbent": None if not st["incumbent"] else {k: st["incumbent"].get(k) for k in ("config_id", "score", "gain_median", "n_seeds")}}


# --------------------------------------------------------------- controller
def spawn_worker(unit_path: Path, *, gpu: int | None, log_path: Path) -> dict[str, Any]:
    """New session (setsid), stdin /dev/null, fixed Python, single-thread env; managed by PID/PGID only [K8]."""

    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    device = "cpu"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        device = "cuda:0"
    env["PYTHONPATH"] = str(repo_root())
    # ``--device`` belongs to the top-level parser and therefore must precede
    # the subcommand.  Putting it after ``worker`` makes argparse reject the
    # process before worker_main can publish a terminal status.
    cmd = [PYTHON, str(SCRIPT), "--device", device, "worker", "--unit", str(unit_path)]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, cwd=str(repo_root()))
    return {"pid": proc.pid, "pgid": proc.pid, "cmd": cmd}


def write_agent_status(agent_root: Path, results_dir: Path | None, **fields: Any) -> Path:
    payload = {"agent": "agent_b", "updated_epoch": time.time(), "heartbeat_epoch": time.time(),
               "sealed_partition_opened": False, "controller_pid": os.getpid(), **fields}
    path = Path(agent_root) / "agent_b.status.json"
    atomic_write_json(path, payload)
    if results_dir is not None:
        atomic_write_json(Path(results_dir) / "agent_b.status.json", payload)
    return path


class Controller:
    def __init__(self, shared_root: Path = SHARED_ROOT, agent_root: Path = AGENT_B_ROOT, *, registered: Sequence[str] | None = None,
                 head_commit: str | None = None, release_present: Callable[[], bool] | None = None,
                 spawner: Callable[..., dict[str, Any]] = spawn_worker, lease: Mapping[str, Any] | None = None,
                 results_index: Path | None = None, gpu_ids: Sequence[int] | None = None, poll_seconds: float = 30.0,
                 device_hint: str | None = None,
                 snapshotter: Callable[[], Mapping[str, Any]] | None = None) -> None:
        self.shared_root, self.agent_root = Path(shared_root), Path(agent_root)
        self.registered = tuple(registered or TRAINABLE_REGISTRY)
        self.head_commit = head_commit or current_commit()
        self.release_present = release_present or (lambda: bool(release_status()["present"]))
        self.spawner = spawner
        self.fixed_lease = None if lease is None else dict(lease)
        self.results_index = Path(results_index) if results_index is not None else None
        self.poll_seconds = float(poll_seconds)
        self.drivers: dict[str, SearchDriver] = {}
        self.running: dict[str, dict[str, Any]] = {}
        self.finished: dict[str, str] = {}
        self.stable_cycles = 0
        self.current_cap = 1
        self.device_hint = device_hint
        # Keep the live resource guard in production, but make the resource
        # observation an explicit dependency so queue tests do not depend on
        # unrelated host load from concurrently running scientific jobs.
        self.snapshotter = snapshotter or snapshot
        (self.agent_root / "controller").mkdir(parents=True, exist_ok=True)
        (self.agent_root / "units").mkdir(parents=True, exist_ok=True)
        (self.agent_root / "logs").mkdir(parents=True, exist_ok=True)
        saved = read_json(self.agent_root / "controller" / "running.json") or {}
        for uid, rec in saved.items():
            rec["unit"] = Unit.from_dict(rec["unit"])
            self.running[uid] = rec
        atomic_write_json(self.agent_root / "controller" / "controller.json",
                          {"pid": os.getpid(), "pgid": os.getpgid(0), "started_epoch": time.time(), "commit": self.head_commit})

    # ------------------------------------------------------------- helpers
    def lease(self) -> dict[str, Any]:
        return dict(self.fixed_lease) if self.fixed_lease is not None else read_supervisor_lease(self.shared_root)

    def sentinel_for(self, workload_class: str) -> dict[str, Any]:
        found = read_json(self.agent_root / "sentinels" / f"{workload_class}.json")
        return found or dict(DEFAULT_SENTINEL, workload_class=workload_class)

    def _persist_running(self) -> None:
        atomic_write_json(self.agent_root / "controller" / "running.json",
                          {uid: {**rec, "unit": rec["unit"].to_dict()} for uid, rec in self.running.items()})

    def _poll_running(self, now: float) -> list[Unit]:
        """Read every running unit's atomic status; return follow-up units (backoff / nan diagnostics / resume)."""

        follow: list[Unit] = []
        for uid, rec in list(self.running.items()):
            unit: Unit = rec["unit"]
            status = read_json(unit.status_path) or {}
            state = status.get("status")
            alive = pid_alive(rec.get("pid"))
            if state in (JobStatus.COMPLETE.value, JobStatus.SKIPPED_EXISTING.value, JobStatus.FAILED.value,
                         JobStatus.HELD_NO_RELEASE.value):
                self.finished[uid] = state
                del self.running[uid]
                continue
            if state == JobStatus.OOM_RETRYABLE.value:
                del self.running[uid]
                self.finished[uid] = state
                attempt = int(unit.attempt) + 1
                params = oom_backoff(unit.params, attempt)
                if params is None:
                    write_unit_status(unit, JobStatus.RESOURCE_UNRESOLVED.value, finished_epoch=now)
                    self.finished[uid] = JobStatus.RESOURCE_UNRESOLVED.value
                else:
                    self.current_cap = max(1, self.current_cap - 1)
                    follow.append(Unit(**{**unit.to_dict(), "params": params, "attempt": attempt, "backoff": {"attempt": attempt}}))
                continue
            if state == JobStatus.NAN.value:
                del self.running[uid]
                self.finished[uid] = state
                if not unit.diagnostic_rerun:
                    follow.append(derive_nan_diagnostic_unit(unit))
                continue
            if not alive:
                if detect_stale({**status, "status": JobStatus.RUNNING.value, "pid": rec.get("pid")}, now=now) or \
                        (state == JobStatus.RUNNING.value and not alive and now - float(status.get("heartbeat_epoch", rec.get("started_epoch", now))) > STALE_TIMEOUT_SECONDS):
                    write_unit_status(unit, JobStatus.STALE.value, stale_detected_epoch=now)
                    del self.running[uid]
                    self.finished[uid] = JobStatus.STALE.value
                    if not (read_json(unit.result_path) or {}).get("status") == JobStatus.COMPLETE.value:
                        follow.append(unit)                      # resume: same unit, same out_dir (train_recipe resumes)
                elif state == JobStatus.RUNNING.value and \
                        now - float(rec.get("started_epoch", now)) > max(5.0, 2.0 * self.poll_seconds):
                    # A worker that exits before publishing any terminal state
                    # is a launch/runtime failure, not a long-running job.  Do
                    # not leave the request falsely RUNNING until the much
                    # longer stale-heartbeat timeout expires.
                    write_unit_status(unit, JobStatus.FAILED.value, finished_epoch=now,
                                      error="worker process exited without terminal status")
                    del self.running[uid]
                    self.finished[uid] = JobStatus.FAILED.value
                elif state != JobStatus.RUNNING.value:
                    # process gone without a terminal status and heartbeat still fresh -> treat as failed exit
                    write_unit_status(unit, JobStatus.FAILED.value, finished_epoch=now, error="worker exited without status")
                    del self.running[uid]
                    self.finished[uid] = JobStatus.FAILED.value
        return follow

    def _spawn(self, unit: Unit, gpu: int | None) -> None:
        unit_path = self.agent_root / "units" / f"{unit.unit_id}.json"
        unit.gpu = gpu
        atomic_write_json(unit_path, unit.to_dict())
        log_path = self.agent_root / "logs" / f"{unit.unit_id}.log"
        info = self.spawner(unit_path, gpu=gpu, log_path=log_path)
        write_unit_status(unit, JobStatus.RUNNING.value, pid=info["pid"], pgid=info.get("pgid"), started_epoch=time.time(),
                          heartbeat_epoch=time.time(), log=str(log_path), gpu=gpu)
        self.running[unit.unit_id] = {"unit": unit, "pid": info["pid"], "pgid": info.get("pgid"), "started_epoch": time.time(),
                                      "gpu": gpu, "log": str(log_path)}

    # ---------------------------------------------------------------- step
    def step(self) -> dict[str, Any]:
        now = time.time()
        release = bool(self.release_present())
        ingested = {s["request_id"]: s["status"] for s in ingest_requests(self.shared_root, self.agent_root, registered=self.registered,
                                                                          release_present=release, head_commit=self.head_commit)}
        follow = self._poll_running(now)
        pending: list[Unit] = list(follow)
        active_ids = {rec["unit"].unit_id for rec in self.running.values()} | {u.unit_id for u in pending}
        request_status: dict[str, Any] = {}
        for rid, status in ingested.items():
            if status not in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
                request_status[rid] = {"status": status}
                continue
            request = read_json(self.agent_root / "requests" / rid / "request.json")
            if request is None:
                continue
            driver = self.drivers.get(rid)
            if driver is None:
                driver = self.drivers[rid] = SearchDriver(request, self.agent_root, shared_root=self.shared_root,
                                                          device_hint=self.device_hint or "cpu")
            for unit in driver.next_units():
                if unit.unit_id in active_ids:
                    continue
                if is_human_view(request["input_view"]) and not release:              # [K6]
                    continue
                pending.append(unit)
                active_ids.add(unit.unit_id)
            request_status[rid] = {"status": status, **driver.status()}
            if driver.state["phase"] not in ("done", "failed") and status == JobStatus.PENDING.value:
                update_job_status(self.shared_root, rid, status=JobStatus.RUNNING.value, driver=driver.status())
        lease = self.lease()
        snap = dict(self.snapshotter())
        workload = pending[0].workload_class if pending else "cpu_train_fixed_leaky"
        sentinel = self.sentinel_for(workload)
        threads = int(lease.get("threads_per_worker", 1))
        ceilings = [int(r.get("resource_ceiling", {}).get("max_workers", 10 ** 6)) for r in
                    (read_json(self.agent_root / "requests" / rid / "request.json") or {} for rid in request_status)]
        plan = plan_concurrency(snap, sentinel, lease, pending=len(pending), threads=threads,
                                ceiling=min(ceilings) if ceilings else None, my_running_threads=len(self.running) * threads)
        # ramp: one slot at a time after two stable cycles (supervisor runbook §4.2)
        self.stable_cycles = self.stable_cycles + 1 if plan["slots"] >= self.current_cap else 0
        if self.stable_cycles >= 2 and self.current_cap < plan["slots"]:
            self.current_cap += 1
            self.stable_cycles = 0
        allowed = min(plan["slots"], self.current_cap)
        free = max(allowed - len(self.running), 0)
        gpu_ids = [int(g) for g in lease.get("gpu_ids", [])]
        gpu_in_use = [rec.get("gpu") for rec in self.running.values() if rec.get("gpu") is not None]
        spawned = 0
        for unit in pending[:free]:
            gpu = None
            if unit.workload_class.startswith("gpu") and gpu_ids:
                gpu = min(gpu_ids, key=lambda g: gpu_in_use.count(g))
                gpu_in_use.append(gpu)
            self._spawn(unit, gpu)
            spawned += 1
        self._persist_running()
        counts = {"pending": max(len(pending) - spawned, 0), "running": len(self.running), "spawned_this_step": spawned,
                  "finished": {k: sum(1 for v in self.finished.values() if v == k) for k in set(self.finished.values())}}
        rationale = (f"slots={plan['slots']} binding={plan['binding']} cap={self.current_cap} running={len(self.running)} "
                     f"pending={counts['pending']} release={'present' if release else 'absent'}")
        write_agent_status(self.agent_root, self.results_index, commit=self.head_commit, release_present=release, counts=counts,
                           resources={"snapshot": snap, "plan": plan, "sentinel": sentinel}, lease=lease,
                           requests=request_status, next_batch_rationale=rationale,
                           running_units={uid: {"pid": rec["pid"], "gpu": rec.get("gpu"), "unit_type": rec["unit"].unit_type}
                                          for uid, rec in self.running.items()})
        write_agent_lease(self.shared_root, {"running_units": len(self.running), "gpu_workers": {str(g): gpu_in_use.count(g) for g in gpu_ids},
                                             "threads_per_worker": threads, "lease_source": lease.get("lease_source")})
        return {"ingested": ingested, "spawned": spawned, "running": len(self.running), "slots": plan["slots"],
                "binding": plan["binding"], "pending_units": counts["pending"], "requests": request_status}

    def run(self, *, once: bool = False, stop_file: Path | None = None, max_idle_cycles: int | None = None) -> None:
        stop_file = stop_file or (self.agent_root / "controller" / "STOP")
        idle = 0
        while True:
            report = self.step()
            atomic_write_json(self.agent_root / "controller" / "heartbeat.json", {"epoch": time.time(), "pid": os.getpid(), **report})
            if once or stop_file.exists():
                return
            idle = idle + 1 if (report["running"] == 0 and report["pending_units"] == 0) else 0
            if max_idle_cycles is not None and idle >= max_idle_cycles:
                return
            time.sleep(self.poll_seconds)


# ------------------------------------------------------------------- worker
def worker_main(unit_path: Path, *, device: str = "cpu") -> int:
    set_single_thread_env()
    unit = Unit.from_dict(json.loads(Path(unit_path).read_text()))
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            previous = read_json(unit.status_path) or {}
            if previous.get("status") == JobStatus.RUNNING.value:
                atomic_write_json(unit.status_path, {**previous, "heartbeat_epoch": time.time()})

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        return execute_unit(unit, device=device)
    finally:
        stop.set()
