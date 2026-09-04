"""Small, explicit contracts for the v0.3.4 spatial-state pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import random

import numpy as np
import torch

from src.topic5_group_event_state.v033_training_lab.paths import (
    atomic_write_json,
    file_hash,
    payload_hash,
)


FORMAT_PREFIX = "group_event_state_v0_3_4_spatial_state"
TUNING_SUBJECTS = ("epilepsiae_253", "epilepsiae_916")
EVALUATION_SUBJECTS = (
    "epilepsiae_1146", "epilepsiae_583", "epilepsiae_548", "epilepsiae_922",
)
ALLOWED_PHASES = ("STATE_TRAIN", "STATE_SELECTION")
FORBIDDEN_PHASE_TOKENS = ("DEVELOPMENT", "SEALED", "TEST", "SEIZURE")
RUNGS = (300, 900, 2700)
LOCKED_SEEDS = tuple(range(20260903, 20260908))
SEED_CONTRACT = "python_numpy_torch_seeded_before_model_construction_v2"


LOCKED_ARCH_KEYS = (
    "width", "depth", "write_width", "adapter_rank", "residual", "taus_seconds",
)
LOCKED_TRAIN_KEYS = (
    "max_steps", "validate_every", "patience_checks", "pair_batch_size",
    "anchors_per_step", "events_per_anchor", "burn_in_seconds", "chunk_seconds",
    "grammar_weight", "extent_weight", "lag_weight",
)


def seed_before_model_construction(seed: int) -> None:
    """Set every RNG before constructing a module or optimizer.

    Training repeats the call as a defensive measure, but it is too late to
    control module initialisation there.  All public runners call this first.
    """

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class ArchConfig:
    width: int = 64
    depth: int = 2
    write_width: int = 4
    adapter_rank: int = 4
    residual: bool = True
    taus_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)

    @property
    def state_dim(self) -> int:
        return len(self.taus_seconds) * self.write_width

    def validate(self) -> "ArchConfig":
        if self.width not in (32, 64, 128) or self.depth not in (1, 2, 4):
            raise ValueError("spatial width/depth is outside the pilot surface")
        if self.write_width < 1 or self.adapter_rank < 1:
            raise ValueError("write_width and adapter_rank must be positive")
        if not self.taus_seconds or any(float(x) <= 0 for x in self.taus_seconds):
            raise ValueError("all fixed time constants must be positive")
        if tuple(sorted(self.taus_seconds)) != tuple(self.taus_seconds):
            raise ValueError("time constants must be increasing")
        return self


@dataclass(frozen=True)
class OptimizerConfig:
    """Independent LRs; no inherited hidden multiplier is permitted."""

    lr_encoder: float = 3e-4
    lr_state_adapter: float = 1e-3
    lr_auxiliary: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    gradient_clip: float = 1.0

    def validate(self) -> "OptimizerConfig":
        if min(self.lr_encoder, self.lr_state_adapter, self.lr_auxiliary) <= 0:
            raise ValueError("all spatial-state learning rates must be positive")
        if self.weight_decay < 0 or self.eps <= 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer regularisation/numerics")
        if len(self.betas) != 2 or not all(0 < x < 1 for x in self.betas):
            raise ValueError("AdamW betas must be in (0,1)")
        return self


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 300
    validate_every: int = 25
    patience_checks: int = 8
    pair_batch_size: int = 512
    anchors_per_step: int = 128
    events_per_anchor: int = 16
    burn_in_seconds: float = 1800.0
    chunk_seconds: float = 3600.0
    seed: int = 20260903
    grammar_weight: float = 1.0
    extent_weight: float = 0.2
    lag_weight: float = 0.2

    def validate(self, *, allow_tiny: bool = False) -> "TrainConfig":
        if not allow_tiny and self.max_steps not in RUNGS:
            raise ValueError(f"human/synthetic rung must be one of {RUNGS}")
        if self.max_steps < 1 or self.validate_every < 1 or self.patience_checks < 1:
            raise ValueError("step/validation settings must be positive")
        if min(self.pair_batch_size, self.anchors_per_step, self.events_per_anchor) < 1:
            raise ValueError("batch sizes must be positive")
        if self.burn_in_seconds < 0 or self.chunk_seconds <= 0:
            raise ValueError("burn-in/chunk time must be valid")
        if min(self.grammar_weight, self.extent_weight, self.lag_weight) < 0:
            raise ValueError("loss weights cannot be negative")
        return self


def assert_safe_phases(phases: Sequence[str]) -> None:
    seen = {str(x) for x in phases}
    bad = sorted(x for x in seen if any(t in x.upper() for t in FORBIDDEN_PHASE_TOKENS))
    unknown = sorted(seen - set(ALLOWED_PHASES))
    if bad or unknown:
        raise PermissionError(f"v0.3.4 S_P allows only {ALLOWED_PHASES}; bad={bad}, unknown={unknown}")


def lr_search_cells() -> list[dict[str, float]]:
    """A bounded paired search, not a Cartesian product."""

    return [
        {"lr_encoder": 1e-4, "lr_state_adapter": 3e-4, "lr_auxiliary": 3e-4},
        {"lr_encoder": 3e-4, "lr_state_adapter": 1e-3, "lr_auxiliary": 1e-3},
        {"lr_encoder": 1e-3, "lr_state_adapter": 3e-3, "lr_auxiliary": 1e-3},
        {"lr_encoder": 3e-4, "lr_state_adapter": 3e-3, "lr_auxiliary": 3e-4},
        {"lr_encoder": 1e-3, "lr_state_adapter": 1e-3, "lr_auxiliary": 3e-3},
    ]


def optimizer_contract(config: OptimizerConfig) -> dict[str, Any]:
    config.validate()
    requested = {
        "encoder": config.lr_encoder,
        "state_adapter": config.lr_state_adapter,
        "auxiliary": config.lr_auxiliary,
    }
    # Deliberately duplicate requested/effective so a future multiplier cannot
    # be introduced silently (the v0.3.3 E253 failure mode).
    return {
        "family": "adamw",
        "requested_lr": requested,
        "effective_lr": dict(requested),
        "hidden_lr_multiplier": 1.0,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "eps": config.eps,
        "gradient_clip": config.gradient_clip,
    }


def _valid_card(path: Path, expected_kind: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != f"{FORMAT_PREFIX}_{expected_kind}_card_v1":
        raise ValueError(f"unexpected {expected_kind} card format")
    if payload.get("status") != "PASS":
        raise PermissionError(f"{expected_kind} did not pass")
    if payload.get("development_targets_read") is not False \
            or payload.get("sealed_partition_opened") is not False \
            or payload.get("seizure_outcomes_read") is not False:
        raise PermissionError(f"{expected_kind} card violates data scope")
    return payload, file_hash(Path(path))


def require_synthetic_recovery(path: Path) -> Mapping[str, Any]:
    """A human-input canary may start only after S_P synthetic recovery."""

    payload, _sha = _valid_card(Path(path), "synthetic_recovery")
    return payload


def build_human_release_gate(*, synthetic_card: Path, canary_card: Path, output: Path) -> dict[str, Any]:
    synthetic, synthetic_sha = _valid_card(synthetic_card, "synthetic_recovery")
    canary, canary_sha = _valid_card(canary_card, "tiny_canary")
    payload = {
        "format": f"{FORMAT_PREFIX}_human_release_gate_v1",
        "status": "PASS",
        "scope": "tuning_subjects_only",
        "allowed_subjects": list(TUNING_SUBJECTS),
        "synthetic_card": str(Path(synthetic_card)),
        "synthetic_card_sha256": synthetic_sha,
        "canary_card": str(Path(canary_card)),
        "canary_card_sha256": canary_sha,
        "synthetic_contract_hash": synthetic.get("contract_hash"),
        "canary_contract_hash": canary.get("contract_hash"),
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    payload["content_hash"] = payload_hash(payload)
    atomic_write_json(Path(output), payload)
    return payload


def require_human_release_gate(path: Path, *, subject: str) -> Mapping[str, Any]:
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    if gate.get("format") != f"{FORMAT_PREFIX}_human_release_gate_v1" \
            or gate.get("status") != "PASS" or gate.get("scope") != "tuning_subjects_only":
        raise PermissionError("human S_P requires a valid v0.3.4 release gate")
    if subject not in gate.get("allowed_subjects", []) or subject not in TUNING_SUBJECTS:
        raise PermissionError(f"human S_P is not authorized for {subject}")
    for name in ("synthetic", "canary"):
        source = Path(str(gate[f"{name}_card"]))
        if not source.is_file() or file_hash(source) != gate[f"{name}_card_sha256"]:
            raise PermissionError(f"{name} gate evidence changed")
    if gate.get("development_targets_read") is not False \
            or gate.get("sealed_partition_opened") is not False \
            or gate.get("seizure_outcomes_read") is not False:
        raise PermissionError("human release gate has forbidden data provenance")
    return gate


def _assert_no_forbidden_provenance(payload: Mapping[str, Any], *, label: str) -> None:
    """Reject any evidence that admits later outcomes into fitting/selection."""

    for key in ("development_targets_read", "sealed_partition_opened", "seizure_outcomes_read"):
        if payload.get(key) is not False:
            raise PermissionError(f"{label} has forbidden provenance: {key}")
    contract = payload.get("contract", {})
    if isinstance(contract, Mapping):
        source = contract.get("input_provenance", {})
        if isinstance(source, Mapping):
            for key in (
                "development_targets_read", "development_targets_exposed",
                "sealed_partition_opened", "seizure_outcomes_read",
            ):
                if key in source and source.get(key) is not False:
                    raise PermissionError(f"{label} has forbidden nested provenance: {key}")
            phases = source.get("target_phases")
            if phases is not None:
                assert_safe_phases(tuple(str(v) for v in phases))


def _locked_recipe_from_card(card: Mapping[str, Any]) -> dict[str, Any]:
    contract = card.get("contract", {})
    arch = contract.get("arch", {})
    optimizer = contract.get("optimizer", {})
    train = contract.get("train", {})
    effective = optimizer.get("effective_lr", {})
    return {
        "arch": {
            key: arch.get(key) for key in LOCKED_ARCH_KEYS
        },
        "optimizer": {
            "family": optimizer.get("family"),
            "lr_encoder": effective.get("encoder"),
            "lr_state_adapter": effective.get("state_adapter"),
            "lr_auxiliary": effective.get("auxiliary"),
            "weight_decay": optimizer.get("weight_decay"),
            "betas": optimizer.get("betas"),
            "eps": optimizer.get("eps"),
            "gradient_clip": optimizer.get("gradient_clip"),
        },
        "train": {
            key: train.get(key) for key in LOCKED_TRAIN_KEYS
        },
        "allowed_seeds": list(LOCKED_SEEDS),
        "seed_contract": card.get("seed_contract"),
    }


def _validate_locked_tuning_card(
    path: Path,
    *,
    expected_subject: str,
    require_seed_contract: bool = True,
) -> tuple[dict[str, Any], str]:
    card, sha = _valid_card(path, "human_tuning")
    _assert_no_forbidden_provenance(card, label=str(path))
    contract = card.get("contract", {})
    if not isinstance(contract, Mapping) \
            or card.get("contract_hash") != payload_hash(contract):
        raise PermissionError(f"{path}: tuning card contract hash differs")
    if contract.get("subject") != expected_subject:
        raise ValueError(f"{path}: expected {expected_subject} tuning evidence")
    got = _locked_recipe_from_card(card)
    if require_seed_contract and got.get("seed_contract") != SEED_CONTRACT:
        raise ValueError(f"{path}: tuning evidence predates the fixed seed contract")
    if got.get("train", {}).get("max_steps") not in RUNGS:
        raise ValueError(f"{path}: tuning evidence has an unknown rung")
    return card, sha


def build_locked_recipe_manifest(
    *,
    e253_cards: Sequence[Path],
    e916_diagnostic_cards: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    """Freeze one recipe from E253; retain E916 only as no-learning context."""

    if len(e253_cards) != len(LOCKED_SEEDS):
        raise ValueError(f"locked recipe requires {len(LOCKED_SEEDS)} E253 seed cards")
    selected_rows: list[dict[str, Any]] = []
    seen_seed: set[int] = set()
    chosen_recipe: dict[str, Any] | None = None
    for path in e253_cards:
        card, sha = _validate_locked_tuning_card(path, expected_subject="epilepsiae_253")
        candidate = _locked_recipe_from_card(card)
        if candidate["train"]["max_steps"] != 900:
            raise ValueError("E253 recipe selection cards must all be rung900")
        if chosen_recipe is None:
            chosen_recipe = candidate
        elif candidate != chosen_recipe:
            raise ValueError("E253 seed-fixed cards do not share one recipe")
        seed = int(card["contract"]["train"]["seed"])
        if seed in seen_seed or seed not in LOCKED_SEEDS:
            raise ValueError("E253 recipe evidence has duplicate or unregistered seeds")
        seen_seed.add(seed)
        gain = float(card.get("selection_gain", float("nan")))
        if not np.isfinite(gain) or gain <= 0:
            raise PermissionError("E253 selected recipe is not positive in every locked seed")
        selected_rows.append({
            "path": str(Path(path)), "sha256": sha, "seed": seed,
            "selection_gain": gain, "selected_step": int(card.get("selected_step", -1)),
        })
    if seen_seed != set(LOCKED_SEEDS):
        raise ValueError("E253 recipe evidence does not cover the locked seed set")

    if not e916_diagnostic_cards:
        raise ValueError("at least one E916 no-learning diagnostic card is required")
    diagnostic_rows: list[dict[str, Any]] = []
    for path in e916_diagnostic_cards:
        card, sha = _validate_locked_tuning_card(
            path, expected_subject="epilepsiae_916", require_seed_contract=False,
        )
        diagnostic_rows.append({
            "path": str(Path(path)), "sha256": sha,
            "seed": int(card["contract"]["train"]["seed"]),
            "rung": int(card["contract"]["train"]["max_steps"]),
            "selection_gain": float(card.get("selection_gain", float("nan"))),
            "selected_step": int(card.get("selected_step", -1)),
            "role": "no_learning_diagnostic_only",
            "seed_contract": card.get("seed_contract", "legacy_post_model_construction"),
        })

    gains = np.asarray([row["selection_gain"] for row in selected_rows], dtype=float)
    if chosen_recipe is None:  # pragma: no cover - guarded by the five-card requirement
        raise RuntimeError("no recipe-selection evidence")
    payload: dict[str, Any] = {
        "format": f"{FORMAT_PREFIX}_locked_recipe_v1",
        "status": "LOCKED",
        "recipe": chosen_recipe,
        "recipe_hash": payload_hash(chosen_recipe),
        "selection_rule": (
            "E253 five-seed STATE_SELECTION performance selected the recipe; "
            "E916 is retained only as a no-learning diagnostic and did not veto, "
            "select, or redefine the recipe"
        ),
        "selection_subject": "epilepsiae_253",
        "selection_evidence": sorted(selected_rows, key=lambda row: row["seed"]),
        "selection_n_positive": int(np.sum(gains > 0)),
        "selection_n_total": int(gains.size),
        "selection_gain_median": float(np.median(gains)),
        "diagnostic_subject": "epilepsiae_916",
        "diagnostic_evidence": diagnostic_rows,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    payload["content_hash"] = payload_hash(payload)
    atomic_write_json(Path(output), payload)
    return payload


def require_locked_recipe_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    content_hash = payload.pop("content_hash", None)
    if content_hash != payload_hash(payload):
        raise PermissionError("locked recipe manifest content hash differs")
    payload["content_hash"] = content_hash
    if payload.get("format") != f"{FORMAT_PREFIX}_locked_recipe_v1" \
            or payload.get("status") != "LOCKED":
        raise PermissionError("not a locked v0.3.4 S_P recipe manifest")
    _assert_no_forbidden_provenance(payload, label="locked recipe manifest")
    recipe = payload.get("recipe")
    if not isinstance(recipe, Mapping) or payload.get("recipe_hash") != payload_hash(recipe):
        raise PermissionError("locked recipe hash is invalid")
    if recipe.get("seed_contract") != SEED_CONTRACT \
            or recipe.get("allowed_seeds") != list(LOCKED_SEEDS) \
            or recipe.get("train", {}).get("max_steps") != 900:
        raise PermissionError("locked recipe violates the seed/rung contract")
    for row in [*payload.get("selection_evidence", []), *payload.get("diagnostic_evidence", [])]:
        source = Path(str(row.get("path", "")))
        if not source.is_file() or file_hash(source) != row.get("sha256"):
            raise PermissionError("locked recipe source tuning card changed")
    return payload


def _validate_evaluation_input_manifest(path: Path, *, subject: str) -> tuple[dict[str, Any], str]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != "group_event_state_v0_3_3_human_r0_input_manifest" \
            or manifest.get("subject") != subject \
            or manifest.get("role") != "explicit_non_tuning_override" \
            or manifest.get("sealed") is not False \
            or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError(f"{subject}: evaluation input has forbidden provenance")
    artifact = Path(str(manifest.get("input_path", "")))
    if not artifact.is_file() or file_hash(artifact) != manifest.get("input_npz_sha256"):
        raise ValueError(f"{subject}: locked evaluation input bytes differ")
    return manifest, file_hash(Path(path))


def build_evaluation_release_gate(
    *,
    recipe_manifest: Path,
    input_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Release exactly four non-tuning patients under one immutable recipe."""

    recipe = require_locked_recipe_manifest(recipe_manifest)
    inputs: dict[str, Any] = {}
    for subject in EVALUATION_SUBJECTS:
        path = Path(input_root) / subject / "manifest_v3.json"
        manifest, sha = _validate_evaluation_input_manifest(path, subject=subject)
        inputs[subject] = {
            "path": str(path), "sha256": sha,
            "input_path": str(manifest["input_path"]),
            "input_sha256": str(manifest["input_npz_sha256"]),
        }
    payload: dict[str, Any] = {
        "format": f"{FORMAT_PREFIX}_locked_evaluation_release_gate_v1",
        "status": "PASS",
        "scope": "locked_recipe_evaluation_only",
        "allowed_subjects": list(EVALUATION_SUBJECTS),
        "recipe_manifest": str(Path(recipe_manifest)),
        "recipe_manifest_sha256": file_hash(Path(recipe_manifest)),
        "recipe_hash": recipe["recipe_hash"],
        "inputs": inputs,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    payload["content_hash"] = payload_hash(payload)
    atomic_write_json(Path(output), payload)
    return payload


def require_evaluation_release_gate(
    path: Path,
    *,
    subject: str,
    requested_recipe: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    content_hash = gate.pop("content_hash", None)
    if content_hash != payload_hash(gate):
        raise PermissionError("evaluation release gate content hash differs")
    gate["content_hash"] = content_hash
    if gate.get("format") != f"{FORMAT_PREFIX}_locked_evaluation_release_gate_v1" \
            or gate.get("status") != "PASS" \
            or gate.get("scope") != "locked_recipe_evaluation_only":
        raise PermissionError("evaluation S_P requires its separate locked-evaluation gate")
    if subject not in EVALUATION_SUBJECTS or subject not in gate.get("allowed_subjects", []):
        raise PermissionError(f"locked S_P evaluation is not authorized for {subject}")
    _assert_no_forbidden_provenance(gate, label="evaluation release gate")
    recipe_path = Path(str(gate.get("recipe_manifest", "")))
    if not recipe_path.is_file() or file_hash(recipe_path) != gate.get("recipe_manifest_sha256"):
        raise PermissionError("evaluation recipe manifest changed")
    recipe = require_locked_recipe_manifest(recipe_path)
    if gate.get("recipe_hash") != recipe.get("recipe_hash"):
        raise PermissionError("evaluation gate and recipe identity differ")
    if requested_recipe is not None and dict(requested_recipe) != recipe.get("recipe"):
        raise PermissionError("locked evaluation CLI attempted to change the recipe")
    node = gate.get("inputs", {}).get(subject, {})
    manifest_path = Path(str(node.get("path", "")))
    if not manifest_path.is_file() or file_hash(manifest_path) != node.get("sha256"):
        raise PermissionError(f"{subject}: evaluation input manifest changed")
    manifest, _ = _validate_evaluation_input_manifest(manifest_path, subject=subject)
    artifact = Path(str(manifest.get("input_path", "")))
    if str(artifact) != node.get("input_path") or file_hash(artifact) != node.get("input_sha256"):
        raise PermissionError(f"{subject}: evaluation input artifact changed")
    return gate
