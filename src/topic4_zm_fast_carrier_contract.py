"""Immutable input contract for the Phase-D Z/M fast-carrier repair.

This module is simulator-free.  It verifies the accepted Phase-C futility
evidence and resolves only real seed-1 checkpoints.  The new Phase-D state may
carry every classified source field byte-for-byte and insert exactly one new
deterministic field: an all-zero E-cell dynamic-threshold increment.

The hashes named ``source_semantic_hashes`` bind canonical configuration
objects.  They are deliberately not described as hashes of realised
connectivity/anatomy arrays; those realised-array hashes are added by the
runner after it reconstructs the substrate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from src import topic4_zm_phasec_contract as PHASEC
from src.topic4_zm_checkpoint import load_state_npz, read_manifest
from src.topic4_zm_noise_bank import build_noise_bank


INPUT_SCHEMA = "zm_fast_carrier_input_v1.2_2026-07-31"
IMPLEMENTATION_START_GIT_SHA = "12add24f"
SOURCE_SEED = 1
SOURCE_DT_MS = 0.1
SOURCE_PANEL = (
    ("pre_entry", "natural"),
    ("bounded_mid", "rising"),
    ("bounded_mid", "peak"),
    ("bounded_late", "rising"),
    ("bounded_late", "peak"),
)
PRE_ENTRY_NOISE = ("noise_replay", "noise_resample_1")
BOUNDED_FIRST_PASS_NOISE = ("noise_replay",)

PHASEC_ROOT = Path("results/topic4_sef_hfo/zm_phase_c_tonic_identity")
PHASEC_MANIFEST = PHASEC_ROOT / "phasec_manifest.json"
PHASEC_FUTILITY = PHASEC_ROOT / "phasec_futility_verdict.json"
PHASEC_COORDINATE = PHASEC_ROOT / "phasec1_coordinate_manifest_dt.json"
CANONICAL_CONFIG = Path(
    "results/topic4_sef_hfo/zm_branch_decision/phase0/canonical_config.json"
)
ANCHOR = Path(
    "results/topic4_sef_hfo/zm_branch_decision/anchors/seed1/anchor.json"
)
SPEC = Path(
    "docs/superpowers/specs/"
    "2026-07-31-topic4-zm-snn-fast-carrier-repair-design.md"
)
PLAN = Path(
    "docs/superpowers/plans/"
    "2026-07-31-topic4-zm-snn-fast-carrier-repair.md"
)
CALIBRATION_AMENDMENT = Path(
    "docs/superpowers/specs/"
    "2026-07-31-topic4-zm-fast-carrier-baseline-anchor-amendment.md"
)
DEFAULT_INPUT_OUTPUT = Path(
    "results/topic4_sef_hfo/zm_fast_carrier_repair/"
    "phaseD_input_manifest_v1_2.json"
)
SUPERSEDED_INPUT_LOCK = {
    "path": (
        "results/topic4_sef_hfo/zm_fast_carrier_repair/"
        "phaseD_input_manifest_v1_1.json"
    ),
    "file_sha256": (
        "a60d70ac5bbc3d0bb6353943bc874e109ef9c6ff667581c8214128bfd7027c0d"
    ),
    "manifest_sha256": (
        "a40225184201fba663aa41eb148211514300bb8187d823b97e7c0e51fae7c2d6"
    ),
    "reason": (
        "v1.1 predates the locked baseline magnitude-anchor amendment and "
        "therefore binds the superseded signed-point calibration plan; source "
        "states, arm identities and the length-40000 phi migration are unchanged"
    ),
    "production_authorized": False,
}

SOURCE_CARRIED_FIELDS = (
    "I_E",
    "I_E_rec",
    "I_I",
    "V",
    "_es_ema",
    "_es_run",
    "ref",
    "ring_sE",
    "ring_sI",
    "rng_state",
    "s_E",
    "s_E_rec",
    "s_I",
    "slow.H",
    "slow.S_G",
    "slow._I_I_last",
    "slow._t",
    "slow.a_shunt",
    "slow.g_K",
    "slow.h_G",
    "slow.m",
    "slow.mu_G",
    "slow.n_load",
    "slow.p",
    "slow.q_I",
    "slow.rE",
    "slow.rE_fast",
    "slow.rI",
    "slow.z",
    "t",
    "xi",
)

CONNECTIVITY_PARAMETER_KEYS = (
    "C_EE",
    "C_IE",
    "C_EI",
    "C_II",
    "w_EE",
    "w_IE",
    "g",
    "l_EE",
    "l_IE",
    "l_EI",
    "l_II",
    "rho_EE",
    "rho_IE",
    "rho_EI",
    "rho_II",
    "v_axon",
    "delay_dt",
)


class ContractInputError(RuntimeError):
    """Raised when Phase-D input evidence or semantics drift."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractInputError(f"cannot read required JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractInputError(f"required JSON is not an object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractInputError(message)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ContractInputError(f"path escapes repository root: {path}") from exc


def _validate_self_hash(
    payload: Mapping[str, Any], *, hash_field: str, label: str
) -> None:
    body = {key: value for key, value in payload.items() if key != hash_field}
    _require(
        payload.get(hash_field) == canonical_sha(body),
        f"{label} self-hash mismatch",
    )


def _validate_futility_evidence(root: Path) -> tuple[dict, dict, dict]:
    phasec_path = root / PHASEC_MANIFEST
    coordinate_path = root / PHASEC_COORDINATE
    verdict_path = root / PHASEC_FUTILITY
    phasec = _read_json(phasec_path)
    PHASEC.validate_manifest(phasec)

    coordinate = _read_json(coordinate_path)
    _validate_self_hash(
        coordinate, hash_field="manifest_sha256", label="coordinate manifest"
    )
    verdict = _read_json(verdict_path)
    _validate_self_hash(
        verdict, hash_field="verdict_sha256", label="futility verdict"
    )
    _require(
        verdict.get("status") == "post_result_futility_stopped_incomplete",
        "Phase-C futility status drift",
    )
    coverage = verdict.get("execution_coverage", {})
    _require(
        coverage.get("completed_runs") == 59
        and coverage.get("seed1_primary_completed") == 59
        and coverage.get("seed1_primary_expected") == 60
        and coverage.get("complete_phasec1_negative") is False,
        "Phase-C futility coverage drift",
    )
    proof = verdict.get("seed1_primary_futility", {})
    _require(
        proof.get("established") is True
        and proof.get("n_cells") == 10
        and proof.get("n_unrescuable_cells") == 10
        and proof.get("phenotype_counts") == {"tonic_non_AI": 59},
        "Phase-C futility proof drift",
    )
    _require(
        verdict.get("phasec_manifest_file_sha256") == sha256_file(phasec_path)
        and verdict.get("phasec_manifest_sha256")
        == phasec.get("manifest_sha256"),
        "Phase-C manifest reference drift",
    )
    _require(
        verdict.get("coordinate_manifest_file_sha256")
        == sha256_file(coordinate_path)
        and verdict.get("coordinate_manifest_sha256")
        == coordinate.get("manifest_sha256"),
        "coordinate manifest reference drift",
    )

    rows = verdict.get("run_rows")
    _require(isinstance(rows, list) and len(rows) == 59, "futility rows drift")
    hash_maps = {
        "parts": {},
        "resource_receipts": {},
        "observables": {},
    }
    field_map = {
        "parts": ("part_path", "part_file_sha256"),
        "resource_receipts": ("receipt_path", "receipt_file_sha256"),
        "observables": ("observables_path", "observables_file_sha256"),
    }
    for row in rows:
        _require(
            row.get("seed") == 1
            and row.get("tier") == "primary_convex"
            and row.get("phenotype") == "tonic_non_AI",
            "futility run-row identity drift",
        )
        for group, (path_key, hash_key) in field_map.items():
            rel = row.get(path_key)
            claimed = row.get(hash_key)
            _require(
                isinstance(rel, str) and isinstance(claimed, str),
                f"futility {group} reference missing",
            )
            path = root / rel
            _require(path.is_file(), f"futility evidence missing: {rel}")
            _require(
                sha256_file(path) == claimed,
                f"futility evidence hash drift: {rel}",
            )
            hash_maps[group][rel] = claimed
    for group, values in hash_maps.items():
        _require(
            canonical_sha(values)
            == verdict.get("evidence_set_sha256", {}).get(group),
            f"futility {group} evidence-set hash drift",
        )
    return phasec, coordinate, verdict


def _semantic_hashes(seed_config: Mapping[str, Any]) -> dict[str, str]:
    config = seed_config["config"]
    params = config["params"]
    substrate = config["substrate"]
    engine = config["engine_sha256"]
    connectivity = {
        "parameters": {
            key: params[key] for key in CONNECTIVITY_PARAMETER_KEYS
        },
        "anisotropy": {
            "AR": substrate["AR"],
            "theta_deg": substrate["placement"]["theta_deg"],
        },
        "producer_sha256": {
            key: engine[key]
            for key in (
                "src/snn_engine/connectivity.py",
                "src/snn_engine/connectivity_rot.py",
            )
        },
    }
    threshold = {
        "V_th": params["V_th"],
        "core_mean": substrate["core_mean"],
        "core_std": substrate["core_std"],
        "core_r": substrate["core_r"],
        "base_mean": substrate["base_mean"],
        "vth_core_seed_offsets": substrate["vth_core_seed_offsets"],
    }
    ee = {
        key: params[key]
        for key in ("C_EE", "w_EE", "l_EE", "rho_EE")
    }
    ee["AR"] = substrate["AR"]
    ee["theta_deg"] = substrate["placement"]["theta_deg"]
    return {
        "kind": "canonical_config_semantic_not_realised_array",
        "anatomy": canonical_sha(substrate),
        "connectivity": canonical_sha(connectivity),
        "threshold_field": canonical_sha(threshold),
        "ee_substrate": canonical_sha(ee),
    }


def _source_panel(
    root: Path,
    *,
    phasec: Mapping[str, Any],
    seed_config: Mapping[str, Any],
) -> list[dict]:
    anchor_path = root / ANCHOR
    anchor = _read_json(anchor_path)
    config_sha = seed_config["config_sha"]
    _require(anchor.get("seed") == SOURCE_SEED, "anchor seed drift")
    _require(anchor.get("config_sha") == config_sha, "anchor config drift")
    expected_engine = seed_config["config"]["engine_sha256"][
        "src/snn_engine/kick_probe.py"
    ]
    lookup = {
        (row.get("bin_name"), row.get("fast_phase")): row
        for row in anchor.get("captured_states", [])
    }
    _require(len(lookup) == len(anchor.get("captured_states", [])), "duplicate state")
    phasec_seed = phasec["per_seed"][str(SOURCE_SEED)]
    _require(
        phasec_seed["canonical_config_sha"] == config_sha,
        "Phase-C/source config drift",
    )

    rows: list[dict] = []
    carried_keys: tuple[str, ...] | None = None
    for bin_name, fast_phase in SOURCE_PANEL:
        source = lookup.get((bin_name, fast_phase))
        _require(
            isinstance(source, Mapping),
            f"source panel lacks real {bin_name}__{fast_phase}",
        )
        rel = source.get("path")
        _require(isinstance(rel, str) and rel, "source state path missing")
        path = root / rel
        _require(path.is_file(), f"source state missing: {rel}")
        manifest = read_manifest(path)
        _require(manifest.get("schema") == "zm_sim_state_v1", "source schema drift")
        _require(manifest.get("seed") == SOURCE_SEED, "source state seed drift")
        _require(
            manifest.get("config_sha") == config_sha, "source config SHA drift"
        )
        _require(
            manifest.get("engine_sha") == expected_engine,
            "source engine SHA drift",
        )
        _require(float(manifest.get("dt", -1)) == SOURCE_DT_MS, "source dt drift")
        _require(
            (manifest.get("bin_name"), manifest.get("fast_phase"))
            == (bin_name, fast_phase),
            "source panel phase drift",
        )
        _require(
            manifest.get("state_hash") == source.get("state_hash"),
            "source state semantic hash drift",
        )
        state, verified = load_state_npz(
            path,
            expected_config_sha=config_sha,
            expected_engine_sha=expected_engine,
            expected_dt=SOURCE_DT_MS,
        )
        del state
        _require(
            verified.get("state_hash") == source.get("state_hash"),
            "source state reload hash drift",
        )
        these_keys = tuple(sorted(manifest.get("keys", [])))
        _require(
            set(these_keys) | {"rng_state"} == set(SOURCE_CARRIED_FIELDS),
            f"unclassified source simulator state in {rel}",
        )
        if carried_keys is None:
            carried_keys = these_keys
        _require(these_keys == carried_keys, "source state key inventory drift")
        noise_names = (
            PRE_ENTRY_NOISE
            if bin_name == "pre_entry"
            else BOUNDED_FIRST_PASS_NOISE
        )
        noise = [
            {
                key: value
                for key, value in build_noise_bank(
                    config_sha,
                    SOURCE_SEED,
                    int(manifest["t_step"]),
                    replicate,
                ).items()
                if key != "rng_state"
            }
            for replicate in noise_names
        ]
        rows.append(
            {
                "bin_name": bin_name,
                "fast_phase": fast_phase,
                "path": _relative(path, root),
                "file_sha256": sha256_file(path),
                "state_hash": manifest["state_hash"],
                "t_step": int(manifest["t_step"]),
                "t_ms": float(manifest["t_ms"]),
                "source_state_manifest": manifest,
                "first_pass_noise_banks": noise,
            }
        )
    return rows


def _arms() -> dict[str, dict]:
    common = {"ee_mutation_allowed": False, "z_m_applied_exactly_once": True}
    return {
        "A": {
            **common,
            "mode": "current_exact_control",
            "gamma_global_gaba": None,
            "dynamic_threshold": False,
        },
        "B": {
            **common,
            "mode": "conductance_local_gaba",
            "gamma_global_gaba": 0.0,
            "dynamic_threshold": False,
        },
        "C": {
            **common,
            "mode": "conductance_local_plus_weak_global_gaba",
            "gamma_global_gaba": 1.0 / 6.0,
            "dynamic_threshold": False,
        },
        "D": {
            **common,
            "mode": "conductance_local_plus_weak_global_gaba_phi",
            "gamma_global_gaba": 1.0 / 6.0,
            "dynamic_threshold": True,
            "phi_grid": [
                {"tau_phi_ms": tau, "fraction": fraction}
                for tau in (60.0, 100.0, 160.0)
                for fraction in (0.15, 0.30)
            ],
        },
    }


def build_input_manifest(root: Path | str) -> dict:
    root = Path(root)
    phasec, coordinate, verdict = _validate_futility_evidence(root)
    canonical = _read_json(root / CANONICAL_CONFIG)
    seed_config = canonical.get("seeds", {}).get(str(SOURCE_SEED))
    _require(isinstance(seed_config, Mapping), "canonical seed-1 config missing")
    _require(
        seed_config.get("config_sha")
        == phasec["per_seed"]["1"]["canonical_config_sha"],
        "canonical/Phase-C config SHA drift",
    )
    panel = _source_panel(root, phasec=phasec, seed_config=seed_config)
    arms = _arms()
    body = {
        "schema": INPUT_SCHEMA,
        "production_authorized": False,
        "supersedes_input_lock": SUPERSEDED_INPUT_LOCK,
        "implementation_start_git_sha": IMPLEMENTATION_START_GIT_SHA,
        "locked_documents": {
            "spec_path": str(SPEC),
            "spec_file_sha256": sha256_file(root / SPEC),
            "plan_path": str(PLAN),
            "plan_file_sha256": sha256_file(root / PLAN),
            "calibration_amendment_path": str(CALIBRATION_AMENDMENT),
            "calibration_amendment_file_sha256": sha256_file(
                root / CALIBRATION_AMENDMENT
            ),
        },
        "source": {
            "seed": SOURCE_SEED,
            "dt_ms": SOURCE_DT_MS,
            "canonical_config_path": str(CANONICAL_CONFIG),
            "canonical_config_file_sha256": sha256_file(root / CANONICAL_CONFIG),
            "canonical_config_sha": seed_config["config_sha"],
            "canonical_seed_object_sha256": canonical_sha(seed_config),
            "source_engine_sha256": seed_config["config"]["engine_sha256"],
            "anchor_path": str(ANCHOR),
            "anchor_file_sha256": sha256_file(root / ANCHOR),
            "phasec_manifest_path": str(PHASEC_MANIFEST),
            "phasec_manifest_file_sha256": sha256_file(root / PHASEC_MANIFEST),
            "phasec_manifest_sha256": phasec["manifest_sha256"],
            "coordinate_manifest_path": str(PHASEC_COORDINATE),
            "coordinate_manifest_file_sha256": sha256_file(
                root / PHASEC_COORDINATE
            ),
            "coordinate_manifest_sha256": coordinate["manifest_sha256"],
            "futility_verdict_path": str(PHASEC_FUTILITY),
            "futility_verdict_file_sha256": sha256_file(root / PHASEC_FUTILITY),
            "futility_verdict_sha256": verdict["verdict_sha256"],
            "futility_evidence_set_sha256": verdict["evidence_set_sha256"],
            "phasec_status": verdict["status"],
            "completed_phasec_runs": 59,
            "complete_phasec1_negative": False,
        },
        "source_semantic_hashes": _semantic_hashes(seed_config),
        "source_panel": panel,
        "state_migration": {
            "source_schema": "zm_sim_state_v1",
            "target_schema": "zm_fast_carrier_state_v1",
            "population_sizes": {"N": 40000, "NE": 32000, "NI": 8000},
            "carried_fields": list(SOURCE_CARRIED_FIELDS),
            "inserted_fields": {
                "slow.phi_increment": {
                    "dtype": "float64",
                    "shape": [40000],
                    "fill": 0.0,
                    "target": "E_active_I_exact_zero",
                }
            },
            "source_config_sha_preserved": True,
            "intervention_config_recorded_separately": True,
        },
        "arms": arms,
        "phaseD_arm_config_sha256": canonical_sha(arms),
        "resource_policy": {
            "max_full_snn_workers": 12,
            "mem_available_reserve_gb": 96.0,
            "swap_growth_allowed": False,
            "threads_per_worker": 1,
        },
        "claim_boundary": {
            "fast_carrier_supported": False,
            "entry_tested": False,
            "offset_tested": False,
            "recovery_tested": False,
            "ictal_lifecycle_established": False,
        },
    }
    return {**body, "manifest_sha256": canonical_sha(body)}


def validate_input_manifest(
    manifest: Mapping[str, Any],
    root: Path | str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(manifest, Mapping):
        raise ContractInputError("input manifest must be an object")
    _validate_self_hash(
        manifest, hash_field="manifest_sha256", label="input manifest"
    )
    # Production callers omit ``expected`` and therefore re-audit every source
    # artifact from disk.  Tests that mutate an already disk-verified fixture
    # may inject that immutable expected object to avoid repeatedly inflating
    # five large delay-ring checkpoints.
    expected = build_input_manifest(root) if expected is None else expected
    _require(manifest.get("schema") == INPUT_SCHEMA, "input schema drift")
    _require(
        manifest.get("production_authorized") is False,
        "input manifest cannot authorize production",
    )
    _require(
        manifest.get("source_semantic_hashes", {}).get("connectivity")
        == expected["source_semantic_hashes"]["connectivity"],
        "source connectivity semantic hash drift",
    )
    _require(
        manifest.get("source_semantic_hashes", {}).get("threshold_field")
        == expected["source_semantic_hashes"]["threshold_field"],
        "source threshold-field semantic hash drift",
    )
    migration = manifest.get("state_migration", {})
    _require(
        migration.get("carried_fields") == list(SOURCE_CARRIED_FIELDS),
        "migration carried-field inventory drift",
    )
    phi = migration.get("inserted_fields", {}).get("slow.phi_increment")
    _require(
        phi == expected["state_migration"]["inserted_fields"][
            "slow.phi_increment"
        ],
        "inserted phi contract drift",
    )
    rows = manifest.get("source_panel")
    _require(
        isinstance(rows, list) and len(rows) == len(SOURCE_PANEL),
        "source panel coverage drift",
    )
    for got, want in zip(rows, expected["source_panel"], strict=True):
        _require(
            (got.get("bin_name"), got.get("fast_phase"))
            == (want["bin_name"], want["fast_phase"]),
            "source panel identity drift",
        )
        got_state = got.get("source_state_manifest", {})
        _require(
            got_state.get("seed") == SOURCE_SEED,
            "source state seed drift",
        )
        _require(
            got_state.get("engine_sha")
            == want["source_state_manifest"]["engine_sha"],
            "source engine SHA drift",
        )
        _require(
            got.get("first_pass_noise_banks")
            == want["first_pass_noise_banks"],
            "source noise bank drift",
        )
        _require(got == want, "source panel provenance drift")
    _require(manifest.get("arms") == expected["arms"], "arm contract drift")
    _require(
        manifest.get("resource_policy") == expected["resource_policy"],
        "resource contract drift",
    )
    _require(dict(manifest) == expected, "Phase-D input manifest drift")


def publish_once(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically publish an immutable JSON object.

    An exact rerun is idempotent.  Any content drift at an existing path fails
    rather than overwriting the scientific lock.
    """
    path = Path(path)
    if path.exists():
        existing = _read_json(path)
        if existing != dict(payload):
            raise ContractInputError(
                f"refusing to overwrite non-identical input lock: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            existing = _read_json(path)
            if existing != dict(payload):
                raise ContractInputError(
                    f"concurrent non-identical input lock exists: {path}"
                )
    finally:
        tmp.unlink(missing_ok=True)
