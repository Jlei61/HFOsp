"""Diagnostic plotting utilities for Topic-4 Z/M Phase C.

The figures produced here are source-space identity/maturation diagnostics.
They do not establish seizure entry, offset, recovery, observation matching,
or a lifecycle.  This module only reads completed JSON/NPZ artifacts and never
runs the SNN.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from src.topic4_zm_phasec_contract import (
    PRIMARY_CELL_NAMES as CONTRACT_PRIMARY_CELL_NAMES,
    SHELL_CELL_NAMES as CONTRACT_SHELL_CELL_NAMES,
)


BOUNDARY = "source-space identity/maturation; not lifecycle"
FIGURE_FILENAMES = (
    "01_ceiling_gain_audit.png",
    "02_irregularity_refnull_audit.png",
    "03_currents_psd_audit.png",
    "04_primary_convex_atlas.png",
    "05_secondary_shell_atlas_extrapolated.png",
    "06_representative_spatiotemporal.png",
    "07_seed_modal_panels.png",
    "08_phasec_status_claim_boundary.png",
)

_PHASE_ORDER = {
    "bounded_mid__rising": 0,
    "bounded_mid__peak": 1,
}
_NOISE_ORDER = {
    "noise_replay": 0,
    "noise_resample_1": 1,
    "noise_resample_2": 2,
}
EXPECTED_SEEDS = (1, 3, 4)
PRIMARY_CELL_NAMES = tuple(CONTRACT_PRIMARY_CELL_NAMES)
SHELL_CELL_NAMES = tuple(CONTRACT_SHELL_CELL_NAMES)

C0_SCHEMA = "zm_phasec_c0_summary_v1"
C1_SCHEMA = "zm_phasec1_summary_v1_2026-07-28"
MODAL_SCHEMA = "zm_phasec_seed_modal_v1_2026-07-28"
FINAL_SCHEMA = "zm_phasec_final_adjudication_v1_2026-07-28"
C0_PART_SCHEMA = "zm_phasec_identity_cell_v1"
C1_PART_SCHEMA = "zm_phasec1_base_part_v1_2026-07-28"


class PlotEvidenceError(RuntimeError):
    """Raised when a referenced immutable artifact is absent or has drifted."""


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    sha256: str | None
    data: dict[str, Any] | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.data is not None and self.error is None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_artifact(path: str | Path) -> LoadedArtifact:
    path = Path(path)
    if not path.is_file():
        return LoadedArtifact(path, None, None, "missing_json")
    digest = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return LoadedArtifact(path, digest, None, f"invalid_json:{exc}")
    if not isinstance(value, dict):
        return LoadedArtifact(path, digest, None, "json_root_not_object")
    return LoadedArtifact(path, digest, value, None)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def select_representative_run(
    c1_summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Select one complete C1 run without outcome- or appearance-based ranking.

    Eligibility is technical/scientific completeness.  The fixed ordering is:
    seed 1 first, primary before shell, then cell/path/phase/noise
    lexicographically.  Phenotype, rate, PSD, and spatial appearance never
    participate in selection.
    """
    candidates: list[dict[str, Any]] = []
    for cell in c1_summary.get("cells", []) or []:
        if not isinstance(cell, Mapping):
            continue
        for run in cell.get("run_rows", []) or []:
            if not isinstance(run, Mapping) or run.get("status") != "complete":
                continue
            row = {
                "seed": int(cell.get("seed", run.get("seed", -1))),
                "tier": str(cell.get("tier", run.get("tier", ""))),
                "cell_id": str(cell.get("cell_id", run.get("cell_id", ""))),
                "path_index": int(cell.get("path_index", run.get("path_index", -1))),
                "path_direction": str(
                    cell.get("path_direction", run.get("path_direction", ""))
                ),
                "phase": str(run.get("phase", "")),
                "noise": str(run.get("noise", "")),
                "part_path": run.get("part_path"),
                "part_sha256": run.get("part_sha256"),
                "cell_status": str(cell.get("status", "missing")),
            }
            if isinstance(row["part_path"], str) and row["part_path"]:
                candidates.append(row)
    if not candidates:
        return None

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            0 if row["seed"] == 1 else 1,
            0 if row["tier"] == "primary_convex" else 1,
            row["cell_id"],
            row["path_index"],
            row["path_direction"],
            _PHASE_ORDER.get(row["phase"], 99),
            row["phase"],
            _NOISE_ORDER.get(row["noise"], 99),
            row["noise"],
        )

    return min(candidates, key=key)


def c0_whole_sheet_ceiling(
    repo_root: str | Path,
    c0: Mapping[str, Any],
) -> tuple[dict[int, list[float]], str | None]:
    """Load six locked continuation-level whole-sheet ceiling summaries/seed.

    These are continuation summaries, not independent biological replicates.
    The plot therefore shows their individual values plus IQR, never a
    pooled-neuron confidence interval.
    """
    root = Path(repo_root)
    output: dict[int, list[float]] = {}
    for seed_row in c0.get("seed_rows", []) or []:
        if not isinstance(seed_row, Mapping):
            continue
        seed = int(seed_row.get("seed", -1))
        values = []
        for run in seed_row.get("rows", []) or []:
            if (
                not isinstance(run, Mapping)
                or run.get("scientific_failure") not in {None, ""}
            ):
                continue
            path_value = run.get("identity_path")
            if not isinstance(path_value, str) or not path_value:
                return {}, f"whole_sheet_part_path_missing:seed{seed}"
            artifact = load_json_artifact(_resolve(root, path_value))
            if (
                not artifact.valid
                or artifact.data.get("schema") != C0_PART_SCHEMA
                or artifact.data.get("status") != "complete"
            ):
                return {}, f"whole_sheet_part_invalid:seed{seed}"
            value = _finite(
                (artifact.data.get("spike_metrics") or {})
                .get("firing", {})
                .get("rho80_all_median")
            )
            if value is None:
                return {}, f"whole_sheet_ceiling_missing:seed{seed}"
            values.append(value)
        if len(values) != 6:
            return {}, (
                f"whole_sheet_continuation_count:seed{seed}:"
                f"{len(values)}!=6"
            )
        output[seed] = values
    if set(output) != set(EXPECTED_SEEDS):
        return {}, "whole_sheet_seed_coverage_mismatch"
    return output, None


def _select_c0_part(c0: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for seed_row in c0.get("seed_rows", []) or []:
        if not isinstance(seed_row, Mapping):
            continue
        for run in seed_row.get("rows", []) or []:
            if not isinstance(run, Mapping):
                continue
            if run.get("scientific_failure") not in {None, ""}:
                continue
            path = run.get("identity_path")
            if isinstance(path, str) and path:
                candidates.append({
                    "seed": int(seed_row.get("seed", -1)),
                    "phase": str(run.get("phase", "")),
                    "noise": str(run.get("noise", "")),
                    "part_path": path,
                })
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            0 if row["seed"] == 1 else 1,
            _PHASE_ORDER.get(row["phase"], 99),
            row["phase"],
            _NOISE_ORDER.get(row["noise"], 99),
            row["noise"],
        ),
    )


def load_part_npz(
    repo_root: str | Path,
    reference: Mapping[str, Any] | None,
    *,
    expected_part_schema: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None, str | None]:
    if reference is None:
        return None, None, "no_complete_representative"
    root = Path(repo_root)
    path_value = reference.get("part_path")
    if not isinstance(path_value, str) or not path_value:
        return None, None, "representative_missing_part_path"
    part_path = _resolve(root, path_value)
    artifact = load_json_artifact(part_path)
    if not artifact.valid:
        return None, None, artifact.error
    expected_part_sha = reference.get("part_sha256")
    if expected_part_sha is not None and artifact.sha256 != expected_part_sha:
        return None, None, "representative_part_sha256_mismatch"
    part = artifact.data
    assert part is not None
    if (
        expected_part_schema is not None
        and part.get("schema") != expected_part_schema
    ):
        return None, None, "representative_part_schema_mismatch"
    if part.get("status") != "complete":
        return None, None, "representative_part_not_complete"
    npz_value = part.get("observables_path")
    if not isinstance(npz_value, str) or not npz_value:
        return part, None, "representative_missing_observables_path"
    npz_path = _resolve(root, npz_value)
    if not npz_path.is_file():
        return part, None, "representative_missing_observables_npz"
    expected_npz_sha = part.get("observables_sha256")
    if not isinstance(expected_npz_sha, str) or sha256_file(npz_path) != expected_npz_sha:
        return part, None, "representative_observables_sha256_mismatch"
    try:
        with np.load(npz_path, allow_pickle=False) as handle:
            arrays = {key: np.asarray(handle[key]) for key in handle.files}
    except (OSError, TypeError, ValueError) as exc:
        return part, None, f"invalid_representative_npz:{exc}"
    return part, arrays, None


def _required_fields(
    value: Mapping[str, Any],
    required: tuple[str, ...],
    *,
    label: str,
) -> str | None:
    missing = [key for key in required if key not in value]
    return (
        None if not missing
        else f"{label}_missing_fields:" + ",".join(missing)
    )


def _validate_summary_contract(
    key: str,
    artifact: LoadedArtifact,
) -> str | None:
    if not artifact.valid:
        return artifact.error or f"{key}_invalid"
    value = artifact.data
    assert value is not None
    if key == "c0":
        if value.get("schema") != C0_SCHEMA:
            return "c0_schema_mismatch"
        return _required_fields(
            value,
            (
                "manifest_sha256", "panel_manifest_sha256", "resolution",
                "seed_rows", "aggregate", "claim_boundary",
            ),
            label="c0",
        )
    if key == "c1":
        if value.get("schema") != C1_SCHEMA:
            return "c1_schema_mismatch"
        return _required_fields(
            value,
            (
                "phasec_manifest_sha256",
                "coordinate_manifest_sha256",
                "coordinate_manifest_semantic_sha256",
                "resolution",
                "cells",
                "primary_adjudication",
                "secondary_shell_adjudication",
                "verdict",
                "claim_boundary",
            ),
            label="c1",
        )
    if key == "modal":
        if value.get("schema") != MODAL_SCHEMA:
            return "modal_schema_mismatch"
        return _required_fields(
            value,
            (
                "phasec_manifest_sha256", "status", "seed_results",
                "routes_by_seed", "input_provenance", "claim_boundary",
            ),
            label="modal",
        )
    if key == "final":
        if value.get("schema") != FINAL_SCHEMA:
            return "final_schema_mismatch"
        missing = _required_fields(
            value,
            (
                "fine_verdict", "next_route", "layers",
                "input_file_provenance", "phasec_manifest_provenance",
                "trigger_provenance", "wrapper_provenance_issues",
                "entry", "offset", "recovery_lifecycle",
                "phase_c2_authorized", "actuator_authorized",
            ),
            label="final",
        )
        if missing is not None:
            return missing
        expected_boundary = {
            "entry": "not_tested",
            "offset": "not_tested",
            "recovery_lifecycle": "not_established",
            "phase_c2_authorized": False,
            "actuator_authorized": False,
        }
        drift = [
            key for key, expected in expected_boundary.items()
            if value.get(key) != expected
        ]
        if drift:
            return "final_claim_boundary_violation:" + ",".join(drift)
        return None
    return f"unknown_summary_contract:{key}"


def _validate_final_references(
    repo_root: Path,
    final: Mapping[str, Any],
) -> str | None:
    refs = final.get("input_file_provenance")
    required = ("c0", "c1_primary", "c1_shell", "modal", "coverage")
    if not isinstance(refs, Mapping):
        return "final_input_file_provenance_invalid"
    for name in required:
        ref = refs.get(name)
        if not isinstance(ref, Mapping):
            return f"final_input_reference_missing:{name}"
        path_value = ref.get("path")
        expected_sha = ref.get("file_sha256")
        if (
            ref.get("status") != "complete"
            or
            not isinstance(path_value, str)
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            return f"final_input_reference_invalid:{name}"
        path = _resolve(repo_root, path_value)
        if not path.is_file() or sha256_file(path) != expected_sha:
            return f"final_input_reference_hash_drift:{name}"
    if final.get("wrapper_provenance_issues") not in ([], ()):
        return "final_wrapper_provenance_incomplete"
    trigger_ref = final.get("trigger_provenance")
    if not isinstance(trigger_ref, Mapping):
        return "final_trigger_provenance_invalid"
    trigger_path_value = trigger_ref.get("path")
    trigger_sha = trigger_ref.get("file_sha256")
    if (
        trigger_ref.get("status") != "complete"
        or not isinstance(trigger_path_value, str)
        or not isinstance(trigger_sha, str)
        or len(trigger_sha) != 64
    ):
        return "final_trigger_reference_invalid"
    trigger_path = _resolve(repo_root, trigger_path_value)
    if (
        not trigger_path.is_file()
        or sha256_file(trigger_path) != trigger_sha
    ):
        return "final_trigger_reference_hash_drift"
    phasec_ref = final.get("phasec_manifest_provenance")
    if not isinstance(phasec_ref, Mapping):
        return "final_phasec_manifest_provenance_invalid"
    path_value = phasec_ref.get("path")
    expected_sha = phasec_ref.get("file_sha256")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
    ):
        return "final_phasec_manifest_reference_invalid"
    path = _resolve(repo_root, path_value)
    if not path.is_file() or sha256_file(path) != expected_sha:
        return "final_phasec_manifest_hash_drift"
    return None


def _validate_parent_identity(
    c0: Mapping[str, Any],
    c1: Mapping[str, Any],
    modal: Mapping[str, Any],
    final: Mapping[str, Any],
) -> str | None:
    phasec_sha = c0.get("manifest_sha256")
    if (
        not isinstance(phasec_sha, str)
        or len(phasec_sha) != 64
        or c1.get("phasec_manifest_sha256") != phasec_sha
        or modal.get("phasec_manifest_sha256") != phasec_sha
        or (final.get("phasec_manifest_provenance") or {}).get(
            "manifest_sha256"
        ) != phasec_sha
    ):
        return "phasec_parent_semantic_identity_mismatch"
    return None


def _validate_plot_part(
    part: Mapping[str, Any] | None,
    arrays: Mapping[str, np.ndarray] | None,
    error: str | None,
    *,
    kind: str,
) -> str | None:
    if error is not None:
        return error
    if part is None or arrays is None:
        return f"{kind}_representative_missing"
    if kind == "c0":
        required_part = (
            "spike_metrics", "threshold_margin_initial",
            "threshold_margin_final",
        )
        required_arrays = (
            "raw_sample_time_ms",
            "raw_raw_ampa_core_mean_mV",
            "raw_raw_gaba_core_mean_mV",
            "effective_sample_time_ms",
            "effective_effective_excitation_core_mean_mV",
            "effective_effective_outward_total_core_mean_mV",
            "effective_effective_net_drive_core_mean_mV",
            "E_rate_grid", "I_rate_grid", "fine_bin_ms",
            "lfp_raw_synaptic_proxy", "lfp_fs_hz",
        )
    elif kind == "c1":
        required_part = ()
        required_arrays = (
            "phasec1_observables_schema", "bin_ms",
            "E_rate_grid", "I_rate_grid", "source_rate_hz",
            "kymograph", "axis_positions",
            "lfp_raw_synaptic_proxy", "lfp_fs_hz",
        )
    else:
        return f"unknown_plot_part_kind:{kind}"
    missing_part = [key for key in required_part if key not in part]
    missing_arrays = [key for key in required_arrays if key not in arrays]
    if missing_part:
        return f"{kind}_representative_part_missing:" + ",".join(missing_part)
    if missing_arrays:
        return f"{kind}_representative_npz_missing:" + ",".join(missing_arrays)
    return None


def _blocked(ax, reason: str) -> None:
    ax.set_facecolor("#E6E6E6")
    ax.text(
        0.5,
        0.5,
        f"BLOCKED\n{reason}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#555555",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#AAAAAA")


def _footer(fig) -> None:
    fig.text(0.995, 0.006, BOUNDARY, ha="right", va="bottom", fontsize=8, color="#555555")


def _save(fig, path: Path, dpi: int) -> None:
    _footer(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _plot_ci(ax, seed_rows, metric: str, ylabel: str, thresholds=()) -> bool:
    shown = False
    for row in sorted(seed_rows, key=lambda value: int(value.get("seed", -1))):
        ci = (row.get("hierarchical_ci") or {}).get(metric) or {}
        point, lo, hi = (_finite(ci.get(key)) for key in ("point", "lo", "hi"))
        if point is None or lo is None or hi is None:
            continue
        seed = int(row["seed"])
        ax.errorbar(
            seed,
            point,
            yerr=[[point - lo], [hi - point]],
            fmt="o",
            color="#2166AC",
            capsize=3,
            ms=5,
        )
        shown = True
    for value, label in thresholds:
        ax.axhline(value, color="#777777", linestyle="--", linewidth=0.8, label=label)
    ax.set_xlabel("seed")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({int(row.get("seed", -1)) for row in seed_rows}))
    ax.margins(x=0.15)
    return shown


def _gain_value(row: Mapping[str, Any], key: str) -> float | None:
    gain = row.get(key) or {}
    value = _finite(gain.get("gain_hz_per_mV"))
    if value is not None:
        return value
    blocks = np.asarray(gain.get("gain_hz_per_mV_blocks", []), float)
    blocks = blocks[np.isfinite(blocks)]
    return float(np.median(blocks)) if blocks.size else None


def _figure_ceiling_gain(
    c0: Mapping[str, Any] | None,
    whole_sheet: Mapping[int, list[float]] | None,
    whole_sheet_error: str | None,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    if c0 is None:
        for ax in axes:
            _blocked(ax, "C0 summary missing/invalid")
    else:
        seeds = c0.get("seed_rows", []) or []
        core_shown = False
        for row in sorted(
            seeds, key=lambda value: int(value.get("seed", -1))
        ):
            ci = (row.get("hierarchical_ci") or {}).get(
                "rho80_active_core"
            ) or {}
            point, lo, hi = (
                _finite(ci.get(key)) for key in ("point", "lo", "hi")
            )
            if point is None or lo is None or hi is None:
                continue
            seed = int(row["seed"])
            axes[0].errorbar(
                seed - 0.10,
                point,
                yerr=[[point - lo], [hi - point]],
                fmt="o",
                color="#2166AC",
                capsize=3,
                ms=5,
                label=(
                    "active core: seed-hierarchical 95% CI"
                    if not core_shown else None
                ),
            )
            core_shown = True
        whole_shown = False
        if whole_sheet_error is None and whole_sheet:
            for seed in EXPECTED_SEEDS:
                values = np.asarray(whole_sheet.get(seed, []), float)
                values = values[np.isfinite(values)]
                if values.size != 6:
                    continue
                x = seed + 0.10
                jitter = np.linspace(-0.035, 0.035, values.size)
                axes[0].scatter(
                    x + jitter,
                    values,
                    color="#B2182B",
                    alpha=0.45,
                    s=12,
                    zorder=2,
                )
                q25, median, q75 = np.quantile(
                    values, [0.25, 0.50, 0.75]
                )
                axes[0].errorbar(
                    x,
                    median,
                    yerr=[[median - q25], [q75 - median]],
                    fmt="s",
                    color="#B2182B",
                    capsize=3,
                    ms=4,
                    label=(
                        "whole sheet: 6 locked continuations + IQR"
                        if not whole_shown else None
                    ),
                    zorder=3,
                )
                whole_shown = True
        if not core_shown:
            _blocked(axes[0], "ceiling CI unavailable")
        else:
            if whole_sheet_error is not None:
                axes[0].text(
                    0.02,
                    0.04,
                    f"whole sheet BLOCKED: {whole_sheet_error}",
                    transform=axes[0].transAxes,
                    fontsize=6.5,
                    color="#8C2D04",
                    va="bottom",
                )
            axes[0].axhline(
                0.20, color="#777777", linestyle="--", linewidth=0.8,
                label="AI max",
            )
            axes[0].axhline(
                0.50, color="#777777", linestyle=":", linewidth=0.8,
                label="saturation min",
            )
            axes[0].set_xlabel("seed")
            axes[0].set_ylabel("ceiling occupancy")
            axes[0].set_xticks(EXPECTED_SEEDS)
            axes[0].margins(x=0.15)
            axes[0].legend(frameon=False, fontsize=6.5)
        if not _plot_ci(
            axes[1],
            seeds,
            "gain_relative_to_preentry",
            "carrier / pre-entry gain",
            ((0.20, "saturation max"), (0.50, "AI min")),
        ):
            _blocked(axes[1], "paired-gain CI unavailable")
        else:
            axes[1].legend(frameon=False, fontsize=7)
        shown = False
        for seed_row in seeds:
            seed = int(seed_row.get("seed", -1))
            for run in seed_row.get("rows", []) or []:
                denominator = _gain_value(run, "gain_preentry")
                carrier = _gain_value(run, "gain_carrier")
                if denominator is None or carrier is None:
                    continue
                axes[2].plot(
                    [seed - 0.08, seed + 0.08],
                    [denominator, carrier],
                    color="#999999",
                    linewidth=0.7,
                    alpha=0.55,
                )
                axes[2].scatter(seed - 0.08, denominator, marker="^", color="#777777", s=16)
                axes[2].scatter(seed + 0.08, carrier, marker="o", color="#B2182B", s=16)
                shown = True
        if not shown:
            _blocked(axes[2], "gain numerator/denominator unavailable")
        else:
            axes[2].set_xlabel("seed")
            axes[2].set_ylabel("local gain (Hz/mV)")
            axes[2].scatter([], [], marker="^", color="#777777", label="pre-entry denominator")
            axes[2].scatter([], [], marker="o", color="#B2182B", label="carrier")
            axes[2].legend(frameon=False, fontsize=7)
    axes[0].set_title("ceiling")
    axes[1].set_title("paired relative gain")
    axes[2].set_title("locked denominator")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path, dpi)


def _part_fano20(repo_root: Path, c0: Mapping[str, Any]) -> dict[int, list[float]]:
    values: dict[int, list[float]] = {}
    for seed_row in c0.get("seed_rows", []) or []:
        seed = int(seed_row.get("seed", -1))
        for run in seed_row.get("rows", []) or []:
            path_value = run.get("identity_path")
            if not isinstance(path_value, str):
                continue
            artifact = load_json_artifact(_resolve(repo_root, path_value))
            if not artifact.valid:
                continue
            metric = (
                artifact.data.get("spike_metrics", {})
                .get("fano", {})
                .get("fano_by_bin", {})
                .get("20ms", {})
                .get("median")
            )
            metric = _finite(metric)
            if metric is not None:
                values.setdefault(seed, []).append(metric)
    return values


def _figure_irregularity(
    repo_root: Path,
    c0: Mapping[str, Any] | None,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3))
    if c0 is None:
        for ax in axes:
            _blocked(ax, "C0 summary missing/invalid")
    else:
        seeds = c0.get("seed_rows", []) or []
        if not _plot_ci(axes[0], seeds, "isi_cv2_median", "ISI CV2", ((0.70, "AI min"),)):
            _blocked(axes[0], "CV2 CI unavailable")
        if not _plot_ci(
            axes[1],
            seeds,
            "refractory_isi_fraction",
            "refractory-lock fraction",
            ((0.80, "saturation min"),),
        ):
            _blocked(axes[1], "refractory CI unavailable")
        fano = _part_fano20(repo_root, c0)
        if not fano:
            _blocked(axes[2], "20-ms Fano unavailable")
        else:
            for seed, points in sorted(fano.items()):
                axes[2].scatter(
                    np.full(len(points), seed), points, color="#2166AC", alpha=0.55, s=14
                )
                axes[2].scatter(seed, np.median(points), color="#B2182B", marker="_", s=80)
            axes[2].axhline(0.60, color="#777777", linestyle="--", linewidth=0.8)
            axes[2].set_xlabel("seed")
            axes[2].set_ylabel("20-ms Fano")
        shown = False
        for row in sorted(seeds, key=lambda value: int(value.get("seed", -1))):
            seed = int(row.get("seed", -1))
            ci = row.get("hierarchical_ci") or {}
            excess = ci.get("pairwise_stratum_max_excess") or {}
            point, lo, hi = (
                _finite(excess.get(k)) for k in ("point", "lo", "hi")
            )
            if point is None or lo is None or hi is None:
                continue
            axes[3].errorbar(
                seed,
                point,
                yerr=[[point - lo], [hi - point]],
                fmt="o",
                color="#B2182B",
                capsize=3,
            )
            shown = True
        if not shown:
            _blocked(axes[3], "stratum max-excess CI unavailable")
        else:
            axes[3].axhline(
                0, color="#777777", linestyle="--", linewidth=0.8,
                label="all strata below null if UCB < 0",
            )
            axes[3].set_xlabel("seed")
            axes[3].set_ylabel("max stratum excess over null q97.5")
            axes[3].legend(frameon=False, fontsize=7)
    for ax, title in zip(
        axes, ("ISI irregularity", "refractory locking", "count variability", "pairwise null")
    ):
        ax.set_title(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path, dpi)


def _array(arrays: Mapping[str, np.ndarray] | None, *keys: str) -> np.ndarray | None:
    if arrays is None:
        return None
    for key in keys:
        if key in arrays:
            value = np.asarray(arrays[key], float)
            if value.size and np.any(np.isfinite(value)):
                return value
    return None


def _periodogram(trace: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trace, float).reshape(-1)
    x = np.nan_to_num(x - np.nanmean(x))
    if x.size < 8 or fs_hz <= 0:
        return np.asarray([]), np.asarray([])
    taper = np.hanning(x.size)
    power = np.abs(np.fft.rfft(x * taper)) ** 2
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    return freq, power


def _margin_median(part: Mapping[str, Any], which: str) -> float | None:
    return _finite(
        (part.get(which) or {})
        .get("core_free_E", {})
        .get("quantiles_mV", {})
        .get("50.0")
    )


def align_current_vseeg(
    effective_time_ms: np.ndarray,
    effective_current: np.ndarray,
    lfp: np.ndarray,
    fs_hz: float,
    *,
    max_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair current and vSEEG only after alignment on their real time axes."""
    t_current = np.asarray(effective_time_ms, float).reshape(-1)
    current = np.asarray(effective_current, float).reshape(-1)
    signal = np.asarray(lfp, float)
    if signal.ndim != 2 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    n = min(t_current.size, current.size)
    t_current = t_current[:n]
    current = current[:n]
    vseeg = np.nanmean(signal, axis=1)
    t_vseeg = np.arange(vseeg.size, dtype=float) * (1000.0 / fs_hz)
    valid_current = np.isfinite(t_current) & np.isfinite(current)
    valid_vseeg = np.isfinite(t_vseeg) & np.isfinite(vseeg)
    if valid_current.sum() < 2 or valid_vseeg.sum() < 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    t_current = t_current[valid_current]
    current = current[valid_current]
    order = np.argsort(t_current, kind="stable")
    t_current = t_current[order]
    current = current[order]
    t_vseeg = t_vseeg[valid_vseeg]
    vseeg = vseeg[valid_vseeg]
    lo = max(float(t_current[0]), float(t_vseeg[0]))
    hi = min(float(t_current[-1]), float(t_vseeg[-1]))
    overlap = (t_current >= lo) & (t_current <= hi)
    t_aligned = t_current[overlap]
    current_aligned = current[overlap]
    if t_aligned.size < 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    vseeg_aligned = np.interp(t_aligned, t_vseeg, vseeg)
    if t_aligned.size > max_points:
        keep = np.linspace(
            0, t_aligned.size - 1, max_points, dtype=int
        )
        t_aligned = t_aligned[keep]
        current_aligned = current_aligned[keep]
        vseeg_aligned = vseeg_aligned[keep]
    return t_aligned, current_aligned, vseeg_aligned


def _figure_currents(
    part: Mapping[str, Any] | None,
    arrays: Mapping[str, np.ndarray] | None,
    error: str | None,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    axes = axes.ravel()
    if part is None or arrays is None:
        for ax in axes:
            _blocked(ax, error or "representative current evidence unavailable")
    else:
        t_raw = _array(arrays, "raw_sample_time_ms")
        raw_e = _array(arrays, "raw_raw_ampa_core_mean_mV")
        raw_i = _array(arrays, "raw_raw_gaba_core_mean_mV")
        if t_raw is None or raw_e is None or raw_i is None:
            _blocked(axes[0], "raw E/I traces missing")
        else:
            n = min(t_raw.size, raw_e.size, raw_i.size)
            axes[0].plot(t_raw[:n] * 1e-3, raw_e[:n], color="#B2182B", lw=0.8, label="raw E")
            axes[0].plot(t_raw[:n] * 1e-3, raw_i[:n], color="#2166AC", lw=0.8, label="raw I")
            axes[0].set_xlabel("time (s)")
            axes[0].set_ylabel("raw synaptic drive (mV)")
            axes[0].legend(frameon=False, fontsize=7)
        t_eff = _array(arrays, "effective_sample_time_ms")
        eff_e = _array(arrays, "effective_effective_excitation_core_mean_mV")
        eff_i = _array(arrays, "effective_effective_outward_total_core_mean_mV")
        eff_net = _array(arrays, "effective_effective_net_drive_core_mean_mV")
        if t_eff is None or eff_e is None or eff_i is None or eff_net is None:
            _blocked(axes[1], "effective membrane-drive traces missing")
        else:
            n = min(t_eff.size, eff_e.size, eff_i.size, eff_net.size)
            axes[1].plot(t_eff[:n] * 1e-3, eff_e[:n], color="#B2182B", lw=0.8, label="effective E")
            axes[1].plot(t_eff[:n] * 1e-3, eff_i[:n], color="#2166AC", lw=0.8, label="outward")
            axes[1].plot(t_eff[:n] * 1e-3, eff_net[:n], color="#333333", lw=0.8, label="net")
            axes[1].set_xlabel("time (s)")
            axes[1].set_ylabel("membrane drive (mV)")
            axes[1].legend(frameon=False, fontsize=7)
        margins = [
            _margin_median(part, "threshold_margin_initial"),
            _margin_median(part, "threshold_margin_final"),
        ]
        if any(value is None for value in margins):
            _blocked(axes[2], "membrane-distance snapshots missing")
        else:
            axes[2].bar(["initial", "final"], margins, color=["#999999", "#A35E48"])
            axes[2].axhline(0, color="#555555", lw=0.7)
            axes[2].set_ylabel(r"core free-E median $V_{th}-V$ (mV)")
        e_grid = _array(arrays, "E_rate_grid")
        i_grid = _array(arrays, "I_rate_grid")
        bin_ms = _array(arrays, "bin_ms", "fine_bin_ms")
        if (
            e_grid is None
            or i_grid is None
            or e_grid.ndim < 2
            or i_grid.ndim < 2
            or bin_ms is None
        ):
            _blocked(axes[3], "fine E/I rate fields missing")
        else:
            e_rate = np.nanmean(e_grid, axis=tuple(range(1, e_grid.ndim)))
            i_rate = np.nanmean(i_grid, axis=tuple(range(1, i_grid.ndim)))
            n = min(e_rate.size, i_rate.size)
            t = np.arange(n) * float(bin_ms.reshape(-1)[0]) * 1e-3
            axes[3].plot(t, e_rate[:n], color="#B2182B", lw=0.8, label="E")
            axes[3].plot(t, i_rate[:n], color="#2166AC", lw=0.8, label="I")
            axes[3].set_xlabel("time (s)")
            axes[3].set_ylabel("fine field rate (Hz)")
            axes[3].legend(frameon=False, fontsize=7)
        lfp = _array(arrays, "lfp_raw_synaptic_proxy")
        fs = _array(arrays, "lfp_fs_hz")
        if lfp is None or fs is None or lfp.ndim != 2:
            _blocked(axes[4], "virtual-SEEG PSD input missing")
        else:
            trace = np.nanmean(lfp, axis=1)
            freq, power = _periodogram(trace, float(fs.reshape(-1)[0]))
            band = (freq >= 1) & (freq <= min(150, 0.45 * float(fs.reshape(-1)[0])))
            if not np.any(band):
                _blocked(axes[4], "PSD band unavailable")
            else:
                axes[4].plot(freq[band], 10 * np.log10(power[band] + 1e-20), color="#6A3D9A", lw=0.8)
                axes[4].set_xlabel("frequency (Hz)")
                axes[4].set_ylabel("raw-synaptic proxy PSD (dB)")
                axes[4].margins(x=0)
        if (
            lfp is None
            or eff_net is None
            or t_eff is None
            or fs is None
            or lfp.ndim != 2
        ):
            _blocked(axes[5], "current-vSEEG pairing unavailable")
        else:
            _, x_aligned, y_aligned = align_current_vseeg(
                t_eff,
                eff_net,
                lfp,
                float(fs.reshape(-1)[0]),
            )
            if x_aligned.size < 8:
                _blocked(axes[5], "current-vSEEG pairing too short")
            else:
                axes[5].scatter(
                    x_aligned,
                    y_aligned,
                    s=4,
                    alpha=0.25,
                    color="#555555",
                )
                axes[5].set_xlabel("effective net drive (mV)")
                axes[5].set_ylabel("raw-synaptic vSEEG proxy")
    for ax, title in zip(
        axes,
        (
            "raw E/I",
            "effective E/I/current",
            "membrane distance",
            "fine E/I rate",
            "fine PSD",
            "current vs vSEEG",
        ),
    ):
        ax.set_title(title)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, path, dpi)


_CLASS_ORDER = (
    "invalid_physical_cell",
    "missing",
    "rest_or_silence",
    "probabilistically_indeterminate",
    "tonic_gain_indeterminate",
    "refractory_saturated",
    "tonic_non_AI",
    "spike_AI_screen_candidate",
    "balanced_AI_tonic_cell",
    "periodic_non_tonic_carrier",
    "clonic_or_bursting_carrier",
    "spatial_relay",
    "hfo_like_relaxation_train",
    "runaway",
)
_CLASS_COLORS = (
    "#BDBDBD",
    "#E6E6E6",
    "#F7F7F7",
    "#FDD49E",
    "#FDAE6B",
    "#D7301F",
    "#FC8D59",
    "#91BFDB",
    "#4575B4",
    "#66BD63",
    "#1A9850",
    "#542788",
    "#8073AC",
    "#67001F",
)


def _cell_final_class(cell: Mapping[str, Any]) -> str:
    gain = cell.get("conditional_gain") or {}
    value = gain.get("final_cell_class")
    return str(value if value is not None else cell.get("cell_class", "missing"))


def atlas_matrix(
    c1: Mapping[str, Any],
    tier: str,
) -> dict[str, Any]:
    """Build a fixed-denominator atlas from the locked expected inventory."""
    if tier == "primary_convex":
        cell_ids = PRIMARY_CELL_NAMES
    elif tier == "secondary_shell":
        cell_ids = SHELL_CELL_NAMES
    else:
        raise ValueError(f"unknown atlas tier: {tier}")
    expected = {
        (seed, cell_id)
        for seed in EXPECTED_SEEDS
        for cell_id in cell_ids
    }
    lookup: dict[tuple[int, str], Mapping[str, Any]] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    unknown_labels: list[str] = []
    for cell in c1.get("cells", []) or []:
        if not isinstance(cell, Mapping) or cell.get("tier") != tier:
            continue
        key = (int(cell.get("seed", -1)), str(cell.get("cell_id", "")))
        if key not in expected:
            unexpected.append(f"{key[0]}:{key[1]}")
            continue
        if key in lookup:
            duplicates.append(f"{key[0]}:{key[1]}")
            continue
        lookup[key] = cell
    matrix = np.full(
        (len(EXPECTED_SEEDS), len(cell_ids)),
        _CLASS_ORDER.index("missing"),
        int,
    )
    technical_complete = 0
    for iy, seed in enumerate(EXPECTED_SEEDS):
        for ix, cell_id in enumerate(cell_ids):
            cell = lookup.get((seed, cell_id))
            if cell is None:
                continue
            label = _cell_final_class(cell)
            if label not in _CLASS_ORDER:
                unknown_labels.append(f"{seed}:{cell_id}:{label}")
                label = "missing"
            matrix[iy, ix] = _CLASS_ORDER.index(label)
            technical_complete += int(cell.get("status") == "complete")
    errors = []
    if duplicates:
        errors.append("duplicate:" + ",".join(sorted(duplicates)))
    if unexpected:
        errors.append("unexpected:" + ",".join(sorted(unexpected)))
    if unknown_labels:
        errors.append("unknown_label:" + ",".join(sorted(unknown_labels)))
    return {
        "matrix": matrix,
        "seeds": EXPECTED_SEEDS,
        "cell_ids": cell_ids,
        "present": len(lookup),
        "technical_complete": technical_complete,
        "expected": len(expected),
        "missing": len(expected) - len(lookup),
        "error": ";".join(errors) if errors else None,
    }


def _figure_atlas(
    c1: Mapping[str, Any] | None,
    tier: str,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    shell = tier == "secondary_shell"
    title = "secondary-shell phenotype atlas — extrapolated sensitivity" if shell else "primary-convex phenotype atlas"
    ax.set_title(title)
    if c1 is None:
        _blocked(ax, "C1 summary missing/invalid")
    else:
        atlas = atlas_matrix(c1, tier)
        matrix = atlas["matrix"]
        ids = atlas["cell_ids"]
        cmap = ListedColormap(_CLASS_COLORS)
        norm = BoundaryNorm(
            np.arange(len(_CLASS_ORDER) + 1) - 0.5, cmap.N
        )
        ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xticks(
            np.arange(len(ids)), ids, rotation=55, ha="right", fontsize=7
        )
        ax.set_yticks(
            np.arange(len(EXPECTED_SEEDS)),
            [f"seed {seed}" for seed in EXPECTED_SEEDS],
        )
        ax.set_xlabel("locked slow-state cell (contract order)")
        ax.set_ylabel("independent seed")
        ax.text(
            0.995,
            1.02,
            (
                f"inventory {atlas['present']}/{atlas['expected']} | "
                f"technical complete {atlas['technical_complete']}/"
                f"{atlas['expected']}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#8C2D04" if atlas["missing"] else "#333333",
        )
        if atlas["error"] is not None:
            ax.text(
                0.005,
                1.02,
                f"contract: {atlas['error']}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
                color="#8C2D04",
            )
        present = sorted(set(matrix.reshape(-1)))
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                color=_CLASS_COLORS[index],
                label=_CLASS_ORDER[index],
                markersize=6,
            )
            for index in present
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.33),
            ncol=min(4, max(1, len(handles))),
            frameon=False,
            fontsize=7,
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path, dpi)


def _figure_spatiotemporal(
    representative: Mapping[str, Any] | None,
    arrays: Mapping[str, np.ndarray] | None,
    error: str | None,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.7))
    if representative is None or arrays is None:
        for ax in axes:
            _blocked(ax, error or "representative C1 evidence unavailable")
    else:
        source = _array(arrays, "source_rate_hz")
        bin_ms = _array(arrays, "bin_ms", "fine_bin_ms")
        if source is None or bin_ms is None:
            _blocked(axes[0], "source-rate trace missing")
        else:
            t = np.arange(source.size) * float(bin_ms.reshape(-1)[0]) * 1e-3
            axes[0].plot(t, source, color="#B2182B", lw=0.9)
            axes[0].set_xlabel("time (s)")
            axes[0].set_ylabel("source rate (Hz)")
        e_grid = _array(arrays, "E_rate_grid")
        if e_grid is None or e_grid.ndim != 3:
            _blocked(axes[1], "E-rate grid missing")
        else:
            image = axes[1].imshow(
                np.nanmean(e_grid, axis=0),
                origin="lower",
                cmap="plasma",
                aspect="equal",
            )
            fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03, label="mean E rate (Hz)")
            axes[1].set_xlabel("grid x")
            axes[1].set_ylabel("grid y")
        kymo = _array(arrays, "kymograph")
        axis_pos = _array(arrays, "axis_positions")
        if kymo is None or kymo.ndim != 2:
            _blocked(axes[2], "kymograph missing")
        else:
            extent = None
            if axis_pos is not None and axis_pos.size == kymo.shape[1]:
                extent = [0, kymo.shape[0], axis_pos[0], axis_pos[-1]]
            image = axes[2].imshow(
                kymo.T,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                extent=extent,
            )
            fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03, label="E rate (Hz)")
            axes[2].set_xlabel("fine time bin")
            axes[2].set_ylabel("shared-axis position")
        lfp = _array(arrays, "lfp_raw_synaptic_proxy")
        fs = _array(arrays, "lfp_fs_hz")
        if lfp is None or fs is None or lfp.ndim != 2:
            _blocked(axes[3], "virtual-SEEG traces missing")
        else:
            n_contact = min(8, lfp.shape[1])
            if n_contact == 0:
                _blocked(axes[3], "virtual-SEEG has no contacts")
            else:
                x = lfp[:, :n_contact]
                scale = np.nanpercentile(np.abs(x), 95)
                scale = 1.0 if not np.isfinite(scale) or scale <= 0 else scale
                t = np.arange(x.shape[0]) / float(fs.reshape(-1)[0])
                for index in range(n_contact):
                    axes[3].plot(t, x[:, index] / scale + index, color="#444444", lw=0.45)
                axes[3].set_yticks(range(n_contact), [f"C{index + 1}" for index in range(n_contact)])
                axes[3].set_xlabel("time (s)")
                axes[3].set_ylabel("virtual contacts")
                axes[3].margins(x=0)
    axes[0].set_title("source")
    axes[1].set_title("spatial grid")
    axes[2].set_title("kymograph")
    axes[3].set_title("raw-synaptic vSEEG proxy")
    if representative is not None:
        fig.suptitle(
            "fixed representative: "
            f"seed {representative['seed']} | {representative['tier']} | "
            f"{representative['cell_id']} | {representative['phase']} | "
            f"{representative['noise']} | "
            f"cell adjudication: {representative['cell_status']}",
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, path, dpi)


def _modal_route_kind(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", ""))
    route = str(row.get("route", ""))
    if status == "identified":
        return "identified"
    if (
        route == "saturated_sensitivity_only"
        or status == "summarized_without_operator"
    ):
        return "saturated_sensitivity_only"
    if route == "descriptive_only" or status == "descriptive_only":
        return "descriptive_only"
    return "not_identifiable"


def _scalar_items(value: Mapping[str, Any]) -> list[tuple[str, float]]:
    output = []
    for key, item in value.items():
        scalar = _finite(item)
        if scalar is not None:
            output.append((str(key), scalar))
            continue
        if isinstance(item, Mapping):
            point = _finite(item.get("point"))
            if point is not None:
                output.append((str(key), point))
    return output


def _plot_modal_seed(ax, row: Mapping[str, Any]) -> None:
    seed = int(row.get("seed", -1))
    route = _modal_route_kind(row)
    ax.set_title(f"seed {seed}: {route.replace('_', ' ')}", fontsize=9)
    if route == "identified":
        operator = row.get("operator_summary") or {}
        heldout = row.get("noise_heldout") or {}
        items = [
            ("spectral radius", _finite(operator.get("spectral_radius"))),
            (
                "spectral abscissa /ms",
                _finite(operator.get("spectral_abscissa_per_ms")),
            ),
            ("finite-time gain", _finite(operator.get("finite_time_gain"))),
            (
                "held-out relative error",
                _finite(heldout.get("heldout_relative_error")),
            ),
        ]
        if any(value is None for _, value in items):
            _blocked(ax, "identified operator fields incomplete")
            return
        ax.axis("off")
        for index, (name, value) in enumerate(items):
            ax.text(
                0.04,
                0.84 - index * 0.18,
                name,
                transform=ax.transAxes,
                fontsize=8,
                color="#333333",
            )
            ax.text(
                0.96,
                0.84 - index * 0.18,
                f"{value:.4g}",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                color="#2166AC",
            )
        return
    if route == "saturated_sensitivity_only":
        items = _scalar_items(
            row.get("locked_local_gain_and_refractory_sensitivity") or {}
        )
        if not items:
            _blocked(ax, "locked sensitivity summary unavailable")
            return
        names, values = zip(*items[:6])
        y = np.arange(len(values))
        ax.barh(y, values, color="#D7301F", alpha=0.75)
        ax.set_yticks(y, names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("locked sensitivity value")
        return
    if route == "descriptive_only":
        runs = [
            item for item in row.get("descriptive_runs", []) or []
            if isinstance(item, Mapping)
        ]
        values = [
            (
                str(item.get("noise", f"run {index + 1}")),
                _finite(item.get("mean_rate_hz")),
                _finite(item.get("rate_sd_hz")),
            )
            for index, item in enumerate(runs)
        ]
        values = [value for value in values if value[1] is not None]
        if not values:
            _blocked(ax, "descriptive run summary unavailable")
            return
        x = np.arange(len(values))
        mean = np.asarray([value[1] for value in values], float)
        sd = np.asarray([
            0.0 if value[2] is None else value[2] for value in values
        ])
        ax.errorbar(
            x, mean, yerr=sd, fmt="o", color="#542788", capsize=3
        )
        ax.set_xticks(
            x, [value[0] for value in values], rotation=35, ha="right",
            fontsize=7,
        )
        ax.set_ylabel("descriptive rate (Hz)")
        return
    _blocked(ax, str(row.get("reason", "modal route not identifiable")))


def _figure_modal(modal: Mapping[str, Any] | None, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.7))
    rows = [] if modal is None else modal.get("seed_results", []) or []
    by_seed = {
        int(row.get("seed", -1)): row
        for row in rows if isinstance(row, Mapping)
    }
    if modal is None or not rows:
        for ax in axes:
            _blocked(ax, "modal summary missing/incomplete")
    else:
        for ax, seed in zip(axes[:3], EXPECTED_SEEDS):
            row = by_seed.get(seed)
            if row is None:
                _blocked(ax, f"seed {seed} modal result missing")
            else:
                _plot_modal_seed(ax, row)
        axes[3].axis("off")
        routes = modal.get("routes_by_seed") or {}
        axes[3].set_title("aggregate routing", fontsize=9)
        axes[3].text(
            0.02,
            0.88,
            f"status: {modal.get('status', 'unknown')}",
            transform=axes[3].transAxes,
            fontsize=9,
        )
        axes[3].text(
            0.02,
            0.75,
            f"class disagreement: {modal.get('class_disagreement', 'N/A')}",
            transform=axes[3].transAxes,
            fontsize=8,
        )
        for index, seed in enumerate(EXPECTED_SEEDS):
            axes[3].text(
                0.02,
                0.60 - index * 0.14,
                f"seed {seed}: {routes.get(str(seed), 'missing')}",
                transform=axes[3].transAxes,
                fontsize=8,
            )
        axes[3].text(
            0.02,
            0.12,
            "Routes are seed-specific;\nno cross-route eigenvalue pooling.",
            transform=axes[3].transAxes,
            fontsize=8,
            color="#555555",
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path, dpi)


def _layer_rows(final: Mapping[str, Any]) -> list[tuple[str, str]]:
    layers = final.get("layers") or {}
    out = []
    for key in (
        "source_identity",
        "primary_neighbourhood",
        "secondary_shell",
        "seed_specific_modal",
        "observation_match",
        "entry",
        "offset",
        "recovery_lifecycle",
    ):
        value = layers.get(key, final.get(key, "missing"))
        if isinstance(value, Mapping):
            value = value.get("verdict", value.get("status", "present"))
        out.append((key, str(value)))
    return out


def _figure_status(final: Mapping[str, Any] | None, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.axis("off")
    if final is None:
        ax.set_facecolor("#E6E6E6")
        ax.text(0.5, 0.5, "BLOCKED\nfinal Phase-C verdict missing/invalid", ha="center", va="center")
    else:
        verdict = str(final.get("fine_verdict", final.get("verdict", "no_evidence")))
        route = str(final.get("next_route", "no_evidence"))
        ax.text(0.02, 0.94, "Phase-C status", fontsize=14, fontweight="bold", va="top")
        ax.text(0.02, 0.84, f"verdict: {verdict}", fontsize=11, va="top")
        ax.text(0.02, 0.77, f"next route: {route}", fontsize=10, va="top")
        rows = _layer_rows(final)
        y0 = 0.66
        for index, (name, value) in enumerate(rows):
            y = y0 - index * 0.065
            color = "#BDBDBD" if value in {"missing", "blocked", "not_tested", "not_established"} else "#91BFDB"
            ax.add_patch(
                plt.Rectangle((0.03, y - 0.025), 0.27, 0.045, facecolor=color, edgecolor="#777777", lw=0.5)
            )
            ax.text(0.04, y, name.replace("_", " "), va="center", fontsize=8)
            ax.text(0.32, y, value, va="center", fontsize=8)
        boundaries = (
            ("entry", final.get("entry")),
            ("offset", final.get("offset")),
            ("recovery/lifecycle", final.get("recovery_lifecycle")),
            ("Phase C2 authorized", final.get("phase_c2_authorized")),
            ("actuator authorized", final.get("actuator_authorized")),
        )
        ax.text(0.66, 0.88, "claim boundary", fontsize=11, fontweight="bold", va="top")
        for index, (name, value) in enumerate(boundaries):
            ax.text(0.66, 0.78 - index * 0.09, f"{name}: {value}", fontsize=9, va="top")
        ax.text(
            0.66,
            0.28,
            "No observation match.\nNo seizure entry or offset test.\n"
            "No recovery or lifecycle claim.",
            fontsize=10,
            color="#8C2D04",
            va="top",
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path, dpi)


def _readme_text() -> str:
    entries = {
        FIGURE_FILENAMES[0]: (
            "比较 core 与全片的 ceiling occupancy：core 使用 seed 内分层 95% CI；"
            "全片只展示 6 个 locked continuation 值及 IQR，不把它们伪装成独立重复。"
            "第三个面板保留 pre-entry denominator 与 carrier numerator，避免只显示比值。\n\n"
            "**关注点**：判断 tonic 支路更接近 refractory saturation 还是仍保留局部可增益性。"
        ),
        FIGURE_FILENAMES[1]: (
            "并列展示 ISI CV2、refractory locking、20-ms Fano，以及三个空间 stratum 中"
            "最不利的 pairwise observed-minus-shift-null q97.5 分层 95% CI。"
            "神经元或 pair 只用于 seed 内不确定性，图上不把它们当独立重复。\n\n"
            "**关注点**：AI 判据要求 max-excess 的 CI 上界低于零；周期锁定与相关性过强不能被 pooled median 掩盖。"
        ),
        FIGURE_FILENAMES[2]: (
            "展示 raw synaptic 与 effective membrane-drive 的 E/I 分量、阈值距离、fine-rate、PSD 和 current-vSEEG proxy。"
            "这里的 current/vSEEG 都是模型诊断量，不是生物物理 transmembrane current 或真实观测匹配。\n\n"
            "**关注点**：膜方程实际看到的净驱动是否与 raw synaptic proxy 给出一致的 carrier 身份。"
        ),
        FIGURE_FILENAMES[3]: (
            "按 locked slow-state path 排列 primary-convex cell，并逐 seed 显示最终 phenotype。"
            "图中同时报告 exact coverage；缺失、无效或 indeterminate cell 保持灰色/独立颜色。\n\n"
            "**关注点**：primary reachable neighbourhood 内是否存在连续、跨 seed 的成熟窗口。"
        ),
        FIGURE_FILENAMES[4]: (
            "以相同语法展示 secondary-shell phenotype atlas，但明确标为 extrapolated sensitivity。"
            "这一层不等同于 primary reachable path，也不能单独授权后续 slow-path 机制。\n\n"
            "**关注点**：主邻域以外是否只有外推候选，还是同样为有界负结果。"
        ),
        FIGURE_FILENAMES[5]: (
            "代表 run 由固定规则选择：完整 run 中优先 seed 1、primary tier，再按 cell/path/phase/noise 字典序。"
            "四个面板只展示 source rate、空间 grid、kymograph 与 raw-synaptic virtual-SEEG proxy。\n\n"
            "**关注点**：空间 carrier 的形态是否与 atlas 分类一致，而不是挑选最漂亮的个例。"
        ),
        FIGURE_FILENAMES[6]: (
            "逐 seed 按预先分配的 route 展示证据：identified 显示 operator 与 held-out 误差，"
            "saturated_sensitivity_only 显示 locked sensitivity，descriptive_only 显示 rate 摘要。"
            "modal 层不替代 carrier identity 或 lifecycle 证据。\n\n"
            "**关注点**：每个 seed 是否在自己的合法 route 内得到可解释结果，而不是强迫共享 operator。"
        ),
        FIGURE_FILENAMES[7]: (
            "汇总 C0、primary、shell、modal 和最终 verdict，并把未测试的层明确列出。"
            "右侧固定写出 entry、offset、recovery/lifecycle 与 actuator 的授权边界。\n\n"
            "**关注点**：区分本轮执行完成、机制身份结论和仍未建立的完整发作生命周期。"
        ),
    }
    lines = [
        "# Z/M Phase C 诊断图",
        "",
        f"全部图的固定语义：`{BOUNDARY}`。这些图不建立 entry、offset、recovery、observation matching 或 lifecycle。",
        "",
    ]
    for filename in FIGURE_FILENAMES:
        lines.extend((f"### {filename}", "", entries[filename], ""))
    return "\n".join(lines).rstrip() + "\n"


def render_phasec_figures(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    c0_summary_path: str | Path,
    c1_summary_path: str | Path,
    modal_summary_path: str | Path,
    final_verdict_path: str | Path,
    dpi: int = 160,
) -> dict[str, Any]:
    """Render all eight diagnostic figures and a provenance manifest."""
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "c0": load_json_artifact(c0_summary_path),
        "c1": load_json_artifact(c1_summary_path),
        "modal": load_json_artifact(modal_summary_path),
        "final": load_json_artifact(final_verdict_path),
    }
    contract_errors = {
        key: _validate_summary_contract(key, artifact)
        for key, artifact in artifacts.items()
    }
    c0 = (
        artifacts["c0"].data
        if contract_errors["c0"] is None else None
    )
    c1 = (
        artifacts["c1"].data
        if contract_errors["c1"] is None else None
    )
    modal = (
        artifacts["modal"].data
        if contract_errors["modal"] is None else None
    )
    final = (
        artifacts["final"].data
        if contract_errors["final"] is None else None
    )
    if final is not None:
        contract_errors["final_references"] = _validate_final_references(
            root, final
        )
    else:
        contract_errors["final_references"] = (
            contract_errors["final"] or "final_invalid"
        )
    if all(value is not None for value in (c0, c1, modal, final)):
        contract_errors["parent_identity"] = _validate_parent_identity(
            c0, c1, modal, final
        )
    else:
        contract_errors["parent_identity"] = "summary_contract_blocked"

    representative = select_representative_run(c1 or {})
    rep_part, rep_arrays, rep_error = load_part_npz(
        root,
        representative,
        expected_part_schema=C1_PART_SCHEMA,
    )
    rep_error = _validate_plot_part(
        rep_part, rep_arrays, rep_error, kind="c1"
    )
    c0_reference = _select_c0_part(c0 or {})
    c0_part, c0_arrays, c0_error = load_part_npz(
        root,
        c0_reference,
        expected_part_schema=C0_PART_SCHEMA,
    )
    c0_error = _validate_plot_part(
        c0_part, c0_arrays, c0_error, kind="c0"
    )
    whole_sheet, whole_sheet_error = c0_whole_sheet_ceiling(
        root, c0 or {}
    )
    primary_atlas = atlas_matrix(c1 or {}, "primary_convex")
    shell_atlas = atlas_matrix(c1 or {}, "secondary_shell")
    atlas_errors = {}
    for name, atlas in (
        ("primary_atlas", primary_atlas),
        ("shell_atlas", shell_atlas),
    ):
        reasons = []
        if atlas["error"] is not None:
            reasons.append(str(atlas["error"]))
        if atlas["missing"]:
            reasons.append(
                f"missing_expected_cells:{atlas['missing']}/"
                f"{atlas['expected']}"
            )
        atlas_errors[name] = ";".join(reasons) if reasons else None

    _figure_ceiling_gain(
        c0,
        whole_sheet,
        whole_sheet_error,
        out / FIGURE_FILENAMES[0],
        dpi,
    )
    _figure_irregularity(root, c0, out / FIGURE_FILENAMES[1], dpi)
    _figure_currents(
        c0_part, c0_arrays, c0_error, out / FIGURE_FILENAMES[2], dpi
    )
    _figure_atlas(c1, "primary_convex", out / FIGURE_FILENAMES[3], dpi)
    _figure_atlas(c1, "secondary_shell", out / FIGURE_FILENAMES[4], dpi)
    _figure_spatiotemporal(
        representative, rep_arrays, rep_error, out / FIGURE_FILENAMES[5], dpi
    )
    _figure_modal(modal, out / FIGURE_FILENAMES[6], dpi)
    _figure_status(final, out / FIGURE_FILENAMES[7], dpi)

    missing_figures = [
        filename for filename in FIGURE_FILENAMES
        if not (out / filename).is_file() or (out / filename).stat().st_size == 0
    ]
    if missing_figures:
        raise PlotEvidenceError(
            "figure render incomplete: " + ",".join(missing_figures)
        )
    all_errors = {
        **contract_errors,
        "representative": rep_error,
        "c0_representative": c0_error,
        "whole_sheet_ceiling": whole_sheet_error,
        **atlas_errors,
    }
    complete = all(value is None for value in all_errors.values())
    manifest = {
        "schema": "zm_phasec_diagnostic_figures_v1_2026-07-28",
        "claim_boundary": BOUNDARY,
        "status": "complete" if complete else "BLOCKED",
        "blocked_reasons": {
            key: value for key, value in all_errors.items()
            if value is not None
        },
        "inputs": {
            key: {
                "path": str(artifact.path),
                "sha256": artifact.sha256,
                "status": (
                    "valid"
                    if contract_errors[key] is None else "BLOCKED"
                ),
                "reason": contract_errors[key],
            }
            for key, artifact in artifacts.items()
        },
        "representative_selection_rule": (
            "run technical complete plus immutable part provenance only; "
            "cell adjudication is annotation, not eligibility; seed1 first; "
            "primary first; cell/path/phase/noise lexical order"
        ),
        "representative": representative,
        "representative_error": rep_error,
        "c0_representative": c0_reference,
        "c0_representative_error": c0_error,
        "whole_sheet_ceiling": {
            "source_field": "spike_metrics.firing.rho80_all_median",
            "unit_of_display": "locked continuation",
            "uncertainty": "within-seed IQR, not pooled-neuron CI",
            "values_by_seed": whole_sheet,
            "error": whole_sheet_error,
        },
        "atlas_inventory": {
            "primary_convex": {
                key: value for key, value in primary_atlas.items()
                if key != "matrix"
            },
            "secondary_shell": {
                key: value for key, value in shell_atlas.items()
                if key != "matrix"
            },
        },
        "figures": [
            {
                "filename": filename,
                "sha256": sha256_file(out / filename),
            }
            for filename in FIGURE_FILENAMES
        ],
        "readme": "README.md",
    }
    readme_path = out / "README.md"
    manifest_path = out / "phasec_figure_manifest.json"
    if complete:
        readme_path.write_text(_readme_text(), encoding="utf-8")
        tmp_path = out / ".phasec_figure_manifest.json.tmp"
        tmp_path.write_text(
            json.dumps(
                manifest, indent=2, sort_keys=True, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(manifest_path)
    else:
        # Never leave stale complete acceptance metadata beside a blocked
        # re-render.  The eight PNGs remain as explicit grey diagnostics.
        readme_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    return manifest
