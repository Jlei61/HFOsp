import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from src.topic4_zm_phasec_plot import (
    BOUNDARY,
    EXPECTED_SEEDS,
    FIGURE_FILENAMES,
    PRIMARY_CELL_NAMES,
    SHELL_CELL_NAMES,
    _modal_route_kind,
    align_current_vseeg,
    atlas_matrix,
    render_phasec_figures,
    select_representative_run,
    sha256_file,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _write_observables(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    n_t, n_grid, n_contact = 64, 5, 4
    time_ms = np.arange(n_t) * 5.0
    e_grid = 20.0 + 5.0 * rng.normal(size=(n_t, n_grid, n_grid))
    i_grid = 15.0 + 3.0 * rng.normal(size=(n_t, n_grid, n_grid))
    lfp = rng.normal(size=(n_t * 2, n_contact))
    np.savez_compressed(
        path,
        phasec1_observables_schema=np.asarray(
            "zm_phasec1_observables_v1_2026-07-28"
        ),
        raw_sample_time_ms=time_ms,
        raw_raw_ampa_core_mean_mV=1.0 + 0.2 * np.sin(time_ms / 30.0),
        raw_raw_gaba_core_mean_mV=0.8 + 0.1 * np.cos(time_ms / 35.0),
        effective_sample_time_ms=time_ms,
        effective_effective_excitation_core_mean_mV=0.9 + 0.1 * np.sin(time_ms / 31.0),
        effective_effective_outward_total_core_mean_mV=0.7 + 0.1 * np.cos(time_ms / 29.0),
        effective_effective_net_drive_core_mean_mV=0.2 + 0.1 * np.sin(time_ms / 17.0),
        E_rate_grid=e_grid,
        I_rate_grid=i_grid,
        bin_ms=np.asarray(5.0),
        fine_bin_ms=np.asarray(5.0),
        source_rate_hz=e_grid[:, 2, 2],
        rest_mask=np.zeros(n_t, bool),
        active_area_fraction=np.mean(e_grid > 5.0, axis=(1, 2)),
        kymograph=np.mean(e_grid, axis=1),
        axis_positions=np.linspace(-5.0, 5.0, n_grid),
        lfp_raw_synaptic_proxy=lfp,
        lfp_fs_hz=np.asarray(200.0),
    )


def _margin(value):
    return {
        "core_free_E": {
            "quantiles_mV": {"50.0": value},
        }
    }


def _part(path: Path, npz_path: Path, *, schema):
    payload = {
        "schema": schema,
        "status": "complete",
        "observables_path": str(npz_path),
        "observables_sha256": sha256_file(npz_path),
        "spike_metrics": {
            "firing": {
                "rho80_active_core_median": 0.10,
                "rho80_all_median": 0.12,
            },
            "fano": {"fano_by_bin": {"20ms": {"median": 0.82}}},
        },
        "threshold_margin_initial": _margin(0.9),
        "threshold_margin_final": _margin(0.6),
    }
    _write_json(path, payload)
    return payload


def _ci(point, lo=None, hi=None):
    return {
        "point": point,
        "lo": point - 0.05 if lo is None else lo,
        "hi": point + 0.05 if hi is None else hi,
    }


def _seed_row(seed, part_path):
    return {
        "seed": seed,
        "klass": "balanced_AI_tonic_candidate",
        "hierarchical_ci": {
            "rho80_active_core": _ci(0.10, 0.07, 0.14),
            "gain_relative_to_preentry": _ci(0.72, 0.62, 0.84),
            "isi_cv2_median": _ci(0.91, 0.80, 1.02),
            "refractory_isi_fraction": _ci(0.13, 0.08, 0.19),
            "pairwise_observed_median": _ci(0.03, 0.01, 0.05),
            "pairwise_null_q97_5": _ci(0.10, 0.08, 0.12),
            "active_area_fraction": _ci(0.22, 0.18, 0.27),
        },
        "rows": [
            {
                "phase": phase,
                "noise": noise,
                "identity_path": str(part_path),
                "gain_preentry": {
                    "gain_hz_per_mV": 8.0,
                    "gain_hz_per_mV_blocks": [7.8, 8.2],
                },
                "gain_carrier": {
                    "gain_hz_per_mV": 6.0,
                    "gain_hz_per_mV_blocks": [5.8, 6.2],
                },
            }
            for phase in ("bounded_mid__rising", "bounded_mid__peak")
            for noise in (
                "noise_replay", "noise_resample_1", "noise_resample_2"
            )
        ],
    }


def test_representative_selection_is_fixed_and_not_outcome_ranked():
    summary = {
        "cells": [
            {
                "seed": 1,
                "tier": "secondary_shell",
                "cell_id": "a",
                "path_index": 0,
                "path_direction": "forward",
                "status": "complete",
                "cell_class": "periodic_non_tonic_carrier",
                "run_rows": [
                    {
                        "status": "complete",
                        "phase": "bounded_mid__rising",
                        "noise": "noise_replay",
                        "part_path": "shell.json",
                        "part_sha256": "a" * 64,
                    }
                ],
            },
            {
                "seed": 1,
                "tier": "primary_convex",
                "cell_id": "b",
                "path_index": 1,
                "path_direction": "reverse",
                "status": "complete",
                "cell_class": "tonic_non_AI",
                "run_rows": [
                    {
                        "status": "complete",
                        "phase": "bounded_mid__peak",
                        "noise": "noise_resample_2",
                        "part_path": "primary_b.json",
                        "part_sha256": "b" * 64,
                    }
                ],
            },
            {
                "seed": 1,
                "tier": "primary_convex",
                "cell_id": "a",
                "path_index": 8,
                "path_direction": "forward",
                "status": "indeterminate",
                "cell_class": "runaway",
                "run_rows": [
                    {
                        "status": "complete",
                        "phase": "bounded_mid__rising",
                        "noise": "noise_resample_1",
                        "part_path": "primary_a.json",
                        "part_sha256": "c" * 64,
                    }
                ],
            },
        ]
    }
    chosen = select_representative_run(summary)
    assert chosen["part_path"] == "primary_a.json"
    assert chosen["cell_status"] == "indeterminate"
    assert "cell_class" not in chosen


def test_atlas_uses_locked_denominator_and_preserves_rest_and_missing():
    cells = [
        {
            "seed": 1,
            "tier": "primary_convex",
            "cell_id": PRIMARY_CELL_NAMES[0],
            "status": "complete",
            "cell_class": "rest_or_silence",
        }
    ]
    atlas = atlas_matrix({"cells": cells}, "primary_convex")
    assert atlas["matrix"].shape == (3, 10)
    assert atlas["expected"] == 30
    assert atlas["present"] == 1
    assert atlas["missing"] == 29
    assert atlas["matrix"][0, 0] != atlas["matrix"][0, 1]
    shell = atlas_matrix({"cells": []}, "secondary_shell")
    assert shell["matrix"].shape == (3, 8)
    assert shell["expected"] == 24


def test_current_vseeg_alignment_uses_physical_time():
    t_ms = np.asarray([100.0, 150.0, 200.0, 250.0])
    current = np.asarray([1.0, 2.0, 3.0, 4.0])
    lfp = np.arange(8, dtype=float)[:, None]
    t, paired_current, paired_lfp = align_current_vseeg(
        t_ms, current, lfp, 20.0
    )
    assert np.array_equal(t, t_ms)
    assert np.array_equal(paired_current, current)
    assert np.allclose(paired_lfp, [2.0, 3.0, 4.0, 5.0])


def test_modal_route_detection_accepts_all_three_scientific_branches():
    assert _modal_route_kind({
        "route": "AI_observational_DMD", "status": "identified"
    }) == "identified"
    assert _modal_route_kind({
        "route": "saturated_sensitivity_only",
        "status": "summarized_without_operator",
    }) == "saturated_sensitivity_only"
    assert _modal_route_kind({
        "route": "descriptive_only", "status": "descriptive_only"
    }) == "descriptive_only"


def test_render_eight_phasec_diagnostics_and_readme(tmp_path):
    repo = tmp_path / "repo"
    out = repo / "results/topic4_sef_hfo/zm_phase_c_tonic_identity/figures"
    c0_npz = repo / "parts/c0_observables.npz"
    c0_part = repo / "parts/c0_identity.json"
    c1_npz = repo / "parts/c1_observables.npz"
    c1_part = repo / "parts/c1_phenotype.json"
    _write_observables(c0_npz)
    _write_observables(c1_npz)
    _part(
        c0_part,
        c0_npz,
        schema="zm_phasec_identity_cell_v1",
    )
    _part(
        c1_part,
        c1_npz,
        schema="zm_phasec1_base_part_v1_2026-07-28",
    )

    phasec_body = {
        "per_seed": {
            "1": {"canonical_config_sha": "1" * 64},
            "3": {"canonical_config_sha": "3" * 64},
            "4": {"canonical_config_sha": "4" * 64},
        },
        "provenance": {
            "producer_file_sha256": {"producer.py": "9" * 64}
        },
        "production_authorized": True,
    }
    phasec_sha = _canonical_sha(phasec_body)
    c0 = {
        "schema": "zm_phasec_c0_summary_v1",
        "manifest_sha256": phasec_sha,
        "panel_manifest_sha256": "b" * 64,
        "resolution": "dt",
        "seed_rows": [_seed_row(seed, c0_part) for seed in (1, 3, 4)],
        "aggregate": {"verdict": "balanced_AI_tonic_candidate_supported"},
        "claim_boundary": BOUNDARY,
    }
    primary_cells = []
    for seed in EXPECTED_SEEDS:
        for path_index, cell_id in enumerate(PRIMARY_CELL_NAMES):
            representative = seed == 1 and path_index == 0
            final_class = (
                "periodic_non_tonic_carrier"
                if representative else "rest_or_silence"
            )
            primary_cells.append({
                "seed": seed,
                "tier": "primary_convex",
                "cell_id": cell_id,
                "path_index": path_index,
                "path_direction": "forward",
                "status": (
                    "probabilistically_indeterminate"
                    if representative else "complete"
                ),
                "cell_class": final_class,
                "conditional_gain": {"final_cell_class": final_class},
                "run_rows": (
                    [{
                        "status": "complete",
                        "phase": "bounded_mid__rising",
                        "noise": "noise_replay",
                        "part_path": str(c1_part),
                        "part_sha256": sha256_file(c1_part),
                    }] if representative else []
                ),
            })
    shell_cells = [
        {
            "seed": seed,
            "tier": "secondary_shell",
            "cell_id": cell_id,
            "path_index": path_index,
            "path_direction": "forward",
            "status": "complete",
            "cell_class": "probabilistically_indeterminate",
            "conditional_gain": {},
            "run_rows": [],
        }
        for seed in EXPECTED_SEEDS
        for path_index, cell_id in enumerate(SHELL_CELL_NAMES)
    ]
    c1 = {
        "schema": "zm_phasec1_summary_v1_2026-07-28",
        "phasec_manifest_sha256": phasec_sha,
        "coordinate_manifest_sha256": "c" * 64,
        "coordinate_manifest_semantic_sha256": "d" * 64,
        "resolution": "dt",
        "cells": primary_cells + shell_cells,
        "primary_adjudication": {
            "status": "isolated_maturation_candidate",
        },
        "secondary_shell_adjudication": {
            "status": "no_local_maturation_window",
        },
        "verdict": "isolated_maturation_candidate",
        "claim_boundary": BOUNDARY,
    }
    modal = {
        "schema": "zm_phasec_seed_modal_v1_2026-07-28",
        "phasec_manifest_sha256": phasec_sha,
        "status": "complete",
        "routes_by_seed": {
            "1": "AI_observational_DMD",
            "3": "saturated_sensitivity_only",
            "4": "descriptive_only",
        },
        "input_provenance": {},
        "seed_results": [
            {
                "seed": 1,
                "route": "AI_observational_DMD",
                "status": "identified",
                "operator_summary": {
                    "spectral_radius": 0.81,
                    "spectral_abscissa_per_ms": -0.02,
                    "finite_time_gain": 1.4,
                },
                "noise_heldout": {
                    "status": "ok",
                    "heldout_relative_error": 0.1,
                },
            },
            {
                "seed": 3,
                "route": "saturated_sensitivity_only",
                "status": "summarized_without_operator",
                "locked_local_gain_and_refractory_sensitivity": {
                    "gain_relative_to_preentry": _ci(0.2),
                    "refractory_isi_fraction": _ci(0.9),
                    "rho80_active_core": _ci(0.8),
                    "source": "locked_C0_hierarchical_ci",
                },
            },
            {
                "seed": 4,
                "route": "descriptive_only",
                "status": "descriptive_only",
                "descriptive_runs": [{
                    "noise": "noise_replay",
                    "mean_rate_hz": 18.0,
                    "rate_sd_hz": 3.0,
                }],
            },
        ],
        "claim_boundary": BOUNDARY,
    }
    input_dir = repo / "summaries"
    c0_path = input_dir / "c0.json"
    c1_path = input_dir / "c1.json"
    modal_path = input_dir / "modal.json"
    final_path = input_dir / "final.json"
    for path, payload in (
        (c0_path, c0),
        (c1_path, c1),
        (modal_path, modal),
    ):
        _write_json(path, payload)
    coverage_path = input_dir / "coverage.json"
    coverage = {
        "c0": {"status": "complete"},
        "c1_primary": {"status": "complete"},
        "c1_shell": {"status": "complete"},
        "modal": {"status": "complete"},
    }
    _write_json(coverage_path, coverage)
    phasec_path = input_dir / "phasec_manifest.json"
    _write_json(phasec_path, {
        **phasec_body,
        "manifest_sha256": phasec_sha,
    })
    trigger_path = input_dir / "phasec1_gain_trigger_manifest.json"
    trigger_body = {
        "schema": "zm_phasec1_gain_trigger_manifest_v1_2026-07-28",
        "phasec_manifest_sha256": phasec_sha,
        "selection_is_closed": True,
        "producer_file_sha256": {"producer.py": "0" * 64},
        "triggered_cells": [],
    }
    trigger_sha = _canonical_sha(trigger_body)
    _write_json(trigger_path, {
        **trigger_body,
        "manifest_sha256": trigger_sha,
    })

    refs = {
        "c0": c0_path,
        "c1_primary": c1_path,
        "c1_shell": c1_path,
        "modal": modal_path,
        "coverage": coverage_path,
    }
    final = {
        "schema": "zm_phasec_final_adjudication_v1_2026-07-28",
        "version": "zm_phasec_verdict_v1_2026-07-28",
        "fine_verdict": "isolated_maturation_candidate",
        "next_route": "mixed_identity_requires_refinement",
        "layers": {
            "source_identity": {
                "verdict": "balanced_AI_tonic_candidate_supported"
            },
            "primary_neighbourhood": {
                "verdict": "isolated_maturation_candidate"
            },
            "secondary_shell": {"verdict": "no_maturation"},
            "seed_specific_modal": {"status": "complete"},
            "observation_match": "blocked",
            "entry": "not_tested",
            "offset": "not_tested",
            "recovery_lifecycle": "not_established",
        },
        "entry": "not_tested",
        "offset": "not_tested",
        "recovery_lifecycle": "not_established",
        "phase_c2_authorized": False,
        "actuator_authorized": False,
        "input_file_provenance": {
            name: {
                "status": "complete",
                "path": str(path),
                "file_sha256": sha256_file(path),
                "artifact_sha256": "e" * 64,
                "parent_phasec_manifest_sha256": phasec_sha,
            }
            for name, path in refs.items()
        },
        "phasec_manifest_provenance": {
            "path": str(phasec_path),
            "file_sha256": sha256_file(phasec_path),
            "manifest_sha256": phasec_sha,
        },
        "trigger_provenance": {
            "status": "complete",
            "path": str(trigger_path),
            "file_sha256": sha256_file(trigger_path),
            "manifest_sha256": trigger_sha,
            "producer_file_sha256": {"producer.py": "0" * 64},
            "parent_phasec_manifest_sha256": phasec_sha,
        },
        "wrapper_provenance_issues": [],
    }
    _write_json(final_path, final)

    manifest = render_phasec_figures(
        repo_root=repo,
        output_dir=out,
        c0_summary_path=c0_path,
        c1_summary_path=c1_path,
        modal_summary_path=modal_path,
        final_verdict_path=final_path,
        dpi=55,
    )

    assert manifest["status"] == "complete"
    assert manifest["representative"]["cell_id"] == PRIMARY_CELL_NAMES[0]
    assert manifest["representative"]["cell_status"] == (
        "probabilistically_indeterminate"
    )
    assert manifest["representative_selection_rule"].startswith(
        "run technical complete"
    )
    assert manifest["atlas_inventory"]["primary_convex"]["expected"] == 30
    assert manifest["atlas_inventory"]["secondary_shell"]["expected"] == 24
    assert manifest["whole_sheet_ceiling"]["uncertainty"].endswith(
        "not pooled-neuron CI"
    )
    for filename in FIGURE_FILENAMES:
        image_path = out / filename
        assert image_path.is_file()
        image = mpimg.imread(image_path)
        assert image.shape[0] > 100
        assert image.shape[1] > 200
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.count("### ") == 8
    assert readme.count("**关注点**：") == 8
    assert BOUNDARY in readme
    assert "extrapolated sensitivity" in readme
    assert (out / "phasec_figure_manifest.json").is_file()

    # A referenced NPZ hash drift is a technical blocker.  Diagnostic PNGs
    # remain visible, but acceptance metadata must not be published.
    with c1_npz.open("ab") as handle:
        handle.write(b"drift")
    blocked_out = repo / "blocked-hash"
    blocked = render_phasec_figures(
        repo_root=repo,
        output_dir=blocked_out,
        c0_summary_path=c0_path,
        c1_summary_path=c1_path,
        modal_summary_path=modal_path,
        final_verdict_path=final_path,
        dpi=45,
    )
    assert blocked["status"] == "BLOCKED"
    assert "sha256_mismatch" in blocked["blocked_reasons"]["representative"]
    assert not (blocked_out / "README.md").exists()
    assert not (blocked_out / "phasec_figure_manifest.json").exists()
    assert all((blocked_out / filename).is_file() for filename in FIGURE_FILENAMES)

    # Missing current-schema final fields are also blocked; they are never
    # silently treated as a complete status figure.
    bad_final = dict(final)
    bad_final.pop("input_file_provenance")
    bad_final_path = input_dir / "bad_final.json"
    _write_json(bad_final_path, bad_final)
    missing_out = repo / "blocked-missing"
    missing = render_phasec_figures(
        repo_root=repo,
        output_dir=missing_out,
        c0_summary_path=c0_path,
        c1_summary_path=c1_path,
        modal_summary_path=modal_path,
        final_verdict_path=bad_final_path,
        dpi=45,
    )
    assert missing["status"] == "BLOCKED"
    assert "input_file_provenance" in missing["blocked_reasons"]["final"]
    assert not (missing_out / "README.md").exists()
    assert not (missing_out / "phasec_figure_manifest.json").exists()
