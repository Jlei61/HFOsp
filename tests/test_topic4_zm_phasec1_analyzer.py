"""Pure tests for the C1 atlas, window and conditional-gain glue."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.analyze_topic4_zm_phasec1 as A  # noqa: E402
import scripts.analyze_topic4_zm_phasec1_v2 as A2  # noqa: E402
import scripts.lock_topic4_zm_phasec1_gain_triggers as L  # noqa: E402
import src.topic4_zm_phasec_neighbourhood as N  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402


def _write_receipted_part(path, payload, *, manifest_sha256, task_key):
    path = Path(path)
    token = "test-" + str(abs(hash((str(path), task_key))))
    runtime = dict(payload.get("runtime_provenance", {}))
    runtime.update({
        "coordinator_run_id": "c1-analyzer-test",
        "coordinator_launch_token": token,
        "self_pid_at_publish": 12345,
        "self_vm_swap_kb_at_publish": 0,
    })
    payload = {**payload, "runtime_provenance": runtime}
    path.write_text(json.dumps(payload, sort_keys=True))
    audit = {
        "pid": 12345,
        "task_key": task_key,
        "coordinator_run_id": "c1-analyzer-test",
        "coordinator_launch_token": token,
        "n_samples": 1,
        "first_sample_at": 1.0,
        "last_sample_at": 1.0,
        "observed_max_kb": 0,
        "final_publish_swap_kb": 0,
    }
    receipt = PRES.build_resource_receipt(
        artifact_path=path,
        artifact_root=A.ROOT,
        artifact_sha256=A._sha256(path),
        manifest_sha256=manifest_sha256,
        task_key=task_key,
        run_id="c1-analyzer-test",
        launch_token=token,
        pid=12345,
        audit_row=audit,
        sampled_allowed_bytes=0,
    )
    receipt_path = PRES.resource_receipt_path(path)
    if receipt_path.exists():
        valid, reason, _ = PRES.validate_resource_receipt(
            receipt_path,
            artifact_path=path,
            artifact_root=A.ROOT,
            manifest_sha256=manifest_sha256,
            task_key=task_key,
        )
        assert valid, reason
    else:
        PRES.publish_resource_receipt_once(receipt_path, receipt)
    return payload


def _hierarchical(ai=True):
    return {
        "rho80_active_core_by_block_window": np.full(
            (16, 6), 0.10 if ai else 0.40
        ),
        "block_isi_cv2_by_panel_neuron": np.full((16, 8), 0.80),
        "block_refractory_isi_numerator_by_stratum": np.full(
            (16, 2), 10, np.int64
        ),
        "block_refractory_isi_denominator_by_stratum": np.full(
            (16, 2), 100, np.int64
        ),
        "refractory_isi_stratum_names": ("core", "surround"),
        "pair_corr_by_block_and_pair": np.full((16, 12), 0.05),
        "pair_null_median_by_block_and_draw": np.full((16, 3, 99), 0.20),
        "pair_strata": np.resize(np.asarray([0, 1, 2], np.int8), 12),
        "active_area_fraction_by_block_window": np.full((16, 20), 0.20),
    }


def _coordinate(cell_id="primary__rising__bounded_mid", tier="primary_convex"):
    return {
        "seed": 1,
        "cell_id": cell_id,
        "tier": tier,
        "trajectory_id": "rising",
        "path_index": 2,
        "path_direction": "forward",
        "state_sha256": "a" * 64,
        "status": "valid",
    }


def test_phenotype_loader_requires_separate_all_sheet_runaway_trace(
    tmp_path, monkeypatch
):
    """Core morphology and whole-sheet runaway must remain separate signals."""
    monkeypatch.setattr(A2, "ROOT", tmp_path)
    path = tmp_path / "observables.npz"
    n = 16
    base = {
        "phasec1_observables_schema": np.asarray(A2.C1_OBSERVABLES_SCHEMA),
        "bin_ms": np.asarray(2.0),
        "E_rate_grid": np.zeros((n, 2, 2), np.float32),
        "I_rate_grid": np.zeros((n, 2, 2), np.float32),
        "source_rate_hz": np.full(n, 440.0, np.float32),
        "rest_mask": np.zeros(n, bool),
        "active_area_fraction": np.full(n, 0.25, np.float32),
        "kymograph": np.zeros((n, 4), np.float32),
        "axis_positions": np.arange(4, dtype=np.float32),
        "readout_kernel_width_mm": np.asarray(0.278),
    }
    np.savez_compressed(path, **base)
    part = {
        "observables_path": str(path),
        "observables_sha256": A2._sha256(path),
    }
    blocked = A2._load_phenotype_arrays(part)
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == (
        "missing_phenotype_npz_fields:carrier_gate_r_all_hz,"
        "carrier_gate_bin_ms"
    )

    np.savez_compressed(
        path,
        **base,
        carrier_gate_r_all_hz=np.full(4, 150.0, np.float32),
        carrier_gate_bin_ms=np.asarray(25.0),
    )
    part["observables_sha256"] = A2._sha256(path)
    loaded = A2._load_phenotype_arrays(part)
    assert loaded["status"] == "ok"
    assert loaded["all_sheet_rate_hz"].shape == (4,)
    assert loaded["all_sheet_bin_ms"] == 25.0


def _six_runs(label="tonic_non_AI", spike_pass=False):
    phase = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    base_signature = np.column_stack([
        np.sin(phase),
        np.sin(phase + 0.4),
        np.sin(phase + 0.9),
        np.sin(phase + 1.4),
    ])
    rows = []
    for phase in A.PHASES:
        for noise in A.NOISES:
            rows.append({
                "seed": 1,
                "phase": phase,
                "noise": noise,
                "status": "complete",
                "terminal_class": label,
                "path_direction": "forward",
                "spike_ai_screen": {"pass": spike_pass},
                "_hierarchical": _hierarchical(ai=spike_pass),
                "spatial_relay": {"is_spatial_relay": False},
                "part_path": f"parts/{phase}/{noise}.json",
                "part_sha256": "b" * 64,
                "locked_arm_identity": {
                    "schema": A.C1_BASE_PART_SCHEMA,
                    "phasec_manifest_sha256": "c" * 64,
                    "coordinate_manifest_sha256": "d" * 64,
                    "coordinate_manifest_semantic_sha256": "8" * 64,
                    "seed": 1,
                    "cell_id": "primary__rising__bounded_mid",
                    "tier": "primary_convex",
                    "trajectory_id": "rising",
                    "path_index": 2,
                    "path_direction": "forward",
                    "phase": phase,
                    "noise": noise,
                    "resolution": "dt",
                    "slow_state_sha256": "a" * 64,
                    "coordinate_npz_file_sha256": "6" * 64,
                    "coordinate_npz_semantic_sha256": "7" * 64,
                    "config_sha": "1" * 64,
                    "fast_base_state_hash": "2" * 64,
                    "state_file_sha256": "3" * 64,
                    "noise_bank_sha": "4" * 64,
                    "burn_in_ms": 500.0,
                    "measure_ms": 8000.0,
                },
                "phenotype": {
                    "temporal_diagnostics": {
                        "periodic": {
                            "median_period_ms": (
                                50.0 if phase == "rising" else 55.0
                            ),
                            "source_phase_signature": {
                                "status": "ok",
                                "profile": np.roll(
                                    base_signature,
                                    2 if phase == "peak" else 0,
                                    axis=0,
                                ).tolist(),
                            },
                        }
                    }
                },
            })
    return rows


def _cell(seed, tier, trajectory, index, label, cell_id=None):
    return {
        "seed": seed,
        "tier": tier,
        "trajectory_id": trajectory,
        "path_index": index,
        "path_direction": "forward",
        "cell_id": cell_id or f"{tier}__{trajectory}__{index}",
        "slow_state_sha256": "a" * 64,
        "status": "complete",
        "cell_class": label,
        "gain_trigger_eligible": label == "spike_AI_screen_candidate",
        "run_rows": [
            {
                key: value for key, value in row.items()
                if key != "_hierarchical"
            }
            for row in _six_runs(
                "tonic_non_AI",
                spike_pass=label == "spike_AI_screen_candidate",
            )
        ],
        "spike_ai_screen_support": {
            "passes_locked_cell_gate": label == "spike_AI_screen_candidate",
            "k": 6 if label == "spike_AI_screen_candidate" else 0,
            "n": 6,
            "posterior_median": (
                0.95 if label == "spike_AI_screen_candidate" else 0.05
            ),
            "per_phase_pass_count": {
                phase: (
                    3 if label == "spike_AI_screen_candidate" else 0
                )
                for phase in A.PHASES
            },
        },
    }


def test_spike_ai_screen_is_the_c0_conjunction_without_gain():
    passed = A.spike_ai_screen(
        _hierarchical(ai=True), terminal_class="tonic_non_AI"
    )
    assert passed["pass"] is True
    assert A.spike_ai_screen(
        _hierarchical(ai=False), terminal_class="tonic_non_AI"
    )["pass"] is False
    assert A.spike_ai_screen(
        _hierarchical(ai=True),
        terminal_class="periodic_non_tonic_carrier",
    )["pass"] is False


def test_cell_gate_requires_five_of_six_and_two_per_phase():
    rows = _six_runs("periodic_non_tonic_carrier")
    rows[-1]["terminal_class"] = "tonic_non_AI"
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["cell_class"] == "periodic_non_tonic_carrier"
    support = out["terminal_support"]["periodic_non_tonic_carrier"]
    assert (support["k"], support["n"]) == (5, 6)
    assert support["posterior_median"] > 0.80
    assert support["per_phase_pass_count"] == {"rising": 3, "peak": 2}
    assert out["periodic_fast_phase_consistency"]["pass"] is True

    rows[-2]["terminal_class"] = "tonic_non_AI"
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["cell_class"] == "probabilistically_indeterminate"


def test_cell_ai_trigger_requires_hierarchical_uncertainty_gate(monkeypatch):
    monkeypatch.setattr(A.C0, "N_BOOT", 100)
    rows = _six_runs("tonic_non_AI", spike_pass=True)
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["cell_class"] == "spike_AI_screen_candidate"
    assert out["spike_ai_screen_support"]["hierarchical_ci_pass"] is True
    assert out["spike_ai_hierarchical_ci"]["n_boot"] == 100

    for row in rows:
        row["_hierarchical"] = _hierarchical(ai=False)
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["cell_class"] == "tonic_non_AI"
    assert out["spike_ai_screen_support"]["passes_locked_cell_gate"] is True
    assert out["spike_ai_screen_support"]["hierarchical_ci_pass"] is False


def test_periodic_window_fails_when_fast_phase_periods_disagree():
    rows = _six_runs("periodic_non_tonic_carrier")
    for row in rows:
        if row["phase"] == "peak":
            row["phenotype"]["temporal_diagnostics"]["periodic"][
                "median_period_ms"
            ] = 90.0
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["cell_class"] == "probabilistically_indeterminate"
    assert out["periodic_fast_phase_consistency"]["pass"] is False
    assert (
        out["terminal_support"]["periodic_non_tonic_carrier"][
            "phase_consistency_blocked"
        ]
        is True
    )


def test_periodic_window_fails_when_source_phase_structure_disagrees():
    rows = _six_runs("periodic_non_tonic_carrier")
    phase = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    orthogonal = np.column_stack([
        np.sin(2.0 * phase + offset)
        for offset in (0.0, 0.3, 0.8, 1.1)
    ])
    for row in rows:
        if row["phase"] == "peak":
            row["phenotype"]["temporal_diagnostics"]["periodic"][
                "source_phase_signature"
            ]["profile"] = orthogonal.tolist()
    out = A.aggregate_cell_rows(rows, _coordinate())
    consistency = out["periodic_fast_phase_consistency"]
    assert out["cell_class"] == "probabilistically_indeterminate"
    assert consistency["pass"] is False
    assert consistency["reason"] == "fast_phase_source_structure_differs"


def test_periodic_source_phase_threshold_is_decisive(monkeypatch):
    rows = _six_runs("periodic_non_tonic_carrier")
    baseline = A._periodic_phase_consistency(rows)
    similarity = baseline["source_phase_similarity_median"]
    assert baseline["pass"] is True
    monkeypatch.setattr(
        A, "PERIODIC_PHASE_STRUCTURE_CORR_MIN",
        min(1.01, similarity + 0.01),
    )
    assert A._periodic_phase_consistency(rows)["pass"] is False


def test_whole_sheet_early_stop_is_not_a_refractory_diagnosis():
    assert A._scientific_terminal("whole_sheet_plateau") == (
        "probabilistically_indeterminate"
    )


def test_ai_tonic_never_forms_a_maturation_window():
    cells = []
    for seed in A.SEEDS:
        for idx in range(3):
            label = (
                "spike_AI_screen_candidate" if idx in (0, 1)
                else "tonic_non_AI"
            )
            cells.append(_cell(
                seed, "primary_convex", "rising", idx, label
            ))
    out = A.adjudicate_tier(cells, "primary_convex")
    assert out["status"] == "no_window"


def test_window_requires_same_non_tonic_class_and_preserves_shell_semantics():
    primary = []
    shell = []
    for seed in A.SEEDS:
        primary.extend([
            _cell(
                seed, "primary_convex", "rising", 0,
                "periodic_non_tonic_carrier",
            ),
            _cell(
                seed, "primary_convex", "rising", 1,
                (
                    "periodic_non_tonic_carrier"
                    if seed in (1, 3) else "tonic_non_AI"
                ),
            ),
        ])
        shell.extend([
            _cell(
                seed, "secondary_shell", "shell_a", 0,
                "clonic_or_bursting_carrier",
            ),
            _cell(
                seed, "secondary_shell", "shell_a", 1,
                "clonic_or_bursting_carrier",
            ),
        ])
    # Two matching seeds cannot overrule a complete tonic phenotype in the
    # third seed.
    assert A.adjudicate_tier(
        primary, "primary_convex"
    )["status"] == "seed_heterogeneous_maturation"
    primary[-1]["cell_class"] = "periodic_non_tonic_carrier"
    assert A.adjudicate_tier(
        primary, "primary_convex"
    )["status"] == "local_maturation_window"
    assert A.adjudicate_tier(
        shell, "secondary_shell"
    )["status"] == "maturation_candidate_in_secondary_shell"

    # Adjacent cells with different temporal classes are not a window.
    mixed = [
        _cell(
            seed, "primary_convex", "rising", 0,
            "periodic_non_tonic_carrier",
        )
        for seed in A.SEEDS
    ] + [
        _cell(
            seed, "primary_convex", "rising", 1,
            "clonic_or_bursting_carrier",
        )
        for seed in A.SEEDS
    ]
    assert A.adjudicate_tier(
        mixed, "primary_convex"
    )["status"] == "isolated_maturation_candidate"


def _two_seed_primary_window(third_labels, *, third_direction="forward"):
    rows = []
    for seed in (1, 3):
        rows.extend([
            _cell(
                seed, "primary_convex", "rising", idx,
                "periodic_non_tonic_carrier",
            )
            for idx in (0, 1)
        ])
    for idx, label in enumerate(third_labels):
        row = _cell(
            4, "primary_convex", "rising", idx, label,
        )
        row["path_direction"] = third_direction
        rows.append(row)
    return rows


def test_primary_majority_allows_only_explicit_indeterminate_third_seed():
    rows = _two_seed_primary_window([
        "probabilistically_indeterminate",
        "probabilistically_indeterminate",
    ])
    for row in rows:
        if row["seed"] == 4:
            row["status"] = "indeterminate"
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "local_maturation_window"
    candidate = out["candidates"][0]
    assert candidate["supporting_seeds"] == [1, 3]
    assert candidate["third_seed_assessment"]["4"]["disposition"] == (
        "probabilistically_indeterminate"
    )


def test_primary_non_dt2_seed_pair_is_confirmation_unavailable():
    rows = []
    for seed in (1, 4):
        rows.extend([
            _cell(
                seed, "primary_convex", "rising", idx,
                "periodic_non_tonic_carrier",
            )
            for idx in (0, 1)
        ])
    for idx in (0, 1):
        row = _cell(
            3, "primary_convex", "rising", idx,
            "probabilistically_indeterminate",
        )
        row["status"] = "indeterminate"
        rows.append(row)
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "resolution_confirmation_unavailable"
    assert out["dt2_eligible_candidates"] == []
    assert out["candidates"][0]["supporting_seeds"] == [1, 4]
    assert out["candidates"][0]["dt2_eligible"] is False


def test_primary_majority_allows_concordant_but_incomplete_third_seed():
    rows = _two_seed_primary_window([
        "periodic_non_tonic_carrier",
        "probabilistically_indeterminate",
    ])
    rows[-1]["status"] = "indeterminate"
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "local_maturation_window"
    assert out["candidates"][0]["third_seed_assessment"]["4"][
        "disposition"
    ] == "probabilistically_indeterminate"


def test_primary_majority_does_not_hide_tonic_cell_behind_concordant_cell():
    rows = _two_seed_primary_window([
        "periodic_non_tonic_carrier",
        "tonic_non_AI",
    ])
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


def _ten_cell_primary_fixture(
    third_pair,
    *,
    third_pair_direction="forward",
    reverse_window=False,
):
    rows = []
    for seed in A.SEEDS:
        for trajectory in ("rising", "peak"):
            for index in range(5):
                label = "tonic_non_AI"
                if trajectory == "rising" and index in (0, 1):
                    if seed in (1, 3):
                        label = "periodic_non_tonic_carrier"
                    else:
                        label = third_pair[index]
                row = _cell(
                    seed, "primary_convex", trajectory, index, label,
                )
                if (
                    seed == 4 and trajectory == "rising"
                    and index in (0, 1)
                ):
                    row["path_direction"] = third_pair_direction
                    if label == "probabilistically_indeterminate":
                        row["status"] = "indeterminate"
                if (
                    reverse_window and seed == 4
                    and trajectory == "peak" and index in (0, 1)
                ):
                    row["cell_class"] = "periodic_non_tonic_carrier"
                    row["path_direction"] = "reverse"
                rows.append(row)
    return rows


def test_primary_real_ten_cell_indeterminate_pair_ignores_unrelated_tonic():
    rows = _ten_cell_primary_fixture((
        "probabilistically_indeterminate",
        "probabilistically_indeterminate",
    ))
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "local_maturation_window"
    assert out["candidates"][0]["supporting_seeds"] == [1, 3]
    assessment = out["candidates"][0]["third_seed_assessment"]["4"]
    assert assessment["disposition"] == "probabilistically_indeterminate"
    assert assessment["homologous_cells"] == [
        "primary_convex__rising__0",
        "primary_convex__rising__1",
    ]


def test_primary_real_ten_cell_corresponding_tonic_pair_blocks():
    out = A.adjudicate_tier(
        _ten_cell_primary_fixture(("tonic_non_AI", "tonic_non_AI")),
        "primary_convex",
    )
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


def test_primary_real_ten_cell_different_phenotype_pair_blocks():
    out = A.adjudicate_tier(
        _ten_cell_primary_fixture((
            "clonic_or_bursting_carrier",
            "clonic_or_bursting_carrier",
        )),
        "primary_convex",
    )
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


def test_primary_real_ten_cell_reverse_window_blocks():
    out = A.adjudicate_tier(
        _ten_cell_primary_fixture((
            "probabilistically_indeterminate",
            "probabilistically_indeterminate",
        ), reverse_window=True),
        "primary_convex",
    )
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


def test_primary_real_ten_cell_three_of_three_passes():
    out = A.adjudicate_tier(
        _ten_cell_primary_fixture((
            "periodic_non_tonic_carrier",
            "periodic_non_tonic_carrier",
        )),
        "primary_convex",
    )
    assert out["status"] == "local_maturation_window"
    assert out["candidates"][0]["supporting_seeds"] == [1, 3, 4]


def test_primary_shifted_windows_across_dt2_seeds_are_not_homologous():
    rows = _ten_cell_primary_fixture((
        "probabilistically_indeterminate",
        "probabilistically_indeterminate",
    ))
    for row in rows:
        if (
            row["seed"] == 3
            and row["trajectory_id"] == "rising"
        ):
            row["cell_class"] = (
                "periodic_non_tonic_carrier"
                if row["path_index"] in (3, 4) else "tonic_non_AI"
            )
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


@pytest.mark.parametrize(
    "third_labels,third_direction",
    [
        (["periodic_non_tonic_carrier", "clonic_or_bursting_carrier"], "forward"),
        (["periodic_non_tonic_carrier", "tonic_non_AI"], "forward"),
        (["periodic_non_tonic_carrier", "refractory_saturated"], "forward"),
        (["periodic_non_tonic_carrier", "runaway"], "forward"),
        (["periodic_non_tonic_carrier", "periodic_non_tonic_carrier"], "reverse"),
    ],
)
def test_primary_majority_rejects_contradictory_complete_third_seed(
    third_labels, third_direction,
):
    out = A.adjudicate_tier(
        _two_seed_primary_window(
            third_labels, third_direction=third_direction,
        ),
        "primary_convex",
    )
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


def test_primary_majority_rejects_non_explicit_indeterminate_third_seed():
    rows = _two_seed_primary_window([
        "probabilistically_indeterminate",
        "tonic_gain_indeterminate",
    ])
    for row in rows:
        if row["seed"] == 4:
            row["status"] = "indeterminate"
    out = A.adjudicate_tier(rows, "primary_convex")
    assert out["status"] == "seed_heterogeneous_maturation"
    assert out["candidates"] == []


def test_spatial_relay_modifier_requires_a_reproducible_direction():
    rows = _six_runs("periodic_non_tonic_carrier")
    for index, row in enumerate(rows):
        row["spatial_relay"] = {
            "is_spatial_relay": True,
            "direction_sign": 1 if index < 5 else -1,
        }
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["spatial_relay_modifier"]["supported"] is True
    assert out["spatial_relay_modifier"]["support"]["direction_sign"] == 1

    rows[4]["spatial_relay"]["direction_sign"] = -1
    out = A.aggregate_cell_rows(rows, _coordinate())
    assert out["spatial_relay_modifier"]["supported"] is False


def test_real_shell_cells_replicate_same_cell_across_seeds_without_adjacency():
    cell_id = N.SHELL_CELL_NAMES[0]
    rows = [
        _cell(
            seed, "secondary_shell", "secondary_shell", 0,
            "periodic_non_tonic_carrier", cell_id,
        )
        for seed in (1, 3)
    ]
    third = _cell(
        4, "secondary_shell", "secondary_shell", 0,
        "probabilistically_indeterminate", cell_id,
    )
    third["status"] = "indeterminate"
    rows.append(third)
    out = A.adjudicate_tier(rows, "secondary_shell")
    assert out["status"] == "maturation_candidate_in_secondary_shell"
    assert out["candidates"][0]["cell_id"] == cell_id
    assert out["candidates"][0]["supporting_seeds"] == [1, 3]
    assert out["primary_reachability_established"] is False


def test_shell_single_seed_or_different_cells_do_not_replicate():
    cell_a, cell_b = N.SHELL_CELL_NAMES[:2]
    rows = [
        _cell(
            1, "secondary_shell", "secondary_shell", 0,
            "clonic_or_bursting_carrier", cell_a,
        ),
        _cell(
            3, "secondary_shell", "secondary_shell", 0,
            "clonic_or_bursting_carrier", cell_b,
        ),
    ]
    for seed in (4,):
        for cell_id in (cell_a, cell_b):
            row = _cell(
                seed, "secondary_shell", "secondary_shell", 0,
                "probabilistically_indeterminate", cell_id,
            )
            row["status"] = "indeterminate"
            rows.append(row)
    out = A.adjudicate_tier(rows, "secondary_shell")
    assert out["status"] == "isolated_maturation_candidate"
    assert out["candidates"] == []


def test_complete_negative_is_fail_closed_by_conditional_gain():
    cells = []
    for seed in A.SEEDS:
        for idx, cell_id in enumerate(N.PRIMARY_CELL_NAMES):
            row = _cell(
                seed, "primary_convex",
                "rising" if idx < 5 else "peak", idx % 5,
                "tonic_non_AI", cell_id,
            )
            row["conditional_gain"] = {"status": "not_triggered"}
            cells.append(row)
    ok, reason = A._strict_negative(cells, tier="primary_convex")
    assert ok is True
    assert reason == "complete_bounded_negative"

    cells[0]["cell_class"] = "spike_AI_screen_candidate"
    cells[0]["conditional_gain"] = {
        "status": "C1_blocked_conditional_gain"
    }
    ok, reason = A._strict_negative(cells, tier="primary_convex")
    assert ok is False
    assert reason == "conditional_gain_not_terminal_resolved"

    cells[0]["status"] = "indeterminate"
    cells[0]["cell_class"] = "tonic_gain_indeterminate"
    cells[0]["conditional_gain"] = {"status": "scientific_indeterminate"}
    ok, _ = A._strict_negative(cells, tier="primary_convex")
    assert ok is False


def test_empty_trigger_manifest_must_be_locked_before_negative(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    base_path = tmp_path / "base_atlas.json"
    base_path.write_text("{}\n")
    base = {
        "phasec_manifest_sha256": "a" * 64,
        "phasec_manifest_file_sha256": "b" * 64,
        "coordinate_manifest_sha256": "c" * 64,
        "coordinate_manifest_semantic_sha256": "e" * 64,
        "coordinate_manifest_file_sha256": "d" * 64,
        "resolution": "dt",
        "matrix": {"complete": True},
        "cells": [],
        "primary_base_adjudication": {
            "tier": "primary_convex", "status": "no_window",
            "candidates": [], "seed_results": {},
        },
        "secondary_shell_base_adjudication": {
            "tier": "secondary_shell", "status": "no_window",
            "candidates": [], "seed_results": {},
        },
        "claim_boundary": "test",
    }
    out = A.apply_conditional_gain(
        base,
        base_atlas_path=base_path,
        trigger_manifest_path=tmp_path / "missing_trigger.json",
    )
    assert out["verdict"] == "C1_gain_trigger_not_locked"
    assert out["reason"] == "write_once_trigger_decision_missing"


def test_base_part_requires_exact_runtime_producer_map(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    path = tmp_path / "part.json"
    phasec = {
        "manifest_sha256": "c" * 64,
        "provenance": {"producer_file_sha256": {"runner.py": "d" * 64}},
        "per_seed": {
            "1": {
                "canonical_config_sha": "f" * 64,
                "c0_carrier_states": {
                    "rising": {
                        "state": {
                            "state_hash": "2" * 64,
                            "file_sha256": "3" * 64,
                        },
                        "noise_banks": [{
                            "replicate": "noise_replay",
                            "bank_sha": "1" * 64,
                        }],
                    }
                },
            }
        },
    }
    coord = {
        "manifest_sha256": "e" * 64,
        "semantic_sha256": "9" * 64,
        "producer_file_sha256": {"builder.py": "7" * 64},
        "seeds": {
            "1": {
                "config_sha": "f" * 64,
                "npz_file_sha256": "8" * 64,
                "npz_semantic_sha256": "6" * 64,
            }
        },
    }
    coordinate = _coordinate()
    payload = {
        "schema": A.C1_BASE_PART_SCHEMA,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "phasec_manifest_file_sha256": "5" * 64,
        "coordinate_manifest_sha256": coord["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coord["semantic_sha256"],
        "coordinate_manifest_file_sha256": "4" * 64,
        "seed": 1,
        "cell_id": coordinate["cell_id"],
        "tier": coordinate["tier"],
        "trajectory_id": coordinate["trajectory_id"],
        "path_index": coordinate["path_index"],
        "path_direction": coordinate["path_direction"],
        "phase": "rising",
        "noise": "noise_replay",
        "resolution": "dt",
        "slow_state_sha256": coordinate["state_sha256"],
        "coordinate_npz_file_sha256": "8" * 64,
        "coordinate_npz_semantic_sha256": "6" * 64,
        "config_sha": "f" * 64,
        "fast_base_state_hash": "2" * 64,
        "state_file_sha256": "3" * 64,
        "noise_bank_sha": "1" * 64,
        "burn_in_ms": 500.0,
        "measure_ms": 8000.0,
        "status": "scientific_failure",
        "scientific_end_reason": "runaway",
        "runtime_provenance": {
            "manifest_sha256": phasec["manifest_sha256"],
            "manifest_file_sha256": "5" * 64,
            "producer_sha256": {"runner.py": "x" * 64},
            "state_file_sha256": "3" * 64,
            "noise_bank_sha": "1" * 64,
            "coordinate_manifest_sha256": coord["manifest_sha256"],
            "coordinate_manifest_semantic_sha256": coord[
                "semantic_sha256"
            ],
            "coordinate_manifest_file_sha256": "4" * 64,
            "coordinate_npz_file_sha256": "8" * 64,
            "coordinate_npz_semantic_sha256": "6" * 64,
            "coordinate_producer_sha256": coord[
                "producer_file_sha256"
            ],
        },
    }
    path.write_text(json.dumps(payload))
    blocked = A.classify_base_part(
        path,
        coordinate=coordinate,
        coordinate_manifest=coord,
        phasec_manifest=phasec,
        panels={},
        seed=1,
        phase="rising",
        noise="noise_replay",
        resolution="dt",
        phasec_manifest_file_sha256="5" * 64,
        coordinate_manifest_file_sha256="4" * 64,
        coordinate_ref={
            "file_sha256": "4" * 64,
            "manifest_sha256": coord["manifest_sha256"],
            "semantic_sha256": coord["semantic_sha256"],
        },
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "runtime_provenance_mismatch:producer_sha256"

    payload["runtime_provenance"]["producer_sha256"] = (
        phasec["provenance"]["producer_file_sha256"]
    )
    payload["fast_base_state_hash"] = "x" * 64
    path.write_text(json.dumps(payload))
    fast_blocked = A.classify_base_part(
        path,
        coordinate=coordinate,
        coordinate_manifest=coord,
        phasec_manifest=phasec,
        panels={},
        seed=1,
        phase="rising",
        noise="noise_replay",
        resolution="dt",
        phasec_manifest_file_sha256="5" * 64,
        coordinate_manifest_file_sha256="4" * 64,
        coordinate_ref={
            "file_sha256": "4" * 64,
            "manifest_sha256": coord["manifest_sha256"],
            "semantic_sha256": coord["semantic_sha256"],
        },
    )
    assert fast_blocked["reason"] == (
        "base_part_field_mismatch:fast_base_state_hash"
    )
    payload["fast_base_state_hash"] = "2" * 64
    path.write_text(json.dumps(payload))
    receipt_blocked = A.classify_base_part(
        path,
        coordinate=coordinate,
        coordinate_manifest=coord,
        phasec_manifest=phasec,
        panels={},
        seed=1,
        phase="rising",
        noise="noise_replay",
        resolution="dt",
        phasec_manifest_file_sha256="5" * 64,
        coordinate_manifest_file_sha256="4" * 64,
        coordinate_ref={
            "file_sha256": "4" * 64,
            "manifest_sha256": coord["manifest_sha256"],
            "semantic_sha256": coord["semantic_sha256"],
        },
    )
    assert receipt_blocked["status"] == "blocked"
    assert receipt_blocked["reason"] == "missing_resource_audit_receipt"

    monkeypatch.setattr(
        A, "_resource_receipt_failure",
        lambda _path, *, manifest_sha256, task_key: None,
    )
    accepted = A.classify_base_part(
        path,
        coordinate=coordinate,
        coordinate_manifest=coord,
        phasec_manifest=phasec,
        panels={},
        seed=1,
        phase="rising",
        noise="noise_replay",
        resolution="dt",
        phasec_manifest_file_sha256="5" * 64,
        coordinate_manifest_file_sha256="4" * 64,
        coordinate_ref={
            "file_sha256": "4" * 64,
            "manifest_sha256": coord["manifest_sha256"],
            "semantic_sha256": coord["semantic_sha256"],
        },
    )
    assert accepted["status"] == "complete"
    assert accepted["terminal_class"] == "runaway"


def _base_atlas_with_trigger(tmp_path):
    cells = []
    for seed in A.SEEDS:
        for idx, cell_id in enumerate(N.PRIMARY_CELL_NAMES):
            label = (
                "spike_AI_screen_candidate"
                if seed == 1 and cell_id == "primary__rising__bounded_mid"
                else "tonic_non_AI"
            )
            cells.append(_cell(
                seed,
                "primary_convex",
                "rising" if idx < 5 else "peak",
                idx % 5,
                label,
                cell_id,
            ))
        for idx, cell_id in enumerate(N.SHELL_CELL_NAMES):
            cells.append(_cell(
                seed,
                "secondary_shell",
                f"shell_{idx // 2}",
                idx,
                "tonic_non_AI",
                cell_id,
            ))
    base = {
        "schema": A.C1_BASE_ATLAS_SCHEMA,
        "phasec_manifest_sha256": "c" * 64,
        "phasec_manifest_file_sha256": "f" * 64,
        "coordinate_manifest_sha256": "d" * 64,
        "coordinate_manifest_semantic_sha256": "8" * 64,
        "coordinate_manifest_file_sha256": "9" * 64,
        "phasec_producer_file_sha256": {"runner.py": "1" * 64},
        "coordinate_producer_file_sha256": {"builder.py": "2" * 64},
        "coordinate_npz_provenance_by_seed": {
            str(seed): {
                "coordinate_npz_file_sha256": "6" * 64,
                "coordinate_npz_semantic_sha256": "7" * 64,
            }
            for seed in A.SEEDS
        },
        "resolution": "dt",
        "matrix": {"complete": True},
        "cells": cells,
        "primary_base_adjudication": {
            "status": "no_window",
            "tier": "primary_convex",
        },
        "secondary_shell_base_adjudication": {
            "status": "no_window",
            "tier": "secondary_shell",
        },
        "claim_boundary": "test",
    }
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base, sort_keys=True))
    return base, path


def _denominators(_resolution, _seed):
    rows = []
    for noise in A.NOISES:
        for delta in L.DELTAS_MV:
            path = Path(L.ROOT) / f"c0/{noise}/{delta}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
            "schema": "zm_phasec_gain_cell_v1",
            "phasec_manifest_sha256": "c" * 64,
            "phasec_manifest_file_sha256": "f" * 64,
            "phasec_producer_file_sha256": {"runner.py": "1" * 64},
            "resolution": _resolution,
            "seed": int(_seed),
            "state_tag": "pre_entry__natural",
            "noise": noise,
            "replicate": noise,
            "delta_mV": float(delta),
            "threshold_offset_mV": float(delta),
            "signed_delta_abs_mV": abs(float(delta)),
            "sign": int((delta > 0) - (delta < 0)),
            "burn_in_ms": 500.0,
            "measure_ms": 1000.0,
            "config_sha": "1" * 64,
            "fast_base_state_hash": "2" * 64,
            "state_file_sha256": "3" * 64,
            "noise_bank_sha": "4" * 64,
            "path": str(path.relative_to(L.ROOT)),
            }
            task_key = A.C0._gain_task_key(
                int(_seed),
                row["state_tag"],
                row["replicate"],
                row["signed_delta_abs_mV"],
                row["sign"],
            )
            _write_receipted_part(
                path,
                {"noise": noise, "delta": float(delta)},
                manifest_sha256=row["phasec_manifest_sha256"],
                task_key=task_key,
            )
            row["file_sha256"] = A._sha256(path)
            rows.append(row)
    return rows


def test_trigger_manifest_is_write_once_and_contains_30_arms(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "OUT", tmp_path / "results")
    monkeypatch.setattr(L, "ROOT", tmp_path)
    base, path = _base_atlas_with_trigger(tmp_path)

    manifest = L.build_trigger_manifest(
        base, base_atlas_path=path, denominator_provider=_denominators
    )
    assert manifest["n_triggered_cells"] == 1
    assert len(
        manifest["triggered_cells"][0]["expected_carrier_gain_arms"]
    ) == 30
    arm = manifest["triggered_cells"][0][
        "expected_carrier_gain_arms"
    ][0]
    assert arm["fast_base_state_hash"] == "2" * 64
    assert arm["state_file_sha256"] == "3" * 64
    assert arm["noise_bank_sha"] == "4" * 64
    assert arm["coordinate_npz_file_sha256"] == "6" * 64
    assert arm["coordinate_npz_semantic_sha256"] == "7" * 64
    output = tmp_path / "trigger.json"
    assert N.write_json_once(output, manifest) == "created"
    assert N.write_json_once(output, manifest) == "reused"
    changed = dict(manifest)
    changed["n_triggered_cells"] = 2
    try:
        N.write_json_once(output, changed)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a write-once trigger manifest must reject drift")


def test_missing_conditional_gain_blocks_negative_but_not_nontonic_positive(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "OUT", tmp_path / "results")
    monkeypatch.setattr(L, "ROOT", tmp_path)
    base, path = _base_atlas_with_trigger(tmp_path)

    trigger = L.build_trigger_manifest(
        base, base_atlas_path=path, denominator_provider=_denominators
    )
    trigger_path = tmp_path / "trigger.json"
    N.write_json_once(trigger_path, trigger)
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=trigger_path
    )
    assert out["verdict"] == "C1_blocked_conditional_gain"

    base["primary_base_adjudication"] = {
        "status": "local_maturation_window",
        "tier": "primary_convex",
    }
    path.write_text(json.dumps(base, sort_keys=True))
    trigger = L.build_trigger_manifest(
        base, base_atlas_path=path, denominator_provider=_denominators
    )
    trigger_path.unlink()
    N.write_json_once(trigger_path, trigger)
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=trigger_path
    )
    assert out["verdict"] == "primary_maturation_candidate_requires_dt2"
    base["matrix"]["complete"] = False
    path.write_text(json.dumps(base, sort_keys=True))
    # The old trigger no longer matches the mutated base, so no trigger is
    # supplied; a scientific window still cannot rescue technical coverage.
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=tmp_path / "absent.json"
    )
    assert out["verdict"] == "C1_blocked_manifest"


def test_scientific_gain_unresolved_is_not_coerced_to_zero_or_technical(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "OUT", tmp_path / "results")
    monkeypatch.setattr(L, "ROOT", tmp_path)
    base, path = _base_atlas_with_trigger(tmp_path)
    trigger = L.build_trigger_manifest(
        base, base_atlas_path=path, denominator_provider=_denominators
    )
    trigger_path = tmp_path / "trigger.json"
    N.write_json_once(trigger_path, trigger)
    trigger_cell = trigger["triggered_cells"][0]
    gain_path = A.gain_status_path(
        "dt",
        trigger_cell["seed"],
        trigger_cell["tier"],
        trigger_cell["cell_id"],
    )
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    carrier_hashes = {}
    resource_tasks = []
    for arm in trigger_cell["expected_carrier_gain_arms"]:
        arm_path = tmp_path / arm["path"]
        arm_path.parent.mkdir(parents=True, exist_ok=True)
        task_key = (
            f"gain|s{arm['seed']}|{arm['tier']}|{arm['cell_id']}|"
            f"{arm['phase']}|{arm['noise']}|"
            f"{float(arm['delta_mV']):+g}"
        )
        _write_receipted_part(
            arm_path,
            {"path": arm["path"], "delta": arm["delta_mV"]},
            manifest_sha256=trigger["phasec_manifest_sha256"],
            task_key=task_key,
        )
        carrier_hashes[arm["path"]] = A._sha256(arm_path)
        resource_tasks.append((
            task_key,
            arm_path,
            (
                f"c1_gain_numerator|s{trigger_cell['seed']}|"
                f"{trigger_cell['tier']}|{trigger_cell['cell_id']}"
            ),
        ))
    for ref in trigger_cell["reused_c0_preentry_denominators"]:
        resource_tasks.append((
            A.C0._gain_task_key(
                int(ref["seed"]),
                ref["state_tag"],
                ref["replicate"],
                float(ref["signed_delta_abs_mV"]),
                int(ref["sign"]),
            ),
            tmp_path / ref["path"],
            (
                f"c1_gain_preentry_denominator|s{trigger_cell['seed']}|"
                f"{trigger_cell['tier']}|{trigger_cell['cell_id']}"
            ),
        ))
    payload = {
        "schema": A.C1_GAIN_STATUS_SCHEMA,
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "trigger_manifest_file_sha256": A._sha256(trigger_path),
        "phasec_manifest_sha256": trigger["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": trigger[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        "resolution": "dt",
        "seed": trigger_cell["seed"],
        "tier": trigger_cell["tier"],
        "cell_id": trigger_cell["cell_id"],
        "slow_state_sha256": trigger_cell["slow_state_sha256"],
        "phasec_producer_file_sha256": trigger[
            "phasec_producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": trigger[
            "coordinate_producer_file_sha256"
        ],
        "trigger_producer_file_sha256": trigger[
            "producer_file_sha256"
        ],
        "gain_class": "tonic_gain_indeterminate",
        "completed_arm_file_sha256": carrier_hashes,
        "reused_c0_preentry_denominator_sha256": {
            row["path"]: row["file_sha256"]
            for row in trigger_cell["reused_c0_preentry_denominators"]
        },
        "resource_receipt_index": A.build_resource_receipt_index(
            resource_tasks,
            manifest_sha256=trigger["phasec_manifest_sha256"],
        ),
    }
    assert payload["resource_receipt_index"]["expected_task_count"] == 45
    assert (
        payload["resource_receipt_index"][
            "expected_logical_consumption_count"
        ]
        == 45
    )
    roles = [
        row["role"]
        for row in payload["resource_receipt_index"][
            "logical_consumptions"
        ]
    ]
    assert sum("numerator" in role for role in roles) == 30
    assert sum("denominator" in role for role in roles) == 15
    gain_path.write_text(json.dumps(payload))
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=trigger_path
    )
    triggered = next(
        row for row in out["cells"] if row["gain_trigger_eligible"]
    )
    assert triggered["conditional_gain"]["status"] == "scientific_indeterminate"
    assert out["verdict"] == "C1_incomplete_or_indeterminate"

    payload["gain_class"] = "balanced_AI_tonic_cell"
    gain_path.write_text(json.dumps(payload))
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=trigger_path
    )
    triggered = next(
        row for row in out["cells"] if row["gain_trigger_eligible"]
    )
    assert triggered["cell_class"] == "balanced_AI_tonic_cell"
    assert out["verdict"] == "no_maturation_in_tested_primary_neighbourhood"

    # The final consumer must not trust a previously written status map.
    first_arm = trigger_cell["expected_carrier_gain_arms"][0]["path"]
    first_arm_path = tmp_path / first_arm
    original_arm = first_arm_path.read_bytes()
    first_arm_path.write_text("tampered\n")
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=trigger_path
    )
    triggered = next(
        row for row in out["cells"] if row["gain_trigger_eligible"]
    )
    assert (
        triggered["conditional_gain"]["reason"]
        == "conditional_gain_arm_file_hash_drift"
    )

    first_arm_path.write_bytes(original_arm)
    first_denominator = trigger_cell[
        "reused_c0_preentry_denominators"
    ][0]["path"]
    (tmp_path / first_denominator).write_text("tampered denominator\n")
    out = A.apply_conditional_gain(
        base, base_atlas_path=path, trigger_manifest_path=trigger_path
    )
    triggered = next(
        row for row in out["cells"] if row["gain_trigger_eligible"]
    )
    assert (
        triggered["conditional_gain"]["reason"]
        == "conditional_gain_denominator_file_hash_drift"
    )


def test_resolution_gate_requires_same_homologous_window_in_two_seeds():
    windows = {
        str(seed): {
            "windows": [{
                "phenotype": "periodic_non_tonic_carrier",
                "direction": "forward",
                "cells": ["c1", "c2"],
            }],
        }
        for seed in A.SEEDS
    }
    adjudication = {
        "status": "local_maturation_window",
        "candidates": [{
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "homologous_cells": ["c1", "c2"],
            "supporting_seeds": [1, 3, 4],
        }],
        "seed_results": windows,
    }
    common = {
        "schema": A.C1_SUMMARY_SCHEMA,
        "phasec_manifest_sha256": "a" * 64,
        "coordinate_manifest_sha256": "b" * 64,
        "primary_adjudication": adjudication,
        "secondary_shell_adjudication": {"status": "no_window"},
        "claim_boundary": "test",
    }
    native = {
        **common,
        "resolution": "dt",
        "verdict": "primary_maturation_candidate_requires_dt2",
    }
    assert A.combine_resolution_summaries(
        native, None
    )["verdict"] == "C1_window_pending_dt2"
    unavailable_native = json.loads(json.dumps(native))
    unavailable_native["primary_adjudication"]["candidates"][0][
        "supporting_seeds"
    ] = [1, 4]
    unavailable = A.combine_resolution_summaries(
        unavailable_native, None
    )
    assert unavailable["verdict"] == "C1_blocked_resolution_gate"
    assert unavailable["reason"] == "resolution_confirmation_unavailable"
    assert unavailable["resolution_gate"] == (
        "insufficient_homologous_native_support"
    )
    assert unavailable["primary_gate"]["status"] == "blocked"
    assert unavailable["shell_gate"]["status"] == "not_required"
    dt2 = {
        "schema": A.C1_DT2_CONFIRMATION_SUMMARY_SCHEMA,
        "resolution": "dt2_confirmation_only",
        "phasec_manifest_sha256": common["phasec_manifest_sha256"],
        "verdict": "maturation_window_at_primary_convex_states",
        "matches": [{
            "tier": "primary_convex",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "homologous_cells": ["c1", "c2"],
            "homologous_supporting_seeds": [1, 3],
        }],
    }
    assert A.combine_resolution_summaries(
        native, dt2
    )["primary_gate"]["status"] == "confirmed"

    dt2["matches"] = []
    dt2["verdict"] = "C1_window_pending_dt2"
    indeterminate = A.combine_resolution_summaries(native, dt2)
    assert indeterminate["verdict"] == "C1_window_pending_dt2"
    assert indeterminate["resolution_gate"] == (
        "scientifically_indeterminate_dt2_confirmation"
    )

    dt2["matches"] = []
    dt2["verdict"] = "resolution_sensitive_maturation"
    dt2["label_assessments"] = [{
        "tier": "primary_convex",
        "phenotype": "periodic_non_tonic_carrier",
        "direction": "forward",
        "closure": "terminal_contradiction",
    }]
    assert A.combine_resolution_summaries(
        native, dt2
    )["verdict"] == "resolution_sensitive_maturation"
    assert A.combine_resolution_summaries(
        native, dt2
    )["primary_gate"]["status"] == "contradicted"


def test_primary_confirmation_cannot_close_indeterminate_shell_gate():
    def adjudication(status, tier):
        direction = "forward" if tier == "primary_convex" else "shell_a"
        return {
            "status": status,
            "candidates": [{
                "phenotype": "periodic_non_tonic_carrier",
                "direction": direction,
                "homologous_cells": (
                    ["c1", "c2"]
                    if tier == "primary_convex" else [direction]
                ),
                "supporting_seeds": [1, 3],
            }],
        }

    native = {
        "schema": A.C1_SUMMARY_SCHEMA,
        "phasec_manifest_sha256": "a" * 64,
        "phasec_manifest_file_sha256": "b" * 64,
        "primary_adjudication": adjudication(
            A.PRIMARY_POSITIVE_STATUS, "primary_convex"
        ),
        "secondary_shell_adjudication": adjudication(
            A.SHELL_POSITIVE_STATUS, "secondary_shell"
        ),
        "verdict": "primary_maturation_candidate_requires_dt2",
        "claim_boundary": "test",
    }
    dt2 = {
        "schema": A.C1_DT2_CONFIRMATION_SUMMARY_SCHEMA,
        "resolution": "dt2_confirmation_only",
        "phasec_manifest_sha256": "a" * 64,
        "phasec_manifest_file_sha256": "b" * 64,
        "matches": [{
            "tier": "primary_convex",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "homologous_cells": ["c1", "c2"],
            "homologous_supporting_seeds": [1, 3],
        }],
        "label_assessments": [{
            "tier": "secondary_shell",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "shell_a",
            "closure": "scientific_indeterminate",
        }],
        "technical_blockers": [],
    }
    out = A.combine_resolution_summaries(native, dt2)
    assert out["primary_gate"]["status"] == "confirmed"
    assert out["shell_gate"]["status"] == "indeterminate"
    assert out["verdict"] == "maturation_window_at_primary_convex_states"


def test_dt2_multi_candidate_closure_keeps_unresolved_label_pending():
    rows = [
        {
            "tier": "secondary_shell",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "shell_a",
            "cells": ["shell_a"],
            "assessment": "terminal_contradiction",
        },
        {
            "tier": "secondary_shell",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "shell_b",
            "cells": ["shell_b"],
            "assessment": "scientific_indeterminate",
        },
    ]
    verdict, labels = A._close_dt2_candidate_labels(rows, [], [])
    assert verdict == "C1_window_pending_dt2"
    assert {row["closure"] for row in labels} == {
        "terminal_contradiction", "scientific_indeterminate"
    }

    rows[1]["assessment"] = "terminal_contradiction"
    verdict, labels = A._close_dt2_candidate_labels(rows, [], [])
    assert verdict == "resolution_sensitive_maturation"
    assert all(row["closure"] == "terminal_contradiction" for row in labels)


def test_dt2_shifted_primary_windows_do_not_match():
    confirmed = [
        {
            "tier": "primary_convex",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "seed": 1,
            "cells": ["c0", "c1"],
        },
        {
            "tier": "primary_convex",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "seed": 3,
            "cells": ["c3", "c4"],
        },
    ]
    assert A._homologous_window_matches(confirmed) == []


def test_dt2_same_label_contradiction_plus_indeterminate_window_is_pending():
    rows = [
        {
            "tier": "primary_convex",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "cells": ["c0", "c1"],
            "assessment": "terminal_contradiction",
        },
        {
            "tier": "primary_convex",
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "forward",
            "cells": ["c2", "c3"],
            "assessment": "scientific_indeterminate",
        },
    ]
    verdict, labels = A._close_dt2_candidate_labels(rows, [], [])
    assert verdict == "C1_window_pending_dt2"
    assert labels[0]["closure"] == "scientific_indeterminate"
    assert {
        tuple(row["homologous_cells"]): row["closure"]
        for row in labels[0]["homologous_windows"]
    } == {
        ("c0", "c1"): "terminal_contradiction",
        ("c2", "c3"): "scientific_indeterminate",
    }


def test_dt2_selection_uses_only_two_native_supporting_homologous_seeds():
    native = {
        "primary_adjudication": {
            "status": "local_maturation_window",
            "candidates": [{
                "phenotype": "periodic_non_tonic_carrier",
                "direction": "forward",
            }],
            "seed_results": {
                str(seed): {
                    "windows": [{
                        "phenotype": "periodic_non_tonic_carrier",
                        "direction": "forward",
                        "cells": ["c1", "c2"],
                    }]
                }
                for seed in A.SEEDS
            },
        },
        "secondary_shell_adjudication": {"status": "no_window"},
    }
    selected = A._native_positive_windows(native)
    assert {row["seed"] for row in selected} == {1, 3}
    assert all(row["cells"] == ["c1", "c2"] for row in selected)

    native["primary_adjudication"]["seed_results"]["3"]["windows"] = []
    assert A._native_positive_windows(native) == []


def test_dt2_selection_preserves_single_cell_secondary_shell_semantics():
    cell_id = N.SHELL_CELL_NAMES[0]
    native = {
        "primary_adjudication": {"status": "no_window"},
        "secondary_shell_adjudication": {
            "status": "maturation_candidate_in_secondary_shell",
            "candidates": [{
                "phenotype": "clonic_or_bursting_carrier",
                "direction": cell_id,
                "cell_id": cell_id,
            }],
            "seed_results": {
                str(seed): {
                    "windows": [{
                        "phenotype": "clonic_or_bursting_carrier",
                        "direction": cell_id,
                        "cells": [cell_id],
                    }]
                }
                for seed in (1, 3)
            },
        },
    }
    selected = A._native_positive_windows(native)
    assert {row["seed"] for row in selected} == {1, 3}
    assert all(row["tier"] == "secondary_shell" for row in selected)
    assert all(row["cells"] == [cell_id] for row in selected)


def test_dt2_builder_lock_analyzer_round_trip(monkeypatch, tmp_path):
    """The legacy analyzer entry point must only delegate to the dedicated lock."""
    expected = {"manifest_sha256": "a" * 64}
    seen = {}

    def build_payload(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(A.DT2LOCK, "build_payload", build_payload)
    phasec_path = tmp_path / "phasec.json"
    native_path = tmp_path / "native.json"
    trigger_path = tmp_path / "trigger.json"
    assert A.build_dt2_confirmation_manifest(
        native_summary_path=native_path,
        phasec_manifest_path=phasec_path,
        gain_trigger_path=trigger_path,
    ) is expected
    assert seen == {
        "phasec_path": phasec_path,
        "native_summary_path": native_path,
        "gain_trigger_path": trigger_path,
    }
