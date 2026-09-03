from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import compute_topic5_signed_broadband_similarity as compute
from scripts.paper_figures import plot_fig3_peri_onset_field_similarity as render
from scripts.paper_figures import run_fig3_peri_onset_all_subjects as batch
from scripts import run_topic5_fig3b_maxab_spatial_null as spatial_null


def _frozen_record(
    ds_sid: str = "epilepsiae_demo",
    *,
    geometry_2d: bool = True,
    model_names: tuple[str, ...] = ("own_a", "own_b", "shared_a", "shared_b"),
) -> dict:
    dataset, subject = ds_sid.split("_", 1)
    rank = 2 if geometry_2d else 1
    shafts = 2 if geometry_2d else 1
    return {
        "dataset": dataset,
        "subject": subject,
        "contract": "topic5_interictal_template_fields_v1",
        "axis_definition": "template_propagation_axis_v2",
        "axis_direction_convention": "positive_early_to_late",
        "axis_pair": {
            "geometry_2d_supported": geometry_2d,
            "strict_stability_pass": True,
            "axis_a": {"effective_rank": rank, "n_shafts": shafts},
            "axis_b": {"effective_rank": rank, "n_shafts": shafts},
        },
        "interictal_field": {
            "status": "ok",
            "contact_order": [f"C{i}" for i in range(6)],
            "fingerprint_sha256": "fingerprint-demo",
            "field_models": {name: {} for name in model_names},
        },
    }


def _fake_scorers(record: dict) -> dict:
    return {
        name: {"kind": "shared" if name.startswith("shared_") else "own"}
        for name in record["interictal_field"]["field_models"]
    }


def test_frozen_loader_exposes_shared_scorers_only(
    tmp_path: Path, monkeypatch
) -> None:
    record = _frozen_record()
    (tmp_path / "epilepsiae_demo.json").write_text(json.dumps(record))
    monkeypatch.setattr(compute, "FROZEN_FIELD_DIR", tmp_path)
    monkeypatch.setattr(compute, "scorers_from_interictal_record", _fake_scorers)

    loaded, scorers = compute._load_frozen_shared("epilepsiae_demo")

    assert loaded["axis_pair"]["geometry_2d_supported"] is True
    assert set(scorers) == {"shared_a", "shared_b"}
    assert all(value["kind"] == "shared" for value in scorers.values())


def test_frozen_loader_fails_without_complete_shared_pair(
    tmp_path: Path, monkeypatch
) -> None:
    record = _frozen_record(model_names=("own_a", "own_b", "shared_a"))
    (tmp_path / "epilepsiae_demo.json").write_text(json.dumps(record))
    monkeypatch.setattr(compute, "FROZEN_FIELD_DIR", tmp_path)
    monkeypatch.setattr(compute, "scorers_from_interictal_record", _fake_scorers)

    with pytest.raises(ValueError, match="missing_shared_a_or_shared_b_field"):
        compute._load_frozen_shared("epilepsiae_demo")


def test_frozen_loader_rejects_geometry_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    record = _frozen_record(geometry_2d=False)
    (tmp_path / "epilepsiae_demo.json").write_text(json.dumps(record))
    monkeypatch.setattr(compute, "FROZEN_FIELD_DIR", tmp_path)
    monkeypatch.setattr(compute, "scorers_from_interictal_record", _fake_scorers)

    with pytest.raises(ValueError, match="geometry_2d_unsupported"):
        compute._load_frozen_shared("epilepsiae_demo")


def test_frozen_loader_rejects_subject_identity_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    record = _frozen_record("epilepsiae_other")
    (tmp_path / "epilepsiae_demo.json").write_text(json.dumps(record))
    monkeypatch.setattr(compute, "FROZEN_FIELD_DIR", tmp_path)
    monkeypatch.setattr(compute, "scorers_from_interictal_record", _fake_scorers)

    with pytest.raises(ValueError, match="frozen_subject_identity_mismatch"):
        compute._load_frozen_shared("epilepsiae_demo")


def test_explicit_peri_onset_stop_is_not_truncated_at_seizure_offset() -> None:
    stop_at, post_sec = compute._shared_window_extent(
        offset=2.0,
        stop_sec=20.0,
        smooth_sec=10.0,
    )
    assert stop_at == 20.0
    assert post_sec == 50.0


def test_single_seizure_diagnostic_defaults_to_offset() -> None:
    stop_at, post_sec = compute._shared_window_extent(
        offset=12.5,
        stop_sec=None,
        smooth_sec=5.0,
    )
    assert stop_at == 12.5
    assert post_sec == 13.0


def test_batch_discovery_requires_shared_fingerprint_and_2d(
    tmp_path: Path, monkeypatch
) -> None:
    records = {
        "epilepsiae_shared_2d": _frozen_record("epilepsiae_shared_2d"),
        "epilepsiae_shared_1d": _frozen_record(
            "epilepsiae_shared_1d", geometry_2d=False
        ),
        "epilepsiae_own_only": _frozen_record(
            "epilepsiae_own_only", model_names=("own_a", "own_b")
        ),
    }
    for subject, payload in records.items():
        (tmp_path / f"{subject}.json").write_text(json.dumps(payload))
    monkeypatch.setattr(batch, "FROZEN_FIELD_DIR", tmp_path)
    monkeypatch.setattr(batch, "scorers_from_interictal_record", _fake_scorers)

    assert batch._discover_subjects() == ["epilepsiae_shared_2d"]


def _renderer_rows(
    *,
    subject: str = "epilepsiae_demo",
    field_plane: object = "shared",
    fallback: object = False,
    fingerprint: str = "fingerprint-demo",
) -> list[dict]:
    rows = []
    for start in np.arange(-120.0, 10.0 + 1e-9, 2.0):
        rows.append({
            "subject": subject,
            "window_start_sec": start,
            "window_end_sec": start + 10.0,
            "window_center_sec": start + 5.0,
            "seizure_idx": 0,
            "field_plane": field_plane,
            "field_scorers": "shared_a,shared_b",
            "field_contract": "topic5_interictal_template_fields_v1",
            "field_fingerprint_sha256": fingerprint,
            "axis_definition": "template_propagation_axis_v2",
            "axis_direction_convention": "positive_early_to_late",
            "own_field_fallback": fallback,
            "geometry_2d_supported": True,
            "geometry_quality_tier": "strict_2d",
            "minimum_axis_n_shafts": 2,
            "minimum_axis_effective_rank": 2,
            "A_abs_corr": 0.4,
            "B_abs_corr": 0.7,
            "maxAB_abs_corr": 0.7,
            "A_signed_corr": -0.4,
            "B_signed_corr": 0.7,
            "A_abs_projection_z": 0.2,
            "B_abs_projection_z": 0.6,
            "maxAB_abs_projection_z": 0.6,
            "A_signed_projection_z": -0.2,
            "B_signed_projection_z": 0.6,
        })
    return rows


@pytest.fixture
def renderer_frozen(monkeypatch):
    record = _frozen_record()
    monkeypatch.setattr(
        render,
        "_load_frozen_shared",
        lambda _subject: (record, {"shared_a": {}, "shared_b": {}}),
    )
    return record


def _write_renderer_csv(tmp_path: Path, rows: list[dict]) -> Path:
    src = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    return src


@pytest.mark.parametrize(
    ("field_plane", "fallback", "message"),
    [
        ("own", False, "field_plane"),
        ("shared", True, "own-field fallback is forbidden"),
        ("shared", "yes", "own-field fallback is forbidden"),
    ],
)
def test_renderer_rejects_non_shared_or_fallback_csv(
    tmp_path: Path,
    renderer_frozen,
    field_plane: object,
    fallback: object,
    message: str,
) -> None:
    src = _write_renderer_csv(
        tmp_path,
        _renderer_rows(field_plane=field_plane, fallback=fallback),
    )
    with pytest.raises(RuntimeError, match=message):
        render._load_peri_onset(src, "epilepsiae_demo")


@pytest.mark.parametrize("column", ["field_plane", "own_field_fallback"])
def test_renderer_rejects_null_provenance(
    tmp_path: Path, renderer_frozen, column: str
) -> None:
    rows = _renderer_rows()
    rows[-1][column] = np.nan
    src = _write_renderer_csv(tmp_path, rows)
    with pytest.raises(RuntimeError, match="null provenance"):
        render._load_peri_onset(src, "epilepsiae_demo")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("field_fingerprint_sha256", "stale", "field_fingerprint_sha256"),
        ("axis_definition", "old_axis", "axis_definition"),
        ("subject", "epilepsiae_other", "subject"),
    ],
)
def test_renderer_rejects_mixed_or_stale_provenance(
    tmp_path: Path,
    renderer_frozen,
    column: str,
    value: str,
    message: str,
) -> None:
    rows = _renderer_rows()
    rows[-1][column] = value
    src = _write_renderer_csv(tmp_path, rows)
    with pytest.raises(RuntimeError, match=message):
        render._load_peri_onset(src, "epilepsiae_demo")


def test_renderer_rejects_incomplete_window_grid(
    tmp_path: Path, renderer_frozen
) -> None:
    src = _write_renderer_csv(tmp_path, _renderer_rows()[:-1])
    with pytest.raises(RuntimeError, match="66-window grid"):
        render._load_peri_onset(src, "epilepsiae_demo")


def test_journal_clean_design_moves_titles_to_axes_and_omits_panel_text() -> None:
    df = pd.DataFrame(_renderer_rows())
    agg = render._agg(df)

    fig = render._make_figure(
        df,
        agg,
        subject_label="E-demo",
        design_variant=render.DESIGN_JOURNAL_CLEAN,
    )
    try:
        ax0, ax1 = fig.axes
        assert ax0.get_title() == ""
        assert ax1.get_title() == ""
        assert len(ax0.texts) == 0
        assert len(ax1.texts) == 0
        assert not ax0.spines["top"].get_visible()
        assert not ax0.spines["right"].get_visible()
        assert not ax1.spines["top"].get_visible()
        assert not ax1.spines["right"].get_visible()
        assert ax0.get_ylabel().startswith("Expression |q|")
        assert "|q|" in ax0.get_ylabel()
        assert ax1.get_ylabel() == "Signed q\n(baseline z)"
        assert ax0.get_xlabel() == "Time (s)"
        assert ax1.get_xlabel() == "Time (s)"
        assert [text.get_text() for text in ax1.get_legend().get_texts()] == ["TA", "TB"]
        assert ax0.get_legend()._loc == 2
        assert ax1.get_legend()._loc == 2
        assert ax0.get_legend()._ncols == 1
        assert ax1.get_legend()._ncols == 1
        assert len(fig.texts) == 0
        assert tuple(fig.get_size_inches()) == pytest.approx((14.0, 5.6))
    finally:
        render.plt.close(fig)


def test_legacy_similarity_readout_remains_available() -> None:
    df = pd.DataFrame(_renderer_rows())
    agg = render._agg(df, readout=render.READOUT_SIMILARITY)
    fig = render._make_figure(
        df,
        agg,
        subject_label="E-demo",
        design_variant=render.DESIGN_JOURNAL_CLEAN,
        readout=render.READOUT_SIMILARITY,
    )
    try:
        ax0, ax1 = fig.axes
        assert ax0.get_ylabel().startswith("Field similarity")
        assert ax1.get_ylabel() == "Signed field similarity, r"
    finally:
        render.plt.close(fig)


def test_progress_index_cannot_look_complete(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(batch, "INDEX_COLUMNS", ["subject", "status"])
    audit = [{
        "subject": "epilepsiae_demo",
        "identity_valid": True,
        "shared_pair": True,
        "fingerprint_valid": True,
        "geometry_2d_supported": True,
    }]
    csv_path = tmp_path / "progress.csv"
    json_path = tmp_path / "progress.json"
    payload = batch._write_index(
        [{"subject": "epilepsiae_demo", "status": "complete_ok"}],
        csv_path,
        json_path,
        run_id="test-run",
        run_complete=False,
        planned_subjects=["epilepsiae_demo", "epilepsiae_next"],
        frozen_audit=audit,
        canonical_run=False,
    )
    assert payload["run_complete"] is False
    assert payload["n_planned"] == 2
    assert payload["n_processed_records"] == 1
    assert json.loads(json_path.read_text())["run_complete"] is False


def test_spatial_null_batch_engine_uses_shared_scorers_only(monkeypatch) -> None:
    record = _frozen_record()
    shared = {
        "shared_a": {"label": "a"},
        "shared_b": {"label": "b"},
    }
    monkeypatch.setattr(
        spatial_null,
        "_load_frozen_shared",
        lambda _subject: (record, shared),
    )

    def fake_score(scorer, activations):
        values = np.asarray(activations, float)
        base = np.nanmean(values, axis=1)
        offset = 0.1 if scorer["label"] == "a" else 0.2
        return {"abs_r": base + offset}

    monkeypatch.setattr(spatial_null, "score_field_batch", fake_score)
    engine = spatial_null.build_engine("epilepsiae_demo", ["C1", "C4"])
    observed = spatial_null.maxab_batch(engine, np.array([[0.2, 0.4]]))

    assert set(engine["shared"]) == {"shared_a", "shared_b"}
    assert observed == pytest.approx([0.5])


def test_vectorized_spatial_permutations_preserve_required_exchangeability() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    names = ["A1", "A2", "B1", "B2"]
    all_contact = spatial_null._permutation_batch(
        values, names, np.random.default_rng(1), "all_contact", 50
    )
    within_shaft = spatial_null._permutation_batch(
        values, names, np.random.default_rng(2), "within_shaft", 50
    )

    assert all_contact.shape == (50, 4)
    assert all(np.array_equal(np.sort(row), values) for row in all_contact)
    assert all(
        np.array_equal(np.sort(row[:2]), values[:2])
        and np.array_equal(np.sort(row[2:]), values[2:])
        for row in within_shaft
    )


@pytest.mark.parametrize("model", ["all_contact", "within_shaft"])
def test_spatial_mapping_is_fixed_across_all_windows(model: str) -> None:
    names = ["A1", "A2", "B1", "B2"]
    mapping = spatial_null._permutation_indices(
        names,
        np.random.default_rng(7),
        model,
        25,
    )
    first_window = np.array([1.0, 2.0, 3.0, 4.0])
    later_window = first_window + 10.0

    first_shuffled = first_window[mapping]
    later_shuffled = later_window[mapping]

    assert mapping.shape == (25, 4)
    assert np.array_equal(later_shuffled - first_shuffled, np.full((25, 4), 10.0))


def test_cached_null_rejects_source_trajectory_drift(
    tmp_path: Path, monkeypatch
) -> None:
    record = _frozen_record()
    expected = {
        "source_per_seizure_csv": "source.csv",
        "source_per_seizure_csv_sha256": "current-source",
        "trajectory_run_id": "run-current",
        "trajectory_manifest_sha256": "manifest-current",
        "source_seizure_ids": [0],
        "source_seizure_ids_sha256": "seizure-set",
        "window_grid_sha256": "grid-current",
        "spatial_null_algorithm_version": spatial_null.SPATIAL_NULL_ALGORITHM_VERSION,
        "permutation_coupling_version": spatial_null.PERMUTATION_COUPLING_VERSION,
    }
    source = pd.DataFrame({"seizure_idx": [0] * 66})
    monkeypatch.setattr(spatial_null, "_ROOT", tmp_path)
    monkeypatch.setattr(
        spatial_null,
        "_load_frozen_shared",
        lambda _subject: (record, {"shared_a": {}, "shared_b": {}}),
    )
    monkeypatch.setattr(
        spatial_null,
        "_trajectory_source_provenance",
        lambda _subject: (tmp_path / "source.csv", source, expected),
    )
    outputs = {}
    for role, name in {
        "figure_png": "figure.png",
        "figure_pdf": "figure.pdf",
        "stats_csv": "stats.csv",
        "null_matrices_npz": "null.npz",
    }.items():
        (tmp_path / name).write_bytes(b"present")
        outputs[role] = name
    summary = {
        "subject": "epilepsiae_demo",
        "field_plane": "shared",
        "own_field_fallback": False,
        "geometry_2d_supported": True,
        "field_fingerprint_sha256": "fingerprint-demo",
        **expected,
        "source_per_seizure_csv_sha256": "stale-source",
        "n_seizures": 1,
        "n_windows": 66,
        "n_perm": 1000,
        "seed": 0,
        "outputs": outputs,
    }

    with pytest.raises(ValueError, match="source_per_seizure_csv_sha256"):
        spatial_null._validate_cached_summary(
            "epilepsiae_demo", summary, n_perm=1000, seed=0
        )


def test_batch_subject_outputs_are_run_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    record = _frozen_record()
    run_field_dir = tmp_path / "run" / "artifacts" / "field_dynamics_signed"
    run_figure_dir = tmp_path / "run" / "artifacts" / "figures"
    calls = []
    monkeypatch.setattr(
        batch,
        "_load_frozen_shared",
        lambda _subject: (record, {"shared_a": {}, "shared_b": {}}),
    )
    monkeypatch.setattr(
        batch,
        "_eligibility_status",
        lambda _subject: {
            "inventory_n": 1,
            "cache_path": "cache.json",
            "eligible_idxs": [0],
            "reason_code": None,
        },
    )

    def fake_run(cmd):
        calls.append(cmd)
        if str(batch.PAPER_SCRIPT) in cmd:
            run_figure_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "coverage_status": "complete_ok",
                "n_eligible_requested": 1,
                "n_seizures": 1,
                "n_seizure_drops": 0,
                "coverage_fraction": 1.0,
                "n_windows": 66,
                "source_csv": str(run_field_dir / "source.csv"),
                "outputs": {
                    "png": str(run_figure_dir / "figure.png"),
                    "pdf": str(run_figure_dir / "figure.pdf"),
                },
                "readouts": {
                    key: {
                        "median_of_window_medians": 0.5,
                        "median_of_window_variances": 0.1,
                    }
                    for key in ("maxAB_abs", "signed_A", "signed_B")
                },
            }
            (run_figure_dir / batch.PAPER_SUMMARY.format(sid="epilepsiae_demo")).write_text(
                json.dumps(summary)
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(batch, "_run", fake_run)
    result = batch._process_subject(
        "epilepsiae_demo",
        run_field_dir=run_field_dir,
        run_figure_dir=run_figure_dir,
    )

    assert result["status"] == "complete_ok"
    assert calls[0][calls[0].index("--out-dir") + 1] == str(run_field_dir)
    assert calls[1][calls[1].index("--out-dir") + 1] == str(run_figure_dir)
    assert calls[1][calls[1].index("--source-csv") + 1].startswith(str(run_field_dir))
    assert str(batch.FIELD_DIR) not in " ".join(calls[0])
    assert str(batch.FIG_DIR) not in " ".join(calls[1])


def test_null_manifest_hashes_summary_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir = tmp_path / "null"
    out_dir.mkdir()
    subject = "epilepsiae_demo"
    outputs = {
        "figure_png": "null/figure.png",
        "figure_pdf": "null/figure.pdf",
        "stats_csv": "null/stats.csv",
        "null_matrices_npz": "null/matrices.npz",
    }
    for relative in outputs.values():
        path = tmp_path / relative
        path.write_bytes(relative.encode())
    summary_path = out_dir / f"{subject}_maxab_spatial_null_summary.json"
    summary_path.write_text(json.dumps({
        "subject": subject,
        "n_perm": 10,
        "trajectory_run_id": "run-demo",
        "outputs": outputs,
    }))
    monkeypatch.setattr(spatial_null, "_ROOT", tmp_path)
    monkeypatch.setattr(spatial_null, "OUT_DIR", out_dir)
    monkeypatch.setattr(spatial_null, "INDEX_CSV", out_dir / "index.csv")
    monkeypatch.setattr(spatial_null, "INDEX_JSON", out_dir / "index.json")
    monkeypatch.setattr(spatial_null, "MANIFEST_JSON", out_dir / "manifest.json")
    monkeypatch.setattr(
        spatial_null,
        "_validate_cached_summary",
        lambda _subject, _summary: {},
    )

    spatial_null._write_cohort_index([
        {"subject": subject, "status": "ok", "drop_reason": ""}
    ])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    roles = {item["role"] for item in manifest["artifacts"]}
    assert roles == {
        "summary_json",
        "figure_png",
        "figure_pdf",
        "stats_csv",
        "null_matrices_npz",
    }
    assert len(manifest["artifacts"]) == 5
