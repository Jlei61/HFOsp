import importlib.util
import json
import os


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOT_PATH = os.path.join(_ROOT, "scripts", "plot_topic4_zm_branch_decision.py")


def _load_plotter():
    spec = importlib.util.spec_from_file_location(
        "topic4_zm_branch_plotter_for_test", _PLOT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, relative_path, payload):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_phase_status_renders_completed_and_no_evidence_semantics(
        tmp_path, monkeypatch):
    plotter = _load_plotter()
    _write(tmp_path, "branch_verdict.json", {
        "verdict": "phase3_driver_selection_required",
        "layers": {"source_space_carrier": "carrier_window"},
        "eligible_seeds": [1, 3, 4],
        "reference_artifacts": "blocked",
        "confirmation": {"status": "passed"},
    })
    _write(tmp_path, "effective_rank/effective_rank_summary.json", {
        "verdict": "no_evidence_incomplete_central_pairs",
    })
    _write(tmp_path, "source_rhythm/source_rhythm_summary.json", {
        "status": "class_disagreement",
    })
    _write(tmp_path, "modal_operator/modal_operator_summary.json", {
        "status": "insufficient_seeds",
        "n_complete_seeds": 0,
    })
    _write(tmp_path, "boundaries/entry/entry_boundary_summary.json", {
        "verdict": "conditional_Z_entry_boundary_unresolved",
        "n_complete_seeds": 3,
    })
    _write(tmp_path, "boundaries/offset/offset_boundary_summary.json", {
        "verdict": "M_shapes_but_no_offset_surface",
        "n_complete_seeds": 3,
    })
    monkeypatch.setattr(plotter, "OUT", str(tmp_path))
    monkeypatch.setattr(plotter, "FIG", str(tmp_path / "figures"))
    (tmp_path / "figures").mkdir()

    phases, _ = plotter._phase_status_rows()
    statuses = dict(phases)
    assert statuses["1B minimal-subsystem forks"] == "carrier window complete"
    assert statuses["1.5A functional rank"] == (
        "no evidence: central-pair boundary"
    )
    assert statuses["1.5B modal / gain"] == (
        "skipped: source-class disagreement"
    )
    assert statuses["2A Z-entry boundary"] == "unresolved"
    assert statuses["2B offset boundary"] == "no evidence: offset surface"

    made = plotter.fig_phase_status([])
    assert made == str(tmp_path / "figures" / "phase_completion_status.png")
    assert os.path.getsize(made) > 0


def test_phase_status_distinguishes_existing_but_unreached_offset(
        tmp_path, monkeypatch):
    plotter = _load_plotter()
    _write(tmp_path, "branch_verdict.json", {})
    _write(tmp_path, "boundaries/offset/offset_boundary_summary.json", {
        "verdict": "M_Z_recovery_boundary_exists_but_unreached",
        "n_complete_seeds": 3,
    })
    monkeypatch.setattr(plotter, "OUT", str(tmp_path))

    phases, _ = plotter._phase_status_rows()
    assert dict(phases)["2B offset boundary"] == "exists / unreached"
