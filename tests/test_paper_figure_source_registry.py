from __future__ import annotations

import hashlib
import json

import pytest

from scripts.paper_figures import build_fig3_current
from scripts.paper_figures import build_main_figure_3_timing_plus_space as fig3_builder
from scripts.paper_figures import paper_figure_source_registry as source_registry
from scripts.paper_figures import plot_fig2e_all_event_shared_fields as fig2e
from scripts.paper_figures import plot_fig2f_all_event_shared_field_reversal as fig2f
from scripts.paper_figures import plot_interictal_spatial_information_gain as fig2b
from scripts.paper_figures.paper_figure_source_registry import (
    ROOT,
    active_contract,
    load_registry,
    registered_path,
    validate_active_sources,
)


def test_active_contract_forbids_legacy_fallback() -> None:
    registry = load_registry()
    contract_id, contract = active_contract()
    assert contract_id == "all_event_timing_plus_space_no_hard_qc_v1"
    assert contract["hard_event_qc_used"] is False
    assert contract["fig2"]["updated_panels"] == ["b", "e", "f"]
    assert contract["fig3"]["updated_panels"] == list("abcde")
    assert contract["fig3"]["version"] == "visual_alignment_compact_rows_20261010"
    assert contract["fig3"]["status"] == "AUTHOR_ACCEPTED_FINAL"
    assert registry["current_figure1"]["version"] == "y1_final_20261010"
    assert registry["current_figures"]["fig4"]["version"] == "core_search_final_20261011"
    assert registry["policy"]["fallback_to_legacy_sources"] is False
    assert all(
        record["allowed_as_default"] is False
        for record in registry["legacy_contracts"].values()
    )


def test_fig2_plot_defaults_resolve_through_active_registry() -> None:
    expected_gain = ROOT / (
        "results/interictal_propagation_masked/"
        "spatial_information_gain_all_events"
    )
    expected_fields = ROOT / (
        "results/interictal_propagation_masked/"
        "template_gradient_fields_all_events_timing_plus_space"
    )
    assert fig2b.DEFAULT_ANALYSIS_ROOT == expected_gain
    assert fig2b.DEFAULT_ANALYSIS_ROOT == registered_path("fig2", "b", "analysis_root")
    assert fig2e.DEFAULT_INPUT == expected_fields
    assert fig2f.DEFAULT_INPUT == expected_fields
    assert fig2e.DEFAULT_INPUT == registered_path("fig2", "e", "analysis_root")
    assert fig2f.DEFAULT_INPUT == registered_path("fig2", "f", "analysis_root")


def test_fig3_compatibility_builder_uses_current_release_without_legacy_fallback() -> None:
    assert fig3_builder.main is build_fig3_current.main
    with pytest.raises(KeyError, match="unregistered paper panel"):
        registered_path("fig3", "f", "source_pdf")
    report = validate_active_sources(figures=("fig3",))
    assert report["ok"]
    assert any(item["name"] == "fig3.figures/fig3-complete-layout.pdf"
               for item in report["checked"])


def test_registered_producers_are_tracked_source_files() -> None:
    _, contract = active_contract()
    for figure in ("fig2", "fig3"):
        if contract[figure].get("status") == "AUTHOR_ACCEPTED_FINAL":
            assert (ROOT / contract[figure]["producer"]).is_file()
            continue
        for panel in contract[figure]["updated_panels"]:
            producer = ROOT / contract[figure][panel]["producer"]
            assert producer.is_file(), f"missing producer for {figure}.{panel}: {producer}"


@pytest.mark.parametrize("failure", ["modified", "missing"])
def test_frozen_release_validation_rejects_damaged_export(tmp_path, monkeypatch, failure):
    package = tmp_path / "release"
    package.mkdir()
    export = package / "panel.bin"
    export.write_bytes(b"accepted export")
    for name in ("producer.py", "pointer.json", "panels.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {
        "panel.bin": {"sha256": hashlib.sha256(export.read_bytes()).hexdigest()}
    }}))
    record = dict(status="AUTHOR_ACCEPTED_FINAL", package="release", producer="producer.py",
                  pointer="pointer.json", panel_registry="panels.json", release_manifest="manifest.json")
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(source_registry, "active_contract", lambda: (
        "test", {"hard_event_qc_used": False, "fig3": record}
    ))
    assert validate_active_sources(figures=("fig3",))["ok"]
    if failure == "modified":
        export.write_bytes(b"changed export")
    else:
        export.unlink()
    with pytest.raises(RuntimeError, match="fig3.panel.bin"):
        validate_active_sources(figures=("fig3",))
