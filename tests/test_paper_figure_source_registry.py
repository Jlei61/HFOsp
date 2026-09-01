from __future__ import annotations

from scripts.paper_figures import build_main_figure_3_timing_plus_space as fig3_builder
from scripts.paper_figures import plot_fig2e_all_event_shared_fields as fig2e
from scripts.paper_figures import plot_fig2f_all_event_shared_field_reversal as fig2f
from scripts.paper_figures import plot_interictal_spatial_information_gain as fig2b
from scripts.paper_figures.paper_figure_source_registry import (
    ROOT,
    active_contract,
    load_registry,
    registered_path,
)


def test_active_contract_forbids_legacy_fallback() -> None:
    registry = load_registry()
    contract_id, contract = active_contract()
    assert contract_id == "all_event_timing_plus_space_no_hard_qc_v1"
    assert contract["hard_event_qc_used"] is False
    assert contract["fig2"]["updated_panels"] == ["b", "e", "f"]
    assert contract["fig3"]["updated_panels"] == ["c", "d", "e", "f"]
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


def test_fig3_builder_defaults_are_registered_all_event_sources() -> None:
    assert fig3_builder.ACTIVE_CONTRACT_ID == "all_event_timing_plus_space_no_hard_qc_v1"
    for panel_id in "cdef":
        assert fig3_builder.DEFAULT_PANEL_SOURCES[panel_id] == registered_path(
            "fig3", panel_id, "source_pdf"
        )
        assert "fig3_all_events_timing_plus_space_sources" in str(
            fig3_builder.DEFAULT_PANEL_SOURCES[panel_id]
        )


def test_registered_producers_are_tracked_source_files() -> None:
    _, contract = active_contract()
    for figure in ("fig2", "fig3"):
        for panel in contract[figure]["updated_panels"]:
            producer = ROOT / contract[figure][panel]["producer"]
            assert producer.is_file(), f"missing producer for {figure}.{panel}: {producer}"
