from pathlib import Path


def test_paper_fig5_panel_b_uses_audited_critical_manifold():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/paper_figures/"
              "plot_fig5_spatial_zm_ou_tonic_paper_ready.py").read_text()
    assert "draw_critical_manifold_trajectory" in source
    assert "patient_zm_snn_manifold_projection.json" in source
    assert "patient_zm_delay_stability_audit.json" in source
    assert "patient_zm_grid_convergence.json" in source
    assert "delay-unstable" in source
    assert ".gif" not in source.lower()


def test_paper_fig5_keeps_all_panels_on_seed_1842_assets():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/paper_figures/"
              "plot_fig5_spatial_zm_ou_tonic_paper_ready.py").read_text()
    assert "tonic_b0_v2_s1842.npz" in source
    assert "seed1842_static_panels.npz" in source
    assert "joint_04_control_seed_1801" not in source
    assert "etoi005" not in source.lower()
