from pathlib import Path

from PIL import Image

from scripts.paper_figures import build_main_figures_1_2 as builder


def test_fig1a_is_a_fixed_crop_of_registered_legacy_supplementary_tiff(
    tmp_path: Path,
) -> None:
    metadata = builder._build_figure1a_from_legacy_tiff(tmp_path)

    png = tmp_path / "fig1-panela.png"
    pdf = tmp_path / "fig1-panela.pdf"
    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size == builder.FIG1A_EXPORT_SIZE
    assert metadata["source_asset"] == (
        "scripts/paper_figures/assets/fig1a_legacy_brain_crop.png"
    )
    assert metadata["source_tiff"] == "ReplayIED/tiffs/fig_s6_画板 1.tif"
    assert metadata["source_crop_pixels"] == list(builder.FIG1A_SOURCE_CROP)
    assert metadata["identity_semantics"] == (
        "no patient identity is asserted in Figure 1A"
    )
    assert metadata["rendering"].startswith("registered fixed crop only")
