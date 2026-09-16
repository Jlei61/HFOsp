from __future__ import annotations

from PIL import Image, ImageChops

from scripts.paper_figures import build_main_figures_1_2 as layout


def test_fit_width_panel_uses_exact_column_width(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(layout, "ROOT", tmp_path)
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 80), "black").save(source)
    output_dir = tmp_path / "figures"
    output_dir.mkdir()

    layout._compose_complete_layout(
        figures_dir=output_dir,
        stem="fit-width",
        canvas_size=(300, 200),
        placements={"a": (source, (40, 20, 240, 170))},
        labels={},
        anchors={"a": "top-left"},
        fit_to_cell=True,
        fit_width_panels={"a"},
    )

    rendered = Image.open(output_dir / "fit-width.png").convert("RGB")
    mask = ImageChops.difference(rendered, Image.new("RGB", rendered.size, "white"))
    assert mask.getbbox() == (40, 20, 240, 180)
