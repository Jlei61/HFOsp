from pathlib import Path
from scripts.plot_topic5_directional_replay import plot_subject


def test_plot_subject_442_writes_png():
    out = plot_subject("epilepsiae_442", "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"
