from pathlib import Path
from scripts.plot_topic5_directional_replay import (plot_subject, plot_class_interictal_rose,
                                                    plot_cohort_pooled_main_aligned)


def test_plot_subject_442_writes_png():
    out = plot_subject("epilepsiae_442", "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"


def test_plot_class_interictal_rose_442_writes_png():
    out = plot_class_interictal_rose("epilepsiae_442", "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"


def test_plot_cohort_pooled_main_aligned_writes_png():
    out = plot_cohort_pooled_main_aligned(["epilepsiae_442", "epilepsiae_548"], "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"
