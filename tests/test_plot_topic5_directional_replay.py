from pathlib import Path
from scripts.plot_topic5_directional_replay import (plot_subject, plot_class_interictal_rose,
                                                    plot_cohort_pooled_main_aligned,
                                                    plot_cohort_axis_alignment)


def test_plot_subject_442_writes_png():
    out = plot_subject("epilepsiae_442", "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"


def test_plot_class_interictal_rose_442_writes_png():
    out = plot_class_interictal_rose("epilepsiae_442", "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"


def test_plot_cohort_pooled_main_aligned_writes_png():
    # subdir=_smoketest so the reduced (n=2) smoke output never clobbers the canonical figure
    out = plot_cohort_pooled_main_aligned(["epilepsiae_442", "epilepsiae_548"], "broadband",
                                          subdir="_smoketest")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"


def test_plot_cohort_axis_alignment_writes_png():
    # subdir=_smoketest so the reduced (1-band) smoke output never clobbers the 2-panel canonical
    out = plot_cohort_axis_alignment(bands=("broadband",), subdir="_smoketest")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"
