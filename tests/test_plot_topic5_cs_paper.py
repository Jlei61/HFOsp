"""Smoke tests for scripts/plot_topic5_cs_paper.py — each figure renders with
titles/axis-labels/legend set AND saves to a non-empty PNG. Uses real
epilepsiae_1146 data from the main repo's results/ (gitignored T0 cache + axis
records are not part of this worktree — same precedent as
tests/test_run_topic5_contact_similarity.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plot_topic5_cs_paper import (
    REPRESENTATIVE_SUBJECT, _load_subject_ctx, fig1, fig2, fig3, save_fig,
)

ROOT = "/home/honglab/leijiaxin/HFOsp/results"
CAN = Path(ROOT) / "topic5_ictal_recruitment" / "contact_similarity"

pytestmark = pytest.mark.skipif(
    not (CAN / "cohort_summary_broadband.json").exists(),
    reason="requires gitignored T0 cache / cohort summaries under the main repo results/",
)


@pytest.fixture(scope="module")
def ctx():
    return _load_subject_ctx(REPRESENTATIVE_SUBJECT, "broadband", ROOT)


@pytest.fixture(scope="module")
def cohort_summary():
    return json.load(open(CAN / "cohort_summary_broadband.json"))


@pytest.fixture(scope="module")
def r2b_summary():
    return json.load(open(CAN / "r2b_summary_broadband.json"))


def test_load_subject_ctx_smoke(ctx):
    assert ctx["subject_id"] == REPRESENTATIVE_SUBJECT
    assert ctx["ictal_mean"].shape[0] == len(ctx["matched"])
    assert ctx["n_shafts"] >= 2   # multi-shaft, required for the within_shaft null


def test_fig1_renders(tmp_path, ctx):
    fig = fig1(ctx)
    assert fig._suptitle is not None and fig._suptitle.get_text()
    axes = fig.get_axes()
    assert len(axes) >= 3
    for ax in axes[:3]:
        assert ax.get_title()
    out = save_fig(fig, tmp_path / "fig1_spatial_weighting_schematic.png")
    assert out.exists() and out.stat().st_size > 5000


def test_fig2_renders(tmp_path, ctx, cohort_summary):
    fig = fig2(ctx, cohort_summary)
    axes = fig.get_axes()
    assert len(axes) >= 3   # axL, axR, axL2 (twin)
    axL, axR = axes[0], axes[1]
    assert axL.get_title() and axL.get_xlabel() and axL.get_ylabel()
    assert axL.get_legend() is not None
    assert axR.get_title() and axR.get_xlabel()
    out = save_fig(fig, tmp_path / "fig2_rank_comparison.png")
    assert out.exists() and out.stat().st_size > 5000


def test_fig3_renders(tmp_path, ctx, cohort_summary, r2b_summary):
    fig = fig3(ctx, cohort_summary, r2b_summary)
    axes = fig.get_axes()
    assert len(axes) >= 4
    for ax in axes:
        assert ax.get_title() != "" or ax.images or ax.collections
    out = save_fig(fig, tmp_path / "fig3_vs_field.png")
    assert out.exists() and out.stat().st_size > 5000
