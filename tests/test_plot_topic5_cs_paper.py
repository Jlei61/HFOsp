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
    REPRESENTATIVE_SUBJECT, _load_subject_ctx, fig1, fig2, fig2_sup, fig3, save_fig,
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


def test_fig2_renders(tmp_path, ctx):
    # single vertical rank ladder: 发作时序 vs 空间模板 vs 普通时序模板 (3 lines, one axis)
    fig = fig2(ctx)
    ax = fig.get_axes()[0]
    assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    assert ax.get_legend() is not None
    assert len(ax.get_lines()) >= 3   # the three per-contact order tracks
    out = save_fig(fig, tmp_path / "fig2_rank_comparison.png")
    assert out.exists() and out.stat().st_size > 5000


def test_fig2_sup_renders(tmp_path, cohort_summary):
    fig = fig2_sup(cohort_summary, REPRESENTATIVE_SUBJECT)
    ax = fig.get_axes()[0]
    assert ax.get_title() and ax.get_xlabel()
    assert ax.get_legend() is not None
    out = save_fig(fig, tmp_path / "fig2_sup_maxab_vs_null.png")
    assert out.exists() and out.stat().st_size > 5000


def test_fig3_renders(tmp_path, ctx, cohort_summary, r2b_summary):
    fig = fig3(ctx, cohort_summary, r2b_summary)
    assert fig._suptitle is not None and fig._suptitle.get_text()
    axes = fig.get_axes()
    assert len(axes) >= 5   # 2 maps + distance + 2 equivalence scatters (+ colorbar)
    with_content = [ax for ax in axes if ax.get_title() or ax.images or ax.collections]
    assert len(with_content) >= 5
    out = save_fig(fig, tmp_path / "fig3_vs_field.png")
    assert out.exists() and out.stat().st_size > 5000
