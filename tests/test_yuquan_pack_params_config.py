"""Pin pack-stage parameters for the 21 same-source Yuquan subjects.

Locks the contract that `config/subject_params.json` resolves to the exact
same per-subject `pick_k` / `pack_win_sec` / `pack_drop_channels` /
`pack_top_n` values that used to live inside
`scripts/run_yuquan_lagpat_backfill.py::LEGACY_SUBJECT_PARAMS`. Any drift in
either direction (config edited, or the loader re-introducing fallbacks) will
break this test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    YUQUAN_SAME_SOURCE_SUBJECTS,
    resolve_subject_pack_params,
)

# Snapshot of the legacy `LEGACY_SUBJECT_PARAMS` dict (pre-config-unification)
# plus the canonical empty `pack_drop_channels`. If you intentionally change
# Yuquan pack params, update this snapshot and document why in
# `docs/archive/yuquan_lagpat/`.
EXPECTED: dict = {
    # reference subjects
    "gaolan":       {"pick_k": 1.9, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "dongyiming":   {"pick_k": 0.5, "pack_win_sec": 0.220, "pack_drop_channels": []},
    "wangyiyang":   {"pick_k": 1.0, "pack_win_sec": 0.250, "pack_drop_channels": [], "pack_top_n": 22},
    # main cohort with legacy lagPat
    "chenziyang":   {"pick_k": 1.0, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "hanyuxuan":    {"pick_k": 1.0, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "huanghanwen":  {"pick_k": 1.0, "pack_win_sec": 0.200, "pack_drop_channels": []},
    "huangwanling": {"pick_k": 3.0, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "litengsheng":  {"pick_k": 1.0, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "sunyuanxin":   {"pick_k": 1.0, "pack_win_sec": 0.250, "pack_drop_channels": []},
    "xuxinyi":      {"pick_k": 0.7, "pack_win_sec": 0.200, "pack_drop_channels": []},
    "zhangjinhan":  {"pick_k": 1.0, "pack_win_sec": 0.200, "pack_drop_channels": []},
    # backfill-only subjects
    "chengshuai":   {"pick_k": 1.0, "pack_win_sec": 0.500, "pack_drop_channels": []},
    "liyouran":     {"pick_k": 1.0, "pack_win_sec": 0.250, "pack_drop_channels": []},
    "pengzihang":   {"pick_k": 1.0, "pack_win_sec": 0.500, "pack_drop_channels": []},
    "songzishuo":   {"pick_k": 1.0, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "zhangbichen":  {"pick_k": 0.5, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "zhangjiaqi":   {"pick_k": 1.7, "pack_win_sec": 0.150, "pack_drop_channels": []},
    "zhangkexuan":  {"pick_k": 0.5, "pack_win_sec": 0.500, "pack_drop_channels": []},
    "zhaochenxi":   {"pick_k": 0.5, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "zhaojinrui":   {"pick_k": 1.0, "pack_win_sec": 0.300, "pack_drop_channels": []},
    "zhourongxuan": {"pick_k": 1.0, "pack_win_sec": 0.200, "pack_drop_channels": []},
}


def test_cohort_membership_matches_expected() -> None:
    assert set(YUQUAN_SAME_SOURCE_SUBJECTS) == set(EXPECTED.keys()), (
        f"cohort membership drifted: only_in_code="
        f"{set(YUQUAN_SAME_SOURCE_SUBJECTS) - set(EXPECTED)}, "
        f"only_in_expected={set(EXPECTED) - set(YUQUAN_SAME_SOURCE_SUBJECTS)}"
    )
    assert len(YUQUAN_SAME_SOURCE_SUBJECTS) == 21


def test_per_subject_pack_params_match_legacy_snapshot() -> None:
    for subject, expected in EXPECTED.items():
        resolved = resolve_subject_pack_params(subject)
        assert resolved["pick_k"] == expected["pick_k"], (
            f"{subject}: pick_k {resolved['pick_k']} != {expected['pick_k']}"
        )
        assert resolved["pack_win_sec"] == expected["pack_win_sec"], (
            f"{subject}: pack_win_sec {resolved['pack_win_sec']} != {expected['pack_win_sec']}"
        )
        assert resolved["pack_drop_channels"] == expected["pack_drop_channels"], (
            f"{subject}: pack_drop_channels {resolved['pack_drop_channels']} "
            f"!= {expected['pack_drop_channels']}"
        )
        if "pack_top_n" in expected:
            assert resolved.get("pack_top_n") == expected["pack_top_n"], (
                f"{subject}: pack_top_n {resolved.get('pack_top_n')} "
                f"!= {expected['pack_top_n']}"
            )
        else:
            assert "pack_top_n" not in resolved, (
                f"{subject}: unexpected pack_top_n={resolved.get('pack_top_n')}"
            )


def test_unknown_subject_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        resolve_subject_pack_params("not_a_real_subject_xyz")


def test_defaults_alias_rejected() -> None:
    import pytest

    with pytest.raises(KeyError):
        resolve_subject_pack_params("_defaults")
