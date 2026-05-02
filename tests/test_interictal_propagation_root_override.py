"""Tests for `scripts/run_interictal_propagation.py` Epilepsiae root override.

Target plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md
§5 D.1 Step 2 — verify the `--epilepsiae-root` CLI flag flows through and that
the path-resolution helper accepts both legacy (`<root>/<subj>/all_recs/`) and
flat (`<root>/<subj>/`) layouts.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_interictal_propagation.py"


@pytest.fixture(scope="module")
def script_module():
    """Import the script as a module so we can call its helpers directly."""
    spec = importlib.util.spec_from_file_location(
        "run_interictal_propagation", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_epilepsiae_subject_dir_prefers_legacy_layout_when_present(
    tmp_path: Path, script_module
) -> None:
    subject = "548"
    legacy = tmp_path / subject / "all_recs"
    legacy.mkdir(parents=True)
    (tmp_path / subject / "stray.txt").write_text("flat-only sentinel")  # decoy

    out = script_module._epilepsiae_subject_dir(tmp_path, subject)

    assert out == legacy
    assert out.exists()


def test_epilepsiae_subject_dir_falls_back_to_flat_when_no_all_recs(
    tmp_path: Path, script_module
) -> None:
    subject = "253"
    flat = tmp_path / subject
    flat.mkdir(parents=True)
    (flat / "25300102_0000_lagPat.npz").write_bytes(b"")  # backfill-style file

    out = script_module._epilepsiae_subject_dir(tmp_path, subject)

    assert out == flat
    assert (out / "25300102_0000_lagPat.npz").exists()


def test_subject_dir_dispatches_yuquan_to_root_directly(
    tmp_path: Path, script_module
) -> None:
    """yuquan datasets always use the flat root/subject layout regardless of
    backfill considerations. Guard against accidental Epilepsiae fallback for
    yuquan keys."""
    subject = "chengshuai"
    (tmp_path / subject).mkdir()

    out = script_module._subject_dir("yuquan", tmp_path, subject)

    assert out == tmp_path / subject


def test_cli_parser_accepts_epilepsiae_root_argument(script_module) -> None:
    parser = script_module._build_parser()
    args = parser.parse_args(
        ["--dataset", "epilepsiae", "--subjects", "548",
         "--epilepsiae-root", "/some/path/results/epilepsiae_lagpat_backfill"]
    )

    assert args.epilepsiae_root == Path("/some/path/results/epilepsiae_lagpat_backfill")


def test_cli_parser_default_epilepsiae_root_is_none(script_module) -> None:
    """Default must be None so the script can fall back to the module-level
    EPILEPSIAE_ROOT constant — ensures legacy callers see no behavior change."""
    parser = script_module._build_parser()
    args = parser.parse_args([])

    assert args.epilepsiae_root is None


def test_cli_parser_accepts_output_root_argument(script_module) -> None:
    parser = script_module._build_parser()
    args = parser.parse_args(
        ["--output-root", "/tmp/results/interictal_propagation/sensitivity_new_lagpat"]
    )

    assert args.output_root == Path(
        "/tmp/results/interictal_propagation/sensitivity_new_lagpat"
    )


def test_cli_parser_default_output_root_is_none(script_module) -> None:
    """Default must be None so RESULTS_DIR keeps its module-level value for
    legacy callers."""
    parser = script_module._build_parser()
    args = parser.parse_args([])

    assert args.output_root is None
