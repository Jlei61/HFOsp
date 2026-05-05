"""Phase 5 Task 5.0: verify backfill --gpu-root / --output-root flags."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_epilepsiae_lagpat_backfill.py"


def test_help_lists_gpu_root_and_output_root():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--gpu-root" in out, "missing --gpu-root flag"
    assert "--output-root" in out, "missing --output-root flag"
    # confirm defaults
    assert "results/hfo_detection" in out, "default gpu-root path missing in help"
    assert "results/epilepsiae_lagpat_backfill" in out, "default output-root path missing in help"


def test_module_constants_default_to_legacy_paths():
    """Without CLI override, module-level NEW_GPU_ROOT / OUTPUT_ROOT keep
    legacy defaults so prior callers keep working."""
    sys.path.insert(0, str(REPO))
    # Re-import in case other tests mutated module state
    import importlib
    import scripts.run_epilepsiae_lagpat_backfill as m
    importlib.reload(m)
    assert m.NEW_GPU_ROOT == Path("results/hfo_detection")
    assert m.OUTPUT_ROOT == Path("results/epilepsiae_lagpat_backfill")
