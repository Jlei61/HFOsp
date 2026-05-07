"""Phase 3 Task 3.2: verify Epilepsiae detection defaults to use_gpu=True;
Yuquan stays CPU (use_gpu=False or absent)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_hfo_detection.py"


def test_script_exists_and_readable():
    assert SCRIPT.exists()


def test_epilepsiae_default_use_gpu_is_true():
    """The Epilepsiae call site must default use_gpu to True (v2 cohort)."""
    src = SCRIPT.read_text()
    # find the Epilepsiae block
    epi_idx = src.lower().find("epilepsiae")
    assert epi_idx >= 0
    epi_block = src[epi_idx:]
    # inside the Epilepsiae handler, expect either an explicit `use_gpu=True`
    # default or `params.get("use_gpu", True)` style
    assert (
        'params.get("use_gpu", True)' in epi_block
        or 'use_gpu = True' in epi_block
        or 'use_gpu=True' in epi_block
    ), "Epilepsiae path does not default use_gpu to True"


def test_yuquan_default_use_gpu_is_not_true():
    """The Yuquan call site must NOT default use_gpu to True (CPU baseline)."""
    src = SCRIPT.read_text()
    # find the Yuquan block (between yuquan and the next epilepsiae/end-of-file)
    yq_idx = src.lower().find("yuquan")
    if yq_idx < 0:
        return  # no Yuquan path = vacuously safe
    epi_idx = src.lower().find("epilepsiae", yq_idx + 1)
    yq_block = src[yq_idx : epi_idx if epi_idx > 0 else len(src)]
    assert (
        'params.get("use_gpu", True)' not in yq_block
        and 'use_gpu = True' not in yq_block
        and 'use_gpu=True' not in yq_block
    ), "Yuquan path improperly defaulted to use_gpu=True (should be CPU)"
