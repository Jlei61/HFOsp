"""Fail-closed forbidden-input and split-leakage guards."""
import pytest

from src.topic5_epi_prssm.contracts import (
    FORBIDDEN_INPUTS, ForbiddenInputError, LeakageGuard,
)


def test_seizure_side_fields_are_refused_by_default():
    guard = LeakageGuard(stage="unit")
    for field in ("seizure_time", "time_to_seizure", "early_ictal_order", "soz_core_channels",
                  "ab_axis_label", "snn_template", "old_heldout20"):
        with pytest.raises(ForbiddenInputError):
            guard.check_fields([field])


def test_seizure_side_fields_open_only_after_the_freeze():
    released = LeakageGuard(stage="goal3", allow_seizure_side=True)
    released.check_fields(["seizure_time", "time_to_seizure", "early_ictal_order"])
    with pytest.raises(ForbiddenInputError):
        released.check_fields(["soz_core_channels"])
    with pytest.raises(ForbiddenInputError):
        released.check_fields(["snn_template"])


def test_test_partition_is_unreachable_from_training():
    guard = LeakageGuard(stage="unit")
    guard.check_split(["train", "validation"])
    with pytest.raises(ForbiddenInputError, match="Hard Gate C"):
        guard.check_split(["train", "test"])


def test_forbidden_artifact_paths_are_refused():
    guard = LeakageGuard(stage="unit")
    for path in ("results/yuquan_soz_core_channels.json",
                 "results/seizure_detection/pr1_seizure_gaolan.json",
                 "results/topic4_attractor_masked/step0_audit.csv"):
        with pytest.raises(ForbiddenInputError):
            guard.check_path(path)


def test_geometry_is_not_on_the_forbidden_list():
    """Spec section 4 authorises the symmetric contact-geometry Laplacian."""
    assert not any("geometry" in token for token in FORBIDDEN_INPUTS)
    assert "soz" in FORBIDDEN_INPUTS
