from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic5_continuous_marked_state_h2b.contract import LEAD_MINUTES
from src.topic5_continuous_marked_state_h2b.inventory import (
    build_inference_minute_mask,
    build_yuquan_crosswalk,
    load_r16_checkpoint_inventory,
    summarise_seizure_support,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r16_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "repo"
    root = source / (
        "results/epi_prssm/continuous_marked_state/r1/"
        "optimizer_identifiability_r1_6"
    )
    files = []
    for seed in (1, 3, 4):
        relative = Path(
            "confirmation/prefix_high_lr_e12_c128/nested_extended_budget/"
            f"epilepsiae_384/seed_{seed}/result.json"
        )
        result_path = root / relative
        result_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = result_path.with_name("model.pt")
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        result = {
            "status": "COMPLETE",
            "subject": "epilepsiae_384",
            "seed": seed,
            "seed_role": "independent_confirmation" if seed > 2 else "tuning_recheck",
            "stable_checkpoint": True,
            "formal_test_partition_opened": False,
            "sealed_opened": False,
            "development_validation_used_for_selection": False,
            "checkpoint": str(checkpoint.relative_to(source)),
            "checkpoint_sha256": _sha(checkpoint),
            "revision": "r1_6_optimizer_identifiability_nested_selection_v1",
            "confirmation_revision": "r1_6_frozen_optimizer_confirmation_v1",
            "selected_prefix_config": "prefix_high_lr_e12_c128",
            "selected_config": "nested_extended_budget",
        }
        result_path.write_text(json.dumps(result))
        files.append({"path": str(relative), "sha256": _sha(result_path)})
    audit = {
        "status": "COMPLETE",
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "confirmation": {
            "by_subject": {"epilepsiae_384": {"stable_checkpoints": 3}}
        },
        "manifests": {"confirmation": {"count": 3, "files": files}},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    return source, root, audit_path


def test_checkpoint_inventory_follows_audit_and_recomputes_hashes(tmp_path: Path) -> None:
    source, root, audit = _r16_fixture(tmp_path)
    value = load_r16_checkpoint_inventory(
        audit_path=audit, result_root=root, source_repo_root=source
    )
    assert value["n_checkpoints"] == 3
    assert [row["seed"] for row in value["entries"]] == [1, 3, 4]
    assert all(row["checkpoint_sha256_match"] for row in value["entries"])
    assert all(row["state_source_uses_seizure_labels"] is False for row in value["entries"])


def test_checkpoint_inventory_rejects_checkpoint_hash_drift(tmp_path: Path) -> None:
    source, root, audit = _r16_fixture(tmp_path)
    checkpoint = root / (
        "confirmation/prefix_high_lr_e12_c128/nested_extended_budget/"
        "epilepsiae_384/seed_3/model.pt"
    )
    checkpoint.write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        load_r16_checkpoint_inventory(
            audit_path=audit, result_root=root, source_repo_root=source
        )


def test_yuquan_crosswalk_uses_recording_code_and_requires_zero_delta() -> None:
    state = [
        {"state_seizure_id": "FA0013KQ_0", "onset_epoch": 100.0},
        {"state_seizure_id": "FA0013KQ_1", "onset_epoch": 201.0},
        {"state_seizure_id": "bad", "onset_epoch": 300.0},
    ]
    canonical = [
        {"subject": "gaolan", "record": "FA0013KQ",
         "seizure_id": "gaolan_sz_001", "eeg_onset_epoch": 100.0},
        {"subject": "gaolan", "record": "FA0013KQ",
         "seizure_id": "gaolan_sz_002", "eeg_onset_epoch": 200.0},
    ]
    rows = build_yuquan_crosswalk(state, canonical, subject="yuquan_gaolan")
    assert rows[0]["canonical_seizure_id"] == "gaolan_sz_001"
    assert rows[0]["matched"] is True
    assert rows[0]["onset_difference_seconds"] == 0.0
    assert rows[1]["matched"] is False
    assert rows[1]["match_route"] == "record_code+index_onset_mismatch"
    assert rows[2]["match_route"] == "id_did_not_parse"


@dataclass
class _Coverage:
    subject: str
    start: np.ndarray
    stop: np.ndarray
    session: np.ndarray
    dev_end_epoch: float


def _seizure(identifier: str, onset: float) -> dict:
    return {
        "subject": "epilepsiae_384",
        "canonical_seizure_id": identifier,
        "recording_code": "rec",
        "onset_epoch": onset,
        "matched": True,
        "onset_exact_match": True,
    }


def test_support_is_gap_aware_and_patient_tier_uses_eligible_seizures() -> None:
    coverage = _Coverage(
        subject="epilepsiae_384",
        start=np.asarray([0.0, 2000.0]),
        stop=np.asarray([1000.0, 5000.0]),
        session=np.asarray([0, 1]),
        dev_end_epoch=5000.0,
    )
    seizures = [
        _seizure("s1", 900.0),       # 30-min cut-off is before recorded support
        _seizure("s2", 2500.0),      # 30-min lead crosses the 1000--2000 gap
        _seizure("s3", 3900.0),      # 30-min lead is entirely in segment 1
        _seizure("s4", 4800.0),      # 30-min lead is entirely in segment 1
        _seizure("s5", 6000.0),      # outside development
    ]
    training_anchor_time = np.asarray([300.0])
    training_anchor_session = np.asarray([0])
    inference_anchor_time = np.asarray([300.0, 2050.0, 2800.0, 4500.0])
    summary, funnel, detail = summarise_seizure_support(
        seizures,
        coverage=coverage,
        training_anchor_time=training_anchor_time,
        training_anchor_session=training_anchor_session,
        inference_anchor_time=inference_anchor_time,
        leads=LEAD_MINUTES,
    )
    primary = next(row for row in summary if row["lead_minutes"] == 30)
    assert primary["n_eligible_seizures"] == 2
    assert primary["n_training_guarded_anchor_exists"] == 0
    assert primary["n_h2b_inference_observation_available_at_cutoff"] == 2
    assert primary["support_tier"] == "descriptive_case_series"
    assert funnel["by_lead"]["30"]["lead_window_crosses_gap_or_unrecorded_time"] == 2
    s2 = next(row for row in detail if row["seizure_id"] == "s2" and row["lead_minutes"] == 30)
    assert s2["crosses_gap_or_unrecorded_time"] is True


def test_inference_minute_mask_ignores_training_guard_columns_by_contract() -> None:
    # Row zero is readable for inference even though a separate training guard
    # would reject it.  The label-derived guard is intentionally not an input.
    training_guard_free = np.asarray([False, True, True, True])
    value = build_inference_minute_mask(
        covered=np.asarray([True, True, False, True]),
        session_id=np.asarray([0, -1, 0, 0]),
        cached=np.asarray([True, True, True, True]),
        n_valid_contacts=np.asarray([56, 80, 80, 55]),
        n_contacts=80,
        min_valid_contact_fraction=0.70,
    )
    assert training_guard_free.tolist() == [False, True, True, True]
    assert value.tolist() == [True, False, False, False]


CANONICAL_SOURCE = Path("/home/honglab/leijiaxin/HFOsp")
E384_DESIGN_MANIFEST = CANONICAL_SOURCE / (
    "results/epi_prssm/continuous_marked_state/r1/r1_5/cache/"
    "epilepsiae_384/manifest.json"
)
E384_RAW_CACHE = Path(
    "/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1/epilepsiae_384"
)
E384_RAW_REQUIRED = (
    E384_RAW_CACHE / "raw_256hz.zarr",
    E384_RAW_CACHE / "artifact_mask.zarr",
    E384_RAW_CACHE / "train_stats.json",
    E384_RAW_CACHE / "window_index_refined.parquet",
    E384_RAW_CACHE / "cache_index.parquet",
)


@pytest.mark.skipif(
    not E384_DESIGN_MANIFEST.exists()
    or not all(path.exists() for path in E384_RAW_REQUIRED),
    reason="canonical E384 raw-backed instrument unavailable",
)
def test_e384_inventory_anchor_identity_matches_state_extraction_reader() -> None:
    from src.topic5_continuous_marked_state_h2b.inventory import (
        load_state_support_arrays,
    )
    from src.topic5_continuous_marked_state_h2b.state_extraction import (
        InferenceRawAnchorReader,
        _sha256_arrays,
    )

    arrays = load_state_support_arrays(
        CANONICAL_SOURCE, subject="epilepsiae_384"
    )
    manifest = json.loads(E384_DESIGN_MANIFEST.read_text())
    with np.load(manifest["design"], allow_pickle=False) as design:
        reader = InferenceRawAnchorReader(
            "epilepsiae_384", design["event_time"].astype(np.float64)
        )
    anchor, segment, _, _, _ = reader.inference_anchor_inventory(
        arrays["coverage"]
    )
    assert np.array_equal(arrays["inference_anchor_time"], anchor)
    assert np.array_equal(arrays["inference_anchor_segment"], segment)
    assert arrays["anchor_time_segment_sha256"] == _sha256_arrays(
        anchor.astype(np.float64), segment.astype(np.int64)
    )
    assert arrays["inference_min_valid_contact_fraction"] == pytest.approx(0.70)
