from __future__ import annotations

import json

from scripts.topic5_continuous_marked_state_h2b.run_v03_instrument_cell import (
    _resolve_interictal_design,
)
from src.topic5_continuous_marked_state_h2b.contract import sha256_file


def test_interictal_design_resolution_does_not_require_seizure_support(
    tmp_path,
) -> None:
    v03 = tmp_path / "v0_3"
    v02 = tmp_path / "v0_2"
    subject = "synthetic_no_seizure_support"
    cache = v03 / "upstream_r1_2/cache" / subject
    cache.mkdir(parents=True)
    design = cache / "full_design.npz"
    embedding = cache / "explicit_embedding.npy"
    design.write_bytes(b"design")
    embedding.write_bytes(b"embedding")
    manifest = cache / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "COMPLETE",
        "design_sha256": sha256_file(design),
        "explicit_embedding_sha256": sha256_file(embedding),
    }))
    audit = v03 / "manifests/upstream_rebuild" / f"{subject}.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({
        "status": "COMPLETE",
        "checks": {
            "design_matches_frozen_r1_7b": True,
            "normalised_explicit_matches_frozen_r1_7b": True,
            "history_baseline_bitwise_matches_frozen_r1_7b_checkpoint": True,
        },
        "artifacts": {
            "design": str(design),
            "design_sha256": sha256_file(design),
        },
    }))

    observed_design, observed_manifest, observed_embedding, provenance = (
        _resolve_interictal_design(subject, v02_root=v02, result_root=v03)
    )
    assert observed_design == design
    assert observed_manifest == manifest
    assert observed_embedding == embedding
    assert provenance["route"] == "v0_3_hash_verified_rebuild"
