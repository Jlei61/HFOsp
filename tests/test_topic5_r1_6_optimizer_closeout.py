from scripts.topic5_continuous_marked_state_r1.finalize_r1_6_optimizer_identifiability import (
    CLOSEOUT_REVISION,
    digest_manifest,
    repo_relative,
)
from src.topic5_continuous_marked_state_r1 import contract


def test_closeout_manifest_digest_is_order_invariant(tmp_path):
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text("{}")
    right.write_text("{}")
    assert digest_manifest([left, right], tmp_path)["combined_sha256"] == (
        digest_manifest([right, left], tmp_path)["combined_sha256"]
    )
    assert "r1_6" in CLOSEOUT_REVISION


def test_closeout_paths_are_portable_across_worktrees():
    path = contract.REPO_ROOT / "results/example/result.json"
    assert repo_relative(path) == "results/example/result.json"
