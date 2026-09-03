import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
CONFIG = ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_structural_null_contract_names_every_run_and_node_source():
    config = json.loads(CONFIG.read_text())
    contract = config["structural_nulls"]
    assert contract["status"] == "frozen_before_rev22_candidate_selection"
    assert contract["confirmation_seed_prefix_count"] == 6
    assert contract["fixed_topology_variants"] == [
        "r180", "r90", "matched_norm_row_1", "matched_norm_row_2",
        "merged_midpoint_core", "random_two_core_centers",
    ]
    assert 6 * 2 * 6 + 2 * 6 + 2 * 2 * 6 == 108
    assert contract["expected_new_trajectories"] == 108

    factors = contract["node_blocking_factors"]
    assert [row["candidate_id"] for row in factors] == [
        "exact_dual_anchor", "v62_density_t050",
    ]
    for row in factors:
        source = ARTIFACT_ROOT / row["source_manifest"]
        assert _sha256(source) == row["source_manifest_sha256"]

