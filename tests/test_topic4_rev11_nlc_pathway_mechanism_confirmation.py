from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev11_nlc_pathway_mechanism_confirmation import (
    mechanism_library,
)
from src.topic4_graph_edge_flow import array_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev11_nlc_pathway_mechanism_confirmation.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_mechanism_confirmation_inputs_and_seed_pool_are_frozen():
    config = json.loads(CONFIG.read_text())
    assert config["search"]["confirmation_network_seeds"] == list(
        range(1581, 1593)
    )
    assert config["mechanism_readout"]["trace_dt_ms"] == 1.0
    assert config["search"]["beta"] == "closed"
    assert config["candidate_library"]["copied_from_frozen_confirmation_without_refit"]
    for record in config["inputs"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_mechanism_library_is_an_exact_copy_of_frozen_four_arms():
    config = json.loads(CONFIG.read_text())
    source = json.loads(
        (ROOT / config["inputs"]["frozen_confirmation_manifest"]["path"])
        .read_text()
    )
    candidates = mechanism_library(config, source)
    assert [row["candidate_id"] for row in candidates] == config[
        "candidate_library"
    ]["arms"]
    assert len({row["node_field"]["field_sha256"] for row in candidates}) == 1
    for candidate, frozen in zip(
            candidates, source["candidate_set"]["candidates"]):
        assert candidate == frozen
        coefficients = np.asarray(candidate["coefficients"], float)
        assert array_sha256(coefficients) == candidate["coefficients_sha256"]
