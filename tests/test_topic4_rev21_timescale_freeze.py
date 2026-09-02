import copy
import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev21_timescale_library import (
    build_timescale_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_timescale_grid_preserves_substrate_and_integrated_adaptation():
    config = json.loads((
        ROOT / "config/topic4_rev21_dual_core_zm_transition.json"
    ).read_text())
    manifest = json.loads((
        ARTIFACT_ROOT / config["candidate_manifest"]
    ).read_text())
    coarse = copy.deepcopy(next(
        row for row in manifest["candidates"]
        if row["candidate_id"] == "rev21_si_0p9_sm_1"
    ))
    rows = build_timescale_candidates(config, coarse)
    assert len(rows) == 9
    assert sum(row["is_reference"] for row in rows) == 1
    products = np.asarray([
        row["slow_variables"]["eta_m"]
        * row["slow_variables"]["tau_adp"] for row in rows
    ])
    assert np.allclose(products, products[0])
    assert {row["node_field"]["field_sha256"] for row in rows} == {
        coarse["node_field"]["field_sha256"]
    }
    assert {row["mechanisms"]["g_EE"] for row in rows} == {
        coarse["mechanisms"]["g_EE"]
    }
