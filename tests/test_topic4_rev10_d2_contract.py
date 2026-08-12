import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev10_d2_resource_candidates import candidate_library
from src.topic4_graph_edge_flow import array_sha256

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((ROOT / "config/topic4_rev10_d2_inhibitory_resource_canary.json").read_text())


def test_resource_library_is_three_paired_doses_plus_off():
    rows=candidate_library(_config()); assert len(rows)==7
    keys={(r["inhibitory_resource"]["mode"],r["inhibitory_resource"]["k_q_per_ms"]) for r in rows[1:]}
    for k in (0.0005,0.0015,0.004):
        assert ("local",k) in keys and ("global",k) in keys
    assert {row["candidate_id"] for row in rows[1:]} == {
        f"qresource_{mode}_k{dose}"
        for mode in ("local", "global")
        for dose in ("000500", "001500", "004000")
    }


def test_resource_library_keeps_static_edges_off_and_hashes_canonical():
    for row in candidate_library(_config()):
        values=np.asarray(row["coefficients"],float)
        assert np.array_equal(values,np.zeros(12))
        assert row["coefficients_sha256"]==array_sha256(values)


def test_resource_canary_uses_fresh_networks_and_long_wait():
    config=_config(); assert config["search"]["fit_network_seeds"]==[1141,1142,1143]
    assert config["execution"]["wait_seconds"]>=180
    assert config["inhibitory_resource_library"]["update_interval_ms"] == 1.0
