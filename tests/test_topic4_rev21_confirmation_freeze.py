import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.freeze_topic4_rev21_confirmation_library import (
    build_confirmation_candidates,
)
from src.topic4_zm_ictal_transition import make_slow


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _source_rows():
    config = json.loads((
        ROOT / "config/topic4_rev21_dual_core_zm_transition.json"
    ).read_text())
    manifest = json.loads((ARTIFACT_ROOT / config["candidate_manifest"]).read_text())
    finalist = copy.deepcopy(next(
        row for row in manifest["candidates"]
        if row["candidate_id"] == "rev21_si_0p9_sm_1"
    ))
    off = copy.deepcopy(next(
        row for row in manifest["candidates"]
        if row["candidate_id"] == "rev21_zm_off"
    ))
    return finalist, off


def test_confirmation_library_changes_only_slow_mechanism_flags():
    finalist, off = _source_rows()
    rows = build_confirmation_candidates(finalist, off)
    assert {row["candidate_id"] for row in rows} == {
        "rev21_confirm_z_plus_m", "rev21_confirm_z_only",
        "rev21_confirm_m_only", "rev21_confirm_zm_off",
    }
    active = [row for row in rows if row["slow_variables"]["mode"] != "off"]
    assert {row["node_field"]["field_sha256"] for row in active} == {
        finalist["node_field"]["field_sha256"]
    }
    assert {row["mechanisms"]["g_EE"] for row in active} == {
        finalist["mechanisms"]["g_EE"]
    }
    flags = {
        row["candidate_id"]: (
            row["slow_variables"]["use_z"], row["slow_variables"]["use_m"]
        ) for row in active
    }
    assert flags["rev21_confirm_z_plus_m"] == (True, True)
    assert flags["rev21_confirm_z_only"] == (True, False)
    assert flags["rev21_confirm_m_only"] == (False, True)


def test_slow_factory_respects_z_only_and_m_only_flags():
    finalist, off = _source_rows()
    del off
    rows = build_confirmation_candidates(finalist, _source_rows()[1])
    substrate = SimpleNamespace(
        n_e=3, n_i=1, params=SimpleNamespace(V_th=-50.0),
        h_e=np.asarray([1.0, 0.0, 1.0]),
    )
    configs = {row["candidate_id"]: row["slow_variables"] for row in rows}
    z_only = make_slow(substrate, configs["rev21_confirm_z_only"])
    m_only = make_slow(substrate, configs["rev21_confirm_m_only"])
    assert z_only.cfg.use_z and not z_only.cfg.use_m
    assert not m_only.cfg.use_z and m_only.cfg.use_m
