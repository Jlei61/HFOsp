"""Task 1 (spec rev3.1 §2.1 / §1.2): canonical-config + dynamic-state inventory GATE.

The inventory is only worth anything if it FAILS CLOSED when the engine grows a new mutable
variable. So the tests do not check a hand-written list against itself: they re-derive the set of
mutated names from the ENGINE SOURCE (ast) and require every one of them to be classified exactly
once. A new `self.foo = ...` in SpatialSlowField.step, or a new `bar = ...` in simulate_kick, breaks
the audit until someone classifies it as simulator state / derived / observer / temporary.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.topic4_zm_fork_state as FS  # noqa: E402


# ---------------------------------------------------------------- inventory schema
REQUIRED_ROW_FIELDS = {"name", "category", "shape", "dtype", "time_scale", "role",
                       "dt_dependent", "snapshot", "freeze_semantics", "current_effect"}


def test_inventory_rows_have_all_required_fields():
    rows = FS.build_state_inventory()
    assert rows, "inventory must not be empty"
    for r in rows:
        missing = REQUIRED_ROW_FIELDS - set(r)
        assert not missing, f"{r.get('name')} missing inventory fields {missing}"
        assert r["role"] in ("simulator", "observer")
        assert r["current_effect"] in ("direct", "indirect", "none")
        assert r["freeze_semantics"] in FS.FREEZE_SEMANTICS


def test_names_are_unique():
    names = [r["name"] for r in FS.build_state_inventory()]
    assert len(names) == len(set(names)), "each state appears exactly once"


def test_observer_rows_cannot_claim_membrane_current_effect():
    """An observer-only field claiming to change the membrane current is a contradiction: the audit
    must reject it rather than silently record it."""
    bad = dict(FS.build_state_inventory()[0])
    bad.update(name="fake_observer", role="observer", current_effect="direct")
    with pytest.raises(ValueError, match="observer"):
        FS.validate_inventory([bad])


def test_simulator_state_must_be_snapshotted_or_declared_derived():
    bad = dict(FS.build_state_inventory()[0])
    bad.update(name="fake_state", role="simulator", current_effect="direct", snapshot=False,
               freeze_semantics="dynamic")
    with pytest.raises(ValueError, match="snapshot"):
        FS.validate_inventory([bad])


# ---------------------------------------------------------------- fail-closed source audit
def test_every_simulate_kick_mutable_is_classified():
    unknown = FS.unclassified_engine_names()
    assert unknown == {}, f"unclassified engine mutables (fail closed): {unknown}"


def test_unknown_current_affecting_name_fails_closed():
    """Simulate the engine growing a new variable: the audit must report it, not pass."""
    unknown = FS.unclassified_engine_names(
        extra_names={"simulate_kick": {"brand_new_membrane_term"}})
    assert unknown, "a new engine mutable must be reported"
    assert "brand_new_membrane_term" in unknown["simulate_kick"]


def test_every_simulator_state_name_has_a_snapshot_row():
    """The classification buckets and the inventory must agree: anything the audit calls simulator
    state must have a snapshot row, otherwise the audit passes while the snapshot loses it."""
    rows = {r["name"]: r for r in FS.build_state_inventory()}
    for n in FS._KICK_STATE:
        if n == "t":
            continue  # step index has its own row, checked below
        assert n in rows and rows[n]["snapshot"], f"simulate_kick state {n!r} has no snapshot row"
    for n in FS._SLOW_STATE:
        assert f"slow.{n}" in rows and rows[f"slow.{n}"]["snapshot"], f"slow state {n!r} unrowed"
    assert rows["t"]["snapshot"] and rows["rng_state"]["snapshot"]


def test_audit_verdict_blocks_on_unknown_state():
    ok = FS.audit_dynamic_state()
    assert ok["status"] == "ok", ok
    bad = FS.audit_dynamic_state(extra_names={"SpatialSlowField.step": {"self_new_field"}})
    assert bad["status"] == "blocked_state_inventory"


# ---------------------------------------------------------------- canonical config
def test_canonical_config_is_resolved_from_the_real_builder():
    cfg = FS.build_canonical_config(seed=1, I_th_EI=1.28)
    slow = cfg["slow_field"]
    # locked Z/M working point zA_q75_tz5000__mA0p001_tau500 (spec §1.1)
    assert (slow["use_z"], slow["use_m"], slow["use_qI"], slow["use_gK"]) == (True, True, False, False)
    assert slow["tau_z"] == 5000.0 and slow["tau_adp"] == 500.0 and slow["eta_m"] == 0.001
    assert slow["use_H"] is False and slow["use_persist"] is False and slow["use_A"] is False
    assert slow["use_SG"] is True and slow["alpha_G"] == 16.0
    # substrate
    assert cfg["substrate"]["L"] == 20.0 and cfg["substrate"]["density"] == 100.0
    assert cfg["params"]["dt"] == 0.1 and cfg["params"]["seed"] == 1
    assert cfg["lockpoint"] == "zA_q75_tz5000__mA0p001_tau500"
    assert cfg["I_th_EI"] == 1.28


def test_canonical_config_carries_live_engine_shas():
    cfg = FS.build_canonical_config(seed=1, I_th_EI=1.28)
    for rel, sha in cfg["engine_sha256"].items():
        assert sha == FS.sha256_file(os.path.join(FS.ROOT, rel)), f"stale SHA for {rel}"
    for must in ("src/snn_engine/kick_probe.py", "src/snn_engine/slow_field.py",
                 "src/snn_engine/connectivity_rot.py", "src/snn_engine/lfp.py"):
        assert must in cfg["engine_sha256"]


def test_config_sha_changes_with_any_locked_field():
    a = FS.build_canonical_config(seed=1, I_th_EI=1.28)
    b = FS.build_canonical_config(seed=3, I_th_EI=1.28)
    c = FS.build_canonical_config(seed=1, I_th_EI=1.2800001)
    assert FS.config_sha(a) != FS.config_sha(b)
    assert FS.config_sha(a) != FS.config_sha(c)
    assert FS.config_sha(a) == FS.config_sha(FS.build_canonical_config(seed=1, I_th_EI=1.28))


def test_config_is_json_serialisable_and_flat_enough_to_diff():
    cfg = FS.build_canonical_config(seed=1, I_th_EI=1.28)
    json.loads(json.dumps(cfg))  # raises on numpy scalars / non-serialisable objects
