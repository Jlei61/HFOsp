import pytest

from src.topic5_v3_mode_transition import load_v3_config
from scripts._topic5_v3c_io import load_soz, axis_soz_join, V3C_SUBJECTS, extract_latency_matrix, load_axis_coords


def test_v3c_subject_lists():
    # cohort is DATA-DERIVED: cache ∩ propagation-axis pool ∩ clinical SOZ (yuquan held)
    b, n = V3C_SUBJECTS["broad"], V3C_SUBJECTS["narrow"]
    assert "epilepsiae_1146" in b and "epilepsiae_1084" in b   # 1084 recovered by auto-derive
    assert "epilepsiae_442" in n and "epilepsiae_442" not in b  # narrow-only (no broad pool)
    assert len(b) == 11 and len(n) == 14                        # was hard-coded 7/5 (under-count)
    assert all(s.startswith("epilepsiae_") for s in b + n)      # yuquan held (participation absent)


def test_load_soz_epilepsiae():
    s = load_soz("epilepsiae", "1146")
    assert "ICL1" in s and len(s) == 14
    assert load_soz("epilepsiae", "9999") == []              # absent -> empty, not crash


def test_axis_soz_join_intersects_pool():
    cls = {"is_axis": ["a", "b"], "all_clean": ["a", "b", "c"]}
    j = axis_soz_join(cls, ["a", "z"])                        # z not in pool -> dropped
    assert j["soz_in_pool"] == ["a"] and j["n_covered"] == 1 and j["coverage"] == 1.0


@pytest.mark.integration
def test_extract_latency_matrix_shapes():
    cfg = load_v3_config()
    names = ["HL1", "HL2", "HL3"]                             # 3 real cache names for 139
    mats = extract_latency_matrix("epilepsiae_139", cfg, names, thresholds=[2.0, 1.5])
    assert len(mats) >= 1
    m0 = mats[0]
    assert set(m0["kinds"].keys()) == {2.0, 1.5}
    assert len(m0["kinds"][2.0]) == 3 and len(m0["secs"][2.0]) == 3
    assert all(k in ("finite", "t0", "censored") for k in m0["kinds"][2.0])


@pytest.mark.integration
def test_extract_latency_matrix_fails_closed_on_missing_contact():
    # P1-4: a contact absent from the cache MUST raise, never silently shift
    # the row->name alignment (which would misassign one contact's latency).
    cfg = load_v3_config()
    with pytest.raises(ValueError, match="absent from cache"):
        extract_latency_matrix("epilepsiae_139", cfg, ["HL1", "NOT_A_REAL_CONTACT"], thresholds=[2.0])


def test_load_axis_coords_missing_returns_empty(monkeypatch):
    import scripts._topic5_v3c_io as io
    def boom(*a, **k):
        raise FileNotFoundError("no MRI")
    monkeypatch.setattr(io, "load_subject_coords", boom)
    assert load_axis_coords("epilepsiae", "999", ["A1"]) == {}
