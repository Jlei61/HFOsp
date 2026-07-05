from scripts._topic5_v3c_io import load_soz, axis_soz_join, V3C_SUBJECTS


def test_v3c_subject_lists():
    assert "epilepsiae_1146" in V3C_SUBJECTS["broad"]
    assert "epilepsiae_442" in V3C_SUBJECTS["narrow"]
    assert "epilepsiae_442" not in V3C_SUBJECTS["broad"]     # no broad cache (spec §3.3)


def test_load_soz_epilepsiae():
    s = load_soz("epilepsiae", "1146")
    assert "ICL1" in s and len(s) == 14
    assert load_soz("epilepsiae", "9999") == []              # absent -> empty, not crash


def test_axis_soz_join_intersects_pool():
    cls = {"is_axis": ["a", "b"], "all_clean": ["a", "b", "c"]}
    j = axis_soz_join(cls, ["a", "z"])                        # z not in pool -> dropped
    assert j["soz_in_pool"] == ["a"] and j["n_covered"] == 1 and j["coverage"] == 1.0
