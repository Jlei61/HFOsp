import pytest

from src.sef_hfo_snn_engine_guard import record_versions, assert_versions


def test_record_then_assert_roundtrip(tmp_path):
    f = tmp_path / "engine_a.py"
    f.write_text("x = 1\n")
    rec = record_versions([str(f)])
    assert_versions(rec)                       # matches -> no raise
    f.write_text("x = 2\n")                     # drift
    with pytest.raises(RuntimeError):
        assert_versions(rec)                    # checksum mismatch -> loud fail
