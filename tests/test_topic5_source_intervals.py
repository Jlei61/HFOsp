import inspect

import numpy as np
import pytest


def test_the_shared_module_exposes_the_metadata_resolver():
    import dataclasses

    from src.topic5_source_intervals import SourceSegment, build_source_segments

    assert "never event density" in build_source_segments.__doc__
    # SourceSegment is a frozen dataclass owned by topic5_event_innovation_data and only
    # re-exported here, so introspect it as a dataclass rather than as a namedtuple
    names = [field.name for field in dataclasses.fields(SourceSegment)]
    assert names[:3] == ["source_id", "start_time", "stop_time"]


def test_source_segment_is_re_exported_not_redefined():
    from src import topic5_event_innovation_data as owner
    from src import topic5_source_intervals as shared

    assert shared.SourceSegment is owner.SourceSegment


def test_the_v3_0_audit_script_now_imports_rather_than_duplicates():
    from scripts import audit_topic5_event_innovation_v3_0_phase0 as audit
    from src import topic5_source_intervals as shared

    assert audit.build_source_segments is shared.build_source_segments


def test_interval_bounds_do_not_come_from_event_times():
    from src.topic5_source_intervals import build_source_segments

    source = inspect.getsource(build_source_segments)
    for forbidden in ("event_abs_time", "event_time"):
        assert forbidden not in source, f"{forbidden} would reintroduce event-density bounds"


def test_a_record_without_an_inventory_row_fails_loudly():
    from src.topic5_source_intervals import build_source_segments

    with pytest.raises(RuntimeError):
        build_source_segments(
            "epilepsiae_9999",
            np.array(["ghost"]),
            np.array(["ghost"]),
            {"epilepsiae_block_inventory": "results/epilepsiae_block_inventory.csv"},
        )
