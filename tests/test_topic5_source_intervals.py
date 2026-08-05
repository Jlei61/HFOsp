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


def test_a_record_without_an_inventory_row_fails_loudly(tmp_path):
    # rev3 R3-E: hermetic rewrite. The original version read
    # results/epilepsiae_block_inventory.csv, a data artifact that is not tracked in
    # git -- on a clean checkout that file does not exist and the resolver raises
    # FileNotFoundError instead of RuntimeError, so pytest.raises(RuntimeError) does not
    # match and the test fails. Building a two-row inventory in tmp_path removes that
    # dependency on the environment entirely.
    #
    # _inventory_rows resolves the configured path as `ROOT / config[...]`; when
    # config[...] is already absolute, pathlib's `/` operator discards ROOT and returns
    # the absolute path unchanged, so pointing the config at the tmp_path CSV directly
    # (no monkeypatching of ROOT) is sufficient to make this hermetic.
    from src.topic5_source_intervals import build_source_segments

    inventory_path = tmp_path / "block_inventory.csv"
    inventory_path.write_text(
        "subject,block_stem,block_start_epoch,block_end_epoch,recording_id\n"
        "9999,known_block_1,0.0,100.0,rec1\n"
        "9999,known_block_2,100.0,200.0,rec1\n"
    )

    with pytest.raises(RuntimeError):
        build_source_segments(
            "epilepsiae_9999",
            np.array(["ghost"]),
            np.array(["ghost"]),
            {"epilepsiae_block_inventory": str(inventory_path)},
        )
