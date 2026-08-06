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


@pytest.fixture
def two_row_inventory_csv(tmp_path):
    inventory_path = tmp_path / "block_inventory.csv"
    inventory_path.write_text(
        "subject,block_stem,block_start_epoch,block_end_epoch,recording_id\n"
        "9999,known_block_1,0.0,100.0,rec1\n"
        "9999,known_block_2,100.0,200.0,rec1\n"
    )
    return inventory_path


def test_a_record_without_an_inventory_row_fails_loudly(two_row_inventory_csv):
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
    #
    # fix-round-4 ITEM 3: match= added. Without it, a header-only CSV and a
    # wrong-subject CSV would raise the identical RuntimeError type, so this test alone
    # could not tell "the record genuinely isn't in the inventory" apart from "the
    # inventory wasn't read at all" -- match= at least pins that the message names the
    # specific record that was queried. See test_a_record_present_in_the_inventory_
    # resolves_with_its_own_bounds below for the complementary positive proof that the
    # CSV's rows are not decorative.
    from src.topic5_source_intervals import build_source_segments

    with pytest.raises(RuntimeError, match="no metadata inventory row for ghost"):
        build_source_segments(
            "epilepsiae_9999",
            np.array(["ghost"]),
            np.array(["ghost"]),
            {"epilepsiae_block_inventory": str(two_row_inventory_csv)},
        )


def test_a_record_present_in_the_inventory_resolves_with_its_own_bounds(
    two_row_inventory_csv, monkeypatch
):
    # fix-round-4 ITEM 3 companion: proves the two rows in that same fixture CSV are
    # not decorative. Queries BOTH known records in one call and asserts each resolves
    # its OWN, DIFFERENT interval bounds read straight from the CSV -- ruling out a
    # stub that would return the same bounds (or none) regardless of which record was
    # asked for, which the "absent record" test alone cannot rule out.
    from src import topic5_source_intervals as module

    # _montage_hash reads a real lagPat NPZ from the mounted dataset -- orthogonal to
    # what this test checks (CSV-driven interval resolution) -- so it is monkeypatched
    # to keep this test hermetic like the rest of this file.
    monkeypatch.setattr(
        module,
        "_montage_hash",
        lambda subject, record_name: (f"hash-{record_name}", f"path-{record_name}"),
    )

    segments, records = module.build_source_segments(
        "epilepsiae_9999",
        np.array(["src_a", "src_b"]),
        np.array(["known_block_1", "known_block_2"]),
        {"epilepsiae_block_inventory": str(two_row_inventory_csv)},
    )

    assert len(segments) == 2
    assert segments[0].start_time == pytest.approx(0.0)
    assert segments[0].stop_time == pytest.approx(100.0)
    assert segments[1].start_time == pytest.approx(100.0)
    assert segments[1].stop_time == pytest.approx(200.0)
    assert records[0]["start_time"] == pytest.approx(0.0)
    assert records[0]["stop_time"] == pytest.approx(100.0)
    assert records[1]["start_time"] == pytest.approx(100.0)
    assert records[1]["stop_time"] == pytest.approx(200.0)
