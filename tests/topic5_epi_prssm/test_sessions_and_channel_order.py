"""Session, recorded-interval and channel-order contracts."""
import numpy as np
import pytest

from src.topic5_epi_prssm.event_marks import available_subjects, load_patient
from src.topic5_epi_prssm.sessions import build_sessions

SUBJECT = "epilepsiae_620"


def test_event_silence_is_never_smaller_than_the_metadata_gap():
    for subject in ("epilepsiae_620", "epilepsiae_1073", "yuquan_zhangkexuan"):
        sessions = load_patient(subject).sessions
        gap = sessions.metadata_gap_seconds
        silence = sessions.event_silence_seconds
        finite = np.isfinite(gap) & np.isfinite(silence)
        assert np.all(silence[finite] >= gap[finite] - 1e-6)


def test_sessions_never_span_a_gap_larger_than_the_frozen_join():
    sessions = load_patient(SUBJECT).sessions
    for i in range(1, len(sessions.blocks)):
        if sessions.block_session[i] == sessions.block_session[i - 1]:
            assert sessions.metadata_gap_seconds[i] <= sessions.join_seconds + 1e-6


def test_recorded_coverage_never_comes_from_event_density():
    """A block's recorded interval must be at least as long as the span its
    events occupy -- true for a metadata interval, not for one reconstructed
    from the first and last detected event."""
    sessions = load_patient(SUBJECT).sessions
    for block in sessions.blocks:
        span = block.last_event_time - block.first_event_time
        assert (block.stop_epoch - block.start_epoch) >= span - 1e-6


def test_every_event_has_a_session_and_the_index_is_monotone():
    for subject in available_subjects()[:6]:
        events = load_patient(subject)
        index = events.sessions.session_index
        assert len(index) == events.n_events
        assert np.all(np.diff(index) >= 0)


def test_real_delta_t_is_nan_exactly_at_session_openings():
    events = load_patient(SUBJECT)
    assert np.all(np.isnan(events.delta_t[events.session_opening]))
    assert np.all(np.isfinite(events.delta_t[~events.session_opening]))


def test_an_unknown_block_fails_closed():
    events = load_patient(SUBJECT)
    record_names = np.array(["not_a_real_block"] * events.n_events)
    with pytest.raises(RuntimeError):
        build_sessions(SUBJECT, events.event_time, record_names)


def test_channel_order_is_one_canonical_order_per_patient():
    events = load_patient(SUBJECT)
    assert len(events.contact_names) == events.n_contacts
    assert len(set(events.contact_names.tolist())) == events.n_contacts
    assert events.contact_coords.shape == (events.n_contacts, 3)
    assert events.contact_features.shape[0] == events.n_contacts
