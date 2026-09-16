import numpy as np
from scripts.audit_topic4_xy_raw_propagation_video import comparison, window_frames


def test_common_time_origin_does_not_change_contact_pair_lags():
    t=np.array([0.,10.,40.,np.nan])
    m=comparison(t,t+703.)
    assert m['order_discordance']==0
    assert m['pair_lag_mae_ms']==0
    assert m['common_pairs']==3


def test_reverse_order_is_not_hidden_by_missing_contacts_or_ties():
    m=comparison([0.,10.,30.,np.nan],[30.,20.,0.,45.])
    assert m['order_discordance']==1
    assert m['participation_mismatch']==1
    assert m['common_pairs']==3
    assert comparison([0.,1.],[1.,0.])['order_discordance'] is None


def test_same_direction_with_stretched_delays_is_still_a_timing_error():
    m=comparison([0.,10.,30.],[0.,20.,60.])
    assert m['order_discordance']==0
    assert m['pair_lag_mae_ms']==20


def test_float_window_edges_keep_last_native_frame():
    frames=window_frames(4000,2.,683.9999999999999,933.9999999999999)
    assert len(frames)==125
    assert (frames[0],frames[-1])==(342,466)
