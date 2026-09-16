from types import SimpleNamespace
import numpy as np
from src.topic4_streaming_spike_readout import StreamingSpikes
from src.sef_hfo_snn_adapter import snn_event_envelope
from src.topic4_node_dualmode import sheet_activity_movie


def test_streaming_envelope_and_native_movie_match_dense_with_partial_last_bin():
    rng=np.random.default_rng(231);pos=rng.uniform(0,20,(180,2));montage=SimpleNamespace(contacts=pos[[4,19,87]])
    spikes=rng.random((2003,len(pos)))<.08
    rec=StreamingSpikes(len(spikes),len(pos),pos,montage,.1)
    for t,row in enumerate(spikes):rec[t]=row
    dense,dt,_=snn_event_envelope(spikes,pos,montage,.1)
    online,dt2,_=rec.envelope()
    np.testing.assert_allclose(online,dense,rtol=2e-13,atol=2e-14)
    np.testing.assert_array_equal(rec.native()['activity_counts'],sheet_activity_movie(spikes,pos,dt_ms=.1,frame_ms=2,bin_mm=1,sheet_mm=20)['activity_counts'])
    assert dt==dt2


def test_early_stop_prefix_has_correct_clock_and_partial_movie():
    pos=np.array([[1.,1.],[2.,2.]])
    rec=StreamingSpikes(1000,2,pos,SimpleNamespace(contacts=pos),.1)
    for i in range(401):rec[i]=np.array([i%3==0,False])
    prefix=rec[:401]
    assert len(prefix)==401 and prefix.native()['activity_counts'].shape==(21,20,20)
    assert prefix.envelope()[0].shape==(2,20)
