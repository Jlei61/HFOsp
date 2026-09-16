import numpy as np
from src.topic5_group_event_state.v039.human_data import merge_intervals,subtract_intervals,exposure,contained_events,phase


def test_missing_measurement_is_removed_from_exposure_and_never_counted_as_silence():
    raw=merge_intervals([[0,10],[10,20],[23,40]])
    support=subtract_intervals(raw,merge_intervals([[5,7],[19,25],[35,41]]))
    np.testing.assert_array_equal(support,[[0,5],[7,19],[25,35]])
    assert exposure(support,0,40)==27
    assert exposure(support,18,28)==4
    assert not contained_events(np.array([4.,18.,24.]),np.array([6.,20.,26.]),support).any()


def test_target_at_phase_boundary_cannot_be_assigned_to_previous_phase():
    bounds=dict(zip(('20pct','60pct','70pct','80pct'),(20.,60.,70.,80.)))
    times=np.array([59.999,60.,69.999,70.,80.])
    np.testing.assert_array_equal(phase(times,bounds),['FIT','INNER','INNER','SELECTION','CLOSED'])


def test_processing_boundary_preserves_event_clock_but_not_unknown_fine_marks(tmp_path):
    import hashlib
    from src.topic5_group_event_state.v039.human_data import restore_clock_only_events
    gpu=tmp_path/'block_gpu.npz'
    det=np.empty(2,dtype=object);det[0]=np.array([[199.95,200.05]]);det[1]=np.array([[199.98,200.08]])
    np.savez(gpu,chns_names=np.array(['A1','A2']),whole_dets=det)
    card=dict(segment_crossing_exclusions=[[199.9,200.1]],selected_contacts=['A1','A2'],block_start=1000.,
              n_group_windows_before_raw_segments=2,input_hashes={str(gpu):hashlib.sha256(gpu.read_bytes()).hexdigest()})
    names=['participation_0','participation_1','delay_0','delay_1','first_group_0','first_group_1','coupled_0_1']
    t,e,m,p=restore_clock_only_events(card,np.array([1100.]),np.array([1100.2]),np.zeros((1,7)),np.ones((1,2)),names)
    np.testing.assert_allclose(t,[1100.,1199.9]);np.testing.assert_allclose(e,[1100.2,1200.1])
    np.testing.assert_array_equal(p[1],[1,1]);np.testing.assert_array_equal(m[1,[0,1,6]],[1,1,1])
    assert np.isnan(m[1,2:6]).all()
    # Full raw coverage is retained, despite an unmeasured within-event order.
    assert contained_events(t,e,np.array([[1000.,1400.]])).all()


def test_individual_event_replay_respects_physical_lookback_and_actual_release():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('real_credit',Path(__file__).resolve().parents[1]/'scripts/audit_group_event_state_v039_human_gradient.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=dict(input_dim=1,event_replay_blocks=[
        dict(release=2.5*3600,times=np.array([1.5,2.2])*3600,cumsum=np.array([[0.],[2.],[5.]])),
        dict(release=11.*3600,times=np.array([9.5])*3600,cumsum=np.array([[0.],[99.]]))])
    tokens,times,rows,dt=module.replay_events(data,dict(anchor=10.*3600),8.)
    # The 1.5h event is physically too old; the 9.5h event is not yet released.
    np.testing.assert_array_equal(tokens,[[3.]])
    np.testing.assert_allclose(times,[2.2*3600]);assert rows.shape==(1,)
    assert abs(dt.sum()-8)<1e-10 and dt.max()<=1/12+1e-10


def test_unknown_fine_marks_do_not_admit_event_cores_into_background():
    from src.topic5_group_event_state.v039.human_data import background_core_mask,background_summary
    times=np.array([0.,180.,600.,630.]);omitted=[[599.9,600.1]]
    np.testing.assert_array_equal(background_core_mask(times,omitted),[True,True,False,True])
    np.testing.assert_array_equal(background_core_mask(times,[]),np.ones(4,bool))
    # A truly zero-event block still has independently measured background.
    assert background_summary(np.ones((4,6,8))).shape==(24,)
