import numpy as np

from scripts.audit_group_event_state_v039_seizure_support import past_query_support,represented_first_onsets


def test_query_eligibility_ignores_future_seizures_but_resets_after_onset():
    times=np.array([30000.,31000.,33000.]);support=np.array([[0.,50000.]])
    full=np.array([[32000.,32100.]])
    assert past_query_support(times[:2],support,full).tolist()==past_query_support(times[:2],support,[]).tolist()
    assert past_query_support(times,support,full).tolist()==[True,True,False]
    truncated=np.array([[0.,31000.]])
    np.testing.assert_array_equal(past_query_support(times[:2],support,full),past_query_support(times[:2],truncated,[]))


def test_one_query_does_not_count_all_clustered_future_onsets():
    times=np.array([100.,250.]);phases=np.array(['FIT','FIT']);onsets=np.array([200.,300.,400.])
    result=represented_first_onsets(times,phases,onsets,{'60pct':350.,'70pct':500.,'80pct':600.},1000)
    assert result['FIT']['onsets']==[200.,300.]
    assert result['FIT']['positive_queries']==2
    assert result['INNER']['onsets']==[]
