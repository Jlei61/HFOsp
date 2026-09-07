import numpy as np
from src.topic4_joint_xy_kernel_objective import mean_noise_corrected_distance
from scripts.run_topic4_joint_xy_kernel_search import select_racers, assess


def test_ranking_diagonal_correction_matches_off_diagonal_statistic():
    x=np.array([[1.,2.],[3.,-.5],[-1.,4.]])
    mu=np.array([.2,1.])
    exact=((x@x.T).sum()-(x*x).sum())/(len(x)*(len(x)-1))-2*x.mean(axis=0)@mu+mu@mu
    assert np.isclose(mean_noise_corrected_distance(x,mu),exact)
    assert mean_noise_corrected_distance(x[:1],mu) is None


def candidate(name,centers,joint,rank,timing,clearance=1):
    return {'candidate_id':name,'candidate':{'node_field':{'centers_mm':centers}},
            'units':[{'geometry':{'minimum_clearance_mm':clearance}}]*2,'explorable':True,
            'exploration_score':joint,'kernel_distances':{'rank_space':rank,'timing_space':timing}}


def test_racing_includes_manual_control_and_component_optima_without_angle_target():
    rows=[candidate('control_historical_matched',[[4,9],[16,4]],9,9,9),
          candidate('rank_best',[[5,5],[5,13]],3,0,3),
          candidate('timing_best',[[6,14],[14,14]],2,3,0),
          candidate('joint_best',[[12,5],[16,11]],0,2,2),
          candidate('cut_off',[[1,1],[18,1]],-5,-5,-5,-1)]
    chosen=select_racers(rows,0,{'search':{'racers_per_round':4}})
    assert {r['candidate_id'] for r in chosen}=={'control_historical_matched','rank_best','timing_best','joint_best'}


def test_low_event_candidate_cannot_pass_even_with_perfect_kernel(monkeypatch):
    from scripts import run_topic4_joint_xy_kernel_search as controller
    monkeypatch.setattr(controller.v1,'is_qualified',lambda *a,**k:{'checks':{'old':True},'pass':True})
    r={'n_events':20,'kernel_distances':dict(joint=0,support=0,rank_space=0,timing_space=0)}
    cal={'samples':{'64':{'kernel_q95':{k:.1 for k in r['kernel_distances']}}}}
    result=assess(r,cal,{'search':{'minimum_pool_events':64}})
    assert not result['pass'] and not result['checks']['sufficient_events']
    r['n_events']=64
    assert assess(r,cal,{'search':{'minimum_pool_events':64}})['pass']
    r['kernel_distances']['timing_space']=.2
    assert not assess(r,cal,{'search':{'minimum_pool_events':64}})['pass']


def test_real_training_metric_contract_handles_vector_metadata():
    from scripts import run_topic4_joint_xy_adaptive as v1
    from src.topic4_joint_xy_kernel_objective import KernelObjective
    obj=KernelObjective(v1,v1.OUT,v1.OUT/'kernel_qualification/kernel_contract.json')
    result=obj.metrics(obj.patient[:64])
    assert result['n_events']==64 and np.isfinite(result['joint_distance'])
    assert set(result['component_status'])=={'D_support','D_order','D_lag'}
    empty=obj.metrics(obj.patient[:0])
    assert empty['joint_distance'] is None and empty['exploration_score']==2
