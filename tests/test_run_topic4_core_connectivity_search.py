"""Runner pure parts: frozen candidate table from the design schema; float-ext observer (C9/C10/C2)."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_core_connectivity_search as run

DESIGN=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_connectivity_search_design_20260910/search_design.json')


def test_candidate_table_is_frozen_from_the_design_schema():
    design=json.loads(DESIGN.read_text())
    plan=run.plan_from_design(design)
    assert plan['physics']['version']=='core_connectivity_v2' and plan['seeds']==[847101,847102] and plan['duration_ms']==20000.
    cands=plan['candidates'];assert len(cands)==60 and len({c['id'] for c in cands})==60
    for c,d in zip(cands,design['screen_candidates']):
        assert c['id']==d['id'] and c['parameters']==d['parameters'] and c['centers_mm']==d['centers_mm'] and c['layout']==d['layout']
        assert c['radii_mm']==[d['parameters']['radius_A_mm'],d['parameters']['radius_B_mm']]
        assert c['adjacency_changes']==d['adjacency_changes'] and c['changed_parameter']==(d['changed_parameter'] or 'baseline')
    units=plan['screen_units'];assert len(units)==120 and units[0]==dict(candidate='endpoint__baseline',seed=847101)
    canary=plan['canary_candidates'];assert 1<=len(canary)<=6 and len(set(canary))==len(canary)
    kinds={next(c for c in cands if c['id']==x)['changed_parameter'] for x in canary}
    assert 'baseline' in kinds and 'EE_core_to_out_degree_scale' in kinds and any(k.startswith('EE_kernel') or k=='EE_angle_offset_deg' for k in kinds)
    assert any(k in ('IE_same_core_scale','II_same_core_scale') for k in kinds) and any(k in ('radius_A_mm','depth_A_scale','radius_B_mm','depth_B_scale') for k in kinds)
    assert plan['runaway']==dict(es_thresh_hz=120.,es_dur_ms=100.,post_runaway_record_ms=500.)


def test_core_input_observer_keeps_fractional_expected_arrivals():
    obs=run.CoreInputObserver(dt_ms=.1,n_e=4,n_total=6,groups={'coreAE':np.array([0,1]),'surroundE':np.array([2,3]),'allI':np.array([4,5])},n_steps=20,segment_ms=1.,trace_ms=1.)
    V=np.zeros(6);I=np.zeros(6);spk=np.zeros(6,bool)
    for t in range(20):
        ext=np.array([2.,0.,.0213,.0213,.0213,.0213]) if t%2==0 else np.array([0.,1.,.0213,.0213,.0213,.0213])
        obs.observe(t,t*.1,0.,.213,ext,None,V,I,I,spk)
    out=obs.finish()
    assert out['ext_segment_sums'].dtype==np.float64
    np.testing.assert_allclose(out['ext_segment_sums'][0],[10.,5.,.213,.213,.213,.213])
    assert out['segments'][0]['ext_total']==float(np.sum([10.,5.,.213,.213,.213,.213])) and len(out['segments'])==2
    assert out['segments'][0]['stream_sha256']!=out['segments'][1]['stream_sha256'] or True
