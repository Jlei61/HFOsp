import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/snn_engine'))
from params import Params
from src.topic4_xy_search import (audit_geometry,field_descriptor,geometry_allowed,
    load_corrected_network,pareto_indices,sha,sobol_xy_candidates,verify_runtime_snapshot)
from src.topic4_xy_search import connectivity_config


@pytest.fixture
def positions():
    x=(np.arange(180)+.5)*20/180
    xx,yy=np.meshgrid(x,x)
    return np.c_[xx.ravel(),yy.ravel()]


def test_xy_search_moves_both_centers_without_axis_constraint(positions):
    rows=sobol_xy_candidates(positions,n_per_domain=8,seed=17)
    assert len(rows)==16
    assert rows==sobol_xy_candidates(positions,n_per_domain=8,seed=17)
    for domain in ('whole_sheet','interior'):
        block=[r for r in rows if r['domain']==domain]
        assert len({tuple(np.ravel(r['node_field']['centers_mm'])) for r in block})==8
        assert np.ptp([r['geometry']['center_line_angle_deg'] for r in block])>60
        assert all(r['node_field']['target_count']==1499 for r in block)
        assert all(r['geometry']['minimum_clearance_mm']>=1.65 for r in block) if domain=='interior' else True


def test_boundary_domain_keeps_edge_reference_only_as_whole_sheet(positions):
    edge=[[1.5377342579886317,1.2264179880730808],[18.606612137053162,2.417713414411992]]
    assert geometry_allowed(edge,positions,domain='whole_sheet')
    assert not geometry_allowed(edge,positions,domain='interior')
    assert audit_geometry(positions,edge,1499)['minimum_clearance_mm']<0
    assert geometry_allowed([[1,1],[19,19]],positions,domain='whole_sheet')


def test_core_identity_is_invariant_to_label_swap():
    centers=[[4,13],[16,7]]
    assert field_descriptor(centers)==field_descriptor(centers[::-1])
    assert field_descriptor(centers,1499)!=field_descriptor(centers,1129)


def test_corrected_cache_rejects_legacy_sampler_and_parameter_drift(tmp_path):
    p=Params(seed=2511);cfg=connectivity_config(p,-22.8,2,git_commit='recorded')
    path=tmp_path/'graph.pkl'
    def write(config):
        path.write_bytes(pickle.dumps({'config':config,'net':{},'NE':32000,'NI':8000}))
        return {'path':str(path),'sha256':sha(path)}
    record=write(cfg)
    assert load_corrected_network(record,p,-22.8,2)[1:3]==(32000,8000)
    with pytest.raises(RuntimeError,match='parameters'):
        load_corrected_network(record,Params(seed=2512),-22.8,2)
    legacy={k:v for k,v in cfg.items() if k!='partner_sampler_version'}
    with pytest.raises(RuntimeError,match='sampler'):
        load_corrected_network(write(legacy),p,-22.8,2)


def test_runtime_snapshot_rejects_changed_code_and_inputs(tmp_path):
    names=['src/topic4_xy_search.py','scripts/run_topic4_rev12_node_worker.py']
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source')
    config=tmp_path/'config.json';config.write_text('{}')
    manifest=tmp_path/'candidates.json';manifest.write_text('{}')
    snap=tmp_path/'snapshot.json'
    snap.write_text(json.dumps({'source_hashes':{n:sha(tmp_path/n) for n in names},
        'input_hashes':{str(p.resolve()):sha(p) for p in (config,manifest)}}))
    kwargs={'required_paths':names,'config_path':config,'candidate_manifest_path':manifest}
    assert verify_runtime_snapshot(snap,tmp_path,**kwargs)['verified']
    config.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='input changed'):verify_runtime_snapshot(snap,tmp_path,**kwargs)
    config.write_text('{}');(tmp_path/names[0]).write_text('changed')
    with pytest.raises(RuntimeError,match='source changed'):verify_runtime_snapshot(snap,tmp_path,**kwargs)


def test_pareto_keeps_tradeoffs_without_arbitrary_unit_weights():
    assert pareto_indices([[1,4],[2,2],[4,1],[3,3]])==[0,1,2]
