"""core_connectivity_v2 contract tests (design 2026-09-10 §2-§5; checklist C1-C7)."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from params import Params
from model import build_network
from src.topic4_core_field import sample_core_quantiles,core_thresholds
from src import topic4_core_connectivity_v2 as v2

QUANT=dict(quantile_seed=20260806,core_mean=17.5,core_std=1.,v_base=18.)


def small_net(seed=57):
    p=Params(L=4.,density=400.,C_EE=40,C_IE=40,C_EI=16,C_II=16,T=50.,dt=.1,seed=seed)
    net=build_network(p,verbose=False)
    return p,net


def edges(bins,n_e=None):
    rows=[];cols=[];data=[];steps=[]
    for d,m in enumerate(bins):
        c=m.tocoo();rows.append(c.row);cols.append(c.col);data.append(c.data);steps.append(np.full(c.nnz,d))
    return np.concatenate(rows),np.concatenate(cols),np.concatenate(data),np.concatenate(steps)


def edge_dict(bins):
    r,c,d,s=edges(bins)
    return {(int(a),int(b)):(float(w),int(t)) for a,b,w,t in zip(r,c,d,s)}


# ---------------- C1 threshold field ----------------
def test_threshold_field_lowers_only_inside_cores_and_keeps_latent_identity():
    rng=np.random.default_rng(3);pos=rng.uniform(0,20,(3000,2));n_total=3600
    centers=[[4.,9.],[16.,4.]];radii=[2.,2.]
    out=v2.threshold_field(pos,centers,radii,[1.,1.],n_total=n_total,**QUANT)
    vth=out['vtheta'];assert vth.shape==(n_total,)
    d_in=np.linalg.norm(pos[:,None]-np.asarray(centers)[None],axis=2)
    inside=(d_in<=np.asarray(radii)[None]).any(1)
    latent=np.maximum(18-core_thresholds(sample_core_quantiles(3000,QUANT['quantile_seed']),17.5,1.,11.),0)
    np.testing.assert_array_equal(vth[:3000][~inside],18.)
    np.testing.assert_allclose(vth[:3000][inside],18-latent[inside])
    np.testing.assert_array_equal(vth[3000:],18.)
    assert np.all(vth<=18.) and out['audit']['n_raised']==0
    assert out['core_index'].shape==(3000,) and set(np.unique(out['core_index'])).issubset({-1,0,1})
    assert out['audit']['members']==[int((out['core_index']==0).sum()),int((out['core_index']==1).sum())]
    # Swapping the core order must not resample: same neurons, same thresholds.
    swapped=v2.threshold_field(pos,centers[::-1],radii[::-1],[1.,1.],n_total=n_total,**QUANT)
    np.testing.assert_array_equal(swapped['vtheta'],vth)
    # Changing one radius keeps every retained member's threshold byte-identical.
    grown=v2.threshold_field(pos,centers,[3.,2.],[1.,1.],n_total=n_total,**QUANT)
    keep=inside
    np.testing.assert_array_equal(grown['vtheta'][:3000][keep],vth[:3000][keep])
    assert (grown['core_index']>=0).sum()>inside.sum()


def test_threshold_field_depth_scale_and_floor_are_per_core_and_recorded():
    rng=np.random.default_rng(4);pos=rng.uniform(0,20,(3000,2))
    centers=[[4.,9.],[16.,4.]];radii=[2.,2.]
    base=v2.threshold_field(pos,centers,radii,[1.,1.],n_total=3600,**QUANT)
    deep=v2.threshold_field(pos,centers,radii,[3.,1.],n_total=3600,**QUANT)
    a=base['core_index']==0;b=base['core_index']==1
    np.testing.assert_array_equal(deep['vtheta'][:3000][b],base['vtheta'][:3000][b])
    expected=np.maximum(18-3*(18-base['vtheta'][:3000][a]),11.)
    np.testing.assert_allclose(deep['vtheta'][:3000][a],expected)
    assert deep['audit']['floor_clipped_count']==int(np.sum(18-3*(18-base['vtheta'][:3000][a])<11))
    assert deep['audit']['floor_clipped_count']>0 and deep['audit']['floor_clipped_total_mV']>0
    assert deep['audit']['total_lowering_mV']>base['audit']['total_lowering_mV']
    assert deep['audit']['total_lowering_per_core_mV'][1]==pytest.approx(base['audit']['total_lowering_per_core_mV'][1])


def test_threshold_field_overlap_takes_stronger_lowering_not_sum():
    rng=np.random.default_rng(5);pos=rng.uniform(0,10,(2000,2))
    centers=[[5.,5.],[6.,5.]];radii=[2.,2.]
    out=v2.threshold_field(pos,centers,radii,[1.,2.],n_total=2400,**QUANT)
    only_a=v2.threshold_field(pos,centers,[2.,0.],[1.,2.],n_total=2400,**QUANT)['vtheta']
    only_b=v2.threshold_field(pos,centers,[0.,2.],[1.,2.],n_total=2400,**QUANT)['vtheta']
    np.testing.assert_array_equal(out['vtheta'],np.minimum(only_a,only_b))
    assert out['audit']['overlap_members']>0


# ---------------- C4 block-wise pathway scaling ----------------
def membership(net):
    n_e=net['NE'];pos=net['pos']
    e_core=v2.core_index_for(pos[:n_e],[[1.,1.],[3.,3.]],[.8,.8]);i_core=v2.core_index_for(pos[n_e:],[[1.,1.],[3.,3.]],[.8,.8])
    return e_core,i_core


@pytest.mark.parametrize('name,pathway',[('EE_same_core_scale','ampa'),('EE_core_to_out_scale','ampa'),('EE_out_to_out_scale','ampa'),
                                         ('EI_same_core_scale','ampa'),('IE_same_core_scale','gaba'),('II_same_core_scale','gaba')])
def test_scale_pathway_blocks_touches_only_its_named_block(name,pathway):
    p,net=small_net();n_e=net['NE'];e_core,i_core=membership(net)
    factors={k:1. for k in v2.WEIGHT_FACTORS};factors[name]=1.3
    ampa,gaba,audit=v2.scale_pathway_blocks(net['ampa_by_delay'],net['gaba_by_delay'],n_e,e_core,i_core,factors)
    old=edge_dict(net['ampa_by_delay'] if pathway=='ampa' else net['gaba_by_delay']);new=edge_dict(ampa if pathway=='ampa' else gaba)
    assert old.keys()==new.keys()
    src_core=(e_core if pathway=='ampa' else i_core)
    changed=0
    for (row,col),(w,step) in old.items():
        w2,step2=new[(row,col)];assert step2==step
        target_is_e=row<n_e;tc=e_core[row] if target_is_e else i_core[row-n_e];sc=src_core[col]
        expect={'EE_same_core_scale':target_is_e and tc>=0 and sc==tc,
                'EE_core_to_out_scale':target_is_e and tc<0 and sc>=0,
                'EE_out_to_out_scale':target_is_e and tc<0 and sc<0,
                'EI_same_core_scale':(not target_is_e) and tc>=0 and sc==tc,
                'IE_same_core_scale':target_is_e and tc>=0 and sc==tc,
                'II_same_core_scale':(not target_is_e) and tc>=0 and sc==tc}[name]
        assert w2==pytest.approx(w*1.3 if expect else w);changed+=expect
    assert changed>0 and audit['blocks'][name]['n_edges']==changed
    other=edge_dict(gaba if pathway=='ampa' else ampa);ref=edge_dict(net['gaba_by_delay'] if pathway=='ampa' else net['ampa_by_delay'])
    assert other==ref


def test_scale_pathway_blocks_unit_factors_are_exact_noop():
    p,net=small_net();n_e=net['NE'];e_core,i_core=membership(net)
    ampa,gaba,audit=v2.scale_pathway_blocks(net['ampa_by_delay'],net['gaba_by_delay'],n_e,e_core,i_core,{k:1. for k in v2.WEIGHT_FACTORS})
    assert edge_dict(ampa)==edge_dict(net['ampa_by_delay']) and edge_dict(gaba)==edge_dict(net['gaba_by_delay'])
    assert audit['exact_noop']


# ---------------- C5 core-to-out in-degree ----------------
def test_degree_scale_changes_only_core_to_out_edges_deterministically():
    p,net=small_net();n_e=net['NE'];e_core,i_core=membership(net);pos=net['pos']
    kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.)
    base=edge_dict(net['ampa_by_delay'])
    for factor in [1.5,.5]:
        ampa,audit=v2.apply_core_to_out_degree(net['ampa_by_delay'],pos,n_e,e_core,factor,topology_seed=2511,kernel=kernel,p=p)
        again,_=v2.apply_core_to_out_degree(net['ampa_by_delay'],pos,n_e,e_core,factor,topology_seed=2511,kernel=kernel,p=p)
        new=edge_dict(ampa);assert new==edge_dict(again)
        rows,cols,_,_=edges(net['ampa_by_delay'])
        block=(rows<n_e)&(e_core[np.minimum(rows,n_e-1)]<0)&(e_core[cols]>=0)&(rows<n_e)
        base_count=np.bincount(rows[block],minlength=n_e)
        rows2,cols2,data2,steps2=edges(ampa)
        block2=(rows2<n_e)&(e_core[np.minimum(rows2,n_e-1)]<0)&(e_core[cols2]>=0)
        new_count=np.bincount(rows2[block2],minlength=n_e)
        outside=np.flatnonzero(e_core<0)
        for t in outside:
            n0=base_count[t];assert new_count[t]==(int(np.floor(n0*factor+.5)) if n0>0 else 0)
        # untouched blocks identical (weights, delays, membership)
        for key,val in base.items():
            row,col=key
            if row<n_e and e_core[row]<0 and e_core[col]>=0:continue
            assert new[key]==val
        assert not any(r==c for r,c in zip(rows2,cols2))
        assert len(set(zip(rows2.tolist(),cols2.tolist())))==len(rows2)
        # added edges: delay by distance, weight equals the target's baseline EE weight
        for (row,col),(w,step) in new.items():
            if (row,col) in base:continue
            d=np.linalg.norm(pos[col]-pos[row]);assert step==max(1,int(round((p.tau0+d/p.v_axon)/p.delay_dt)))
            ref=[v[0] for k,v in base.items() if k[0]==row and k[1]<n_e]
            assert w==pytest.approx(np.mean(ref))
        assert audit['n_zero_baseline_targets']==int(np.sum(base_count[outside]==0))
        assert audit['factor']==factor and audit['n_edges_added' if factor>1 else 'n_edges_removed']>0


# ---------------- C6 EE kernel resampling ----------------
def test_resample_ee_kernel_keeps_indegree_other_pathways_and_delay_rule():
    p,net=small_net();n_e=net['NE'];pos=net['pos']
    kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.,perp_scale=1.,parallel_scale=1.)
    ampa,audit=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=2511,kernel=kernel,p=p)
    again,_=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=2511,kernel=kernel,p=p)
    assert edge_dict(ampa)==edge_dict(again)
    rows,cols,data,steps=edges(ampa);rows0,cols0,data0,steps0=edges(net['ampa_by_delay'])
    ee=rows<n_e;ee0=rows0<n_e
    np.testing.assert_array_equal(np.bincount(rows[ee],minlength=n_e),np.bincount(rows0[ee0],minlength=n_e))
    assert not np.any(rows[ee]==cols[ee]) and len(set(zip(rows[ee].tolist(),cols[ee].tolist())))==ee.sum()
    ei={(int(r),int(c)):(float(w),int(s)) for r,c,w,s in zip(rows[~ee],cols[~ee],data[~ee],steps[~ee])}
    ei0={(int(r),int(c)):(float(w),int(s)) for r,c,w,s in zip(rows0[~ee0],cols0[~ee0],data0[~ee0],steps0[~ee0])}
    assert ei==ei0
    d=np.linalg.norm(pos[cols[ee]]-pos[rows[ee]],axis=1)
    np.testing.assert_array_equal(steps[ee],np.maximum(1,np.round((p.tau0+d/p.v_axon)/p.delay_dt).astype(int)))
    per_target_w=np.bincount(rows0[ee0],weights=data0[ee0],minlength=n_e)/np.maximum(np.bincount(rows0[ee0],minlength=n_e),1)
    np.testing.assert_allclose(data[ee],per_target_w[rows[ee]])
    assert audit['adjacency_identical_to_baseline'] is False and audit['n_ee_edges']==int(ee.sum())
    # a different topology key gives a different adjacency (random identity is the key, not the graph)
    other,_=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=2512,kernel=kernel,p=p)
    assert edge_dict(other)!=edge_dict(ampa)


def test_resample_ee_kernel_perp_scale_widens_transverse_partner_spread():
    p,net=small_net(seed=11);n_e=net['NE'];pos=net['pos']
    theta=np.deg2rad(30.);u=np.array([np.cos(theta),np.sin(theta)]);v=np.array([-np.sin(theta),np.cos(theta)])
    def spread(bins):
        rows,cols,_,_=edges(bins);ee=rows<n_e;dz=pos[cols[ee]]-pos[rows[ee]]
        return np.std(dz@u),np.std(dz@v)
    narrow,_=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=1,kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.),p=p)
    wide,_=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=1,kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.,perp_scale=3.),p=p)
    long,_=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=1,kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.,parallel_scale=3.),p=p)
    su,sv=spread(narrow);wu,wv=spread(wide);lu,lv=spread(long)
    assert wv>1.5*sv and abs(wu-su)<.3*su
    assert lu>1.5*su and abs(lv-sv)<.3*sv
    k=v2.ee_kernel(p,theta_deg=30.,ar=2.,perp_scale=3.,angle_offset_deg=-15.)
    assert k['l_perp']==pytest.approx(3*p.l_EE/np.sqrt(2)) and k['l_par']==pytest.approx(p.l_EE*np.sqrt(2)) and k['theta_deg']==pytest.approx(15.)


# ---------------- C7 composition from the immutable baseline ----------------
def test_build_candidate_network_composes_kernel_then_degree_then_weights():
    p,net=small_net();n_e=net['NE'];pos=net['pos'];e_core,i_core=membership(net)
    params={k:1. for k in v2.WEIGHT_FACTORS};params.update(EE_kernel_perp_scale=1.5,EE_core_to_out_degree_scale=1.5,EE_core_to_out_scale=1.25)
    reference=v2.ee_kernel(p,theta_deg=30.,ar=2.)
    built,audit=v2.build_candidate_network(net,pos,n_e,e_core,i_core,params,topology_seed=2511,p=p,reference_kernel=reference)
    # Stage 1 alone (kernel only) defines the candidate's EE baseline.
    stage1,_=v2.resample_ee_kernel(net['ampa_by_delay'],pos,n_e,topology_seed=2511,kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.,perp_scale=1.5),p=p)
    stage2,_=v2.apply_core_to_out_degree(stage1,pos,n_e,e_core,1.5,topology_seed=2511,kernel=v2.ee_kernel(p,theta_deg=30.,ar=2.,perp_scale=1.5),p=p)
    f={k:1. for k in v2.WEIGHT_FACTORS};f['EE_core_to_out_scale']=1.25
    stage3,_,_=v2.scale_pathway_blocks(stage2,net['gaba_by_delay'],n_e,e_core,i_core,f)
    assert edge_dict(built['ampa_by_delay'])==edge_dict(stage3)
    assert edge_dict(built['gaba_by_delay'])==edge_dict(net['gaba_by_delay'])
    assert len(built['ampa_by_delay'])==len(built['gaba_by_delay'])==built['max_delay_steps']+1
    assert audit['stages']==['kernel','degree','weights'] and audit['adjacency_changes'] is True
    assert 'ampa_flat' not in built
    # weights-only candidate keeps the cached adjacency and delays exactly
    only=dict(params,EE_kernel_perp_scale=1.,EE_core_to_out_degree_scale=1.)
    fixed,audit2=v2.build_candidate_network(net,pos,n_e,e_core,i_core,only,topology_seed=2511,p=p,reference_kernel=reference)
    assert audit2['adjacency_changes'] is False
    assert {k:v[1] for k,v in edge_dict(fixed['ampa_by_delay']).items()}=={k:v[1] for k,v in edge_dict(net['ampa_by_delay']).items()}
    # never derived from a previous candidate: the input network is untouched
    assert edge_dict(net['ampa_by_delay'])==edge_dict(small_net()[1]['ampa_by_delay'])
