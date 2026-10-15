"""Scientific invariants: timing, masks, alignment, geometry, and native dynamics."""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from data_processing.raw_to_events import event_features,join_blocks
from data_processing.src.group_event_analysis import EventWindow
from analysis.src.lagpat_rank_audit import build_masked_kmeans_features
from analysis.src.topic5_interictal_direction_rose import fit_event_directions_3d
from analysis.src.interictal_spatial_information_gain import fit_evaluate_all_event_crossfit_fold
from modeling.src.snn_engine.params import Params
from modeling.src.snn_engine.model import build_network
from modeling.src.snn_engine.kick_probe_core_input_v2 import simulate_kick
from modeling.cpu_acceleration import accelerated_engine
from modeling.src.snn_engine.mz_historical import MZSlowVars,MZSlowVarsConfig


def main():
    # Known 30 ms shift survives full spectrogram centroiding. No synthetic
    # signal from this test is used in the notebooks or paper evidence.
    fs=800;t=np.arange(800)/fs;centers=[.35,.38,.42]
    x=np.array([np.sin(2*np.pi*120*t)*np.exp(-.5*((t-c)/.02)**2) for c in centers])
    pre=dict(band_V=x,sfreq=fs,names=np.array(['A','B','C']))
    detected=dict(windows=[EventWindow(.1,.7,0)],detections={n:np.array([[c-.05,c+.05]]) for n,c in zip(pre['names'],centers)})
    e=event_features(pre,detected)
    np.testing.assert_allclose(np.diff(e['centroid_ms'][:,0]),[30,40],atol=2)
    # Poisoning a nonparticipant's finite legacy rank cannot affect features.
    r=np.array([[0,2],[1,0],[2,1]]);m=np.array([[1,0],[1,1],[1,1]],bool);q=r.copy();q[~m]=999999
    np.testing.assert_array_equal(build_masked_kmeans_features(r,m),build_masked_kmeans_features(q,m))
    # Cross-record channel permutation is aligned by names, absent channels stay missing.
    a=dict(channel_names=np.array(['B','A']),ranks=r[:2],participating=m[:2],centroid_ms=r[:2].astype(float),block_ids=np.array(['r0','r0']))
    b=dict(channel_names=np.array(['C','A']),ranks=r[1:],participating=m[1:],centroid_ms=r[1:].astype(float),block_ids=np.array(['r1','r1']))
    joined=join_blocks([a,b]);assert joined['channel_names'].tolist()==['A','B','C']
    np.testing.assert_array_equal(joined['ranks'][0,:2],r[1]);assert not joined['participating'][2,:2].any()
    # Analytic rank gradient points EARLY -> LATE, including contact permutation.
    xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[2,0,0.]])
    beta=np.array([1.,2.,3.]);values=(xyz@beta)[:,None]
    direction=fit_event_directions_3d(values,xyz)['directions'][0]
    np.testing.assert_allclose(direction,beta/np.linalg.norm(beta),atol=1e-12)
    # Held-out directions may change the score, never fit parameters/assignments.
    rng=np.random.default_rng(3);n=240;labels=np.arange(n)%2
    ranks=np.array([np.argsort(np.argsort((xyz@beta)*(1 if k==0 else -1)+rng.normal(0,.1,6))) for k in labels]).T
    mask=np.ones_like(ranks,bool);dirs=fit_event_directions_3d(ranks.astype(float),xyz)['directions'];blocks=np.repeat(np.arange(4),60)
    f=fit_evaluate_all_event_crossfit_fold(ranks,mask,dirs,blocks,xyz,fold_index=0)
    changed=dirs.copy();changed[np.isin(blocks,[1,3])]*=-1
    g=fit_evaluate_all_event_crossfit_fold(ranks,mask,changed,blocks,xyz,fold_index=0)
    for method in f['models']:
        np.testing.assert_array_equal(f['models'][method]['templates'],g['models'][method]['templates'])
    # A coordinate-origin change cannot alter a fixed interictal energy score.
    # This would fail if transverse mirror were taken about the sheet edge.
    from analysis.spectral_fields import energy_field_trajectory
    rng=np.random.default_rng(12);xy=rng.normal(size=(12,2))*2+10
    ranks_xy=xy@np.array([1.,.7]);support=np.ones(12)
    samples=rng.normal(size=(12,2000));kwargs=dict(baseline_sec=(.1,1.1),band=(30,80))
    left=energy_field_trajectory(samples,1000,xy,ranks_xy,support,**kwargs)
    right=energy_field_trajectory(samples,1000,xy+np.array([100.,-50.]),ranks_xy,support,**kwargs)
    for key in left['scores']:np.testing.assert_allclose(left['scores'][key],right['scores'][key],atol=1e-11,rtol=1e-11)
    # A post-entry spectral field needs a complete 1s window and variable baseline.
    from analysis.model_transition import onset_broadband_field
    raw=np.random.default_rng(95).normal(size=(15,12000));raw[:,10000:]*=2
    field=onset_broadband_field(raw,1000,10.)
    assert field['status']=='OK' and field['n_baseline_frames']==5 and np.all(field['robust_z']>0)
    for values,onset in [(raw,None),(raw,11.5),(np.zeros_like(raw),10.)]:
        assert onset_broadband_field(values,1000,onset)['status']=='NOT_ESTIMABLE'
    # Full native trajectory, including slow mechanism, must match ordered scatter.
    for enabled in [False,True]:
        p=Params(L=1,density=240,C_EE=20,C_IE=20,C_EI=5,C_II=5,T=200,seed=11)
        runs=[]
        for engine in [simulate_kick,accelerated_engine()]:
            net=build_network(p,verbose=False)
            for matrix in net['ampa_by_delay']:assert not matrix[:net['NE']].diagonal().any()
            for matrix in net['gaba_by_delay']:assert not matrix[net['NE']:].diagonal().any()
            slow=MZSlowVars(240,p.V_th,MZSlowVarsConfig(use_z=enabled,use_m=enabled,tau_z=50,I_th_EI=3,tau_adp=50,eta_m=.1),NE=net['NE'])
            runs.append(engine(p,net,0,slow=slow))
        for key in ['E_spk_bool','rate_E','rate_I']:np.testing.assert_array_equal(runs[0][key],runs[1][key])
    print('PASS: known time shifts, absent-channel invariance, name alignment, signed direction, held-out isolation, fixed-field translation invariance, no autapses, full dynamics parity')

if __name__=='__main__':main()
