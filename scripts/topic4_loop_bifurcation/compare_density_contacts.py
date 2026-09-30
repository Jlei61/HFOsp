#!/usr/bin/env python3
"""Reuse frozen native contact observers, without candidate calibration."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, argparse, time, warnings
from pathlib import Path
import numpy as np
from campaign import ROOT, REPO, read, write, sha
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts/topic4_interictal_surrogate'))
from evaluate_readouts import summarize, electrical_envelope, metrics
from interictal_common import smooth, safe

OLD=REPO/'results/topic4_sef_hfo/interictal_spatial_surrogate_6101_20260916'
OUT=ROOT/'density_contact_replay'
GEO=REPO/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/geometry.npz'


def main():
    qa=read(OUT/'result.json');assert qa['status']=='COMPLETE'
    assert all(qa['final_state_bitwise_equal'].values()) and qa['all_stored_group_observations_bitwise_equal']
    contracts={key:read(OLD/f'observer_{key}.json') for key in ['firing','current_hfo']}
    names=contracts['firing']['contact_names'];assert names==contracts['current_hfo']['contact_names']
    geo=dict(np.load(GEO));weights=geo['contact_rate_weights']
    assert np.allclose(weights.sum(0),1.) and not weights[geo['population']!=0].any()
    rows={};sources={};envelopes={}
    for seed in [9108401,9108402]:
        p=OLD/f'native/{seed}/trajectory.npz';z=np.load(p)
        assert z['contact_names'].tolist()==names
        # Establish current-array identity to the actual Fig5 native reference.
        base=REPO/f'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s{seed}/chunks'
        raw=[]
        for f in sorted(base.glob('*.npz')):
            with np.load(f) as a:
                if a['start_step']>=80000:break
                raw.append(a['lfp_raw'])
        raw=np.concatenate(raw);assert np.array_equal(raw,z['lfp_raw'][:16000])
        label=f'native{seed}';envelopes[label]=dict(firing=z['contact_envelope'][:4000],current_hfo=electrical_envelope(raw)[0])
        sources[label]=str(p)
    for R,seed,folder in [(2048,927611,ROOT/'density_spatial_onset/R2048_num927611'),
                         (2048,927612,ROOT/'density_spatial_onset/R2048_num927612'),
                         (8192,927611,ROOT/'density_spatial_resolution')]:
        with np.load(folder/'trajectory.npz') as z:
            rate=z['group_rate_Hz'][:8000].astype(float)
        raw=(rate.reshape(4000,2,-1).sum(1)*.001)@weights
        label=f'R{R}_num{seed}';envelopes[label]=dict(firing=smooth(raw));sources[label]=str(folder/'trajectory.npz')
        if seed==927612:
            with np.load(OUT/'contacts.npz') as z:current=z['lfp_raw'][:16000]
            envelopes[label]['current_hfo']=electrical_envelope(current)[0]
    for label,envs in envelopes.items():
        rows[label]={}
        for key,env in envs.items():
            assert env.shape==(4000,15) and np.isfinite(env).all()
            rec=summarize(env,contracts[key],key,start_ms=500,stop_ms=8000)
            rows[label][key]=rec
    comparisons=[]
    for mode in ['firing','current_hfo']:
        for left,right,role in [('native9108402','native9108401','native_noise_reference')]+[
                (l,r,'density_native') for l in rows if l.startswith('R') and mode in rows[l]
                for r in ['native9108401','native9108402']]:
            a,b=rows[left][mode]['summary'],rows[right][mode]['summary']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                value=metrics(a,b,names) if a['N'] and b['N'] else [None]*6
            comparisons.append(dict(left=left,right=right,role=role,readout=mode,N_left=a['N'],N_right=b['N'],
                rank=value[0],within_shaft_order=value[1],participation=value[3]))
    write(OUT/'comparison.json',safe(dict(status='COMPLETE',sources=sources,rows=rows,comparisons=comparisons,
        producer_sha256=sha(__file__),frozen_observers={k:dict(path=str(OLD/f'observer_{k}.json'),sha256=sha(OLD/f'observer_{k}.json')) for k in contracts},
        window_ms=[500,8000],candidate_recalibration=False,native_cached_raw_contact_identity=True,
        clocks='Native raw current observes updates at prestep labels0,.5,...; density observes after5,10,... steps with end labels.5,1,... . Under the original native label convention density samples occur .4ms later. Event-level distributions compared, no tracewise equality expected. Rate bins aggregate identical2ms intervals.',
        scope='Original topology6101 and original15contacts. Gaussian firing projection and raw-current HFO observer are separate. Numerical particle streams are not biologicalreplicates. Missing contacts retain NaNs and all15slots. Current observer available for one exact physical-state-preserving replay only.',
        interpretation='Descriptive native-noise comparison, not a new threshold or an automatic equivalence test. Spatially grouped contact weights still assume exchangeability inside each existinggroup.',
        contact_correspondence_certified=False,model_promoted=False,formal_bifurcation_allowed=False,human_review='PENDING')))
    np.savez_compressed(OUT/'envelopes.npz',**{f'{label}__{key}':env for label,envs in envelopes.items() for key,env in envs.items()},contact_names=names)
    print(safe(comparisons),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    while a.wait and not ((OUT/'result.json').exists() and (ROOT/'density_spatial_resolution/result.json').exists()):time.sleep(30)
    main()
