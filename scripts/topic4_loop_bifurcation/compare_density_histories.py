#!/usr/bin/env python3
"""Compare the completed same-input density runs with different native histories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_high_state_continuation import load
from coupled_density_exit import ADAPTED

OUT=ROOT/'density_history_selection_K9p35'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    names=['exit_z0.21_k9.35_fields16p7_high','exit_z0.21_k9.35_fields16p7_held_K9_history']
    roots=[ROOT/'density_exit_bracket_protocol',ROOT/'density_K9p35_held_history']
    native=[ROOT/'native_exit_K_bracket',ROOT/'native_K9p35_held_history']
    contracts=[read(r/'contract.json') for r in roots]
    for key in ['base_sha256','family_sha256','physical_sha256','producer_sha256']:
        assert contracts[0][key]==contracts[1][key],key
    jobs=[read(r/'jobs'/f'{n}.json') for r,n in zip(native,names)]
    for key in ['target_Z','target_K','external_noise_source','Z_template','K_template']:
        assert jobs[0][key]==jobs[1][key],key
    with np.load(jobs[0]['held_fields_file']) as a,np.load(jobs[1]['held_fields_file']) as b:
        assert np.array_equal(a['Z'],b['Z']) and np.array_equal(a['K'],b['K'])
    # Both worker paths call the same target kernel's fixed seed and consume
    # one normal4 per target/replica/step. Identical final RNG state verifies
    # the paired stream advancement rather than relying only on seed metadata.
    with np.load(roots[0]/names[0]/'final_state.npz') as a,np.load(roots[1]/names[1]/'final_state.npz') as b:
        assert np.array_equal(a['rng'],b['rng'])
        assert np.array_equal(a['clock'],b['clock']) and int(a['clock'][0])==100000
    geo=dict(np.load(ADAPTED/'geometry.npz'));E=geo['population']==0
    count=np.bincount(geo['group_cell'][E],weights=geo['group_size'][E],minlength=400)
    data=[];rows=[];fields=[]
    for label,root,name in zip(['native12s_evolving_history','native42s_heldK9_history'],roots,names):
        assert read(root/name/'result.json')['status']=='COMPLETE'
        d=load(root/name,geo);data.append(d);field=d['field_Hz'][5000:10000].mean(0);fields.append(field)
        one=d['rate_Hz'].reshape(10,1000,4).mean(1)
        rows.append(dict(history=label,source=str(root/name),interval_s=[5,10],
            regional_rate_Hz=d['rate_Hz'][5000:10000].mean(0).tolist(),
            mean_Graw=float(d['Graw'][5000:10000].mean()),
            counterfactual_dZ_per_s=d['drift_per_s'][5000:10000].mean(0).tolist(),
            each1s_regional_rate_Hz=one.tolist()))
        np.savez_compressed(OUT/f'{label}.npz',**d)
    result=dict(status='COMPLETE_SAME_INPUT_DENSITY_HISTORY_COMPARISON',rows=rows,
        field_difference_weighted_RMS_Hz=float(np.sqrt(np.average((fields[1]-fields[0])**2,weights=count))),
        QA=dict(held_Z_K_bitwise=True,physical_and_projection_kernels_identical=True,
            expected_external_drive_common='exit_branch_density/drive_0p1ms.npy',
            paired_local_normal_stream_seed=928751,full_final_RNG_arrays_bitwise=True,full_duration_steps=100000),
        interpretation='Different complete endogenous native initial histories select different10s conditional density states under the SAME timevarying expectedexternaldrive and paired Gaussianstreams. This removes the previous constantmean/ramp confound within the density comparison. It does not prove distinct attractors, a basinboundary, nativehistory selection, or temporal stability.',
        statistical_unit='Two initialhistory interventions of one conditional density system; no new independent native seed.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,created_epoch=time.time())
    write(OUT/'result.json',result);print(result,flush=True)


if __name__=='__main__':main()
