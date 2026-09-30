#!/usr/bin/env python3
"""Complete-only protocol/amplitude review, without altering scientific gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha

OUT=ROOT/'held_exit_phase_response_validation'


def main(wait):
    while not (OUT/'result.json').exists():
        if not wait:return
        write(OUT/'review_progress.json',dict(status='WAITING_PHASE_DIAGNOSTIC',pid=os.getpid(),updated_epoch=time.time()))
        time.sleep(15)
    assert not (OUT/'review.json').exists()
    r=read(OUT/'result.json');assert r['status']=='COMPLETE_PHASE_OBSERVER_VALIDATION'
    with np.load(OUT/'derivative_comparison.npz') as z:
        gain=z['gain'];mean=z['means'];sem=z['SEM'];cells=z['cells'];channels=z['channels'];amplitudes=z['amplitudes']
    assert np.array_equal(cells[::2],cells[1::2]) and np.array_equal(channels[::2],channels[1::2])
    assert np.allclose(amplitudes[1::2],amplitudes[::2]/2,rtol=0,atol=0)
    amplitudes_out=[]
    for p in range(4):
        delta=gain[p,1::2]-gain[p,::2];d=delta.mean(-1);se=delta.std(-1,ddof=1)/np.sqrt(gain.shape[-1])
        tol=np.maximum.reduce([.1*abs(mean[p,::2]),2*se,np.full(len(d),1e-7)])
        estimable=abs(mean[p,::2])>=10*np.maximum(sem[p,::2],1e-15)
        for j in range(len(d)):
            amplitudes_out.append(dict(protocol=p,cell=int(cells[2*j]),channel=int(channels[2*j]),
                full_gain=float(mean[p,2*j]),half_gain=float(mean[p,2*j+1]),paired_difference=float(d[j]),
                paired_SEM=float(se[j]),tolerance=float(tol[j]),estimable=bool(estimable[j]),passed=bool(abs(d[j])<=tol[j])))
    summaries=[]
    for before,after in [(0,1),(1,2),(2,3),(0,3)]:
        q=[test for row in r['derivative_rows'] if row['factor']==1 for test in row['protocol_comparisons'] if test['before']==before and test['after']==after]
        eligible=[x for x in q if x['estimable_reference']]
        summaries.append(dict(before=before,after=after,estimable=len(eligible),within_tolerance=sum(x['within_tolerance'] for x in eligible),
            total=len(q),nonestimable=len(q)-len(eligible)))
    amplitude_summary=[]
    for p in range(4):
        q=[x for x in amplitudes_out if x['protocol']==p];eligible=[x for x in q if x['estimable']]
        amplitude_summary.append(dict(protocol=p,total=len(q),estimable=len(eligible),amplitude_pass=sum(x['passed'] for x in eligible),
            amplitude_failed=sum(not x['passed'] for x in eligible),nonestimable=len(q)-len(eligible)))
    # Verify amplitude-pair stream mode against the original alltarget DC kernel.
    import phase_lif_mc as phase
    import measure_all_target_dc as old
    from measure_target_direct_response import run
    with np.load(ROOT/'held_exit_stationarity_K9p35/inputs.npz') as z:
        base=z['pars'];g=z['g'];physical=z['physical']
    qa=[]
    for cell in cells[::2][:3]:
        for channel in range(4 if cell<32000 else 3):
            h=[.02*(base[cell,1]-11),.05*physical[cell,1],.1*physical[cell,2],.002*(1+g[cell])][channel]
            for f in [1.,.5]:
                q=base[cell].copy();q[20]=channel;q[22]=g[cell];q[23]=20. if cell<32000 else 10.
                q[4]=h*f/(physical[cell,channel] if channel in [1,2] else 1.);qa.append(q)
    qa=np.array(qa)
    a=run(old.kernel(),qa,64,100,30,929389)
    b=phase.run(qa,64,100,30,929389,phase_ms=0,stream_mode=2)
    assert np.array_equal(a,b)
    result=dict(status='COMPLETE_PHASE_PROTOCOL_REVIEW',static_duration_agreement=[r['static_duration_agreement_count'],r['static_duration_total']],
        protocol_comparisons=summaries,amplitude_summary=amplitude_summary,amplitude_rows=amplitudes_out,
        paired_amplitude_stream_zero_options_bitwise=True,
        interpretation='Count-window phase is an estimator issue. Keep all protocol and amplitude failures; no formal spectral inference from a mixture of incompatible estimators. A repaired alltarget map, if needed, must be freshly checked rather than relabeling the old map.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,root_gate_changed=False)
    write(OUT/'review.json',result);write(OUT/'review_progress.json',dict(status=result['status'],updated_epoch=time.time()))
    print('PHASE REVIEW',result['static_duration_agreement'],summaries,amplitude_summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
