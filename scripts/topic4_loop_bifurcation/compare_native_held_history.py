#!/usr/bin/env python3
"""Complete-only comparison of two native histories at identical heldZ/K."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from prepare_native_held_history import OUT,NAME
from analyze_actual_G_history import load


def main(wait):
    dest=OUT/'history_comparison';dest.mkdir(exist_ok=True)
    assert not (dest/'result.json').exists()
    while not (OUT/'extended_analysis_summary.json').exists():
        if (OUT/'status.json').exists() and read(OUT/'status.json')['stage']=='FAILED':
            write(dest/'progress.json',dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_NATIVE30S',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'extended_analysis_summary.json')['status']=='COMPLETE'
    original=load(ROOT/'native_exit_K_bracket','exit_z0.21_k9.35_fields16p7_high')
    carried=load(OUT,NAME);assert np.array_equal(carried['inputs'],original['inputs'])
    count=np.load(OUT/'geometry.npz')['cell_e_counts']
    with np.load(ROOT/'carried_exit_lower_holds/analysis/held_K9p35.npz') as z:denfield=z['field_Hz'][-3000:].mean(0)
    rows=[];fields=[]
    for label,d in [('original12s',original),('heldK9end42s',carried)]:
        m=(d['time5']>=20)&(d['time5']<30);g=(d['time1']>=20)&(d['time1']<30)
        md=(d['drift_time']>20)&(d['drift_time']<=30)
        field=d['field'][m].mean(0);fields.append(field)
        rows.append(dict(history=label,name=d['job']['name'],rates_Hz=d['rate'][m].mean(0).tolist(),
            mean_Graw=float(d['G'][g].mean()),counterfactual_mean_dZ_dK_per_s=d['drift'][md].mean(0).tolist(),
            field_RMS_from_carried_density_Hz=float(np.sqrt(np.average((field-denfield)**2,weights=count))),
            finite_window_state=d['generic']['finite_window_state'],tail_brief_events=d['generic']['tail_brief_events'],
            complete_native30s=True))
        np.savez_compressed(dest/f'{label}.npz',time_s=d['time5'],rates_Hz=d['rate'],field_Hz=d['field'],
            time1_s=d['time1'],Graw=d['G'],causal_R_Hz=d['R'])
    result=dict(status='COMPLETE',rows=rows,recorded300_future_inputs_bitwise=True,
        native_history_field_difference_RMS_Hz=float(np.sqrt(np.average((fields[0]-fields[1])**2,weights=count))),
        design='Same heldZ/Kfield and completefutureexternalinput; entire endogenous initialhistory changes. The new condition is an immediate smallK step from heldK9, not a slowramp.',
        interpretation='Finite-history state selection and spatial correspondence only; no equilibrium/stability/basin/fold certification, autonomous release, or independent-seed repetition.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False)
    write(dest/'result.json',result);write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
