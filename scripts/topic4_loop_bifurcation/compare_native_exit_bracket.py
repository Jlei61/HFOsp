#!/usr/bin/env python3
"""Complete-only native correspondence at the conditional K bracket."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_actual_G_history import load

OUT=ROOT/'native_exit_K_bracket'


def main(wait):
    dest=OUT/'density_correspondence';dest.mkdir(exist_ok=True)
    assert not (dest/'result.json').exists()
    while not (OUT/'extended_analysis_summary.json').exists():
        if (OUT/'status.json').exists() and read(OUT/'status.json')['stage']=='FAILED':
            write(dest/'progress.json',dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_NATIVE30S',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'extended_analysis_summary.json')['status']=='COMPLETE'
    baseline=load(ROOT/'exit_return_probes','exit_z0.21_k9_fields16p7_high')
    counts=np.load(OUT/'geometry.npz')['cell_e_counts'];rows=[]
    for K,folder,name in [(9.35,'carried_exit_lower_holds','held_K9p35'),(9.5,'carried_exit_fixed_holds','held_K9p5')]:
        native_name=f'exit_z0.21_k{K:g}_fields16p7_high';d=load(OUT,native_name)
        assert np.array_equal(d['inputs'],baseline['inputs'])
        mask=(d['time5']>=20)&(d['time5']<30);m1=(d['time1']>=20)&(d['time1']<30)
        md=(d['drift_time']>20)&(d['drift_time']<=30)
        nativefield=d['field'][mask].mean(0);nativerate=d['rate'][mask].mean(0)
        density=read(ROOT/folder/'analysis'/f'{name}.json')
        with np.load(ROOT/folder/'analysis'/f'{name}.npz') as z:denfield=z['field_Hz'][-3000:].mean(0)
        row=dict(K=K,native_name=native_name,density_name=name,native_complete30s=True,
            recorded_future_input_samples_bitwise=300,native_tail20to30s_rate_Hz=nativerate.tolist(),
            native_tail_Graw=float(d['G'][m1].mean()),native_tail_counterfactual_dZ_per_s=d['drift'][md,:,0].mean(0).tolist(),
            native_finite_window_state=d['generic']['finite_window_state'],native_tail_brief_events=d['generic']['tail_brief_events'],
            density_final3s_rate_Hz=density['final3s_rates_allE_A_B_surround_Hz'],density_final3s_Graw=density['final3s_Graw'],
            density_final3s_counterfactual_dZ_per_s=density['final3s_dZ_per_s'],
            field_difference_weighted_RMS_Hz=float(np.sqrt(np.average((nativefield-denfield)**2,weights=counts))),
            native_both_core_tailmeans_above300=bool((nativerate[1:3]>300).all()),
            native_both_core_tailmeans_below5=bool((nativerate[1:3]<5).all()),
            density_both_core_active=density['both_cores_active_in_last3s'],density_both_core_low=density['both_cores_low_in_last3s'])
        rows.append(row)
        np.savez_compressed(dest/f'K{K:g}.npz',native_tail_field_Hz=nativefield,density_tail_field_Hz=denfield,
            native_time_s=d['time5'],native_rates_Hz=d['rate'],native_causal_R_Hz=d['R'],native_Graw=d['G'],native_time1_s=d['time1'])
    result=dict(status='COMPLETE_NATIVE_DENSITY_K_BRACKET_COMPARISON',rows=rows,
        protocol_difference='Native uses the original12s high history and immediateK clamp, with original timevarying matchedexternalinput; density carries its ownK9 highstate through a0.15/s ramp under constant expectedexternalmean. Similarity is conditional spatial-state correspondence, not identical trajectories or criticalK certification.',
        statistical_unit='One shared nativehistory and futureinput; two heldK interventions, no independent native seed replication.',
        interpretation='Retain field and core differences. Native anddensity disagreement requires resolving history/input/closure before promoting a branch; neither agreement nor a finite quiet endpoint certifies all equilibria or dynamical stability.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False,counts_as_autonomous_loop=False)
    write(dest/'result.json',result);write(dest/'progress.json',dict(status=result['status'],updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
