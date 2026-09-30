#!/usr/bin/env python3
"""Measure the conditional template's spatial mismatch at equal mean states."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,NATIVE,write
import run_topic4_loop_zk_conditional as native


def main():
    out=ROOT/'spatial_template_audit';out.mkdir(exist_ok=True)
    with np.load(NATIVE/'geometry.npz') as g:
        positions=g['positions_e'];centers=g['centers_mm'];counts=g['region_counts'][:3]
    masks=[np.linalg.norm(positions-c,axis=1)<1.75 for c in centers]
    masks=[np.ones(32000,bool),*masks,~(masks[0]|masks[1])]
    assert [int(m.sum()) for m in masks[1:]]==counts.tolist()
    rows=[];arrays={}
    for sec in [10,30,40,50]:
        source=native.SOURCE/'runs'/native.NAME/'states'/f't{sec}s.pkl'
        e=native.read_pickle(source)['engine'];z=e['slow']['z'][:32000].copy()
        k=e['termination_mechanism']['sahp_g'].copy();assert e['step']==sec*10000
        tz,tk=native.fields(float(z.mean()),float(k.mean()))
        row=dict(source=str(source),time_s=sec,Zmean=float(z.mean()),Kmean=float(k.mean()),
                 same_means_verified=bool(abs(tz.mean()-z.mean())<1e-12 and abs(tk.mean()-k.mean())<1e-12),
                 regions={})
        for label,mask in zip(['allE','coreA','coreB','otherE'],masks):
            row['regions'][label]=dict(neurons=int(mask.sum()),
                native_Z=float(z[mask].mean()),template_Z=float(tz[mask].mean()),
                Z_RMS_difference=float(np.sqrt(np.mean((tz[mask]-z[mask])**2))),
                Z_max_absolute_difference=float(abs(tz[mask]-z[mask]).max()),
                native_K=float(k[mask].mean()),template_K=float(tk[mask].mean()),
                K_RMS_difference_relative_to_allE_mean=float(np.sqrt(np.mean((tk[mask]-k[mask])**2))/k.mean()),
                K_max_absolute_difference=float(abs(tk[mask]-k[mask]).max()))
        arrays.update({f't{sec}_Z_native':z,f't{sec}_K_native':k,
                       f't{sec}_Z_template':tz,f't{sec}_K_template':tk})
        rows.append(row)
    np.savez_compressed(out/'fields.npz',**arrays)
    write(out/'result.json',dict(status='COMPLETE_DIAGNOSTIC',rows=rows,
        question='At equal globalZ/K means, how closely does the common t20template match saved natural-stage fields?',
        limits='Neuron-wise descriptive differences, no calibrated equivalence threshold. t10 is60ms after the9.94s operationalentry, not the exact entry state. No saved state at16.70s is used. No consequence for transition probability is inferred without native continuation.',
        counts_as_autonomous_loop=False,certified_bifurcation=False))
    print([{k:r[k] for k in ['time_s','Zmean','Kmean','regions']} for r in rows],flush=True)


if __name__=='__main__':main()
