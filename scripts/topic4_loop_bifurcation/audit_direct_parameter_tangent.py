#!/usr/bin/env python3
"""Diagnose the failed direct-K predictor; never infer dynamic stability here."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
from campaign import ROOT,read,write,sha
from direct_response_system import DirectDC

OUT=ROOT/'direct_exit_first_K_step'/'tangent_audit'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_PREDICTOR_GUARD_FAILURE',created_epoch=time.time(),
        question='Is the large local K tangent reproducible across measured full/half amplitudes and independent replica halves, and which cells carry it?',
        design='Reassemble the identical DC system with full amplitude, half amplitude, and each independent 128-replica half of the full-amplitude measurements. Solve the same implicit tangent. Save residual and localization. No new nonlinear counts, no stability labels.',
        limits='Numerical perturbation audit of the existing measured derivative, not independent physiological data or native noise. Agreement would not certify a bifurcation, and disagreement requires diagnosing the contributing cells before continuing.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    e=DirectDC()
    with np.load(ROOT/'all_target_dc_direct/measured_dc.npz') as z:
        measurements=[('full',z['gain'][:,:,0]),('half_amplitude',z['gain'][:,:,1]),
            ('replica_half_0',z['replicate_block_gain'][:,:,0,:4].mean(2)),
            ('replica_half_1',z['replicate_block_gain'][:,:,0,4:].mean(2))]
        amplitude_pass=z['amplitude_pass'];sems=z['SEM'][:,:,0]
    source=[];target=[];rows=[];regions=e.geo['group_region'][e.group]
    for name,chi in measurements:
        C,u,D,fk=e.coefficients(chi)
        def target_action(v):return sum(c*(w@v) for c,w in zip(C,e.W))+u*(e.dG@v)
        J=LinearOperator((e.P,e.P),matvec=lambda v:e.S@target_action(v)-v,dtype=float)
        hist=[]
        tangent,info=gmres(J,-e.S@fk,M=e.pre,rtol=1e-7,atol=1e-9,restart=100,maxiter=3,
            callback=lambda x:hist.append(float(x)),callback_type='pr_norm')
        t=target_action(tangent)+fk;mass=abs(t[e.E]);order=np.argsort(-mass)
        row=dict(name=name,gmres_info=int(info),iterations=len(hist),maximum_linear_residual=float(abs(J@tangent+e.S@fk).max()),
            E_mean_tangent=float(e.weights@tangent),maximum_source_tangent=float(abs(tangent).max()),
            maximum_target_tangent=float(abs(t).max()),
            target_tangent_region_absolute_mass=[float(abs(t[e.E&(regions==j)]).sum()/mass.sum()) for j in range(3)],
            E_targets_for_90percent_absolute_tangent=int(np.searchsorted(np.cumsum(mass[order]),.9*mass.sum())+1))
        if target:
            row['target_tangent_relative_difference_from_full']=float(np.linalg.norm(t-target[0])/np.linalg.norm(target[0]))
            row['source_tangent_relative_difference_from_full']=float(np.linalg.norm(tangent-source[0])/np.linalg.norm(source[0]))
            row['target_cosine_to_full']=float(t@target[0]/(np.linalg.norm(t)*np.linalg.norm(target[0])))
        source.append(tangent);target.append(t);rows.append(row)
        print(row,flush=True);write(OUT/'progress.json',dict(status='ASSEMBLING_VARIANTS',completed=len(rows),rows=rows,updated_epoch=time.time()))
    # Fixed set from full-response tangent, selected before any fresh simulations.
    top=np.argsort(-abs(target[0]))[:32]
    cells=[dict(cell=int(i),population='E' if e.E[i] else 'I',region=int(regions[i]),source_group=int(e.group[i]),
                tangent_Hz_per_K=float(target[0][i]),amplitude_pass=amplitude_pass[i].tolist(),
                gains=measurements[0][1][i].tolist(),gain_SEM=sems[i].tolist()) for i in top]
    np.savez_compressed(OUT/'tangents.npz',source=np.array(source),target=np.array(target),top_targets=top)
    result=dict(status='COMPLETE_PARAMETER_TANGENT_AUDIT',rows=rows,top_targets=cells,
        formal_bifurcation_allowed=False,dynamic_stability_established=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result)


if __name__=='__main__':main()
