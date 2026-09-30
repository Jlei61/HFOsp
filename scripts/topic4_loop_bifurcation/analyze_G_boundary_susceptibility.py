#!/usr/bin/env python3
"""Static G-increment feedback at the measured K9.35 working point.

Holding the G perturbation leaves the background G unchanged. It is neither
an actual G=0 equilibrium nor a temporal-stability calculation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from campaign import ROOT,read,write,sha
from held_direct_moments import HeldInputs

OUT=ROOT/'held_K9p35_G_increment_response'
DC=ROOT/'held_exit_phase_dc_operator_K9p35'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(DC/'joint_operator_qa.json')['status']=='PASS_SAME_EQUATION_LINEARISATION'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_STATIC_G_INCREMENT_DECOMPOSITION',created_epoch=time.time(),
        question='Does incremental globalG feedback materially change the localK susceptibility near the R200 activation corner even though its background value is small?',
        method='At the actual measuredK9.35 input, retain all measured targetDC channels andimplicitM. Write the source response asL=A+u*wT. Solve(I-A)a=b and(I-A)v=u; the fullK derivative isa+v*(wTa)/(1-wTv). Compare with the independently assembled existing tangent and full/half modulation estimates. Eight block estimates are reported as sampling variability, not a certified totalerror.',
        comparison='A holds the G perturbation fixed, not its background value atzero. The actualG0 side atR<200 has different coordinates and must be computed separately. This diagnostic cannot prove a fold, a boundaryequilibrium bifurcation, or dynamical stability.',
        scope='Measuredpoint before the small independentlyverified nonlinearcorrection; no newequilibrium claimed andno newK, nativejob, orautomaticparameterextension.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    e=HeldInputs();h=1+e.reference['g'];w=.1*e.causal*e.weights
    assert 200<float(e.reference['stationary_R_Hz'])<500
    with np.load(DC/'measured_dc.npz') as z:allchi=z['gain'];blocks=z['replicate_block_gain'][:,:,0]
    region=e.geo['group_region'];masks=[e.groupE]+[e.groupE&(region==j) for j in range(3)]+[~e.groupE]
    def regional(x):return [float(np.average(x[m],weights=e.sizes[m])) for m in masks]
    def solve(chi):
        D=1+.0005*e.E*chi[:,0]/h;assert D.min()>.5
        C=np.array([chi[:,0]*e.tm*e.area[0]/(1000*h),-chi[:,0]*e.Z*e.tm*e.area[1]/(1000*h),
            chi[:,1]*e.tm*e.area[0]**2/(1000*h**2),chi[:,2]*e.tm*(e.Z*e.area[1])**2/(1000*h**2)])/D
        A=sum(e.S@matrix.multiply(c[:,None]) for c,matrix in zip(C,e.W)).tocsr()
        u=e.S@(e.E*e.Z*chi[:,3]/D)
        fk=e.K/9.35*(chi[:,3]+chi[:,0]*(-30+17.662847938268442)/h)/D
        b=e.S@fk;IminusA=(sparse.eye(e.P)-A).tocsc();lu=splu(IminusA)
        both=lu.solve(np.c_[b,u]);a,v=both[:,0],both[:,1]
        eta=float(w@v);den=1-eta;assert abs(den)>1e-10,'Unresolved static rankone denominator'
        full=a+v*float(w@a)/den
        residual=IminusA@full-u*(w@full)-b
        error=float(np.linalg.norm(residual)/max(np.linalg.norm(b),1e-30));assert error<1e-7
        return dict(eta=eta,one_minus_eta=den,
            fixed_G_increment_regional_Hz_per_K=regional(a),full_regional_Hz_per_K=regional(full),
            fixed_G_increment_dR_Hz_per_K=float(e.causal*(e.weights@a)),full_dR_Hz_per_K=float(e.causal*(e.weights@full)),
            linear_equation_relative_residual=error),full,a,v
    rows=[];arrays={}
    for j,label in enumerate(['full_amplitude','half_amplitude']):
        row,full,a,v=solve(allchi[:,:,j]);row['estimate']=label;rows.append(row)
        arrays[label+'_full_dsource']=full;arrays[label+'_fixedG_dsource']=a;arrays[label+'_G_resolvent']=v
        print(label,row,flush=True)
    with np.load(ROOT/'held_exit_K9p35_tangent/tangent.npz') as z:previous=z['source_Hz_per_K']
    relative=float(np.linalg.norm(arrays['full_amplitude_full_dsource']-previous)/np.linalg.norm(previous))
    assert relative<1e-5
    blockrows=[]
    for b in range(8):
        row,_,_,_=solve(blocks[:,:,b]);row['block']=b;blockrows.append(row)
        write(OUT/'progress.json',dict(status='COMPUTING_BLOCK_VARIABILITY',completed_blocks=b+1,pid=os.getpid(),updated_epoch=time.time()))
    values=np.array([[q['eta'],q['fixed_G_increment_dR_Hz_per_K'],q['full_dR_Hz_per_K']] for q in blockrows])
    np.savez_compressed(OUT/'responses.npz',**arrays,block_summary_eta_fixedGdR_fulldR=values)
    result=dict(status='COMPLETE_STATIC_INCREMENTAL_G_RESPONSE',working_K=9.35,
        background_Graw=float(e.reference['stationary_Graw']),working_R_Hz=float(e.reference['stationary_R_Hz']),
        rows=rows,blocks=blockrows,block_mean_SEM_eta_fixedGdR_fulldR=(values.std(0,ddof=1)/np.sqrt(8)).tolist(),
        independent_existing_tangent_relative_difference=relative,
        uncertainty='Block scatter excludes systematic derivative bias, finitewindowclosure andnativevariation; the gain matrix retains amplitude failures andnonestimable entries. Full/half estimates and allblock values remain visible; no posthoc gate.',
        interpretation='An algebraic static feedback diagnostic at unchangedbackgroundG. Neither Gzero-side behavior, temporal stability, branch existence at anotherK, nor a bifurcation type is established.',
        dynamic_stability_established=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print('STATIC G DECOMPOSITION COMPLETE',relative,flush=True)


if __name__=='__main__':main()
