#!/usr/bin/env python3
"""One bounded K continuation step using direct cell responses, no rate fit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,PYTHON,read,write,sha
from direct_response_system import DirectDC,EG
from audit_target_root_response import density_condition
import lif_mc

OUT=ROOT/'direct_exit_first_K_step'
BASE=ROOT/'direct_newton_validation'


def input_packet(e,folder,source,target,Kmean,source_uncertainty,target_uncertainty,detail):
    folder.mkdir(exist_ok=True)
    assert not (folder/'inputs.npz').exists()
    source=np.clip(source,0,np.where(e.groupE,500.,1000.)*(1-1e-12))
    target=np.clip(target,0,np.where(e.E,500.,1000.)*(1-1e-12));M=np.where(e.E,target,0.)
    physical,g,G,R=e.moments(source,M,Kmean);pars=[]
    for i in range(e.N):
        p,_=density_condition(physical[i],g[i],e.geo['threshold_mv'][e.group[i]],'E' if e.E[i] else 'I',e.p);pars.append(p)
    np.savez_compressed(folder/'inputs.npz',pars=np.array(pars),physical=physical,g=g,M=M,Z=e.Z,K=e.Kshape*Kmean,
        source_Hz=source,target_Hz=target,K_mean=Kmean,Graw=G,causal_R_Hz=R,
        source_prediction_SEM_Hz=source_uncertainty,target_prediction_SEM_Hz=target_uncertainty)
    write(folder/'proposal.json',dict(**detail,K_mean=Kmean,Graw=G,causal_R_Hz=R,
        source_constraint_max_Hz=float(abs(e.S@target-source).max()),formal_bifurcation_allowed=False))


def prepare(delta_requested=.05,maximum_halvings=2):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(BASE/'result.json')['status']=='DIRECT_NEWTON_STEP_REDUCES_RESIDUAL'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_K_STEP',created_epoch=time.time(),
        question='Does the directly corresponding K9 high state extend along the same actual Z/K field family, when a measured DC tangent is tested by fresh nonlinear cell responses?',
        design=dict(initial_K=9.,requested_delta_K=delta_requested,maximum_predictor_halvings=maximum_halvings,
            maximum_nonlinear_evaluations=3,replicas_per_stream=256,streams=2,record_ms=4000,burn_ms=1000,
            evaluation_seeds=[[928911,928912],[928913,928914],[928915,928916]],devices=[0,1]),
        physics='Actual16.7s spatial Z remains fixed atmean.21; K scales its own fixedactualfield. Graph/externalmean unchanged. Source rates, localM and causalR/G follow selfconsistency. Original density cell equations evaluate every nonlinear residual.',
        predictor='Measured all-target DC Jacobian atK9, implicitM and physicalG included. Kderivative uses the different K reversal potential. Finite-difference same-equation parameter check before sampling. Halve requestedstep only if projection would change anyrate by>5Hz or anypredictedchange exceeds100Hz.',
        correction='At most two frozen-Jacobian quasi-Newton corrections at the chosenK, each followed by fresh independent direct counts. Reject a correction that increases both source-population RMS residuals; retain every attempted array. No neural transfer fit and no automatic nextK.',
        numerical_stopping='Near-equilibrium candidate only if each E/I source weightedRMS residual and E individualtargetRMS is <=2 times its combined prediction/newcount samplingSEM (absolute roundoff floor1e-7Hz), and no source group exceeds6 combinedSEM+1e-7Hz. These numerical criteria do not certify exactroot/nativecorrespondence or dynamical stability. All inherited amplitude failures and unestimable components remain.',
        limits='Frozen local derivatives are predictors, not claimed accurate throughout the step. Finite4s record, reset phase, fixedM approximation and source averaging remain distinct from MC sampling error. Full delays/G/M stability and native edge correspondence still required.',
        producer_sha256=sha(__file__),system_sha256=sha(__import__('direct_response_system').__file__),
        baseline_sha256=sha(BASE/'result.json'),formal_bifurcation_allowed=False))
    e=DirectDC();h=1e-4;values=[]
    partialg=e.chi[:,3]-e.chi[:,0]*(EG-e.reference_physical[:,0])/(1+e.reference_g)+2*np.sum(e.chi[:,1:3]*e.reference_physical[:,1:3],axis=1)/(1+e.reference_g)
    for sign in [1,-1]:
        q,g,*_=e.moments(e.reference_source,e.reference_M,9+sign*h)
        values.append(np.sum(e.chi[:,:3]*(q-e.reference_physical),axis=1)+partialg*(g-e.reference_g))
    fd=(values[0]-values[1])/(2*h);pred=e.fk*e.D
    error=float(np.linalg.norm(fd-pred)/np.linalg.norm(pred));assert error<2e-5,error
    tangent,qa=e.solve(-(e.S@e.fk));assert qa['gmres_info']==0,qa
    target_tangent=e.target_action(tangent,1.)
    assert abs(e.S@target_tangent-tangent).max()<1e-4
    with np.load(BASE/'inputs.npz') as z:source=z['proposed_source_rate_Hz'];target=z['proposed_target_rate_Hz']
    with np.load(BASE/'response.npz') as z:base_target_sem=z['cell_SEM_Hz']
    with np.load(BASE/'sampling_precision.npz') as z:base_source_sem=z['combined_sampling_SEM_Hz']
    delta=delta_requested;cap_source=np.where(e.groupE,500.,1000.);cap_target=np.where(e.E,500.,1000.);chosen=False
    attempts=[]
    for halving in range(maximum_halvings+1):
        sp=source+delta*tangent;tp=target+delta*target_tangent
        clip=max(float(abs(np.clip(sp,0,cap_source)-sp).max()),float(abs(np.clip(tp,0,cap_target)-tp).max()))
        maximum=float(max(abs(delta*tangent).max(),abs(delta*target_tangent).max()))
        attempts.append(dict(delta_K=delta,maximum_projection_Hz=clip,maximum_predicted_change_Hz=maximum))
        if clip<=5 and maximum<=100:chosen=True;break
        delta*=.5
    write(OUT/'implementation_qa.json',dict(status='PASS',parameter_JVP_relative_error=error,tangent=qa,
        source_constraint_tangent_error_Hz=float(abs(e.S@target_tangent-tangent).max()),predictor_attempts=attempts))
    np.savez_compressed(OUT/'tangent.npz',source_derivative_Hz_per_K=tangent,target_derivative_Hz_per_K=target_tangent,
        initial_source_Hz=source,initial_target_Hz=target)
    if not chosen:
        write(OUT/'result.json',dict(status='PREDICTOR_OUTSIDE_BOUNDED_STEP',attempts=attempts,formal_bifurcation_allowed=False));return
    block=np.array([e.target_action(delta*tangent,delta,b) for b in range(8)])
    target_sem=np.sqrt(2*base_target_sem**2+block.std(0,ddof=1)**2/8)
    source_sem=np.sqrt(base_source_sem**2+np.array([e.S@v for v in block]).std(0,ddof=1)**2/8)
    input_packet(e,OUT/'evaluation_0',sp,tp,9+delta,source_sem,target_sem,
        dict(kind='Measured_DC_parameter_predictor',delta_K=delta,baseline_K=9.,implementation_qa=qa))
    write(OUT/'progress.json',dict(status='PREPARED',K_mean=9+delta,updated_epoch=time.time()))
    print('DIRECT K PREDICTOR',9+delta,'allE tangent',float(e.weights@tangent),attempts[-1],flush=True)


def sample(iteration,stream,device):
    contract=read(OUT/'contract.json');assert contract['producer_sha256']==sha(__file__)
    folder=OUT/f'evaluation_{iteration}';run=folder/f'stream_{stream}';run.mkdir(exist_ok=True);assert not (run/'progress.json').exists()
    started=time.time();write(run/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    with np.load(folder/'inputs.npz') as z:pars=z['pars']
    counts=[];seed=contract['design']['evaluation_seeds'][iteration][stream]
    for lo in range(0,len(pars),256):
        out=lif_mc.run(pars[lo:lo+256],256,4000,1000,seed+100000*lo,crn=False,device=device,batch=256)
        assert not out[:,:,3].any();counts.append(out[:,:,2].astype('u2'))
        write(run/'progress.json',dict(status='MEASURING',pid=os.getpid(),device=device,completed_targets=min(lo+256,len(pars)),total_targets=len(pars),elapsed_s=time.time()-started,updated_epoch=time.time()))
    counts=np.concatenate(counts);rate=counts/4.
    np.savez_compressed(run/'response.npz',counts=counts,rate_Hz=rate.mean(1),SEM_Hz=rate.std(1,ddof=1)/16.)
    result=dict(status='COMPLETE',elapsed_s=time.time()-started,iteration=iteration,stream=stream)
    write(run/'result.json',result);write(run/'progress.json',result)


def analyze(e,iteration):
    folder=OUT/f'evaluation_{iteration}';means=[];sems=[]
    for stream in range(2):
        assert read(folder/f'stream_{stream}/result.json')['status']=='COMPLETE'
        with np.load(folder/f'stream_{stream}/response.npz') as z:means.append(z['rate_Hz']);sems.append(z['SEM_Hz'])
    mean=np.mean(means,axis=0);sem=np.sqrt(np.sum(np.array(sems)**2,axis=0))/2
    with np.load(folder/'inputs.npz') as z:
        r=z['source_Hz'];rho=z['target_Hz'];M=z['M'];k=float(z['K_mean']);G=float(z['Graw'])
        previous_sem=z['source_prediction_SEM_Hz'];target_previous_sem=z['target_prediction_SEM_Hz']
    source_sem=np.sqrt(np.bincount(e.group,weights=sem**2,minlength=e.P))/e.sizes
    combined=np.sqrt(source_sem**2+previous_sem**2);residual=e.S@mean-r;local=mean-rho
    combined_target=np.sqrt(sem**2+target_previous_sem**2);rows=[]
    for label,mask in [('E',e.groupE),('I',~e.groupE)]:
        rms=float(np.sqrt(np.average(residual[mask]**2,weights=e.sizes[mask])))
        uncertainty=float(np.sqrt(np.average(combined[mask]**2,weights=e.sizes[mask])))
        rows.append(dict(population=label,residual_RMS_Hz=rms,combined_SEM_RMS_Hz=uncertainty,
            maximum_residual_Hz=float(abs(residual[mask]).max()),outside_6SEM=int((abs(residual[mask])>6*combined[mask]+1e-7).sum())))
    target_rms=float(np.sqrt(np.mean(local[e.E]**2)));target_uncertainty=float(np.sqrt(np.mean(combined_target[e.E]**2)))
    near=all(q['residual_RMS_Hz']<=2*q['combined_SEM_RMS_Hz']+1e-7 and q['outside_6SEM']==0 for q in rows) and target_rms<=2*target_uncertainty+1e-7
    regions=e.geo['group_region'];source_mean=e.S@mean
    regional=[float(np.average(source_mean[mask],weights=e.sizes[mask])) for mask in [e.groupE]+[e.groupE&(regions==j) for j in range(3)]+[~e.groupE]]
    np.savez_compressed(folder/'response.npz',cell_rate_Hz=mean,cell_SEM_Hz=sem,source_rate_Hz=source_mean,
        source_residual_Hz=residual,target_residual_Hz=local,source_combined_SEM_Hz=combined,target_combined_SEM_Hz=combined_target,
        field_Hz=e.field(mean))
    result=dict(status='NEAR_EQUILIBRIUM_AT_RECORDED_PRECISION' if near else 'RESIDUAL_REQUIRES_CORRECTION',
        iteration=iteration,K_mean=k,Graw=G,rows=rows,E_target_residual_RMS_Hz=target_rms,
        E_target_combined_SEM_RMS_Hz=target_uncertainty,regional_Hz_allE_A_B_surround_I=regional,
        near_equilibrium=near,formal_bifurcation_allowed=False,stability_established=False)
    write(folder/'analysis.json',result);print('DIRECT K RESPONSE',result,flush=True)
    return result,mean,sem,r,rho,M,k,combined


def supervise():
    assert (OUT/'contract.json').exists() and (OUT/'evaluation_0/inputs.npz').exists()
    assert not (OUT/'supervisor.json').exists(),'No silent restart'
    write(OUT/'supervisor.json',dict(pid=os.getpid(),started_epoch=time.time()))
    e=DirectDC();results=[];reason='Bounded nonlinear evaluation limit'
    for iteration in range(3):
        children=[];folder=OUT/f'evaluation_{iteration}'
        for stream in range(2):
            with (folder/f'stream_{stream}.log').open('x') as log:
                p=subprocess.Popen([PYTHON,__file__,'sample','--output',OUT.name,'--iteration',str(iteration),'--stream',str(stream),'--device',str(stream)],stdout=log,stderr=subprocess.STDOUT)
            children.append(p)
        while any(p.poll() is None for p in children):
            write(OUT/'progress.json',dict(status='NONLINEAR_EVALUATION',iteration=iteration,supervisor_pid=os.getpid(),workers=[dict(pid=p.pid,status=p.poll()) for p in children],updated_epoch=time.time()));time.sleep(5)
        if any(p.returncode!=0 for p in children):
            write(OUT/'result.json',dict(status='FAILED_WORKER_NO_AUTORESTART',iteration=iteration,returncodes=[p.returncode for p in children],formal_bifurcation_allowed=False));return
        result,mean,sem,r,rho,M,k,combined=analyze(e,iteration);results.append(result)
        if result['near_equilibrium']:reason='Fresh nonlinear residual at recorded sampling precision';break
        if iteration and all(result['rows'][j]['residual_RMS_Hz']>results[-2]['rows'][j]['residual_RMS_Hz'] for j in range(2)):
            reason='Both source residuals increased; stop frozen-Jacobian corrector';break
        if iteration==2:break
        local=mean+(.0005*e.E*e.chi[:,0]/(1+e.reference_g))*(M-mean)/e.D
        residual=e.S@local-r;step,qa=e.solve(-residual)
        if qa['gmres_info']!=0:reason='Bounded measured-Jacobian solve did not converge';break
        target=local+e.target_action(step);source=r+step
        projection=max(float(abs(source-np.clip(source,0,np.where(e.groupE,500.,1000.))).max()),float(abs(target-np.clip(target,0,np.where(e.E,500.,1000.))).max()))
        if projection>5 or abs(step).max()>100:reason='Correction left bounded physical neighborhood';break
        blocks=np.array([e.target_action(step,block=b) for b in range(8)])
        prediction_target_sem=np.sqrt((sem/e.D)**2+blocks.std(0,ddof=1)**2/8)
        prediction_source_sem=np.sqrt(np.bincount(e.group,weights=(sem/e.D)**2,minlength=e.P)/e.sizes**2+np.array([e.S@v for v in blocks]).std(0,ddof=1)**2/8)
        input_packet(e,OUT/f'evaluation_{iteration+1}',source,target,k,prediction_source_sem,prediction_target_sem,
            dict(kind='Measured_frozen_Jacobian_correction',parent_iteration=iteration,linear_solve=qa,maximum_projection_Hz=projection))
    final=dict(status='BOUNDED_DIRECT_PARAMETER_STEP_COMPLETE',reason=reason,rows=results,
        candidate_near_equilibrium=results[-1]['near_equilibrium'] if results else False,
        formal_bifurcation_allowed=False,stability_established=False)
    write(OUT/'result.json',final);write(OUT/'progress.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','sample']);p.add_argument('--iteration',type=int);p.add_argument('--stream',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0)
    p.add_argument('--output',default=OUT.name);p.add_argument('--delta-k',type=float,default=.05);p.add_argument('--maximum-halvings',type=int,default=2)
    a=p.parse_args();assert Path(a.output).name==a.output;OUT=ROOT/a.output
    prepare(a.delta_k,a.maximum_halvings) if a.command=='prepare' else supervise() if a.command=='supervise' else sample(a.iteration,a.stream,a.device)
