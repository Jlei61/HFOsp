#!/usr/bin/env python3
"""Two additional directly verified conditional K points; no automatic campaign."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,subprocess,time
import numpy as np
from campaign import ROOT,PYTHON,read,write,sha
from direct_response_system import DirectDC
import direct_parameter_step as stepper

OUT=ROOT/'direct_exit_K_pair'
FIRST=ROOT/'direct_exit_small_K_step'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(FIRST/'result.json')['candidate_near_equilibrium']
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_FIRST_DIRECT_NONLINEAR_K_POINT',created_epoch=time.time(),
        question='Does the directly corresponding high conditional state extend beyond its first tiny verified K step, with nonlinear rather than fitted responses?',
        design=dict(requested_K=[9.0125,9.025],maximum_halvings_per_point=2,maximum_evaluations_per_point=3,
            replicas_per_stream=256,streams=2,record_ms=4000,burn_ms=1000),
        predictor='Secant of the latest two directly checked input states. Keep the same maximum5Hz projection/100Hz predictedchange guard. A failed predictor may halve this point at mosttwice; no increase of tolerances.',
        corrector='Same measured K9 Jacobian, at mosttwo fresh nonlinear corrector validations. Stop if both population RMS residuals worsen, linear solve fails, correction leaves guard, or point remains unresolved. No continuation through a failed point.',
        numerical_stopping=read(FIRST/'contract.json')['numerical_stopping'],
        physics=read(FIRST/'contract.json')['physics'],
        limits='Approximate conditional stationary candidates at recorded sampling precision. Frozen derivatives only generate proposals. No true root, dynamic stability, autonomous native event, or bifurcation type is certified. Source/target/finite-record approximations and inherited derivative failures remain.',
        producer_sha256=sha(__file__),helper_sha256=sha(stepper.__file__),system_sha256=sha(__import__('direct_response_system').__file__),formal_bifurcation_allowed=False))
    print('Registered at mosttwo further direct K points',flush=True)


def worker(point,iteration,stream,device):
    contract=read(OUT/'contract.json');assert contract['producer_sha256']==sha(__file__) and contract['helper_sha256']==sha(stepper.__file__)
    stepper.OUT=OUT/f'point_{point:02d}';stepper.sample(iteration,stream,device)


def supervise():
    contract=read(OUT/'contract.json');assert contract['producer_sha256']==sha(__file__)
    assert not (OUT/'supervisor.json').exists(),'No silent restart'
    write(OUT/'supervisor.json',dict(pid=os.getpid(),started_epoch=time.time()))
    e=DirectDC();previous=[]
    with np.load(ROOT/'direct_newton_validation/inputs.npz') as z:
        previous.append((9.,z['proposed_source_rate_Hz'],z['proposed_target_rate_Hz']))
    first_result=read(FIRST/'result.json');last=FIRST/f"evaluation_{first_result['rows'][-1]['iteration']}"
    with np.load(last/'inputs.npz') as z:previous.append((float(z['K_mean']),z['source_Hz'],z['target_Hz']))
    accepted=[];all_rows=[];stop_reason='Both prescribed points complete'
    for point,wanted in enumerate(contract['design']['requested_K']):
        folder=OUT/f'point_{point:02d}';folder.mkdir(exist_ok=True);stepper.OUT=folder
        seed=928951+20*point
        write(folder/'contract.json',dict(producer_sha256=sha(stepper.__file__),parent_sha256=sha(OUT/'contract.json'),
            design=dict(evaluation_seeds=[[seed+2*j,seed+2*j+1] for j in range(3)]),formal_bifurcation_allowed=False))
        old,current=previous[-2:];den=current[0]-old[0];ds=(current[1]-old[1])/den;dt=(current[2]-old[2])/den
        with np.load(last/'response.npz') as z:
            source_sem=z['source_combined_SEM_Hz'];target_sem=z['cell_SEM_Hz']
        delta=wanted-current[0];attempts=[];chosen=False
        for halving in range(3):
            source=current[1]+delta*ds;target=current[2]+delta*dt
            projection=max(float(abs(source-np.clip(source,0,np.where(e.groupE,500.,1000.))).max()),float(abs(target-np.clip(target,0,np.where(e.E,500.,1000.))).max()))
            maximum=max(float(abs(delta*ds).max()),float(abs(delta*dt).max()))
            attempts.append(dict(delta_K=delta,maximum_projection_Hz=projection,maximum_predicted_change_Hz=maximum))
            if projection<=5 and maximum<=100:chosen=True;break
            delta*=.5
        if not chosen:
            stop_reason='Predictor guard failed at prescribed bounded halvings';write(folder/'result.json',dict(status='PREDICTOR_GUARD_STOP',attempts=attempts));break
        k=current[0]+delta;blocks=np.array([e.target_action(delta*ds,delta,block=b) for b in range(8)])
        target_uncertainty=np.sqrt(2*target_sem**2+blocks.std(0,ddof=1)**2/8)
        source_uncertainty=np.sqrt(source_sem**2+np.array([e.S@v for v in blocks]).std(0,ddof=1)**2/8)
        stepper.input_packet(e,folder/'evaluation_0',source,target,k,source_uncertainty,target_uncertainty,
            dict(kind='Secant_of_directly_checked_conditional_states',requested_K=wanted,previous_K=current[0],attempts=attempts))
        results=[];point_reason='Nonlinear evaluation limit'
        for iteration in range(3):
            sub=folder/f'evaluation_{iteration}';children=[]
            # Keep pilot's GPU0 independent while it runs, then restore two GPUs.
            pilot=ROOT/'direct_multisine_validation/result.json'
            devices=[0,1] if pilot.exists() else [1,1]
            for stream,device in enumerate(devices):
                with (sub/f'stream_{stream}.log').open('x') as log:
                    p=subprocess.Popen([PYTHON,__file__,'worker','--point',str(point),'--iteration',str(iteration),
                        '--stream',str(stream),'--device',str(device)],stdout=log,stderr=subprocess.STDOUT)
                children.append(p)
            while any(p.poll() is None for p in children):
                write(OUT/'progress.json',dict(status='DIRECT_NONLINEAR_EVALUATION',point=point,iteration=iteration,K_mean=k,
                    workers=[dict(pid=p.pid,status=p.poll(),device=d) for p,d in zip(children,devices)],supervisor_pid=os.getpid(),updated_epoch=time.time()));time.sleep(5)
            if any(p.returncode!=0 for p in children):
                point_reason='Worker failed, no autorestart';break
            result,mean,sem,r,rho,M,k,combined=stepper.analyze(e,iteration);results.append(result)
            if result['near_equilibrium']:point_reason='Fresh nonlinear residual at declared precision';last=sub;break
            if iteration and all(result['rows'][j]['residual_RMS_Hz']>results[-2]['rows'][j]['residual_RMS_Hz'] for j in range(2)):
                point_reason='Both source residuals increased';break
            if iteration==2:break
            local=mean+(.0005*e.E*e.chi[:,0]/(1+e.reference_g))*(M-mean)/e.D
            residual=e.S@local-r;change,qa=e.solve(-residual)
            if qa['gmres_info']!=0:point_reason='Measured Jacobian solve failed';break
            target=local+e.target_action(change);source=r+change
            projection=max(float(abs(source-np.clip(source,0,np.where(e.groupE,500.,1000.))).max()),float(abs(target-np.clip(target,0,np.where(e.E,500.,1000.))).max()))
            if projection>5 or abs(change).max()>100:point_reason='Corrector guard failed';break
            blocks=np.array([e.target_action(change,block=b) for b in range(8)])
            target_uncertainty=np.sqrt((sem/e.D)**2+blocks.std(0,ddof=1)**2/8)
            source_uncertainty=np.sqrt(np.bincount(e.group,weights=(sem/e.D)**2,minlength=e.P)/e.sizes**2+np.array([e.S@v for v in blocks]).std(0,ddof=1)**2/8)
            stepper.input_packet(e,folder/f'evaluation_{iteration+1}',source,target,k,source_uncertainty,target_uncertainty,
                dict(kind='Measured_frozen_Jacobian_correction',parent_iteration=iteration,linear_solve=qa,maximum_projection_Hz=projection))
        near=bool(results and results[-1]['near_equilibrium'])
        row=dict(point=point,requested_K=wanted,K_mean=k,candidate_near_equilibrium=near,reason=point_reason,evaluations=results,formal_bifurcation_allowed=False)
        write(folder/'result.json',row);all_rows.append(row)
        if not near:stop_reason=point_reason;break
        with np.load(last/'inputs.npz') as z:previous.append((float(z['K_mean']),z['source_Hz'],z['target_Hz']))
        accepted.append(k)
    final=dict(status='BOUNDED_DIRECT_K_PAIR_COMPLETE',accepted_K=accepted,rows=all_rows,reason=stop_reason,
        dynamic_stability_established=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',final);write(OUT/'progress.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','worker'])
    for name in ['point','iteration','stream','device']:p.add_argument('--'+name,type=int)
    a=p.parse_args();prepare() if a.command=='prepare' else supervise() if a.command=='supervise' else worker(a.point,a.iteration,a.stream,a.device)
