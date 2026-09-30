#!/usr/bin/env python3
"""Fresh nonlinear density-cell evaluation of the measured Newton proposal."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from audit_target_root_response import density_condition
import lif_mc

PARENT=ROOT/'all_target_dc_direct'
OUT=ROOT/'direct_newton_validation'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    parent=read(PARENT/'analysis.json');qa=read(PARENT/'joint_operator_qa.json')
    assert parent['gmres_info']==0 and qa['status']=='PASS_SAME_EQUATION_LINEARISATION'
    assert parent['maximum_proposed_group_change_Hz']<10 and parent['maximum_matrix_action_MC_SEM_Hz']<.1
    geo=dict(np.load(OPS/'geometry.npz'));p=read(OPS/'prepared.json')['params'];group=geo['cell_group'];E=np.arange(len(group))<32000
    with np.load(PARENT/'joint_newton_proposal.npz') as z:
        r=z['proposed_source_rate_Hz'];target=z['proposed_target_rate_Hz'];M=z['proposed_M'];Z=z['Z'];K=z['K'];oldM=z['original_M']
    assert target.min()>-1e-3 and target.max()<1000
    clipped=int((M<0).sum());clip_max=float(max(0,-M.min()));M=np.maximum(M,0);target=np.maximum(target,0)
    with np.load(ROOT/'target_stationary_response_audit/response.npz') as z:
        external=z['mean_external_rate_per_ms'];oldr=z['mean_source_rate_per_ms'].astype(float)*1000;oldphysical=z['physical'];oldg=z['g']
    W=[sparse.load_npz(ROOT/'target_stationary_response_audit'/f'{name}_dc.npz') for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    tm=np.where(E,p['tau_m_E'],p['tau_m_I']);jext=np.where(E,p['J_ext_E'],p['J_ext_I'])
    area=np.array([.1/(p[n]*(-np.expm1(-.1/p[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
    weights=np.where(geo['population']==0,geo['group_size']/32000,0.);causal=.1/(15*(-np.expm1(-.1/15)))
    def moments(rate,adaptation):
        G=30*np.clip((causal*(weights@rate)-200)/300,0,1);g=E*Z*G+K;h=1+g
        a,b,qa,qb=[w@(rate/1000) for w in W]
        IE=tm*area[0]*(a+jext*external);II=tm*area[1]*b
        physical=np.c_[(IE-Z*II-.0005*adaptation+E*Z*G*(-17.662847938268442)-30*K)/h,
            tm*area[0]**2*(qa+jext**2*external)/h**2,tm*(Z*area[1])**2*qb/h**2]
        return physical,g,float(G)
    check,cg,_=moments(oldr,oldM)
    error=float(abs(check-oldphysical).max());assert error<1e-9 and np.allclose(cg,oldg,rtol=1e-13,atol=1e-13),error
    physical,g,G=moments(r,M);pars=[]
    for i in range(len(group)):
        q,_=density_condition(physical[i],g[i],geo['threshold_mv'][group[i]],'E' if E[i] else 'I',p);pars.append(q)
    np.savez_compressed(OUT/'inputs.npz',pars=np.array(pars),physical=physical,g=g,M=M,Z=Z,K=K,
        proposed_source_rate_Hz=r,proposed_target_rate_Hz=target,Graw=G)
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_FRESH_NONLINEAR_RESPONSE',created_epoch=time.time(),
        question='Does the single measured-Jacobian Newton proposal actually reduce the original direct density-cell source/local residuals under fresh independent count paths?',
        design=dict(targets=40000,streams=2,replicas=256,record_ms=4000,burn_ms=1000,seeds=[928901,928902],devices=[0,1]),
        change='Only the proposed source/individualM equilibrium coordinates. HeldZ/K, graph, externalmean, synaptic filters and native cell update unchanged; no neural response function.',
        physical_projection=dict(negative_proposed_M_count=clipped,maximum_projection=clip_max,
            explanation='Tiny negative rate/M predictions from sampling were projected to0 before direct evaluation. Source constraint is remeasured, not assumed exact.'),
        criterion='Compare actual source-group weightedRMS/max residual with the starting point, alongside independent MonteCarlo SEM. A reduced residual supports this step; it is not exact-root, dynamical stability, or native bifurcation certification. No automatic next Newton step.',
        producer_sha256=sha(__file__),proposal_sha256=sha(PARENT/'joint_newton_proposal.npz'),formal_bifurcation_allowed=False))
    write(OUT/'implementation_qa.json',dict(status='PASS',original_input_reconstruction_max_error=error,
        group_rate_units='Hz converted to per_ms before synaptic matrices',physical_target_M_projection=clipped))
    print('Prepared fresh direct Newton validation',G,flush=True)


def worker(stream,device):
    contract=read(OUT/'contract.json');assert sha(__file__)==contract['producer_sha256']
    folder=OUT/f'stream_{stream}';folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists()
    started=time.time();write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    with np.load(OUT/'inputs.npz') as z:pars=z['pars']
    counts=[];seed=contract['design']['seeds'][stream]
    for lo in range(0,len(pars),256):
        out=lif_mc.run(pars[lo:lo+256],256,4000,1000,seed+100000*lo,crn=False,device=device,batch=256)
        assert not out[:,:,3].any();counts.append(out[:,:,2].astype('u2'))
        write(folder/'progress.json',dict(status='MEASURING',pid=os.getpid(),device=device,
            completed_targets=min(lo+256,len(pars)),total_targets=len(pars),elapsed_s=time.time()-started,updated_epoch=time.time()))
    counts=np.concatenate(counts);rates=counts/4.
    np.savez_compressed(folder/'response.npz',counts=counts,rate_Hz=rates.mean(1),SEM_Hz=rates.std(1,ddof=1)/16.)
    result=dict(status='COMPLETE',stream=stream,elapsed_s=time.time()-started);write(folder/'result.json',result);write(folder/'progress.json',result)


def finish(wait):
    while not all((OUT/f'stream_{i}/result.json').exists() for i in range(2)):
        if not wait:return
        write(OUT/'analysis_progress.json',dict(status='WAITING_FIXED_STREAMS',pid=os.getpid(),updated_epoch=time.time()));time.sleep(15)
    means=[];sems=[]
    for i in range(2):
        assert read(OUT/f'stream_{i}/result.json')['status']=='COMPLETE'
        with np.load(OUT/f'stream_{i}/response.npz') as z:means.append(z['rate_Hz']);sems.append(z['SEM_Hz'])
    mean=np.mean(means,axis=0);sem=np.sqrt(np.sum(np.array(sems)**2,axis=0))/2
    geo=dict(np.load(OPS/'geometry.npz'));group=geo['cell_group'];sizes=geo['group_size'];E=geo['population']==0
    with np.load(OUT/'inputs.npz') as z:r=z['proposed_source_rate_Hz'];target=z['proposed_target_rate_Hz'];G=float(z['Graw'])
    with np.load(PARENT/'newton_proposal.npz') as z:old=z['residual_Hz']
    aggregate=lambda x:np.bincount(group,weights=x,minlength=len(sizes))/sizes
    groupmean=aggregate(mean);groupsem=np.sqrt(np.bincount(group,weights=sem**2,minlength=len(sizes)))/sizes
    residual=groupmean-r;rows=[]
    for label,mask in [('E',E),('I',~E)]:
        rows.append(dict(population=label,old_weighted_RMS_Hz=float(np.sqrt(np.average(old[mask]**2,weights=sizes[mask]))),
            new_weighted_RMS_Hz=float(np.sqrt(np.average(residual[mask]**2,weights=sizes[mask]))),
            weighted_MC_SEM_RMS_Hz=float(np.sqrt(np.average(groupsem[mask]**2,weights=sizes[mask]))),
            new_maximum_residual_Hz=float(abs(residual[mask]).max()),
            groups_outside_3MC_SEM=int((abs(residual[mask])>3*groupsem[mask]+1e-7).sum()),groups=int(mask.sum())))
    improved=all(q['new_weighted_RMS_Hz']<q['old_weighted_RMS_Hz'] for q in rows)
    result=dict(status='DIRECT_NEWTON_STEP_REDUCES_RESIDUAL' if improved else 'DIRECT_NEWTON_STEP_NOT_VALIDATED',rows=rows,
        proposed_Graw=G,measured_E_rate_Hz=float(mean[:32000].mean()),
        maximum_target_residual_Hz=float(abs(mean-target).max()),local_target_residual_RMS_Hz=float(np.sqrt(np.mean((mean-target)**2))),
        root_certified=False,dynamic_stability_established=False,formal_bifurcation_allowed=False,
        interpretation='Fresh nonlinear local response to the proposed source/M coordinates. Report finite residual and MC uncertainty. No next step or bifurcation label is automatic.',producer_sha256=sha(__file__))
    np.savez_compressed(OUT/'response.npz',cell_rate_Hz=mean,cell_SEM_Hz=sem,group_rate_Hz=groupmean,group_SEM_Hz=groupsem,
        source_residual_Hz=residual,target_residual_Hz=mean-target,proposed_source_rate_Hz=r,proposed_target_rate_Hz=target)
    write(OUT/'result.json',result);write(OUT/'analysis_progress.json',dict(status=result['status'],updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','finish']);p.add_argument('--stream',type=int,choices=[0,1]);p.add_argument('--device',type=int,default=0);p.add_argument('--wait',action='store_true')
    a=p.parse_args();prepare() if a.command=='prepare' else worker(a.stream,a.device) if a.command=='worker' else finish(a.wait)
