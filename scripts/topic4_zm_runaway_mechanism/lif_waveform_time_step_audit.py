"""Bounded local LIF step refinement; no new fitted rate or spatial model."""
from common import OUT,np,read,write,log
from native_cycle_waveform_response import simulate
from lif_mc import condition
from datetime import datetime
import argparse

DEST=OUT/'lif_waveform_time_step_audit'


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    source=read(OUT/'nonlinear_rate_response/validation/output_bound_diagnostic.json')['rows']
    cases=[]
    for pop in 'EI':
        for period in [100.,200.,400.]:
            rows=[r for r in source if r['pop']==pop and r['period']==period and r['split']=='train']
            row=sorted(rows,key=lambda r:(-r['best_possible_waveform_error_from_ceiling'],r['id']))[0]
            cases.append(dict(label=f'train{row["id"]:03d}',kind='fresh_training',index=row['id'],pop=pop))
    for index,pop in [(0,'E'),(3,'E'),(6,'E'),(9,'I')]:
        cases.append(dict(label=f'original{index:02d}',kind='original_full',index=index,pop=pop))
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Do supra-stationary transient LIF population-rate peaks persist under smaller LIF time steps, and are the four original strong response targets numerically stable?',
        trigger='The rejected bounded-output rate fit has38/288targetprofiles above reciprocal refractory rate; its cap alone forces19waveformfailures (16train,3validation). This does not explain all waveform errors.',
        selection='Six TRAINING profiles: largest output-cap lower-bound defect in eachE/I by100/200/400ms stratum; ties smallestid. Four original full source waveforms retained as controls. No newheldout-targetselection.',
        cases=cases,time_steps_ms=[.1,.05,.025],reuse_original_dt01=True,additional_runs=20,
        physical_invariants='Same input-wave samples, LIF membrane/reset/threshold/refractory constants, synaptic rise/decay and Gaussian driving law. Exact innovation covariance recomputed for eachdt. Burn/record durations and128phase bins retained.',
        noise='Same per-profile seed and replicate count as its parent. Acrossdt this is not pathwise identical SDE noise; differences are population-response comparisons, not a paired trajectory effect.',
        readouts='Rate peaks, the mathematical lower bound on waveform error for r<=1/t_ref, phase-binL2andmean changes across both step halvings; retain exact exposure for each discrete clock.',
        interpretation='Persistent supra-cap peaks reject an instantaneous stationary-rate cap. Large timestep differences flag a numerical-source issue, not a new bifurcation. Agreement alone does not validate a replacement response or model.',
        convergence_reference='Report last-step relativeL2change<=0.02 as numerical diagnostic; no alteration to original local/model acceptance gates.',
        stop='Exactly two additionaldtlevels for10fixedcases. No fitting, spatial network, bifurcation scan or automatic further refinement.'))


def load_case(case):
    if case['kind']=='fresh_training':
        parent=OUT/'nonlinear_rate_response';rows=read(parent/'profiles.json')['rows'];row=rows[case['index']]
        assert row['split']=='train' and row['pop']==case['pop']
        prepared=np.load(parent/'prepared.npz');target=np.load(parent/f'local_data/profile{case["index"]:03d}.npz')
        return dict(wave=prepared['wave'][case['index']],theta=18.,T=row['period_ms'],burn_ms=row['burn_steps']*.1,
                    duration_ms=row['record_steps']*.1,R=row['replicates'],seed=920043,
                    reference_rate=target['rate_hz'],reference_exposure=target['exposure_ms'])
    parent=OUT/'factorial_waveform';c=read(OUT/'factorial_waveform_contract.json');p=np.load(parent/'prepared.npz');target=np.load(parent/'response.npz');T=float(p['T_ms'])
    return dict(wave=p['wave'][case['index']],theta=float(p['pars'][case['index'],1]),T=T,burn_ms=5*T,
                duration_ms=20*T,R=c['replicates'],seed=c['seed'],reference_rate=target['measured_hz'][case['index']],reference_exposure=target['occupancy_ms'])


def run(device):
    c=read(DEST/'contract.json');assert not (DEST/'jobs.json').exists()
    progress=dict(status='RUNNING',expected_additional_runs=20,completed=[]);write(DEST/'jobs.json',progress)
    for case in c['cases']:
        source=load_case(case)
        np.savez_compressed(DEST/f'{case["label"]}_dt0.1.npz',rate_hz=source['reference_rate'],exposure_ms=source['reference_exposure'],dt_ms=.1)
        for dt in [.05,.025]:
            p=condition(0,source['theta'],1.,1.,case['pop'],dt=dt)
            burn=round(source['burn_ms']/dt);steps=round(source['duration_ms']/dt)
            observed=simulate(np.array([p]),source['wave'][None],source['R'],source['T'],dt,burn,steps,source['seed'],128,device)[0]
            phase=((np.arange(steps)+1)*dt/source['T'])%1;bins=np.minimum((phase*128).astype(int),127)
            exposure=np.bincount(bins,minlength=128)*dt
            rates=observed/exposure[None,:]*1000;rate=rates.mean(0);sem=rates.std(0,ddof=1)/np.sqrt(source['R'])
            assert np.isfinite(rate).all() and exposure.min()>0
            path=DEST/f'{case["label"]}_dt{dt:g}.npz';assert not path.exists()
            np.savez_compressed(path,counts=observed,rate_hz=rate,sem_hz=sem,exposure_ms=exposure,dt_ms=dt,steps=steps,burn_steps=burn,
                threshold_mv=source['theta'],period_ms=source['T'],replicates=source['R'],seed=source['seed'])
            progress['completed'].append(dict(label=case['label'],dt_ms=dt,result=path.name));write(DEST/'jobs.json',progress)
            log('LIF STEP AUDIT',len(progress['completed']),20,case['label'],dt)
    progress['status']='COMPLETE';write(DEST/'jobs.json',progress)


def summarize():
    c=read(DEST/'contract.json');rows=[]
    for case in c['cases']:
        values=[]
        for dt in [.1,.05,.025]:
            path=DEST/f'{case["label"]}_dt{dt:g}.npz'
            if not path.exists():continue
            z=np.load(path);r=z['rate_hz'];R=500 if case['pop']=='E' else 1000
            values.append(dict(dt_ms=dt,peak_hz=float(r.max()),mean_hz=float(np.average(r,weights=z['exposure_ms'])),
                supra_cap=bool(r.max()>R),cap_minimum_L2=float(np.linalg.norm(np.maximum(r-R,0))/max(np.linalg.norm(r),np.sqrt(128)))))
        changes=[]
        for low,high in [(.1,.05),(.05,.025)]:
            p0=DEST/f'{case["label"]}_dt{low:g}.npz';p1=DEST/f'{case["label"]}_dt{high:g}.npz'
            if p0.exists() and p1.exists():
                a=np.load(p0);b=np.load(p1);r0=a['rate_hz'];r1=b['rate_hz'];m0=np.average(r0,weights=a['exposure_ms']);m1=np.average(r1,weights=b['exposure_ms'])
                changes.append(dict(dt_pair=[low,high],relative_L2=float(np.linalg.norm(r1-r0)/max(np.linalg.norm(r1),np.sqrt(128))),
                    relative_mean_change=float(abs(m1-m0)/max(m1,1.))))
        rows.append(dict(**case,levels=values,changes=changes))
    complete=all(len(r['levels'])==3 for r in rows)
    result=dict(status='COMPLETE' if complete else 'PENDING',rows=rows,model_promoted=False,
        scope=c['interpretation'],original_validation_gates_changed=False,
        finest_supra_cap_count=sum(r['levels'][-1]['supra_cap'] for r in rows if len(r['levels'])==3),
        last_step_L2_within2percent=sum(r['changes'][-1]['relative_L2']<=.02 for r in rows if len(r['changes'])==2))
    write(DEST/'result.json',result);log('LIF STEP RESULT',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','summarize']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    if a.command=='register':register()
    elif a.command=='run':run(a.device)
    else:summarize()
