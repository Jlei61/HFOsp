"""Bounded local-response bias counterfactual; frozen parent stays immutable.

Two scalar log-hazard offsets target conditional LIF responses, never native
onset timing. The result is a diagnostic, not an accepted replacement model.
"""
from common import OUT, np, read, write, log
from analyze_native_early_surround_inputs import expected_flux
from physical_delay_count_rate import PhysicalDelayCountEngine, projections
from scipy.optimize import minimize_scalar
from datetime import datetime
from pathlib import Path
import argparse, os, time, hashlib

SOURCE = OUT/'native_early_surround_inputs'
DEST = OUT/'early_response_bias_diagnostic'


def register():
    assert read(SOURCE/'local_lif_reference/independent_audit.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can the measured local response underbias account for missing early surround recruitment in closed feedback?',
        hypothesis='One constant E and one constant I log-hazard offset may remove systematic local bias; if transferred inputs improve but network recruitment does not, this simple local bias is insufficient.',
        new_candidate='Exploratory additive ell offsets; all parent weights immutable. This modifies the response function including its stationary gain. It does not fix the previous candidate by definition or inherit its branches.',
        fit_groups=dict(E=[833,682,549],I=[2386]),fit_ms=[500,1500],
        fit_target='Conditional independentGaussianLIF dt0.1 ensemble mean on these inputs, 50ms bins. No native output or onset target enters the objective.',
        fit='One bounded scalar least-squares minimization per population, offset in [-log(2),log(2)], equal relative waveform weight per training group. No group-specific coefficients or subsequent retuning.',
        transfer='All16 groups: full0.5-3s and later1.5-3s. Off-axis E, both cores and other I groups not used in fit. These profiles have already been inspected, so this is development transfer, not blind validation.',
        local_gate='Every group: 50ms waveform relativeL2<=.15 and total-countrelativeerror<=.10 in full and later windows, versus both0.1and0.05LIF references. Passing only permits a diagnostic network run, not promotion.',
        network='If localgate passes: one0-3s trajectory using PhysicalDelayCountEngine, same graph/groups/fineinput/seed1/dt0.05/ZM laws andinitialstate. Only E/I final logitbias differs. Existing unmodified baseline reused.',
        observations='Original20x20 spatialfields, groupcounts, Z/M, strictcompleteevents/quietfraction and regionalrate. No event time alignment. Shortwindow cannot validate onset or entireinterictalstate.',
        numerical='Invert saved own expected flux to logits and reconstruct to roundoff. Verify last-layeronly modification and restored-parent100msbitwise prefix before one network run.',
        budget='Two scalar fits, fixed local transfer checks and at most one3s network diagnostic. No other parameter search, fulltrajectory extension, newseed or branch launch. Failure preserved, no automatic retuning.',
        scientific_acceptance='Current original local/native failures retained; goal remains incomplete.'))


def restore_logits(r):
    r=r/1000;occupied=np.zeros_like(r)
    for j in range(r.shape[1]):
        for lag in range(1,20 if j<11 else 10):
            occupied[lag:,j]+=r[:-lag,j]*.1
    p=r*.1/(1-occupied)
    assert p.min()>0 and p.max()<1
    return np.log(p)-np.log1p(-p)


def local():
    c=read(DEST/'contract.json');assert not (DEST/'local_result.json').exists()
    z=np.load(SOURCE/'fixed_readout.npz');geo=np.load(SOURCE/'membership.npz')
    groups=z['selected_groups'];N=geo['group_size'][groups];ref=np.where(geo['population'][groups]==0,2.,1.)
    ell=restore_logits(z['native_mean_private_variance']);rates,worst=expected_flux(ell,ref)
    err=float(np.max(abs(rates-z['native_mean_private_variance'])))
    assert err<1e-8 and worst<=1+1e-12
    def bins(r):return r.reshape(60,500,16).sum(1)[10:]*.1/1000*N
    references={}
    for dt in [.1,.05]:
        mc=np.load(SOURCE/f'local_lif_reference/dt{dt:g}.npz')
        references[str(dt)]=np.array([mc['counts'][j,:int(n)].mean(0)*N[j] for j,n in enumerate(mc['replicates'])]).T
    target=references['0.1'];offsets={};fits=[]
    for pop in 'EI':
        ids=np.array([np.flatnonzero(groups==g)[0] for g in c['fit_groups'][pop]])
        # Only selected training columns are supplied to the scalar optimizer.
        truth=target[:20,ids];norm=np.sum(truth**2,axis=0)
        def objective(b):
            pred,_=expected_flux(ell[:,ids]+b,ref[ids])
            pred=pred.reshape(60,500,len(ids)).sum(1)[10:30]*.1/1000*N[ids]
            return float(np.mean(np.sum((pred-truth)**2,axis=0)/norm))
        fit=minimize_scalar(objective,bounds=(-np.log(2),np.log(2)),method='bounded',options={'xatol':1e-8,'maxiter':50})
        assert fit.success
        offsets[pop]=float(fit.x);fits.append(dict(pop=pop,offset=float(fit.x),hazard_multiplier=float(np.exp(fit.x)),objective=float(fit.fun),baseline_objective=objective(0.),evaluations=fit.nfev))
    # Coefficients are persisted before transfer scoring, and never changed.
    locked=dict(status='OFFSETS_LOCKED_BEFORE_TRANSFER',offsets=offsets,fit=fits,
        parent_weights=str(OUT/'conditioned_refractory_rate/fit/locked_weights.json'),
        native_outputs_used_in_fit=False,development_not_blind=True)
    write(DEST/'offsets.json',locked)
    b=np.where(geo['population'][groups]==0,offsets['E'],offsets['I'])
    changed,worst=expected_flux(ell+b,ref);pred=bins(changed);base=bins(rates);rows=[]
    for dt,truth in references.items():
        for window,sl in [('full',slice(None)),('later',slice(20,None))]:
            for j,g in enumerate(groups):
                t=truth[sl,j];p=pred[sl,j];q=base[sl,j];den=max(np.linalg.norm(t),1.)
                l2=float(np.linalg.norm(p-t)/den);bias=float(abs(p.sum()-t.sum())/max(t.sum(),1.))
                rows.append(dict(group=int(g),reference_dt_ms=float(dt),window=window,
                    fitted_group=bool(int(g) in c['fit_groups']['E']+c['fit_groups']['I']),
                    L2=l2,count_error=bias,baseline_L2=float(np.linalg.norm(q-t)/den),
                    baseline_count_error=float(abs(q.sum()-t.sum())/max(t.sum(),1.)),
                    passed=bool(l2<=.15 and bias<=.10)))
    np.savez_compressed(DEST/'local_readout.npz',groups=groups,logits=ell,offset_by_group=b,
        rate_hz=changed,counts=pred,baseline_counts=base,bin_start_ms=z['bin_start_ms'],
        MC_dt01_counts=references['0.1'],MC_dt005_counts=references['0.05'])
    passed=all(r['passed'] for r in rows)
    write(DEST/'local_result.json',dict(status='LOCAL_TRANSFER_PASS' if passed else 'LOCAL_TRANSFER_FAIL',
        flux_inverse_max_error_hz=err,maximum_occupancy=worst,rows=rows,passed=sum(r['passed'] for r in rows),total=len(rows),
        fit=fits,network_permitted=passed,model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('EARLY BIAS LOCAL',fits,'gates',sum(r['passed'] for r in rows),'/',len(rows))


def shifted(e,offsets):
    original=e.local.network.get();changed=original.copy()
    for j,p in enumerate('EI'):changed[j,-1]+=offsets[p]
    assert np.array_equal(changed[:,:-1],original[:,:-1])
    e.local.network[:]=e.cp.asarray(changed)
    return original


def causal_register():
    # New scientific question after reviewing the failed local transfer, rather
    # than silently promoting the failed candidate or changing its gate.
    assert read(DEST/'local_result.json')['status']=='LOCAL_TRANSFER_FAIL'
    assert read(DEST/'independent_local_audit.json')['status']=='PASS'
    folder=DEST/'causal_counterfactual';folder.mkdir(exist_ok=True)
    assert not (folder/'contract.json').exists()
    write(folder/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does applying the measured population response bias alter early closed-loop surround recruitment?',
        original_local_status='FAIL: group531 laterwindow L2 .157/.162; original.15gate preserved. No accepted repair or original gate-conditioned network was launched.',
        design_revision='A distinct causal sensitivity experiment, not the gated model-validation run. The local failure does not prevent measuring the effect of a fixed response perturbation. Outcome cannot promote this candidate, even if native correspondence improves.',
        intervention='Use exactly the two alreadylocked ell offsets, no furtherfit. Compare one0-3s PhysicalDelayCountEngine trajectory with existing unmodified trajectory, original seed1/graph/forcing/ZM/initialstate/dt0.05.',
        hypotheses='Recruitment restored: these response differences can strongly affect feedback, though local/model correctness remain unresolved. Recruitment stillweak: the simple populationwide localbias cannot explain the deficit alone. Earlypersistent runaway: correction distorts the regime and cannot be accepted.',
        budget='One3s network sensitivity, no extension, newseed, nextfit or branch. Independent numerical and original earlywindow readouts required.',
        model_promoted=False,acceptance_gates_changed=False))


def permitted(counterfactual):
    if counterfactual:
        assert read(DEST/'local_result.json')['status']=='LOCAL_TRANSFER_FAIL'
        assert read(DEST/'causal_counterfactual/contract.json')['model_promoted'] is False
        return DEST/'causal_counterfactual'
    assert read(DEST/'local_result.json')['network_permitted']
    return DEST


def check(device,counterfactual=False):
    folder=permitted(counterfactual)
    e=PhysicalDelayCountEngine(seed=1,device=device);offsets=read(DEST/'offsets.json')['offsets']
    original=shifted(e,offsets);e.local.network[:]=e.cp.asarray(original);e.graph()
    x=np.concatenate([e.chunk() for _ in range(10)])
    with np.load(OUT/'physical_delay_count_rate/recorded_drive_binomial_seed1/trajectory.npz') as z:
        assert np.array_equal(x[:,0].astype('f4'),z['group_rate_hz'][:100])
        assert np.array_equal(x[:,1].astype('f4'),z['group_expected_rate_hz'][:100])
    shifted(e,offsets);e.graph();prefix=e.chunk()
    assert np.isfinite(prefix).all() and prefix.min()>=0
    write(folder/'implementation_check.json',dict(status='PASS',restored_parent100ms_bitwise=True,
        changed_coefficients='Onlytwofinalpackedbiascoefficients; parentfilesnotmodified',new10msfinite=True))
    log('EARLY BIAS NETWORK IMPLEMENTATION PASS')


def run(device,counterfactual=False):
    folder=permitted(counterfactual)
    assert read(folder/'implementation_check.json')['status']=='PASS'
    assert not (folder/'network_jobs.json').exists()
    jobs=dict(status='RUNNING',pid=os.getpid(),time_ms=0);write(folder/'network_jobs.json',jobs)
    e=PhysicalDelayCountEngine(seed=1,device=device);s=e.s
    shifted(e,read(DEST/'offsets.json')['offsets']);e.graph();projection=projections(s,e.coarse,e.parent)
    R=[];Z=[];M=[];start=time.time()
    for k in range(300):
        x=e.chunk();assert np.isfinite(x).all() and x.min()>=-1e-9
        R.append(x);Z.append(e.syn[5].get());M.append(e.syn[4].get())
        assert Z[-1].min()>=0 and Z[-1].max()<=1
        if (k+1)%50==0:
            jobs['time_ms']=(k+1)*10;write(folder/'network_jobs.json',jobs)
            log('EARLY BIAS NETWORK',jobs['time_ms'],'ms seconds',round(time.time()-start,1))
    r=np.concatenate(R);zs=np.array(Z);m=np.array(M)
    fields={grid:(C@r[:,0].T).T for grid,(C,count) in projection.items()}
    count=projection[20][1];whole=r[:,0,s.E]@s.mean_weights
    assert np.max(abs(fields[20]@(count/count.sum())-whole))<1e-8
    np.savez_compressed(folder/'trajectory.npz',time_ms=np.arange(1,3001.),state_time_ms=np.arange(10,3001.,10),
        group_rate_hz=r[:,0].astype('f4'),group_expected_rate_hz=r[:,1].astype('f4'),
        field_E_hz=fields[20].astype('f4'),field_E_hz_grid40=fields[40].astype('f4'),global_E_hz=whole,
        cell_counts=count,Z=zs.astype('f4'),M_current=m.astype('f4'),D=1-zs[:,s.E]@s.mean_weights,parent_g20=e.parent,
        final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),final_own_history=e.local.history.get(),
        final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
    jobs.update(status='COMPLETE',time_ms=3000,seconds=time.time()-start);write(folder/'network_jobs.json',jobs)
    log('EARLY BIAS NETWORK COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','local','check','run','causal_register','causal_check','causal_run']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'local':local,'check':lambda:check(a.device),'run':lambda:run(a.device),
     'causal_register':causal_register,'causal_check':lambda:check(a.device,True),'causal_run':lambda:run(a.device,True)}[a.command]()
