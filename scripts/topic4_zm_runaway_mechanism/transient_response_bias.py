"""One bounded transient-only correction to the fixed conditioned rate field.

delta ell = b_p H/(H+h_p), H=mean(input-history difference**2).
Both the correction and its first derivative vanish at a stationary input,
so the original stationary response and infinitesimal susceptibility remain.
No physical graph, synaptic state or Z/M equation changes.
"""
from common import OUT,model,np,read,write,log
from analyze_native_early_surround_inputs import features,expected_flux
from native_input_fixed_readout import direct_features
from scipy.optimize import minimize_scalar
from datetime import datetime
import argparse

DEST=OUT/'transient_response_correction'
EARLY=OUT/'native_early_surround_inputs'
LATE=OUT/'native_input_bridge'


def register():
    assert read(OUT/'early_response_bias_diagnostic/late_input_transfer/independent_audit.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can a transient-only response correction recover the early LIF response without degrading the late response that constant bias spoiled?',
        evidence='Constant E/Ibias restored early3s closed-loop recruitment/D but overpredictedlateI14-16percent. Uniformbias is stopped and remains an unaccepted counterfactual.',
        hypothesis='Missing response is associated with nonlinear input transients; this is tested, not assumed proved byearly/latecomparison.',
        equation='deltaell=b_population*H/(H+h_population); H=mean(f[3:39]**2), f are the unchanged normalizedinput-history differences. No new dynamic states.',
        invariants='For historydifferences zero, deltaell=0 and full first derivative=0 exactly: same stationary response and infinitesimal localgain. This preserves their existing successes AND failures; it doesnot certify them.',
        scale='h_population is the median H over the same early training groups and0.5-1.5s, weighting by the existing parent predicted flux. Input andparent output only, no LIF/native target orlate data.',
        training_groups=dict(E=[833,682,549],I=[2386]),training_ms=[500,1500],
        fit='One scalar b perE/I in[-log4,log4], atmost50boundedoptimizer evaluations, same pergroup normalized50ms waveformLS as constantbias diagnostic. Target independentGaussianLIF dt0.1 only. No native output/onset/D target.',
        transfer='Same16early and6late profiles versusboth0.1and0.05LIF. Allfull/later earlywindows andlate9-10.35s use unchanged L2<=.15,relativecountbias<=.10. Developmental because sourceswereinspected; no blindclaim.',
        new_validation='After weightslocked, 14newprescribedinputs: earlygroups531,185,945,2613,2237 andlategroups279,594 eachat timewarp0.8/1.25 and balancedmeanaboutreset/varianceamplitude0.9/1.1. Exactrecipe recorded beforeconstruction; predictionslockedbefore newLIFtargets. Localvalidation only, no native inference.',
        new_validation_recipe='Variant0: index floor(k*0.8),mu=11+.9*(mu-11),rawvariances*1.1; variant1: floor(k*1.25),mu=11+1.1*(mu-11),rawvariances*.9; lastinputheldifindexexceedsrecord. Z samewarpedtime. Initialization allnoiseandhistoryzero/Vreset; burn500msearly/1000mslate.',
        new_validation_reference='IndependentGaussianLIF8192replicates,minroundedtooriginalN,seed920141,dt.1/.05. Same50ms bins,2stepgate max(.05,3MCsplit). Ratefine/coarsebinL2<=.02.',
        stop='One prescribed form andonefitperpopulation only. No scale/feature/weight tuning afterscoring. Ifdevelopmental ornewlocalchecks fail, retainfailure anddo notlaunch a network fromthiscandidate. Existingglobal/localfailurescannotbewaived.',
        budget='Oneanalytic-invariantcheck, two scalarfits, fixedexistingtransfer and14newLIFinputsat2steps. No wholenetworkorbranch launch underthiscontract.',
        model_promoted=False))


def inverse_flux(rate,ref):
    r=rate/1000;occ=np.zeros_like(r)
    for j in range(r.shape[1]):
        for lag in range(1,round(ref[j]/.1)):
            occ[lag:,j]+=r[:-lag,j]*.1
    p=r*.1/(1-occ);assert p.min()>0 and p.max()<1
    ell=np.log(p)-np.log1p(-p);back,w=expected_flux(ell,ref)
    error=float(np.max(abs(back-rate)));assert error<1e-8 and w<=1+1e-12
    return ell,error


def inputs(kind):
    if kind=='early':
        s=model(40);info=np.load(EARLY/'membership.npz');groups=info['selected_groups']
        parts=[]
        for p in sorted((EARLY/'inputs').glob('*.npz')):
            with np.load(p) as z:parts.append(z['moments']);names=z['moment_names'].tolist()
        m=np.concatenate(parts);pr=np.load(EARLY/'projected_inputs.npz')
        scale=2*(s.rise+s.decay)[:,None]/s.tm[groups]
        physical=np.stack([m[:,names.index('net')],scale[0]*pr['filtered_moments'][:,6],
            scale[1]*m[:,names.index('z')]**2*pr['filtered_moments'][:,7]],axis=2)
        f=features(physical,s.theta[groups]);pop=info['population'][groups];ref=np.where(pop==0,2.,1.)
        z=np.load(EARLY/'fixed_readout.npz');ell,error=inverse_flux(z['native_mean_private_variance'],ref)
        raw=np.concatenate([m[:,names.index('net'),None,:],pr['raw_private_variance_forcing'],m[:,names.index('z'),None,:]],axis=1).transpose(2,1,0).copy()
        return dict(groups=groups,N=info['group_size'][groups],pop=pop,ref=ref,theta=s.theta[groups],f=f,ell=ell,
            raw_wave=raw,time_ms=z['time_ms'],starts=z['bin_start_ms'],inverse_error=error)
    info=np.load(LATE/'selected_input_history.npz');z=np.load(LATE/'fixed_readout.npz');m={name:info['moments'][:,i] for i,name in enumerate(info['moment_names'])}
    r=info['reconstructed'];pop=info['population'];tm=np.where(pop==0,20.,10.)
    from lif_mc import PARAMS
    sums=[PARAMS['tau_r_AMPA']+PARAMS['tau_d_AMPA'],PARAMS['tau_r_GABA']+PARAMS['tau_d_GABA']]
    physical=np.stack([r[:,0]-m['z']*r[:,1]-m['mcurrent'],2*sums[0]/tm*r[:,6],2*sums[1]/tm*m['z']**2*r[:,7]],axis=2)
    ref=np.where(pop==0,2.,1.);ell,error=inverse_flux(z['projected_private'],ref)
    raw=np.load(LATE/'selected_raw_variance_forcing.npz')['raw_variances']
    wave=np.stack([physical[:,:,0],raw[:,2],raw[:,3],m['z']],axis=1).transpose(2,1,0).copy()
    return dict(groups=info['groups'],N=info['group_size'],pop=pop,ref=ref,theta=info['theta'],
        f=direct_features(physical,info['theta']),ell=ell,raw_wave=wave,time_ms=z['time_ms'],starts=z['bin_start_ms'],inverse_error=error)


def integrate_bins(r,case):
    return np.stack([r[(case['time_ms']>=lo)&(case['time_ms']<lo+50)].sum(0)*.0001*case['N'] for lo in case['starts']])


def references(kind,dt):
    folder=EARLY/'local_lif_reference' if kind=='early' else LATE/'local_lif'
    z=np.load(folder/f'dt{dt:g}.npz');G=16 if kind=='early' else 6
    return np.stack([z['counts'][j,:int(z['replicates'][j])].sum(0,dtype=np.uint64)/float(z['replicates'][j])*z['group_sizes'][j] for j in range(G)],axis=1)


def fit():
    c=read(DEST/'contract.json');assert not (DEST/'locked.json').exists()
    x=inputs('early');H=np.mean(x['f'][:,:,3:]**2,axis=2);target=references('early',.1)
    rate,_=expected_flux(x['ell'],x['ref']);rows=[];pars={}
    train=(x['time_ms']>=500)&(x['time_ms']<1500)
    for pop in 'EI':
        ids=np.array([np.flatnonzero(x['groups']==g)[0] for g in c['training_groups'][pop]])
        h=H[train][:,ids].ravel();weights=rate[train][:,ids].ravel();order=np.argsort(h)
        scale=float(h[order][np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)])
        assert scale>0
        gate=H[:,ids]/(H[:,ids]+scale);truth=target[:20,ids];norm=np.sum(truth**2,axis=0)
        def objective(b):
            r,_=expected_flux(x['ell'][:,ids]+b*gate,x['ref'][ids])
            predicted=r.reshape(60,500,len(ids)).sum(1)[10:30]*.0001*x['N'][ids]
            return float(np.mean(np.sum((predicted-truth)**2,axis=0)/norm))
        fit=minimize_scalar(objective,bounds=(-np.log(4),np.log(4)),method='bounded',options={'maxiter':50,'xatol':1e-8})
        assert fit.success
        pars[pop]=dict(b=float(fit.x),h=scale)
        rows.append(dict(pop=pop,**pars[pop],objective=float(fit.fun),parent_objective=objective(0.),evaluations=fit.nfev))
    write(DEST/'locked.json',dict(status='LOCKED_BEFORE_TRANSFER_AND_NEW_REFERENCES',parameters=pars,fit=rows,native_outputs_used=False,parent_weight_files_unchanged=True))
    # Explicit independent stationary-gradient and second-order scaling check.
    rng=np.random.default_rng(920140);checks=[]
    for pop,p in pars.items():
        d=rng.normal(size=36);H0=np.mean(np.zeros(36)**2)
        assert p['b']*H0/(H0+p['h'])==0
        def delta(v):q=np.mean(v*v);return p['b']*q/(q+p['h'])
        central=(delta(d*1e-6)-delta(-d*1e-6))/(2e-6);assert central==0
        ratio=delta(d*5e-7)/delta(d*1e-6);assert abs(ratio-.25)<1e-6
        checks.append(dict(pop=pop,stationary_correction=0,first_directional_derivative=central,half_amplitude_ratio=ratio))
    write(DEST/'invariant_check.json',dict(status='PASS',checks=checks,scope='Parentstationaryresponseandlinearizationunchanged; originalfailedvalidationisnotexcused. No stabilitylabelinferred.'))
    log('TRANSIENT RESPONSE FIT',rows)


def transfer():
    locked=read(DEST/'locked.json');assert not (DEST/'transfer_result.json').exists();rows=[]
    for kind in ['early','late']:
        x=inputs(kind);H=np.mean(x['f'][:,:,3:]**2,axis=2)
        b=np.array([locked['parameters']['E' if p==0 else 'I']['b'] for p in x['pop']])
        h=np.array([locked['parameters']['E' if p==0 else 'I']['h'] for p in x['pop']])
        gate=H/(H+h);r,occupancy=expected_flux(x['ell']+b*gate,x['ref']);pred=integrate_bins(r,x)
        assert occupancy<=1+1e-12
        windows=[('full',slice(None)),('later',slice(20,None))] if kind=='early' else [('full',slice(None))]
        base,_=expected_flux(x['ell'],x['ref']);baseline=integrate_bins(base,x)
        for dt in [.1,.05]:
            truth=references(kind,dt)
            for window,sl in windows:
                for j,g in enumerate(x['groups']):
                    t=truth[sl,j];p=pred[sl,j];q=baseline[sl,j];den=max(np.linalg.norm(t),1.)
                    l2=float(np.linalg.norm(p-t)/den);bias=float(abs(p.sum()/t.sum()-1))
                    rows.append(dict(kind=kind,window=window,dt_ms=dt,group=int(g),pop='E' if x['pop'][j]==0 else 'I',
                        L2=l2,count_error=bias,parent_L2=float(np.linalg.norm(q-t)/den),
                        passed=bool(l2<=.15 and bias<=.1)))
        np.savez_compressed(DEST/f'{kind}_readout.npz',time_ms=x['time_ms'],groups=x['groups'],rate_hz=r,counts=pred,
            gate=gate,history_energy=H,bin_start_ms=x['starts'],offset_by_group=b,scale_by_group=h)
    passed=all(r['passed'] for r in rows)
    write(DEST/'transfer_result.json',dict(status='TRANSFER_PASS' if passed else 'TRANSFER_FAIL',rows=rows,
        passed=sum(r['passed'] for r in rows),total=len(rows),model_promoted=False,new_validation_pending=True,
        scope='Previously inspected developmentprofiles; not blindvalidation. No parameterretuning ornetworklaunch allowed.'))
    log('TRANSIENT RESPONSE TRANSFER',sum(r['passed'] for r in rows),'/',len(rows),[r for r in rows if not r['passed']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','fit','transfer']);a=p.parse_args()
    {'register':register,'fit':fit,'transfer':transfer}[a.command]()
