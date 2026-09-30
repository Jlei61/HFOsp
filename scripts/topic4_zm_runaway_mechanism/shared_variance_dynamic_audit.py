"""Exact Poisson current-covariance kernels versus a stationary-only split.

This isolates allocation error by applying the same exact double-exponential
current kernel to both sides. It does not replace the calibrated variance-rate
response or claim that native correlated spikes are independent Poisson.
"""
from common import *
from shared_variance_delay_audit import kernels


def square_transfer(lam,tr,td):
    return (1/(lam+2/td)-2/(lam+1/tr+1/td)+1/(lam+2/tr))/(td-tr)**2


def cross_transfer(lam,d,e,tr,td):
    gap=abs(d-e);ed=np.exp(-gap/td);er=np.exp(-gap/tr)
    return np.exp(-lam*max(d,e))*(ed/(lam+2/td)-(ed+er)/(lam+1/tr+1/td)+er/(lam+2/tr))/(td-tr)**2


def common_coefficients(a,delay_step,tr,td):
    """Collect equal-max-delay terms of every ordered pair without truncation."""
    a=a.toarray();cd=np.empty_like(a);cr=np.empty_like(a)
    pd=np.zeros(len(a));pr=np.zeros(len(a))
    ed=np.exp(-delay_step/td);er=np.exp(-delay_step/tr)
    for j in range(a.shape[1]):
        pd*=ed;pr*=er;x=a[:,j]
        cd[:,j]=x*x+2*x*pd;cr[:,j]=x*x+2*x*pr
        pd+=x;pr+=x
    return cd,cr


def main():
    c=read(OUT/'shared_variance_dynamic_contract.json');s=model()
    orbit=np.load(OUT/c['cycle_source']);rates=orbit['r'].mean(0)
    assert rates.min()>=0 and float(orbit['residual'])<2e-8
    freqs=sorted(set(c['frequencies_Hz']+[1000/float(orbit['T'])]))
    profiles={'uniform_10Hz':np.full(s.P,.01),'old_candidate_cycle_mean':rates}
    masks={'E':s.E,'I':~s.E}
    masks.update({name:s.E&(s.geo['group_region']==k) for k,name in enumerate(['Core A E','Core B E','Surround E'])})
    rows=[];checks=[];saved={}
    for k,kind in enumerate(['ampa','gaba']):
        target,source,a=s.raw[k];qt,qs,q=s.raw[k+2]
        assert np.array_equal(target,qt) and np.array_equal(source,qs)
        tr,td=s.rise[k],s.decay[k];size=s.sizes[source]
        cd,cr=common_coefficients(a,.1,tr,td)
        total0=np.asarray(q.sum(1)).ravel()*square_transfer(0.,tr,td)
        bd0=cd.sum(1);br0=cr.sum(1)
        common0=(bd0/(2/td)-(bd0+br0)/(1/tr+1/td)+br0/(2/tr))/(td-tr)**2/size
        fraction=common0/total0
        assert fraction.min()>=-1e-12 and fraction.max()<=1+1e-12
        kc,_,_,_=kernels(tr,td,.1,len(s.delays))
        delta=abs(np.arange(len(s.delays))[:,None]-np.arange(len(s.delays))[None,:])
        previous=np.asarray(a.multiply(a@kc[delta]).sum(1)).ravel()/size*square_transfer(0.,tr,td)
        dc_error=float(np.max(abs(previous-common0)/np.maximum(total0,1e-30)))
        assert dc_error<1e-12,dc_error
        subset=np.unique(np.linspace(0,len(target)-1,11).astype(int))
        manual_errors=[]
        per_frequency=[]
        for freq in freqs:
            lam=2j*np.pi*freq/1000;phase=np.exp(-lam*s.delays)
            total=(q@phase)*square_transfer(lam,tr,td)
            bd=cd@phase;br=cr@phase
            common=(bd/(lam+2/td)-(bd+br)/(lam+1/tr+1/td)+br/(lam+2/tr))/(td-tr)**2/size
            allocated=(1-fraction)*total+common
            if freq in [0,10,80]:
                for j in subset:
                    sparse_row=a.getrow(j);d=s.delays[sparse_row.indices];w=sparse_row.data
                    direct=sum(w[ii]*w[jj]*cross_transfer(lam,d[ii],d[jj],tr,td)
                               for ii in range(len(w)) for jj in range(len(w)))/size[j]
                    manual_errors.append(float(abs(direct-common[j])/max(total0[j],1e-30)))
            per_frequency.append((total,common,allocated))
            for name,r in profiles.items():
                def aggregate(x):
                    x=x*r[source]
                    return np.bincount(target,weights=x.real,minlength=s.P)+1j*np.bincount(target,weights=x.imag,minlength=s.P)
                dc=np.bincount(target,weights=total0*r[source],minlength=s.P)
                actual=aggregate(total);split=aggregate(allocated);error=split-actual
                for label,mask in masks.items():
                    mask=mask&(dc>0);weights=s.sizes[mask]
                    norm=abs(error[mask])/dc[mask]
                    weights=weights/weights.sum()
                    row=dict(synapse=kind,profile=name,target=label,frequency_Hz=freq,
                             cell_weighted_mean_error_over_DC=float(weights@norm),
                             max_error_over_DC=float(norm.max()),
                             error_over_DC_quantiles=np.quantile(norm,[0,.25,.5,.75,1]).tolist(),
                             cell_weighted_total_response_over_DC=float(weights@(abs(actual[mask])/dc[mask])))
                    rows.append(row)
                saved[f'{kind}_{name}_{freq:.9g}']=np.stack([dc,actual,split,error])
        assert max(manual_errors)<1e-12,max(manual_errors)
        # A direct time-domain impulse check on dispersed group pairs.
        time=np.linspace(0,s.delays[-1]+40*td,12001)
        def h(x):
            xp=np.maximum(x,0.)
            return np.where(x>=0,(np.exp(-xp/td)-np.exp(-xp/tr))/(td-tr),0.)
        impulse=[]
        for j in subset:
            ar=a.getrow(j);qr=q.getrow(j)
            shared_signal=h(time[:,None]-s.delays[ar.indices])@ar.data
            total_signal=h(time[:,None]-s.delays[qr.indices])**2@qr.data
            common_signal=shared_signal**2/size[j];private=total_signal-common_signal
            scale=max(total_signal.max(),1e-30)
            minimum=float(private.min()/scale)
            assert minimum>=-1e-10,minimum
            integrated=float(np.trapz(common_signal,time))
            integral_error=abs(integrated-common0[j])/max(total0[j],1e-30)
            assert integral_error<2e-5,integral_error
            impulse.append(dict(pair_index=int(j),target_group=int(target[j]),source_group=int(source[j]),
                                exact_private_minimum_relative=minimum,common_integral_error_over_total_DC=integral_error))
        checks.append(dict(synapse=kind,DC_identity_error=dc_error,
                           direct_pair_sum_error_max=max(manual_errors),impulse_checks=impulse))
        del cd,cr,per_frequency
    dest=OUT/'shared_variance_dynamic_audit';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'responses.npz',**saved)
    result=dict(status='DYNAMIC_POISSON_SPLIT_DIAGNOSTIC_COMPLETE',frequencies_Hz=freqs,rows=rows,checks=checks,
        exact_formula='Total=sum_d Q_d exp(-lambda*d) Laplace[h^2]; common=sum_de A_d A_e Laplace[h(t-d)h(t-e)]/N; private=total-common.',
        compared_formula='Constant corrected fraction f=common(0)/total(0): allocated=(1-f)*total(lambda)+common(lambda).',
        scope=c['scope'],model_modified=False,network_runs=0,replacement_promoted=False)
    write(dest/'result.json',result)
    log('DYNAMIC SPLIT RESULT',[(r['synapse'],r['target'],r['frequency_Hz'],r['cell_weighted_mean_error_over_DC'])
        for r in rows if r['profile']=='old_candidate_cycle_mean' and r['target']=='E'])


if __name__=='__main__':main()
