"""Stationary covariance identity for the stochastic group's variance split.

The existing split subtracts (sum_d A_ghd)^2/N_h before synaptic filtering.
The sampled group counts instead generate sum_de A_ghd A_ghe K(d-e)/N_h.
Here K is the normalized current-filter autocorrelation. No network is run.
"""
from common import *


def kernels(rise,decay,dt,nlag):
    lag=np.arange(nlag)*dt
    continuous=(decay*np.exp(-lag/decay)-rise*np.exp(-lag/rise))/(decay-rise)
    # Same Heun update with input values at both known delayed endpoints.
    A=np.array([[-1/rise,0.],[1/decay,-1/decay]])
    B=np.array([1/rise,0.]);F=np.eye(2)+dt*A+.5*dt*dt*(A@A)
    b0=.5*dt*(B+dt*A@B);b1=.5*dt*B
    h=np.empty(int(np.ceil(50*decay/dt))+2)
    y=b1.copy();h[0]=y[1]
    y=F@y+b0;h[1]=y[1]
    for j in range(2,len(h)):
        y=F@y;h[j]=y[1]
    dc=float(h.sum());assert abs(dc-1)<1e-12,dc
    power=float(h@h)
    discrete=np.array([power if k==0 else h[:-k]@h[k:] for k in range(nlag)])/power
    assert continuous[0]==1 and discrete[0]==1
    assert np.all(np.diff(continuous)<=0) and np.all(np.diff(discrete)<=1e-15)
    return continuous,discrete,power/dt,dict(DC_gain=dc,
        continuous_C0_per_ms=1/(2*(rise+decay)),Heun_C0_per_ms=power/dt,
        Heun_to_continuous_C0_ratio=power/dt*2*(rise+decay))


def main():
    c=read(OUT/'shared_variance_delay_contract.json');s=model();dt=.1
    z=np.load(OUT/c['cycle_source']);cycle=z['r'].mean(0)
    assert cycle.min()>=0 and float(z['residual'])<2e-8
    profiles={'uniform_10Hz':np.full(s.P,.01),'candidate_cycle_mean':cycle}
    regions=s.geo['group_region'];masks={'E':s.E,'I':~s.E,
        'Core A E':s.E&(regions==0),'Core B E':s.E&(regions==1),'Surround E':s.E&(regions==2)}
    rows=[];arrays={};checks=[]
    for k,kind in enumerate(['ampa','gaba']):
        target,source,a=s.raw[k];tq,sq,q=s.raw[k+2]
        assert np.array_equal(target,tq) and np.array_equal(source,sq)
        aa=np.asarray(a.sum(1)).ravel();qq=np.asarray(q.sum(1)).ravel()
        removed=aa*aa/s.sizes[source]
        assert np.all(removed<=qq*(1+1e-12))
        kc,kd,c0d,meta=kernels(s.rise[k],s.decay[k],dt,len(s.delays))
        distance=abs(np.arange(len(kc))[:,None]-np.arange(len(kc))[None,:])
        covariance=[]
        for kernel in [kc,kd]:
            contracted=a@kernel[distance]
            v=np.asarray(a.multiply(contracted).sum(1)).ravel()/s.sizes[source]
            assert np.all(v>=-1e-14) and np.all(v<=removed*(1+1e-12))
            covariance.append(v)
            del contracted
        # Independent sum over actual nonzero delay pairs for dispersed rows.
        idx=np.unique(np.linspace(0,len(aa)-1,17).astype(int))
        err=0.
        for j in idx:
            one=a.getrow(j);dd=one.indices;w=one.data
            manual=sum(w[ii]*w[jj]*kc[abs(dd[ii]-dd[jj])]
                       for ii in range(len(w)) for jj in range(len(w)))/s.sizes[source[j]]
            err=max(err,abs(manual-covariance[0][j])/max(abs(manual),1e-30))
        assert err<1e-12,err
        checks.append(dict(kind=kind,independent_double_sum_relative_max=err,**meta))
        for name,r in profiles.items():
            group=lambda x:np.bincount(target,weights=x*r[source],minlength=s.P)
            total=group(qq);subtracted=group(removed)
            shared_c=group(covariance[0]);shared_d=group(covariance[1])
            old_continuous=(total-subtracted)+shared_c
            old_heun=(total-subtracted)+shared_d*meta['Heun_to_continuous_C0_ratio']
            delay_error=(old_continuous-total)/np.maximum(total,1e-30)
            implemented_error=(old_heun-total)/np.maximum(total,1e-30)
            for label,mask in masks.items():
                mask=mask&(total>0);weight=s.sizes[mask]
                rows.append(dict(synapse=kind,profile=name,target=label,groups=int(mask.sum()),
                    E_or_I_cell_weighted_delay_error=float(np.average(delay_error[mask],weights=weight)),
                    E_or_I_cell_weighted_Heun_total_error=float(np.average(implemented_error[mask],weights=weight)),
                    delay_error_quantiles=np.quantile(delay_error[mask],[0,.25,.5,.75,1]).tolist(),
                    Heun_total_error_quantiles=np.quantile(implemented_error[mask],[0,.25,.5,.75,1]).tolist(),
                    weighted_old_removed_fraction=float(np.average(subtracted[mask]/total[mask],weights=weight)),
                    weighted_actual_shared_fraction=float(np.average(shared_c[mask]/total[mask],weights=weight))))
            arrays[kind+'_'+name]=np.stack([total,subtracted,shared_c,shared_d,delay_error,implemented_error])
    dest=OUT/'shared_variance_delay_audit';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'group_diagnostics.npz',**arrays)
    result=dict(status='STATIONARY_POISSON_SPLIT_DIAGNOSTIC_COMPLETE',checks=checks,rows=rows,
        formula='Q_total - A_sum^2/N + sum_de A_d*A_e*K(delay_d-delay_e)/N; each term weighted by stationary source rate',
        units='Dimensionless relative error in stationary recurrent current variance; per-source operator intensity before tm and area factors.',
        scope=c['interpretation'],model_modified=False)
    write(dest/'result.json',result);log('SHARED DELAY VARIANCE AUDIT',result)


if __name__=='__main__':main()
