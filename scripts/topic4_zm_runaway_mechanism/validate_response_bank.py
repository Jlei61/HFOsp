"""Fail-fast local validation of the single preregistered response candidate."""
from response_bank import *
from response_voltage_units_audit import response as original_response
from scipy.fft import rfft,irfft


def main():
    fit=read(DEST/'fit_result.json');assert fit['status']=='TRAINING_FIT_COMPLETE'
    c=read(OUT/'response_bank_candidate_contract.json');tabs=tables();s=model()
    coefficients=np.load(DEST/'coefficients.npz');checks=[]
    for pop in 'EI':
        x,e,h=np.meshgrid(coefficients['x_'+pop],coefficients['sE_'+pop],coefficients['sI_'+pop],indexing='ij')
        mu=11+7*x.ravel();ve=(7*e.ravel())**2;vi=(7*h.ravel())**2
        actual=tabs[pop].evaluate(mu,ve,vi,np.full(mu.shape,18.))
        expected=coefficients['values_'+pop].reshape(3,5,-1)
        err=float(np.max(abs(actual-expected)));assert np.allclose(actual,expected,rtol=1e-8,atol=1e-8),err
        instantaneous=1+expected[0].sum(0)
        checks.append(dict(pop=pop,knot_interpolation_max_error=err,
                           instantaneous_mean_factor_min=float(instantaneous.min()),
                           negative_instantaneous_mean_grid_points=int((instantaneous<0).sum())))
    source=read(OUT/'response_error_decomposition.json');reference=read(BASE/'dynamic_assay/validation_result.json')['rows']
    channels=['mean','variance_E','variance_I'];rows=[];tau=np.array(c['filter_times_ms'])
    for old in source['rows']:
        q=old['workpoint'];pop=q['pop'];ch=channels.index(old['channel']);f=old['frequency_hz']
        origin=original_response(s,q,f,ch,False)
        candidates=[r for r in reference if r['kind']==old['kind'] and r['pop']==pop and r['channel']==old['channel']
                    and r['frequency_hz']==f and abs(complex(*r['predicted'])-origin)<1e-10]
        assert len(candidates)==1
        ref=candidates[0];theta=np.array([q['theta']]);mu=np.array([q['mu']]);ve=np.array([q['ve']]);vi=np.array([q['vi']])
        b=tabs[pop].evaluate(mu,ve,vi,theta)[:,:,0];g=s.spline[pop].evaluate(mu,ve,vi,theta)
        w=2j*np.pi*f/1000;basis=w*tau/(1+w*tau)
        pred=g['d_mu'][0]*1000*(1+b[0]@basis) if ch==0 else (
            g['d_ve' if ch==1 else 'd_vi'][0]*1000+g['d_mu'][0]*1000*(b[ch]@basis)/(q['theta']-11))/(1+w*s.tau[ch-1]/2)
        error=abs(pred-complex(*ref['measured']))/max(abs(ref['dc_measured']),1e-12)
        rows.append(dict(kind=old['kind'],pop=pop,channel=old['channel'],frequency_hz=f,counted=old['counted'],
                         old_error=old['original_error'],new_error=float(error),predicted=[pred.real,pred.imag],
                         old_failed=old['original_failed'],new_failed=bool(error>old['tol']) if old['counted'] else None))
    counted=[r for r in rows if r['counted']];failed=sum(r['new_failed'] for r in counted)
    response_summary={ch:dict(n=sum(r['channel']==ch for r in counted),
                              failed=sum(r['new_failed'] for r in counted if r['channel']==ch)) for ch in channels}
    parent=OUT/'native_cycle_waveform_response';z=np.load(parent/'prepared.npz');obs=np.load(parent/'response.npz')
    primary=read(parent/'result.json');groups=read(parent/'preparation.json')['groups'];W=z['wave'].shape[-1];T=float(z['T_ms'])
    lam=2j*np.pi*np.arange(W//2+1)/T
    def filt(x,t):return irfft(rfft(x)/(1+lam*t),n=W)
    phase=((np.arange(primary['steps'])+1)*primary['dt_ms']/T)%1;B=obs['measured_hz'].shape[1]
    bins=np.minimum((phase*B).astype(int),B-1);exposure=np.bincount(bins,minlength=B)
    pos=phase*W;lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;a=pos-np.floor(pos)
    wave_rows=[];outputs=[]
    for j,group in enumerate(groups):
        pop=group['pop'];mu,rawE,rawI=z['wave'][j]
        ve=filt(rawE,s.tau[0]/2);vi=filt(rawI,s.tau[1]/2);p=s.resp.poles[pop]
        anchor=[filt(mu,p['tau_s']),filt(ve,p['tau_vE']),filt(vi,p['tau_vI'])]
        th=np.full(W,group['theta_mv']);b=tabs[pop].evaluate(*anchor,th)
        differences=np.array([[value-filt(value,t) for t in tau] for value in [mu,ve,vi]])
        eff=mu+np.sum(b[0]*differences[0],axis=0)+np.sum(b[1:]*differences[1:],axis=(0,1))/(th-11)
        rate=s.spline[pop].evaluate(eff,ve,vi,th)['rate']*1000
        predicted=np.bincount(bins,weights=(1-a)*rate[lo]+a*rate[hi],minlength=B)/exposure;outputs.append(predicted)
        measured=obs['measured_hz'][j];err=float(np.linalg.norm(predicted-measured)/max(np.linalg.norm(measured),np.sqrt(B)))
        mean=float(predicted@obs['occupancy_ms']/obs['occupancy_ms'].sum());mc=primary['rows'][j]['MC_cycle_mean_hz']
        bias=abs(mean-mc)/max(mc,1)
        wave_rows.append(dict(label=group['label'],group=group['group'],waveform_error=err,relative_mean_error=bias,
                              predicted_mean_hz=mean,MC_mean_hz=mc,passed=bool(err<=.15 and bias<=.1)))
    q=dict(status='LOCAL_VALIDATION_COMPLETE',linear_response_verdict='PASS' if failed<=max(2,int(.1*len(counted))) else 'FAIL',
           n_counted=len(counted),n_failed=failed,by_channel=response_summary,response_rows=rows,
           waveform_verdict='PASS' if all(r['passed'] for r in wave_rows) else 'FAIL',waveform_rows=wave_rows,
           implementation_checks=checks,network_promoted=False,
           scope='One candidate, original independent response tests and previously observed waveform diagnostics. No fit to either target. No autonomous network or bifurcation evidence.')
    np.savez_compressed(DEST/'waveform_predictions.npz',predicted_hz=outputs,measured_hz=obs['measured_hz'],phase_centres=obs['phase_centres'])
    write(DEST/'validation_result.json',q);log('BANK LOCAL VALIDATION',{k:v for k,v in q.items() if k!='response_rows'})


if __name__=='__main__':main()
