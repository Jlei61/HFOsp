"""Open-loop component diagnostics of the frozen response; never a fitted model."""
from native_cycle_waveform_response import DEST
from common import *
from scipy.fft import rfft, irfft


def main():
    contract=read(OUT/'native_cycle_response_components_contract.json')
    s=model();z=np.load(DEST/'prepared.npz');observed=np.load(DEST/'response.npz')
    primary=read(DEST/'result.json');groups=read(DEST/'preparation.json')['groups']
    W=z['wave'].shape[-1];T=float(z['T_ms']);lam=2j*np.pi*np.arange(W//2+1)/T
    def filt(x,tau):return irfft(rfft(x)/(1+lam*tau),n=W)
    phase=((np.arange(primary['steps'])+1)*primary['dt_ms']/T)%1
    B=observed['measured_hz'].shape[1];bins=np.minimum((phase*B).astype(int),B-1)
    exposure=np.bincount(bins,minlength=B)
    pos=phase*W;lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;a=pos-np.floor(pos)
    def aggregate(x):
        return np.bincount(bins,weights=(1-a)*x[lo]+a*x[hi],minlength=B)/exposure
    names=contract['variants'];result=[];curves=[];parity=[]
    for j,g in enumerate(groups):
        mu,rawE,rawI=z['wave'][j];vE=filt(rawE,s.tau[0]/2);vI=filt(rawI,s.tau[1]/2)
        th=np.full(W,g['theta_mv']);p=s.resp.poles[g['pop']]
        weights,_=s.resp.tables[g['pop']].evaluate(mu,vE,vI,th)
        al,aE,aI,eE,eI=weights
        mus=filt(mu,p['tau_s']);vEf=filt(vE,p['tau_cE']);vIf=filt(vI,p['tau_cI'])
        vEv=filt(vE,p['tau_vE']);vIv=filt(vI,p['tau_vI'])
        mean=al*mu+(1-al)*mus;cross=eE*(vE-vEf)+eI*(vI-vIf)
        varE=np.maximum(aE*vE+(1-aE)*vEv,0);varI=np.maximum(aI*vI+(1-aI)*vIv,0)
        def exact_variance(raw,k):
            tr,td=s.rise[k],s.decay[k]
            h=1/((1+lam*tr/2)*(1+lam*tr*td/(tr+td))*(1+lam*td/2))
            return np.maximum(irfft(rfft(raw)*h,n=W),0)
        variants=[(mean+cross,varE,varI),(mu+cross,varE,varI),
                  (mean,varE,varI),(mean+cross,vE,vI),(mu,vE,vI),
                  (mu,exact_variance(rawE,0),exact_variance(rawI,1))]
        outputs=np.array([aggregate(s.spline[g['pop']].evaluate(m,e,h,th)['rate']*1000)
                          for m,e,h in variants])
        reference=observed['predicted_hz'][j]
        delta=float(np.linalg.norm(outputs[0]-reference)/max(np.linalg.norm(reference),1))
        parity.append(delta)
        assert delta<contract['acceptance']['base_reproduction_relative_L2_max'],delta
        measured=observed['measured_hz'][j];denom=max(np.linalg.norm(measured),np.sqrt(B))
        means=outputs@observed['occupancy_ms']/observed['occupancy_ms'].sum()
        mcmean=primary['rows'][j]['MC_cycle_mean_hz']
        rows=[dict(variant=name,relative_waveform_L2=float(np.linalg.norm(out-measured)/denom),
                   predicted_cycle_mean_hz=float(means[k]),
                   relative_mean_error=float(abs(means[k]-mcmean)/max(mcmean,1)))
              for k,(name,out) in enumerate(zip(names,outputs))]
        result.append(dict(group=g['group'],label=g['label'],rows=rows));curves.append(outputs)
    np.savez_compressed(DEST/'component_response.npz',predicted_hz=curves,variants=names,
                        measured_hz=observed['measured_hz'],phase_centres=observed['phase_centres'])
    q=dict(status='DIAGNOSTIC_COMPLETE',base_reproduction_relative_L2=parity,groups=result,
           scope=contract['scope'])
    write(DEST/'component_result.json',q);log('RESPONSE COMPONENT DIAGNOSTIC',q)


if __name__=='__main__':main()
