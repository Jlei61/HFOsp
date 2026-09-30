"""Test history-conditioned response weights without fitting or deploying a model.

At a steady input all filtered moments equal instantaneous moments. Weight
derivatives multiply zero filter differences, so both variants preserve the
existing static curve AND local first-order response (including its failures).
"""
from common import *
from scipy.fft import rfft,irfft


def main():
    c=read(OUT/'native_cycle_history_weights_contract.json')
    dest=OUT/'native_cycle_waveform_response';z=np.load(dest/'prepared.npz')
    obs=np.load(dest/'response.npz');primary=read(dest/'result.json')
    groups=read(dest/'preparation.json')['groups'];s=model()
    W=z['wave'].shape[-1];T=float(z['T_ms']);lam=2j*np.pi*np.arange(W//2+1)/T
    def filt(x,tau):return irfft(rfft(x)/(1+lam*tau),n=W)
    phase=((np.arange(primary['steps'])+1)*primary['dt_ms']/T)%1
    B=obs['measured_hz'].shape[1];bins=np.minimum((phase*B).astype(int),B-1)
    exposure=np.bincount(bins,minlength=B);pos=phase*W
    lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;a=pos-np.floor(pos)
    def aggregate(x):return np.bincount(bins,weights=(1-a)*x[lo]+a*x[hi],minlength=B)/exposure
    rows=[];all_outputs=[]
    for j,g in enumerate(groups):
        mu,rawE,rawI=z['wave'][j];vE=filt(rawE,s.tau[0]/2);vI=filt(rawI,s.tau[1]/2)
        th=np.full(W,g['theta_mv']);p=s.resp.poles[g['pop']];tab=s.resp.tables[g['pop']]
        mus=filt(mu,p['tau_s']);vEf=filt(vE,p['tau_cE']);vIf=filt(vI,p['tau_cI'])
        vEv=filt(vE,p['tau_vE']);vIv=filt(vI,p['tau_vI'])
        old,_=tab.evaluate(mu,vE,vI,th)
        mean_history,_=tab.evaluate(mus,vE,vI,th)
        all_history,_=tab.evaluate(mus,vEv,vIv,th)
        mean_only=old.copy();mean_only[0]=mean_history[0]
        outputs=[]
        for weights in [old,mean_only,all_history]:
            al,aE,aI,eE,eI=weights
            eff=al*mu+(1-al)*mus+eE*(vE-vEf)+eI*(vI-vIf)
            e=np.maximum(aE*vE+(1-aE)*vEv,0);h=np.maximum(aI*vI+(1-aI)*vIv,0)
            outputs.append(aggregate(s.spline[g['pop']].evaluate(eff,e,h,th)['rate']*1000))
        outputs=np.array(outputs);all_outputs.append(outputs)
        parity=np.linalg.norm(outputs[0]-obs['predicted_hz'][j])/np.linalg.norm(obs['predicted_hz'][j])
        assert parity<1e-3,parity
        measured=obs['measured_hz'][j];means=outputs@obs['occupancy_ms']/obs['occupancy_ms'].sum()
        group_rows=[]
        for k,name in enumerate(c['variants']):
            error=float(np.linalg.norm(outputs[k]-measured)/max(np.linalg.norm(measured),np.sqrt(B)))
            bias=float(abs(means[k]-primary['rows'][j]['MC_cycle_mean_hz'])/max(primary['rows'][j]['MC_cycle_mean_hz'],1))
            group_rows.append(dict(variant=name,relative_waveform_L2=error,relative_mean_error=bias,
                                   mean_hz=float(means[k]),passed=bool(error<=.15 and bias<=.1)))
        rows.append(dict(label=g['label'],group=g['group'],base_parity=float(parity),rows=group_rows))
    np.savez_compressed(dest/'history_weights_response.npz',predicted_hz=all_outputs,
                        variants=c['variants'],measured_hz=obs['measured_hz'],phase_centres=obs['phase_centres'])
    q=dict(status='DIAGNOSTIC_COMPLETE',groups=rows,scope=c['scope'],
           variants_all_selected_groups_pass={name:all(g['rows'][k]['passed'] for g in rows)
               for k,name in enumerate(c['variants'])})
    write(dest/'history_weights_result.json',q);log('HISTORY WEIGHTS',q)


if __name__=='__main__':main()
