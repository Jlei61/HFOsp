"""Input-space versus output-space mean mixing at identical linear response.

No coefficients fitted. Static Phi, poles, corrected voltage units, variance
readout and input waveforms are fixed. Output mixing is convex because the
existing alpha is in[0,1], so it preserves physical rate bounds.
"""
from common import *
from scipy.fft import rfft,irfft
from response_voltage_units import VoltageScaledResponseTable


def main():
    c=read(OUT/'nonlinear_mean_readout_contract.json')
    source=OUT/'native_cycle_waveform_response';dest=OUT/'nonlinear_mean_readout';dest.mkdir(exist_ok=True)
    z=np.load(source/'prepared.npz');obs=np.load(source/'response.npz')
    primary=read(source/'result.json');groups=read(source/'preparation.json')['groups'];s=model()
    W=z['wave'].shape[-1];T=float(z['T_ms']);lam=2j*np.pi*np.arange(W//2+1)/T
    def filt(x,tau):return irfft(rfft(x)/(1+lam*tau),n=W)
    phase=((np.arange(primary['steps'])+1)*primary['dt_ms']/T)%1
    B=obs['measured_hz'].shape[1];bins=np.minimum((phase*B).astype(int),B-1)
    exposure=np.bincount(bins,minlength=B);pos=phase*W;lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;frac=pos-np.floor(pos)
    def aggregate(x):return np.bincount(bins,weights=(1-frac)*x[lo]+frac*x[hi],minlength=B)/exposure
    rng=np.random.default_rng(919801);rows=[];outputs=[];identity=[]
    for j,g in enumerate(groups):
        pop=g['pop'];tab=VoltageScaledResponseTable(s.resp.tables[pop]);spline=s.spline[pop]
        p=s.resp.poles[pop];mu,rawE,rawI=z['wave'][j]
        ve=filt(rawE,s.tau[0]/2);vi=filt(rawI,s.tau[1]/2)
        mus=filt(mu,p['tau_s']);vef=filt(ve,p['tau_cE']);vif=filt(vi,p['tau_cI'])
        ves=filt(ve,p['tau_vE']);vis=filt(vi,p['tau_vI'])
        def evaluate(x,history,output_mix):
            mu,mus,ve,vef,ves,vi,vif,vis=x;th=np.full(np.shape(mu),g['theta_mv'])
            coord=(mus,ves,vis) if history else (mu,ve,vi)
            (al,ae,ai,ee,ei),_=tab.evaluate(*coord,th)
            cross=ee*(ve-vef)+ei*(vi-vif)
            e=np.maximum(ae*ve+(1-ae)*ves,0);i=np.maximum(ai*vi+(1-ai)*vis,0)
            if output_mix:
                fast=spline.evaluate(mu+cross,e,i,th)['rate']
                slow=spline.evaluate(mus+cross,e,i,th)['rate']
                r=al*fast+(1-al)*slow
            else:r=spline.evaluate(al*mu+(1-al)*mus+cross,e,i,th)['rate']
            return r*1000
        x=np.array([mu,mus,ve,vef,ves,vi,vif,vis])
        group_curves=[];group_rows=[]
        for name,history,mix in c['variants']:
            raw=evaluate(x,history,mix);pred=aggregate(raw)
            assert raw.min()>=-1e-12 and raw.max()<=1000/s.ref[g['group']]+1e-8
            error=float(np.linalg.norm(pred-obs['measured_hz'][j])/max(np.linalg.norm(obs['measured_hz'][j]),np.sqrt(B)))
            mean=float(pred@obs['occupancy_ms']/obs['occupancy_ms'].sum())
            bias=float(abs(mean-primary['rows'][j]['MC_cycle_mean_hz'])/max(primary['rows'][j]['MC_cycle_mean_hz'],1))
            group_rows.append(dict(variant=name,waveform_L2=error,mean_bias=bias,mean_hz=mean,
                                   passed=bool(error<=c['waveform_gate'] and bias<=c['mean_gate'])))
            group_curves.append(pred)
        # Same steady input and random eight-coordinate perturbations. Both
        # readouts have the same first derivative; no LIF validation implied.
        for hlookup in [False,True]:
            for ratio in [-.3,1.,2.5]:
                span=g['theta_mv']-11.;m0=11.+ratio*span;v0=span**2
                base=np.array([m0,m0,v0,v0,v0,v0,v0,v0])[:,None]
                direction=rng.normal(size=(8,1))*np.array([span,span,*([span**2]*6)])[:,None]
                eqdiff=float(abs(evaluate(base,hlookup,False)-evaluate(base,hlookup,True)).max())
                assert eqdiff<1e-10
                errs=[]
                for eps in [1e-5,5e-6]:
                    a=(evaluate(base+eps*direction,hlookup,False)-evaluate(base-eps*direction,hlookup,False))/(2*eps)
                    b=(evaluate(base+eps*direction,hlookup,True)-evaluate(base-eps*direction,hlookup,True))/(2*eps)
                    errs.append(float(abs(a-b).max()/max(abs(a).max(),1.)))
                assert errs[-1]<2e-6,(g['label'],ratio,errs)
                identity.append(dict(group=g['group'],history_lookup=hlookup,x=ratio,equilibrium_difference_Hz=eqdiff,
                                     derivative_relative_difference=errs))
        outputs.append(group_curves);rows.append(dict(group=g['group'],label=g['label'],rows=group_rows))
    np.savez_compressed(dest/'response.npz',predicted_hz=outputs,measured_hz=obs['measured_hz'],
                        variants=[x[0] for x in c['variants']],phase_centres=obs['phase_centres'])
    result=dict(status='DIAGNOSTIC_COMPLETE',rows=rows,steady_and_first_order_identity=identity,
       all_selected_pass={name:all(row['rows'][i]['passed'] for row in rows) for i,(name,_,_) in enumerate(c['variants'])},
       physical_rate_bounds_pass=True,scope=c['scope'],replacement_promoted=False)
    write(dest/'result.json',result);log('NONLINEAR READOUT',rows)


if __name__=='__main__':main()
