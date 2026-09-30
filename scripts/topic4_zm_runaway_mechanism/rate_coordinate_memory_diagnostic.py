"""Test mean-response memory in rate coordinates without fitting coefficients.

Given u=(mu+cross,vEeff,vIeff), p'=-p/tau_s+Phi_mu(u)*mu',
r=Phi(u)-(1-alpha)*p. For fixed variances/cross and constant alpha this is
alpha*Phi(mu)+(1-alpha)*LP[Phi(mu)]. Its DC and first-order response equal
the original input-memory mean filter. Positivity under varying variances is
not assumed; negative output rejects the diagnostic, without clipping it.
"""
from common import *
from scipy.fft import rfft,irfft
from response_voltage_units import VoltageScaledResponseTable


def main():
    c=read(OUT/'rate_coordinate_memory_contract.json');dest=OUT/'rate_coordinate_memory';dest.mkdir(exist_ok=True)
    source=OUT/'native_cycle_waveform_response';z=np.load(source/'prepared.npz');obs=np.load(source/'response.npz')
    primary=read(source/'result.json');groups=read(source/'preparation.json')['groups'];s=model()
    W=z['wave'].shape[-1];T=float(z['T_ms']);lam=2j*np.pi*np.arange(W//2+1)/T
    def filt(x,tau):return irfft(rfft(x)/(1+lam*tau),n=W)
    def derivative(x):return irfft(rfft(x)*lam,n=W)
    phase=((np.arange(primary['steps'])+1)*primary['dt_ms']/T)%1
    B=obs['measured_hz'].shape[1];bins=np.minimum((phase*B).astype(int),B-1);exposure=np.bincount(bins,minlength=B)
    pos=phase*W;lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;frac=pos-np.floor(pos)
    def aggregate(x):return np.bincount(bins,weights=(1-frac)*x[lo]+frac*x[hi],minlength=B)/exposure
    rows=[];curves=[];equivalence=[]
    for j,g in enumerate(groups):
        tab=VoltageScaledResponseTable(s.resp.tables[g['pop']]);phi=s.spline[g['pop']]
        poles=s.resp.poles[g['pop']];mu,rawE,rawI=z['wave'][j];dmu=derivative(mu)
        ve=filt(rawE,s.tau[0]/2);vi=filt(rawI,s.tau[1]/2);mus=filt(mu,poles['tau_s'])
        vef=filt(ve,poles['tau_cE']);vif=filt(vi,poles['tau_cI']);ves=filt(ve,poles['tau_vE']);vis=filt(vi,poles['tau_vI'])
        th=np.full(W,g['theta_mv']);qrows=[];qcurves=[]
        for name,history in c['variants']:
            coord=(mus,ves,vis) if history else (mu,ve,vi)
            (al,ae,ai,ee,ei),_=tab.evaluate(*coord,th)
            cross=ee*(ve-vef)+ei*(vi-vif)
            ev=np.maximum(ae*ve+(1-ae)*ves,0);iv=np.maximum(ai*vi+(1-ai)*vis,0)
            value=phi.evaluate(mu+cross,ev,iv,th)
            memory=filt(poles['tau_s']*value['d_mu']*dmu,poles['tau_s'])
            raw=(value['rate']-(1-al)*memory)*1000;pred=aggregate(raw)
            error=float(np.linalg.norm(pred-obs['measured_hz'][j])/max(np.linalg.norm(obs['measured_hz'][j]),np.sqrt(B)))
            mean=float(pred@obs['occupancy_ms']/obs['occupancy_ms'].sum())
            bias=float(abs(mean-primary['rows'][j]['MC_cycle_mean_hz'])/max(primary['rows'][j]['MC_cycle_mean_hz'],1))
            valid=raw.min()>=-c['negative_tolerance_Hz'] and raw.max()<=1000/s.ref[g['group']]+c['negative_tolerance_Hz']
            qrows.append(dict(variant=name,waveform_L2=error,mean_bias=bias,mean_hz=mean,
                minimum_raw_rate_Hz=float(raw.min()),maximum_raw_rate_Hz=float(raw.max()),
                negative_phase_fraction=float(np.mean(raw< -c['negative_tolerance_Hz'])),
                physical_bounds_pass=bool(valid),accuracy_pass=bool(error<=.15 and bias<=.1),
                passed=bool(valid and error<=.15 and bias<=.1)))
            qcurves.append(pred)
        # Independent fixed-variance identity on a smooth, bounded mean input:
        # Phi - tau LP[dPhi/dt] = LP[Phi]. No MC data enter this check.
        clock=np.arange(W)*T/W;span=g['theta_mv']-11.
        mm=11.+span*(.6+.1*np.sin(2*np.pi*clock/T));vv=np.full(W,span**2)
        v=phi.evaluate(mm,vv,vv,th);dp=derivative(mm)*v['d_mu']
        from_derivative=v['rate']-filt(poles['tau_s']*dp,poles['tau_s'])
        directly=filt(v['rate'],poles['tau_s'])
        rel=float(np.linalg.norm(from_derivative-directly)/max(np.linalg.norm(directly),1e-30))
        assert rel<1e-8,rel
        equivalence.append(dict(group=g['group'],constant_variance_rate_filter_identity_relative=rel))
        rows.append(dict(group=g['group'],label=g['label'],rows=qrows));curves.append(qcurves)
    np.savez_compressed(dest/'response.npz',predicted_hz=curves,measured_hz=obs['measured_hz'],
                        variants=[v[0] for v in c['variants']],phase_centres=obs['phase_centres'])
    result=dict(status='DIAGNOSTIC_COMPLETE',rows=rows,fixed_variance_identity=equivalence,
        all_selected_pass={name:all(row['rows'][i]['passed'] for row in rows) for i,(name,_) in enumerate(c['variants'])},
        scope=c['scope'],replacement_promoted=False)
    write(dest/'result.json',result);log('RATE COORDINATE MEMORY',rows)


if __name__=='__main__':main()
