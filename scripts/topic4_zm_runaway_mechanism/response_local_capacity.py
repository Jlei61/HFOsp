"""Measure six fixed counterexample workpoints, separating fit from interpolation.

These selected points are diagnostic calibration data, never validation of a
replacement model. Existing old-frequency observations are not fitted here.
"""
from common import *
from lif_mc import condition,run
from scipy.optimize import lsq_linear
import argparse

DEST=OUT/'response_local_capacity'


def main(device):
    c=read(OUT/'response_local_capacity_contract.json');DEST.mkdir(exist_ok=True)
    pars=[];meta=[]
    for j,item in enumerate(c['points']):
        q=item['workpoint'];ch=1 if item['channel']=='variance_E' else 2
        for f in c['frequencies_hz']:
            pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],
                        amplitude=.05,freq_hz=f,channel=ch));meta.append((j,ch,f))
    pars=np.array(pars);R=c['replicates'];T=c['duration_ms']
    obs=run(pars,R,T,c['burn_ms'],c['seed'],device=device,batch=8)
    measurements=[]
    for i,(j,ch,f) in enumerate(meta):
        amp=.05*pars[i,2 if ch==1 else 3]
        samples=(obs[i,:,0]+1j*obs[i,:,1])*1000/(T*amp);mean=samples.mean()
        measurements.append(dict(point=j,channel=ch,frequency_hz=f,measured=[mean.real,mean.imag],
                                 sem=float(np.sqrt(np.mean(abs(samples-mean)**2)/R))))
    np.savez_compressed(DEST/'observations.npz',observations=obs,parameters=pars)
    s=model();tau=np.array(c['filter_times_ms']);targets=read(OUT/'response_variance_precision/independent_audit.json')['rows']
    results=[]
    for j,item in enumerate(c['points']):
        q=item['workpoint'];ch=1 if item['channel']=='variance_E' else 2
        data=[r for r in measurements if r['point']==j];dc=next(r for r in data if r['frequency_hz']==0)
        data=[r for r in data if r['frequency_hz']>0];f=np.array([r['frequency_hz'] for r in data])
        w=2j*np.pi*f/1000;h=1+w*s.tau[ch-1]/2
        measured=np.array([complex(*r['measured']) for r in data])*h
        empirical_dc=dc['measured'][0];sem=np.array([r['sem'] for r in data])*abs(h)
        gain=s.spline[q['pop']].evaluate(np.array([q['mu']]),np.array([q['ve']]),np.array([q['vi']]),np.array([q['theta']]))
        static_dc=gain['d_ve' if ch==1 else 'd_vi'][0]*1000
        matched=[r for r in targets if r['channel']==item['channel'] and all(r['workpoint'][k]==q[k] for k in ['pop','theta','mu','ve','vi'])]
        assert len(matched)==2
        variants=[]
        for label,base in [('spline_DC',static_dc),('measured_DC_oracle',empirical_dc)]:
            basis=w[:,None]*tau/(1+w[:,None]*tau)
            scale=max(abs(empirical_dc),1e-8);weight=scale/np.maximum(sem,scale*.005)
            A=basis*weight[:,None];b=(measured-base)/scale*weight
            coefficients=np.linalg.lstsq(np.r_[A.real,A.imag,np.sqrt(c['ridge'])*np.eye(5)],
                            np.r_[b.real,b.imag,np.zeros(5)],rcond=None)[0]*scale
            predictions=base+basis@coefficients
            held=[]
            for r in matched:
                lam=2j*np.pi*r['frequency_hz']/1000
                pred=(base+(lam*tau/(1+lam*tau))@coefficients)/(1+lam*s.tau[ch-1]/2)
                error=abs(pred-complex(*r['new_measured']))/max(abs(r['new_DC']),1e-12)
                held.append(dict(frequency_hz=r['frequency_hz'],predicted=[pred.real,pred.imag],error=float(error),
                                  passed=bool(error<=r['tolerance']),table_error=r['errors']['bank_candidate']))
            variants.append(dict(DC_source=label,DC=float(base),coefficients=coefficients.tolist(),
                training_RMS_over_DC=float(np.sqrt(np.mean(abs(predictions-measured)**2))/scale),out_of_fit_frequencies=held))
        results.append(dict(point=j,workpoint=q,channel=item['channel'],empirical_DC=empirical_dc,
                            empirical_DC_SEM=dc['sem'],variants=variants))
    out=dict(status='DIAGNOSTIC_COMPLETE',measurements=measurements,rows=results,scope=c['scope'])
    write(DEST/'result.json',out);log('LOCAL CAPACITY',{k:v for k,v in out.items() if k!='measurements'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
