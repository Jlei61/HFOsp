"""Held-out validation of the response closure (registered in dynamic_assay/validation_contract.json).

(1) v2-assayed workpoints at 6 and 25 Hz (independent seeds, never used for fitting): compare the
    closure's predicted complex response with the measured one, normalised by the channel's DC gain.
(2) 12 random workpoints per population (seed 11) at 15 and 60 Hz: new MC run (seed 27182818).
Prediction (V3): mean channel R_mu(0) H_mu(w); variance channel c: [R_c(0)(a_c+(1-a_c)/(1+iw tau_vc)) + R_mu(0) eta_c iw tau_c/(1+iw tau_c)]
times the synaptic variance filter 1/(1+iw tau_syn/2). DC gains from the transfer spline.
"""
from lif_mc import *
from dynamics_v3 import DynamicModel,ResponseParams
import argparse
def predict(s,pop,mu,ve,vi,theta,f,ch):
    ev=s.spline[pop].evaluate(np.array([mu]),np.array([ve]),np.array([vi]),np.array([theta]));w=2j*np.pi*f/1000
    tab=s.resp.tables[pop];wt,_=tab.evaluate(np.array([mu]),np.array([ve]),np.array([vi]),np.array([theta]));al,aE,aI,eE,eI=wt[:,0];pl=s.resp.poles[pop]
    Hm=al+(1-al)/(1+w*pl['tau_s']);Rmu0=ev['d_mu'][0]*1000
    if ch==0:return Rmu0*Hm
    tau=pl['tau_cE'] if ch==1 else pl['tau_cI'];tv=pl['tau_vE'] if ch==1 else pl['tau_vI'];eta=eE if ch==1 else eI;ac=aE if ch==1 else aI
    Rc0=(ev['d_ve'] if ch==1 else ev['d_vi'])[0]*1000;tsyn=s.tau[0] if ch==1 else s.tau[1]
    return (Rc0*(ac+(1-ac)/(1+w*tv))+Rmu0*eta*w*tau/(1+w*tau))/(1+w*tsyn/2)
def main(a):
    contract=read(DEST/'dynamic_assay/validation_contract.json');assert contract['status']=='REGISTERED_BEFORE_VALIDATION'
    resp=ResponseParams(DEST/'response_closure/closure.json');s=DynamicModel(resp=resp,quiet=True);rows=[]
    # (1) v2 assay
    for lab in ['local_response','local_response_additional_mode_groups']:
        d=read(OLDV2/lab/'result.json')
        for r in d['rows']:
            if r['frequency_hz'] not in (6.,25.):continue
            ch=['mean','variance_E','variance_I'].index(r['channel']);meas=complex(*r['measured']) if isinstance(r['measured'],list) else complex(r['measured'])
            dc=[q for q in d['rows'] if q['state']==r['state'] and q['group']==r['group'] and q['channel']==r['channel'] and q['frequency_hz']==0][0]
            dcm=complex(*dc['measured']) if isinstance(dc['measured'],list) else complex(dc['measured']);dcs=dc['complex_sem']
            pred=predict(s,r['population'],r['mu_mv'],r['variance_E'],r['variance_I'],r['threshold_mv'],r['frequency_hz'],ch)
            snr=abs(dcm)/max(dcs,1e-12);err=abs(pred-meas)/max(abs(dcm),1e-12)
            rows.append(dict(kind='v2_assayed',state=r['state'],group=r['group'],pop=r['population'],channel=r['channel'],frequency_hz=r['frequency_hz'],measured=[meas.real,meas.imag],sem=r['complex_sem'],
                predicted=[pred.real,pred.imag],dc_measured=dcm.real,dc_snr=snr,norm_error=float(err),counted=bool(snr>=10),tol=.10,passed=bool(err<=.10) if snr>=10 else None,
                dc_spline=float(predict(s,r['population'],r['mu_mv'],r['variance_E'],r['variance_I'],r['threshold_mv'],0.,ch).real),
                sign_ok=bool(np.sign(pred.real)==np.sign(meas.real)) if r['frequency_hz']<=10 and snr>=10 else None))
    # (2) random points, new MC
    rng=np.random.default_rng(11);pts=[]
    for pop in 'EI':
        for _ in range(12):
            th=18. if rng.uniform()<.5 or pop=='I' else rng.uniform(14.2,17.5);sc=th-11;x=rng.uniform(-1,4);se=np.exp(rng.uniform(np.log(.3),np.log(3)));si=np.exp(rng.uniform(np.log(.1),np.log(4)))
            pts.append(dict(pop=pop,mu=11+sc*x,theta=th,ve=(sc*se)**2,vi=(sc*si)**2,x=x))
    pars=[];meta=[]
    for q in pts:
        for ch in (0,1,2):
            for f in (0.,15.,60.):
                pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.15 if ch==0 else .05,freq_hz=f,channel=ch));meta.append((q,ch,f))
    R=a.replicates;T=a.duration;obs=run(pars,R,T,300,27182818,crn=True,device=a.device);dcs={}
    for i,(q,ch,f) in enumerate(meta):
        p=pars[i];amp=p[4] if ch==0 else p[4]*p[2 if ch==1 else 3];est=(obs[i,:,0]+1j*obs[i,:,1])/(T*amp)*1000;m=est.mean();sem=float(np.sqrt(np.mean(abs(est-m)**2)/R))
        if f==0:dcs[(id(q),ch)]=(m.real,sem);continue
        dcm,dcsem=dcs[(id(q),ch)];pred=predict(s,q['pop'],q['mu'],q['ve'],q['vi'],q['theta'],f,ch);snr=abs(dcm)/max(dcsem,1e-12);err=abs(pred-m)/max(abs(dcm),1e-12);tol=.10 if f<=25 else .15
        rows.append(dict(kind='random',pop=q['pop'],x=q['x'],channel=['mean','variance_E','variance_I'][ch],frequency_hz=f,measured=[m.real,m.imag],sem=sem,predicted=[pred.real,pred.imag],dc_measured=dcm,dc_snr=snr,
            norm_error=float(err),counted=bool(snr>=10),tol=tol,passed=bool(err<=tol) if snr>=10 else None,dc_spline=float(predict(s,q['pop'],q['mu'],q['ve'],q['vi'],q['theta'],0.,ch).real)))
    counted=[r for r in rows if r['counted']];fails=[r for r in counted if not r['passed']]
    dc_cons=[abs(r['dc_spline']-r['dc_measured'])/max(abs(r['dc_measured']),1e-9) for r in counted]
    summary=dict(status='COMPLETE',n_rows=len(rows),n_counted=len(counted),n_failed=len(fails),
        by_channel={c:dict(counted=sum(1 for r in counted if r['channel']==c),failed=sum(1 for r in fails if r['channel']==c),median_norm_error=float(np.median([r['norm_error'] for r in counted if r['channel']==c]) if any(r['channel']==c for r in counted) else np.nan)) for c in ['mean','variance_E','variance_I']},
        dc_consistency_median=float(np.median(dc_cons)),dc_consistency_p90=float(np.percentile(dc_cons,90)),verdict='PASS' if len(fails)<=max(2,int(.1*len(counted))) else 'FAIL',
        candidates_tried=2,rows=rows)
    write(DEST/'dynamic_assay/validation_result.json',summary)
    for r in counted:print(f"{r['kind']:10s} {r['pop']} {r['channel']:10s} f={r['frequency_hz']:4.0f} meas={complex(*r['measured']):+.3f} pred={complex(*r['predicted']):+.3f} dc={r['dc_measured']:+.3f} err={r['norm_error']:.3f} {'ok' if r['passed'] else 'FAIL'}")
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=1))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=2048);p.add_argument('--duration',type=float,default=4000.);p.add_argument('--device',type=int,default=0);main(p.parse_args())
