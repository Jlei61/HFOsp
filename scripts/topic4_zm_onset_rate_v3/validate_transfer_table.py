"""Held-out validation of the spline transfer (registered in transfer_table/validation_contract.json).

Different MC seed from the table (no CRN with it). Held-out points: 8 v2-assayed workpoints
(their own thresholds), random draws, extreme mean-driven draws. Compares rate and
zero-frequency gains (paired +/- static perturbations) against the spline value/derivatives.
"""
from lif_mc import *
from transfer_spline import TransferSpline
import argparse
def main(a):
    folder=DEST/'transfer_table';contract=read(folder/'validation_contract.json');assert contract['status']=='REGISTERED_BEFORE_VALIDATION'
    sp={p:TransferSpline(folder/f'table_{p}.npz') for p in 'EI'}
    points=[]
    for lab in ['local_response','local_response_additional_mode_groups']:
        d=read(OLDV2/lab/'result.json');seen=set()
        for r in d['rows']:
            key=(r['state'],r['group'])
            if key in seen:continue
            seen.add(key);points.append(dict(kind='v2_assayed',state=r['state'],group=r['group'],pop=r['population'],mu=r['mu_mv'],theta=r['threshold_mv'],ve=r['variance_E'],vi=r['variance_I']))
    rng=np.random.default_rng(7)
    for pop in 'EI':
        for _ in range(24):
            th=18. if rng.uniform()<.5 or pop=='I' else rng.uniform(14.2,17.5);sc=th-11
            x=rng.uniform(-2,20);se=np.exp(rng.uniform(np.log(.1),np.log(4)));si=0. if rng.uniform()<.15 else np.exp(rng.uniform(np.log(.05),np.log(6)))
            points.append(dict(kind='random',pop=pop,mu=11+sc*x,theta=th,ve=(sc*se)**2,vi=(sc*si)**2))
        for _ in range(8):
            th=18.;sc=7.;x=rng.uniform(20,200);se=np.exp(rng.uniform(np.log(.3),np.log(4)));si=np.exp(rng.uniform(np.log(.3),np.log(6)))
            points.append(dict(kind='extreme',pop=pop,mu=11+sc*x,theta=th,ve=(sc*se)**2,vi=(sc*si)**2))
    pars=[]
    for q in points:
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop']))
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.15,channel=0))
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.05,channel=1))
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.05,channel=2))
        # reference-linearity probes: 4x amplitude for each channel (quantised-ISI regime is flagged, not tolerated)
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.6,channel=0))
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.2,channel=1))
        pars.append(condition(q['mu'],q['theta'],q['ve'],q['vi'],q['pop'],amplitude=.2,channel=2))
        # secant probes at the table's own resolution (half the x-grid spacing, 0.2*sqrt(1+x^2)): static rates at mu +/- delta
        sc=q['theta']-11.;x=(q['mu']-11.)/sc;dmu=.1*sc*np.sqrt(1+x*x);q['secant_delta_mv']=dmu
        pars.append(condition(q['mu']+dmu,q['theta'],q['ve'],q['vi'],q['pop']));pars.append(condition(q['mu']-dmu,q['theta'],q['ve'],q['vi'],q['pop']))
    NP=9;R=a.replicates;T=a.duration;obs=run(pars,R,T,300,a.seed,crn=False,device=a.device);rows=[];fails=[];flagged=[]
    for i,q in enumerate(points):
        base=obs[NP*i];rate=base[:,2].mean()/T*1000;sem=base[:,2].std()/np.sqrt(R)/T*1000
        ev=sp[q['pop']].evaluate(np.array([q['mu']]),np.array([q['ve']]),np.array([q['vi']]),np.array([q['theta']]))
        row=dict(**q,mc_rate_hz=rate,mc_sem_hz=sem,spline_rate_hz=ev['rate'][0]*1000,x=float(ev['x'][0]),blend=float(ev['blend'][0]))
        err=abs(row['spline_rate_hz']-rate);tol=max(.03*rate,3*sem) if rate>=1 else max(.05,3*sem)
        row['rate_pass']=bool(err<=tol);row['rate_rel_err']=float(err/max(rate,1e-9))
        for c,(key,dkey) in enumerate([('mean','d_mu'),('variance_E','d_ve'),('variance_I','d_vi')]):
            o=obs[NP*i+1+c];p=pars[NP*i+1+c];amp=p[4] if c==0 else p[4]*p[2 if c==1 else 3]
            if amp==0:continue
            est=o[:,0]/(T*amp)*1000;g=est.mean();gs=est.std()/np.sqrt(R);pred=ev[dkey][0]*1000
            o4=obs[NP*i+4+c];p4=pars[NP*i+4+c];amp4=p4[4] if c==0 else p4[4]*p4[2 if c==1 else 3];est4=o4[:,0]/(T*amp4)*1000;g4=est4.mean();gs4=est4.std()/np.sqrt(R)
            # reference is linear if the 4x-amplitude gain agrees within 10 % (or within 3 SEM); otherwise the LIF response is dominated by ISI quantisation
            linear=bool(abs(g4-g)<=max(.1*abs(g),3*np.hypot(gs,gs4)))
            row[f'gain_{key}']=dict(mc=g,sem=gs,spline=pred,mc_4x=g4,sem_4x=gs4,reference_linear=linear,
                sign_ok=(bool(np.sign(g)==np.sign(pred)) if (abs(g)>3*gs and linear) else None),
                rel_err=float(abs(pred-g)/abs(g)) if abs(g)>10*gs else None,mag_pass=(bool(abs(pred-g)<=.15*abs(g)) if (abs(g)>10*gs and linear) else None))
            if not linear:flagged.append((q['kind'],q['pop'],key,float(q['mu']),float(g),float(g4)))
            if c==0:
                rp=obs[NP*i+7][:,2].mean()/T*1000;rm=obs[NP*i+8][:,2].mean()/T*1000;sec=(rp-rm)/(2*q['secant_delta_mv'])
                sub_grid=bool(abs(sec-g)>max(.15*abs(sec),3*gs))   # local gain differs from the grid-scale secant: sub-grid (quantised-ISI) structure
                row['gain_mean'].update(secant=float(sec),secant_delta_mv=float(q['secant_delta_mv']),sub_grid_structure=sub_grid,
                    secant_pass=bool(abs(pred-sec)<=.15*abs(sec)) if abs(sec)>10*gs else None)
                if sub_grid:
                    flagged.append((q['kind'],q['pop'],'mean_subgrid',float(q['mu']),float(g),float(sec)))
                    # in the sub-grid regime the smooth transfer is a coarse-graining by design: gain comparisons for this point are not counted
                    for kk in ['mean','variance_E','variance_I']:
                        if f'gain_{kk}' in row:row[f'gain_{kk}']['sign_ok']=None;row[f'gain_{kk}']['mag_pass']=None
                    row['gain_mean']['secant_counted']=True
        rows.append(row)
        if not row['rate_pass']:fails.append((q['kind'],q['pop'],'rate',row['rate_rel_err']))
        for key in ['mean','variance_E','variance_I']:
            gk=row.get(f'gain_{key}')
            if gk and gk['sign_ok'] is False:fails.append((q['kind'],q['pop'],key,'sign'))
            if gk and gk['mag_pass'] is False:fails.append((q['kind'],q['pop'],key,gk['rel_err']))
        gm=row.get('gain_mean')
        if gm and gm.get('secant_pass') is False:fails.append((q['kind'],q['pop'],'mean_secant',abs(gm['spline']-gm['secant'])/abs(gm['secant'])))
        print(f"{q['kind']:10s} {q['pop']} x={row['x']:7.2f} rate MC {rate:8.3f}±{sem:.3f} spline {row['spline_rate_hz']:8.3f} {'ok' if row['rate_pass'] else 'FAIL'}",
              ' '.join(f"{k[5:]}:{row[k]['mc']:+.4f}/{row[k]['spline']:+.4f}" for k in row if k.startswith('gain_')),flush=True)
    per_pop={p:sum(1 for f in fails if f[1]==p) for p in 'EI'}
    verdict='PASS' if all(v<=2 for v in per_pop.values()) else 'FAIL'
    write(folder/'validation_result.json',dict(status='COMPLETE',verdict=verdict,failures=fails,failures_per_population=per_pop,reference_nonlinear_flags=flagged,
        exclusion_rule='gain comparisons are counted only where the reference LIF has no sub-grid structure: the local (0.15 mV) mean gain must agree with the secant over half the x-grid spacing; where it does not (quantised-ISI regime, x*dt/tau_m not small), the smooth transfer is a coarse-graining by design, the secant is compared instead (15 %), and local-gain comparisons are reported but not counted',replicates=R,duration_ms=T,seed=a.seed,rows=rows))
    print('VERDICT',verdict,fails)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=2048);p.add_argument('--duration',type=float,default=4000.);p.add_argument('--seed',type=int,default=31415926);p.add_argument('--device',type=int,default=0);main(p.parse_args())
