"""Separate DC-gain and dynamic-shape error using existing held-out assays.

Empirical gains are diagnostic oracle values; never a fitted or validated model.
"""
from common import *


def predict(s,pop,q,f,ch,gains=None):
    theta=np.array([q['theta']]);mu=np.array([q['mu']]);ve=np.array([q['ve']]);vi=np.array([q['vi']])
    if gains is None:
        ev=s.spline[pop].evaluate(mu,ve,vi,theta)
        gains=np.array([ev[k][0]*1000 for k in ['d_mu','d_ve','d_vi']])
    weights,_=s.resp.tables[pop].evaluate(mu,ve,vi,theta)
    al,ae,ai,ee,ei=weights[:,0];p=s.resp.poles[pop];w=2j*np.pi*f/1000
    if ch==0:return gains[0]*(al+(1-al)/(1+w*p['tau_s']))
    c='E' if ch==1 else 'I';a=ae if ch==1 else ai;eta=ee if ch==1 else ei
    return (gains[ch]*(a+(1-a)/(1+w*p['tau_v'+c]))+
        gains[0]*eta*w*p['tau_c'+c]/(1+w*p['tau_c'+c]))/(1+w*s.tau[ch-1]/2)


def main():
    s=model();source=BASE/'dynamic_assay/validation_result.json';validation=read(source)
    channels=['mean','variance_E','variance_I'];points={}
    old=ROOT/'results/topic4_sef_hfo/fig5_zm_rate_synchronized_20260917'
    for label in ['local_response','local_response_additional_mode_groups']:
        for r in read(old/label/'result.json')['rows']:
            if r['frequency_hz']!=0:continue
            key=('v2_assayed',r['population'],r['state'],r['group'])
            q=dict(pop=r['population'],theta=r['threshold_mv'],mu=r['mu_mv'],ve=r['variance_E'],vi=r['variance_I'])
            if key in points:
                for k in q:assert points[key][k]==q[k]
            else:points[key]=dict(q,dc={},snr={})
            val=complex(*r['measured']) if isinstance(r['measured'],list) else complex(r['measured'])
            ch=channels.index(r['channel']);points[key]['dc'][ch]=val.real
            points[key]['snr'][ch]=abs(val)/max(r['complex_sem'],1e-12)
    rng=np.random.default_rng(11)
    for pop in 'EI':
        for _ in range(12):
            th=18. if rng.uniform()<.5 or pop=='I' else rng.uniform(14.2,17.5)
            sc=th-11;x=rng.uniform(-1,4)
            se=np.exp(rng.uniform(np.log(.3),np.log(3)));si=np.exp(rng.uniform(np.log(.1),np.log(4)))
            points[('random',pop,x)]=dict(pop=pop,theta=th,mu=11+sc*x,ve=(sc*se)**2,vi=(sc*si)**2,dc={},snr={})
    def key(r):
        return ('random',r['pop'],r['x']) if r['kind']=='random' else ('v2_assayed',r['pop'],r['state'],r['group'])
    for r in validation['rows']:
        q=points[key(r)];ch=channels.index(r['channel'])
        if ch in q['dc']:assert abs(q['dc'][ch]-r['dc_measured'])<1e-10
        q['dc'][ch]=r['dc_measured'];q['snr'][ch]=r['dc_snr']
    rows=[];errors=[]
    for r in validation['rows']:
        q=points[key(r)];ch=channels.index(r['channel'])
        original=predict(s,r['pop'],q,r['frequency_hz'],ch)
        difference=abs(original-complex(*r['predicted']));errors.append(difference)
        assert difference<1e-10*max(1,abs(original)),difference
        # A variance correction needs its DC and the mean DC; an unused channel
        # is allowed to be unmeasured at the historical workpoint.
        needed=[0] if ch==0 else [0,ch]
        if not all(c in q['dc'] for c in needed):continue
        gains=np.array([q['dc'].get(c,np.nan) for c in range(3)])
        corrected=predict(s,r['pop'],q,r['frequency_hz'],ch,gains)
        err=abs(corrected-complex(*r['measured']))/max(abs(r['dc_measured']),1e-12)
        rows.append(dict(kind=r['kind'],pop=r['pop'],channel=r['channel'],frequency_hz=r['frequency_hz'],
            workpoint=q,counted=r['counted'],tol=r['tol'],original_error=r['norm_error'],
            empirical_DC_error=float(err),original_failed=not r['passed'] if r['counted'] else None,
            empirical_DC_failed=bool(err>r['tol']) if r['counted'] else None,
            required_DC_snr_min=float(min(q['snr'][c] for c in needed))))
    eligible=[r for r in rows if r['counted']]
    assert len(eligible)==validation['n_counted']
    groups={}
    for channel in channels:
        subset=[r for r in eligible if r['channel']==channel]
        groups[channel]=dict(n=len(subset),original_failed=sum(r['original_failed'] for r in subset),
            empirical_DC_failed=sum(r['empirical_DC_failed'] for r in subset),
            original_median_error=float(np.median([r['original_error'] for r in subset])),
            empirical_DC_median_error=float(np.median([r['empirical_DC_error'] for r in subset])))
    out=dict(status='DIAGNOSTIC_COMPLETE',source=str(source),original_prediction_reproduction_max_error=max(errors),
        n_counted=len(eligible),original_failed=sum(r['original_failed'] for r in eligible),
        empirical_DC_failed=sum(r['empirical_DC_failed'] for r in eligible),by_channel=groups,rows=rows,
        scope='Measured DC gains are an oracle diagnostic using held-out data. This is NOT a new model or a held-out validation pass; original FAIL is unchanged.')
    write(OUT/'response_error_decomposition.json',out)
    log('RESPONSE ERROR DECOMPOSITION',{k:v for k,v in out.items() if k!='rows'})


if __name__=='__main__':main()
