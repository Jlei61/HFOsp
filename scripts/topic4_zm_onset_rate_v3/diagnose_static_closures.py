"""Phase-1 root-cause diagnostic: which analytic static closure fails at the assayed workpoints?

Compares, at the 6 already-measured local workpoints (v2 local_response assay, colored-LIF truth):
  (a) shifted Siegert with harmonic effective tau (current frozen closure),
  (b) plain white-noise Siegert (no colored shift),
  (c) AMPA-shift + quasi-static Gaussian average over the slow GABA current (v1 'mixed'),
  (d) deterministic LIF rate (no noise).
No fitting. Output: table of baseline rate errors and variance-gain signs.
"""
from common_v3 import *
from scipy.special import erfcx
p=read(OPERATORS/'g20/prepared.json')['params']
VR=p['V_reset'];tauA=p['tau_r_AMPA']+p['tau_d_AMPA'];tauG=p['tau_r_GABA']+p['tau_d_GABA']
nodes,weights=np.polynomial.legendre.leggauss(64)
def siegert(mu,sig,theta,tm,ref,shift=0.):
    lo=(VR-mu)/sig+shift;hi=(theta-mu)/sig+shift
    x=(lo+hi)/2+(hi-lo)/2*nodes
    with np.errstate(over='ignore'):
        integ=(hi-lo)/2*np.sum(weights*erfcx(-x))
    return 1/(ref+tm*np.sqrt(np.pi)*integ)
def deterministic(mu,theta,tm,ref):
    return 0. if mu<=theta else 1/(ref+tm*np.log((mu-VR)/(mu-theta)))
def closure_a(mu,ve,vi,theta,tm,ref):
    q=ve+vi;teff=q/(ve/tauA+vi/tauG);return siegert(mu,np.sqrt(q),theta,tm,ref,1.0325*np.sqrt(teff/tm))
def closure_b(mu,ve,vi,theta,tm,ref):
    return siegert(mu,np.sqrt(ve+vi),theta,tm,ref)
gh_x,gh_w=np.polynomial.hermite_e.hermegauss(31)
def closure_c(mu,ve,vi,theta,tm,ref):
    sd=np.sqrt(tm*vi/(2*tauG));shift=1.0325*np.sqrt(tauA/tm)
    vals=[siegert(mu-sd*x,np.sqrt(ve),theta,tm,ref,shift) for x in gh_x]
    return float(np.sum(gh_w*np.array(vals))/np.sqrt(2*np.pi))
rows=[]
for lab in ['local_response','local_response_additional_mode_groups']:
    d=read(OLDV2/lab/'result.json')
    seen=set()
    for r in d['rows']:
        key=(r['state'],r['group'])
        if key in seen:continue
        seen.add(key)
        mu,ve,vi,th=r['mu_mv'],r['variance_E'],r['variance_I'],r['threshold_mv']
        tm,ref=(20.,2.) if r['population']=='E' else (10.,1.)
        meas=[q for q in d['rows'] if q['state']==r['state'] and q['group']==r['group']]
        rate_meas=np.mean([q['measured_rate_hz'] for q in meas])
        gE=[q['measured'] for q in meas if q['channel']=='variance_E' and q['frequency_hz']==0][0]
        gI=[q['measured'] for q in meas if q['channel']=='variance_I' and q['frequency_hz']==0][0]
        gE=gE[0] if isinstance(gE,list) else gE;gI=gI[0] if isinstance(gI,list) else gI
        out=dict(state=r['state'],group=r['group'],pop=r['population'],mu=mu,theta=th,ve=ve,vi=vi,x=(mu-VR)/(th-VR),
                 sigma_total=np.sqrt(ve+vi),measured_hz=rate_meas,measured_gain_vE=gE,measured_gain_vI=gI)
        for name,f in [('a_shifted_harmonic',closure_a),('b_white_siegert',closure_b),('c_ampa_shift_gaba_quasistatic',closure_c)]:
            base=f(mu,ve,vi,th,tm,ref)*1000
            h=1e-3
            dE=(f(mu,ve*(1+h),vi,th,tm,ref)-f(mu,ve*(1-h),vi,th,tm,ref))/(2*h*ve)*1000
            dI=(f(mu,ve,vi*(1+h),th,tm,ref)-f(mu,ve,vi*(1-h),th,tm,ref))/(2*h*vi)*1000
            out[name]=dict(rate_hz=base,rel_err=(base-rate_meas)/rate_meas,gain_vE=dE,gain_vI=dI,
                           sign_vE_ok=bool(np.sign(dE)==np.sign(gE)),sign_vI_ok=bool(np.sign(dI)==np.sign(gI)))
        out['d_deterministic']=dict(rate_hz=deterministic(mu,th,tm,ref)*1000)
        rows.append(out)
for o in rows:
    print(f"{o['state']:>10} g{o['group']:4d} {o['pop']} x={o['x']:6.2f} sig={o['sigma_total']:5.1f} meas={o['measured_hz']:7.2f} gE={o['measured_gain_vE']:+.4f} gI={o['measured_gain_vI']:+.4f}")
    for k in ['a_shifted_harmonic','b_white_siegert','c_ampa_shift_gaba_quasistatic']:
        q=o[k];print(f"      {k:32s} rate={q['rate_hz']:7.2f} err={q['rel_err']:+6.1%} gE={q['gain_vE']:+.4f}({'ok' if q['sign_vE_ok'] else 'SIGN'}) gI={q['gain_vI']:+.4f}({'ok' if q['sign_vI_ok'] else 'SIGN'})")
    print(f"      deterministic rate={o['d_deterministic']['rate_hz']:7.2f}")
write(DEST/'diagnostics/static_closure_attribution.json',dict(scope='Root-cause attribution at 8 previously assayed colored-LIF workpoints; no fitting',rows=rows,
      definitions=dict(x='(mu-V_reset)/(theta-V_reset)',gains='Hz per mV^2 of diffusion variance, zero frequency',
      a='current frozen closure: Siegert with both bounds shifted by 1.0325*sqrt(teff/tau_m), teff variance-weighted harmonic',
      b='white-noise Siegert, no shift',c='AMPA shift only; GABA current treated as quasi-static Gaussian with variance tau_m*vI/(2*tau_G)')))
