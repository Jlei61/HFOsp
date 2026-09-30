"""Check whether the local mean-filter ablation generalizes to existing assays."""
from response_error_decomposition import predict
from common import *


def main():
    c=read(OUT/'response_mean_filter_removal_contract.json')
    source=read(OUT/'response_error_decomposition.json');s=model();rows=[]
    channels=['mean','variance_E','variance_I']
    for old in source['rows']:
        q=old['workpoint'];ch=channels.index(old['channel'])
        original=predict(s,old['pop'],q,old['frequency_hz'],ch)
        if ch==0:
            ev=s.spline[old['pop']].evaluate(np.array([q['mu']]),np.array([q['ve']]),
                 np.array([q['vi']]),np.array([q['theta']]))
            new=complex(ev['d_mu'][0]*1000)
        else:new=original
        # Recover the exact original target by matching metadata, not ordering.
        candidates=[r for r in read(BASE/'dynamic_assay/validation_result.json')['rows']
                    if r['kind']==old['kind'] and r['pop']==old['pop']
                    and r['channel']==old['channel'] and r['frequency_hz']==old['frequency_hz']
                    and abs(complex(*r['predicted'])-original)<1e-10]
        assert len(candidates)==1,(old,len(candidates))
        ref=candidates[0];error=abs(new-complex(*ref['measured']))/max(abs(ref['dc_measured']),1e-12)
        rows.append(dict(pop=old['pop'],channel=old['channel'],frequency_hz=old['frequency_hz'],
                         counted=old['counted'],original_failed=old['original_failed'],
                         removal_failed=bool(error>old['tol']) if old['counted'] else None,
                         original_error=old['original_error'],removal_error=float(error)))
    eligible=[r for r in rows if r['counted']]
    summary={}
    for channel in channels:
        subset=[r for r in eligible if r['channel']==channel]
        summary[channel]=dict(n=len(subset),original_failed=sum(r['original_failed'] for r in subset),
                              removal_failed=sum(r['removal_failed'] for r in subset))
    q=dict(status='DIAGNOSTIC_COMPLETE',by_channel=summary,rows=rows,scope=c['scope'])
    write(OUT/'response_mean_filter_removal_check.json',q)
    log('MEAN FILTER REMOVAL CHECK',summary)


if __name__=='__main__':main()
