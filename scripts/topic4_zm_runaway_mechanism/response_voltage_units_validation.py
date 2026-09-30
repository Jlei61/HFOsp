"""Rerun only predictions of the original validation under the unit repair."""
from response_voltage_units_audit import response,DEST
from common import *


def main():
    assert read(DEST/'implementation_audit.json')['status']=='UNIT_CORRECTION_AUDIT_PASS'
    s=model();diagnostic=read(OUT/'response_error_decomposition.json')
    reference=read(BASE/'dynamic_assay/validation_result.json')['rows']
    channels=['mean','variance_E','variance_I'];rows=[]
    for old in diagnostic['rows']:
        q=old['workpoint'];ch=channels.index(old['channel']);f=old['frequency_hz']
        original=response(s,q,f,ch,False);corrected=response(s,q,f,ch,True)
        candidates=[r for r in reference if r['kind']==old['kind'] and r['pop']==old['pop']
                    and r['channel']==old['channel'] and r['frequency_hz']==f
                    and abs(complex(*r['predicted'])-original)<1e-10]
        assert len(candidates)==1,(old,len(candidates))
        ref=candidates[0];error=abs(corrected-complex(*ref['measured']))/max(abs(ref['dc_measured']),1e-12)
        rows.append(dict(kind=old['kind'],pop=old['pop'],theta=q['theta'],channel=old['channel'],
                         frequency_hz=f,counted=old['counted'],original_error=old['original_error'],
                         corrected_error=float(error),original_failed=old['original_failed'],
                         corrected_failed=bool(error>old['tol']) if old['counted'] else None))
    eligible=[r for r in rows if r['counted']];summary={}
    for channel in channels:
        ss=[r for r in eligible if r['channel']==channel]
        summary[channel]=dict(n=len(ss),original_failed=sum(r['original_failed'] for r in ss),
                             corrected_failed=sum(r['corrected_failed'] for r in ss))
    failures=sum(r['corrected_failed'] for r in eligible)
    q=dict(status='PREDICTIONS_REEVALUATED',n_counted=len(eligible),original_failed=sum(r['original_failed'] for r in eligible),
           corrected_failed=failures,verdict='PASS' if failures<=max(2,int(.1*len(eligible))) else 'FAIL',
           by_channel=summary,rows=rows,scope='Same existing independent validation observations, eligibility and thresholds. Unit correction is the sole change; no refitting or new measurements.')
    write(DEST/'validation_result.json',q);log('UNIT CORRECTION VALIDATION',{k:v for k,v in q.items() if k!='rows'})


if __name__=='__main__':main()
