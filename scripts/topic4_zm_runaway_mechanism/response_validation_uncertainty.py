"""Describe assay precision without changing the original pass/fail decisions."""
from common import *


def main():
    original=read(BASE/'dynamic_assay/validation_result.json')['rows']
    bank=read(OUT/'response_bank_candidate/validation_result.json')['response_rows']
    units=read(OUT/'response_voltage_units/validation_result.json')['rows']
    assert len(original)==len(bank)==len(units)
    rows=[]
    for ref,b,u in zip(original,bank,units):
        for key in ['kind','pop','channel','frequency_hz','counted']:
            assert ref[key]==b[key]==u[key]
        if not ref['counted']:continue
        sem=ref['sem']/max(abs(ref['dc_measured']),1e-12);dcsem=1/ref['dc_snr']
        # Conservative descriptive margin; it is not a replacement gate or
        # a simultaneous confidence interval across correlated frequencies.
        bound=ref['tol']*(1+3*dcsem)+3*sem
        rows.append(dict(kind=ref['kind'],pop=ref['pop'],channel=ref['channel'],frequency_hz=ref['frequency_hz'],
                         AC_SEM_over_DC=sem,DC_relative_SEM=dcsem,original_error=ref['norm_error'],
                         unit_corrected_error=u['corrected_error'],bank_error=b['new_error'],
                         original_failed=not ref['passed'],bank_failed=b['new_failed'],
                         original_exceeds_three_SEM_margin=bool(ref['norm_error']>bound),
                         bank_exceeds_three_SEM_margin=bool(b['new_error']>bound)))
    summary=[]
    for channel in ['mean','variance_E','variance_I']:
        for kind in ['v2_assayed','random']:
            ss=[r for r in rows if r['channel']==channel and r['kind']==kind]
            summary.append(dict(channel=channel,kind=kind,n=len(ss),
                 original_failed=sum(r['original_failed'] for r in ss),bank_failed=sum(r['bank_failed'] for r in ss),
                 original_exceeds_three_SEM=sum(r['original_exceeds_three_SEM_margin'] for r in ss),
                 bank_exceeds_three_SEM=sum(r['bank_exceeds_three_SEM_margin'] for r in ss),
                 median_AC_SEM_over_DC=float(np.median([r['AC_SEM_over_DC'] for r in ss]))))
    q=dict(status='PRECISION_DIAGNOSTIC_COMPLETE',rows=rows,summary=summary,
           scope='Preserves all original validation failures. Three-SEM margin is descriptive, not a new acceptance rule or a claim that unresolved discrepancies are absent.')
    write(OUT/'response_validation_uncertainty.json',q);log('RESPONSE PRECISION',summary)


if __name__=='__main__':main()
