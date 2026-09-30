#!/usr/bin/env python3
"""Known-assay diagnostic: compress only hazard-history time constants."""
import numpy as np
from campaign import ROOT,read,write,sha
import conductance_linear_v3 as assay


def main():
    result=read(ROOT/'conductance_linear_v3/result.json');rows=result['rows']
    g=np.array([r['g'] for r in rows]);old=assay.transfer_factors
    def scaled(pop,frequency):
        bank,cov,K=old(pop,frequency)
        for i,(f,gg) in enumerate(zip(frequency,g)):
            lam=2j*np.pi*f/1000.;z=np.exp(-lam*.1)
            for j,tau in enumerate(np.array([1.,4.,16.,64.])/(1+gg)):
                v=.1/tau;e=np.exp(-v)
                L=e*np.array([[1,0,0],[v,1,0],[.5*v*v,v,1.]])
                B=np.array([1-e,1-e*(1+v),1-e*(1+v+.5*v*v)])
                bank[i,3*j:3*j+3]=np.linalg.solve(np.eye(3)-L*z,B)-1
        return bank,cov,K
    assay.transfer_factors=scaled
    rate,gain=assay.predictions(rows)
    diagnostic=[]
    for row,h in zip(rows,gain):
        if not row['primary']:continue
        error=abs(h-complex(*row['measured_gain']))
        diagnostic.append(dict(g=row['g'],x=row['x'],channel=row['channel'],frequency_Hz=row['frequency_Hz'],
             unscaled_error_over_tolerance=row['complex_error']/row['tolerance'],
             scaled_error_over_tolerance=float(error/row['tolerance']),
             scaled_prediction=[float(h.real),float(h.imag)],measured=row['measured_gain']))
    write(ROOT/'conductance_history_scaling_diagnostic.json',dict(
        status='KNOWN_ASSAY_DIAGNOSIS_NOT_VALIDATION',source_sha256=sha(__file__),
        question='Does membrane-inspired time compression of the existing hazard histories explain the conductance-dependent dynamic mismatch?',
        change='Historytaus[1,4,16,64]ms divided by1+g; native current-covariance filters, absolute refractory2ms and staticv3 unchanged. No fitted scale.',
        limitation='Parent history bank is empirical and need not consist only of membrane modes. Known failed assay reused for diagnosis; no independent dynamic pass claimed.',
        old_pass=sum(r['unscaled_error_over_tolerance']<=1 for r in diagnostic),
        scaled_pass=sum(r['scaled_error_over_tolerance']<=1 for r in diagnostic),total=len(diagnostic),rows=diagnostic))
    for gg in [0.,2.,8.]:
        for f in [0.,10.,50.]:
            v=[r for r in diagnostic if r['g']==gg and r['frequency_Hz']==f]
            print(gg,f,'old',sum(r['unscaled_error_over_tolerance']<=1 for r in v),
                  'scaled',sum(r['scaled_error_over_tolerance']<=1 for r in v),
                  'ratios',[round(r['scaled_error_over_tolerance'],2) for r in v],flush=True)


if __name__=='__main__':main()
