"""Refit a causal colored-response correction at this network's workpoints.

Static gain stays that of the explicitly defined stationary mean-field; only
frequency dependence is corrected. Core-A assays at 2,4,10,20 Hz fit the two
time constants. Core B, 6 Hz and 40 Hz are reserved checks, not fit targets.
"""
from common import *
from scipy.optimize import least_squares

def main():
    dest=OUT/'local_response_fit';dest.mkdir(exist_ok=True);results=[]
    for pop,source in [('E','local_response_check_v2'),('I','local_response_check_I')]:
        rows=read(OUT/source/'result.json')['rows'];dc={(r['core'],r['group']):r for r in rows if r['frequency_hz']==0}
        values=[]
        for r in rows:
            if not r['frequency_hz']:continue
            d=dc[(r['core'],r['group'])];obs=complex(*r['measured_chi_hz_per_mv']);pred=complex(*r['predicted_chi_hz_per_mv'])
            obs0=complex(*d['measured_chi_hz_per_mv']);pred0=complex(*d['predicted_chi_hz_per_mv'])
            values.append(dict(core=r['core'],group=r['group'],frequency_hz=r['frequency_hz'],ratio=(obs/obs0)/(pred/pred0),
                static_gain_ratio=(obs0/pred0).real,fit=r['core']=='A' and r['frequency_hz'] in [2.,4.,10.,20.]))
        train=[r for r in values if r['fit']]
        def fun(logpars):
            a,b=np.exp(logpars);omega=2j*np.pi*np.array([r['frequency_hz'] for r in train])/1000
            z=(1+a*omega)/(1+b*omega)-np.array([r['ratio'] for r in train]);return np.r_[z.real,z.imag]
        fit=least_squares(fun,np.log([4.,.5]),bounds=(np.log([.001,.001]),np.log([100.,100.])),xtol=1e-13,ftol=1e-13,gtol=1e-13)
        a,b=np.exp(fit.x)
        for r in values:
            z=2j*np.pi*r['frequency_hz']/1000;ratio=(1+a*z)/(1+b*z)
            r['relative_shape_error']=float(abs(ratio-r['ratio'])/abs(r['ratio']))
        result=dict(population=pop,zero_time_ms=a,pole_time_ms=b,rows=values,
            fit_mean_shape_error=float(np.mean([r['relative_shape_error'] for r in values if r['fit']])),
            heldout_mean_shape_error=float(np.mean([r['relative_shape_error'] for r in values if not r['fit']])),
            static_gain_ratio_range=[min(r['static_gain_ratio'] for r in values),max(r['static_gain_ratio'] for r in values)])
        results.append(result);print({k:v for k,v in result.items() if k!='rows'},flush=True)
    write(dest/'result.json',dict(status='COMPLETE',model='C(lambda)=(1+lambda*tau_zero)/(1+lambda*tau_pole)',rows=results,
        approximation='Local amplitude/phase correction only; retains original approximate stationary gain and variance response. Not a full nonlinear time-domain closure.',
        source='Numerical colored-LIF response protocol of Bachschmid-Romano et al.2026, refitted for the actual heterogeneous workpoints'))

if __name__=='__main__':main()
