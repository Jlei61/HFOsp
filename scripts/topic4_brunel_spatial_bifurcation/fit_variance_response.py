"""Reproduce the variance-channel frequency corrections from heldout assays."""
from common import *
from scipy.optimize import least_squares

def main():
    rows=read(OUT/'local_variance_response_check/result.json')['rows'];out=[]
    for pop in ['E','I']:
        for kind in ['E','I']:
            rr=[r for r in rows if r['population']==pop and r['input_variance']==kind]
            dc={(r['core'],r['group']):r for r in rr if r['frequency_hz']==0};values=[]
            for r in rr:
                if not r['frequency_hz']:continue
                d=dc[(r['core'],r['group'])]
                obs=complex(*r['measured_chi_hz_per_mv2']);pred=complex(*r['predicted_chi_hz_per_mv2'])
                obs0=complex(*d['measured_chi_hz_per_mv2']);pred0=complex(*d['predicted_chi_hz_per_mv2'])
                values.append(dict(core=r['core'],group=r['group'],frequency_hz=r['frequency_hz'],ratio=(obs/obs0)/(pred/pred0),
                    dc_ratio=(obs0/pred0).real,fit=r['core']=='A' and r['frequency_hz'] in [2,4,10,20]))
            train=[r for r in values if r['fit']]
            def fun(v):
                a,b=np.exp(v);z=2j*np.pi*np.array([r['frequency_hz'] for r in train])/1000
                d=(1+a*z)/(1+b*z)-np.array([r['ratio'] for r in train]);return np.r_[d.real,d.imag]
            fit=least_squares(fun,np.log([4,.5]),bounds=(np.log([.001,.001]),np.log([100,100])))
            a,b=np.exp(fit.x)
            for r in values:
                z=2j*np.pi*r['frequency_hz']/1000;r['shape_error']=abs((1+a*z)/(1+b*z)-r['ratio'])/abs(r['ratio'])
            item=dict(population=pop,input_variance=kind,zero_time_ms=a,pole_time_ms=b,
                fit_error=np.mean([r['shape_error'] for r in values if r['fit']]),heldout_error=np.mean([r['shape_error'] for r in values if not r['fit']]),
                dc_ratio_range=[min(r['dc_ratio'] for r in values),max(r['dc_ratio'] for r in values)],rows=values)
            out.append(item)
    write(OUT/'local_response_fit/variance_result.json',dict(rows=out,meaning='DC-normalized variance response; 2,4,10,20Hz at A fit, B/6/40Hz heldout'))

if __name__=='__main__':main()
