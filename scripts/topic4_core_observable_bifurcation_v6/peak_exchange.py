"""Refine an extremum-identity exchange using a well-conditioned arc chart."""
from common import *
from folds import Chart,metric
from scipy.signal import resample,find_peaks
from scipy.optimize import brentq,minimize_scalar
import numpy as np

def peaks(r,T):
    N=len(r);n=32768;dense=resample(r,n,axis=0)
    offset=n//2-int(np.argmax(dense[:,2]));aligned=np.roll(dense,offset,axis=0)
    time=(np.arange(n)-n//2)*T/n
    inds=find_peaks(aligned[:,0])[0];R=np.fft.rfft(r[:,0]);freq=2j*np.pi*np.arange(N//2+1)
    def height(x):
        v=R*np.exp(freq*x)
        return float((v[0].real+v[-1].real+2*v[1:-1].real.sum())/N*1000)
    result={}
    for name,window in [('primary',(-160,-25)),('secondary',(-5,35))]:
        valid=inds[(time[inds]>window[0])&(time[inds]<window[1])]
        i=valid[np.argmax(aligned[valid,0])];center=((i-offset)%n)/n
        fit=minimize_scalar(lambda x:-height(x),bounds=(center-2/n,center+2/n),method='bounded',options={'xatol':1e-14})
        result[name+'_hz']=-fit.fun;result[name+'_time_ms']=float(time[i])
    return result

def main():
    N=2048;gscale=.01
    files=[V5/'arcs/surround_recruited_back'/f'point{i:03d}_g{g}.npz' for i,g in [(27,'1.1762753544'),(28,'1.1762710125')]]
    # Resolve the second filename from the recorded branch, never from rounding assumptions.
    rows=read(V5/'arcs/surround_recruited_back/progress.json');files=[Path(rows[i]['source']) for i in (27,28)]
    def vector(path):
        a=np.load(path);return np.r_[(resample(a['r'],N,axis=0)/.01).ravel(),np.log(float(a['T'])),float(a['g'])/gscale]
    a,b=map(vector,files);tan=b-a;tan/=np.sqrt(metric(tan,tan,N));span=metric(b-a,tan,N)
    chart=Chart(System(),a,tan,N,gscale);cache={}
    def at(x):
        if x not in cache:
            z,t,err,*_=chart.solve(a+(b-a)*x/span,x)
            r=z[:-2].reshape(N,6)*.01;T=float(np.exp(z[-2]));p=peaks(r,T)
            cache[x]=(z,t,err,p);print('PEAK_EXCHANGE',x,z[-1]*gscale,p,err,flush=True)
        return cache[x]
    def target(x):
        p=at(x)[3];return p['primary_hz']-p['secondary_hz']
    x=brentq(target,0,span,xtol=1e-10);z,t,err,p=at(x)
    dest=OUT/'periodic/peak_exchange';dest.mkdir(parents=True,exist_ok=True);path=dest/'equal_peaks_N2048.npz'
    np.savez_compressed(path,r=z[:-2].reshape(N,6)*.01,T=np.exp(z[-2]),g=z[-1]*gscale,residual=err)
    write('peak_exchange.json',dict(g=float(z[-1]*gscale),T_ms=float(np.exp(z[-2])),residual=err,arc_dJ=float(t[-1]*gscale),source=str(path),**p))

if __name__=='__main__':main()
