"""Screen full E-field recurrence before attempting any post-transition cycle BVP."""
from common import *
from scipy.fft import rfft,irfft,next_fast_len
from scipy.signal import find_peaks
import argparse


def main(a):
    s=model();rows=[]
    for label in a.labels:
        path=OUT/'runs'/label/'trajectory.npz';z=np.load(path)
        x=z['group_rate_hz'][-6000:,s.E].astype(float);x-=x.mean(0)
        n=len(x);m=next_fast_len(2*n);weights=s.mean_weights
        power=(abs(rfft(x,n=m,axis=0))**2)@weights
        cross=irfft(power,n=m)[:n]
        energy=(x*x)@weights;cs=np.r_[0,np.cumsum(energy)];lags=np.arange(5,min(3000,n//2)+1)
        diff=(cs[n]-cs[lags]+cs[n-lags]-2*cross[lags])/(n-lags)
        recurrence=np.sqrt(np.maximum(diff,0)/max(float(energy.mean()),1e-20))
        minima=find_peaks(-recurrence)[0];best=sorted(minima,key=lambda j:recurrence[j])[:8]
        q=dict(label=label,source=str(path),window_ms=[len(z['group_rate_hz'])-n,len(z['group_rate_hz'])],
               weighting='original E-cell counts across the entire spatial field',
               best_returns=[dict(lag_ms=int(lags[j]),relative_error=float(recurrence[j])) for j in best],
               meaning='Low recurrence error nominates a period for a BVP; no periodicity is inferred from global-rate peaks alone.')
        rows.append(q);log('FIELD RETURN',q)
    dest=OUT/'field_recurrence.json'
    previous={q['label']:q for q in read(dest)} if dest.exists() else {}
    previous.update({q['label']:q for q in rows})
    write(dest,list(previous.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='+');main(p.parse_args())
