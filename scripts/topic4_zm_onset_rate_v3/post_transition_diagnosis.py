"""Diagnose the state reached after a transition in frozen-Z runs: equilibrium / periodic / quasi-periodic /
irregular / long transient. Uses the last part of the run (global E rate at 1 ms and the group field).
- drift: linear trend of the 500-ms block means over the last 3 s relative to the level (transient if |trend| large)
- equilibrium: coefficient of variation of the global rate over the last 2 s < 1e-3 and no drift, and distance to
  the nearest equilibrium on the stored branches at the same D
- periodic: autocorrelation of the detrended global rate has a peak > 0.9 at lag T; Poincaré return (peak-to-peak
  amplitudes) coefficient of variation < 2 %
- quasi-periodic / irregular: peak > 0.5 but return CV >= 2 % (quasi-periodic if two incommensurate spectral peaks
  dominate the spectrum, irregular otherwise)
"""
from common_v3 import *
from scipy.signal import find_peaks,periodogram
import glob,argparse
def diagnose(t,g,tail_ms=3000):
    n=len(g);sl=slice(max(0,n-tail_ms),n);x=g[sl];tt=t[sl]
    blocks=x[:len(x)//500*500].reshape(-1,500).mean(1);trend=np.polyfit(np.arange(len(blocks)),blocks,1)[0]*(len(blocks)-1) if len(blocks)>2 else 0.
    level=max(abs(x.mean()),1e-9);rel_drift=float(trend/level);cv=float(x.std()/level)
    out=dict(tail_ms=tail_ms,mean_hz=float(x.mean()),cv=cv,relative_drift=rel_drift)
    if cv<1e-3 and abs(rel_drift)<1e-3:out['state']='EQUILIBRIUM';return out
    y=x-x.mean();ac=np.correlate(y,y,'full')[len(y)-1:];ac/=ac[0];pk=find_peaks(ac[10:],height=.3)[0]
    if len(pk)==0:out['state']='IRREGULAR_OR_TRANSIENT' if abs(rel_drift)<.05 else 'LONG_TRANSIENT';return out
    lag=int(pk[0]+10);out['autocorr_peak']=float(ac[lag]);out['period_ms']=lag
    peaks=find_peaks(y,distance=max(5,lag//2))[0];amps=y[peaks];out['return_cv']=float(amps.std()/max(abs(amps.mean()),1e-9)) if len(amps)>3 else None
    f,P=periodogram(y,fs=1000.);top=np.argsort(P)[-3:][::-1];out['spectral_peaks_hz']=[float(f[i]) for i in top]
    if abs(rel_drift)>=.05:out['state']='LONG_TRANSIENT'
    elif ac[lag]>.9 and out['return_cv'] is not None and out['return_cv']<.02:out['state']='PERIODIC'
    elif ac[lag]>.5:
        f1,f2=sorted(out['spectral_peaks_hz'][:2]);ratio=f2/max(f1,1e-9);out['state']='QUASI_PERIODIC' if abs(ratio-round(ratio))>.05 else 'PERIODIC_WITH_MODULATION'
    else:out['state']='IRREGULAR'
    return out
def main(a):
    rows=[]
    for path in a.runs:
        z=np.load(path);t=z['time_ms'];g=z['global_E_hz'];d=diagnose(t,g,a.tail);d['run']=path;d['D']=float(z['D'][-1]) if 'D' in z.files and np.ndim(z['D']) else (float(z['D']) if 'D' in z.files else None)
        rows.append(d);print(Path(path).parent.name if path.endswith('trajectory.npz') else Path(path).stem,d['state'],'mean %.1f cv %.3f drift %.3f'%(d['mean_hz'],d['cv'],d['relative_drift']),d.get('period_ms'),d.get('return_cv'))
    write(DEST/'post_transition_diagnosis.json',dict(rows=rows,definitions=__doc__))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--tail',type=int,default=3000);main(p.parse_args())
