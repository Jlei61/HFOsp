"""Separate sample-phase peak changes from changes of a burst waveform."""
from analyze_qualification import *
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
import argparse


def run(a):
    r=np.load(a.source/'trace.npz')['rate_0p1ms'];t=.1*(np.arange(len(r))+1)
    fits=json.load(open(a.period_source/'result.json'))['rows']
    fitted_period=[v['period_ms'] for v in fits if v['order']==3][0]
    period=fitted_period/a.period_divisor
    refs=t[(t>=5)&(t<period-5)];rows=[]
    for width in (1,100):
        y=r if width==1 else uniform_filter1d(r,size=width,axis=0,mode='nearest')
        curve=CubicSpline(t,y,extrapolate=False);base=curve(refs)
        for k in (1,2,3):
            if refs[-1]+k*period>t[-1]-5:continue
            later=curve(refs+k*period);diff=later-base
            assert np.isfinite(diff).all()
            rows.append(dict(observation_window_ms=width*.1,cycle_lag=k,
                relative_RMS=np.sqrt(np.mean(diff*diff,axis=0)/np.maximum(np.mean(base*base,axis=0),1e-24)),
                absolute_RMS_hz=np.sqrt(np.mean(diff*diff,axis=0)),base_max_hz=base.max(0),lagged_max_hz=later.max(0)))
    report=dict(status='COMPLETE_RATE_WAVEFORM_DIAGNOSTIC',source=str(a.source),period_source=str(a.period_source),period_ms=period,
        fitted_full_return_period_ms=fitted_period,period_divisor=a.period_divisor,
        observable_order=['global_E','coreA_E','coreB_E','other_E'],rows=rows,
        scope='Cubic phase alignment of native-step rate readouts across bursts; not an exact-map periodicity or stability test')
    (OUT/f'{a.label}.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    for row in rows:print(row['observation_window_ms'],row['cycle_lag'],row['relative_RMS'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--period-source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--period-divisor',type=float,default=1.,help='Use a fractional candidate period only to test subcycle waveform equality')
    run(ap.parse_args())
