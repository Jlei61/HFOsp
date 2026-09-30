"""Inspect spatial, rather than only global, variation in existing held-Z runs.

Finite-window recurrence is a diagnostic, never a continued periodic branch.
Historical runs using step-averaged delay histories remain explicitly flagged.
"""
from common import *
from scipy.signal import find_peaks
from scipy.fft import next_fast_len


def describe(path):
    z=np.load(path);field=z['field_E_hz'][-4000:].astype(float)
    weights=z['cell_counts']/z['cell_counts'].sum();mean=field.mean(0)
    x=(field-mean)*np.sqrt(weights);energy=np.sum(x*x,axis=1)
    size=next_fast_len(2*len(x)-1,real=True)
    freq=np.fft.rfft(x,n=size,axis=0)
    corr=np.fft.irfft(np.sum(abs(freq)**2,axis=1),n=size)[:len(x)]
    cum=np.r_[0.,np.cumsum(energy)];lags=np.arange(5,1001)
    left=cum[len(x)-lags];right=cum[-1]-cum[lags]
    ac=corr[lags]/np.maximum(np.sqrt(left*right),1e-30)
    err=np.sqrt(np.maximum(2*(left+right-2*corr[lags])/(left+right+1e-30),0))
    peaks=find_peaks(ac)[0];best=sorted(peaks,key=lambda j:-ac[j])[:5]
    global_rate=field@weights
    contract=read(path.parent/'contract.json')
    return dict(source=str(path),D=contract['D_initial'],global_Z=1-contract['D_initial'],
        global_mean_hz=float(global_rate.mean()),global_std_hz=float(global_rate.std()),
        weighted_spatial_temporal_RMS_hz=float(np.sqrt(energy.mean())),
        temporal_RMS_over_spatial_mean_RMS=float(np.sqrt(energy.mean()/np.sum(weights*mean*mean))),
        spatial_recurrence_peaks=[dict(lag_ms=int(lags[j]),correlation=float(ac[j]),
            normalized_return_error=float(err[j])) for j in best],
        initial_history=contract['initial'],time_step_ms=contract['dt_ms'],
        history_scheme=contract.get('rate_history_scheme','Historical step-average scheme'),
        tail_duration_ms=len(field),Z=contract['Z'],M=contract['M'])


def main():
    names=['endpoint_D0.2190000_dt0.05','endpoint_D0.2196000_dt0.05',
           'endpoint_D0.2220000_dt0.05','endpoint_D0.2288448_dt0.05',
           'native_Z9870_carried','native_Z10370_carried']
    rows=[describe(OUT/'runs'/name/'trajectory.npz') for name in names]
    q=dict(status='DESCRIPTIVE_COMPLETE',rows=rows,
        observable='Original-E-count-weighted spatial-field variation and recurrence, last4s,1ms field averages',
        limitation='Different carried histories and integration schemes are retained. No attractor, exact period, bifurcation or continuity across parameter gaps inferred.')
    write(OUT/'native_post_transition_geometry.json',q)
    for row in rows:log(Path(row['source']).parent.name,row['global_mean_hz'],
        row['weighted_spatial_temporal_RMS_hz'],row['spatial_recurrence_peaks'][:2])


if __name__=='__main__':main()
