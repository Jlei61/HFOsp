"""Independently reaggregate the assay and check input interpolation refinement."""
from native_cycle_waveform_response import DEST,simulate
from common import *
from scipy.signal import resample
import argparse


def main(device):
    c=read(OUT/'native_cycle_waveform_response_contract.json')
    check=read(OUT/'native_cycle_waveform_sampling_contract.json')
    q=read(DEST/'result.json');z=np.load(DEST/'prepared.npz');obs=np.load(DEST/'response.npz')
    assert read(DEST/'implementation_check.json')['constant_wave_per_replicate_count_bitwise']
    T=float(z['T_ms']);dt=q['dt_ms'];R=c['replicates'];B=c['phase_bins']
    assert q['steps']==round(c['record_cycles']*T/dt) and q['burn_steps']==round(c['burn_cycles']*T/dt)
    phases=((np.arange(q['steps'])+1)*dt/T)%1;indices=np.minimum((phases*B).astype(int),B-1)
    exposure=np.bincount(indices,minlength=B)*dt
    assert np.array_equal(exposure,obs['occupancy_ms'])
    counts=obs['counts'];rate=counts/exposure[None,None,:]*1000;mean=rate.mean(1)
    assert np.array_equal(mean,obs['measured_hz'])
    for i,row in enumerate(q['rows']):
        mean_rate=float(counts[i].sum()/(R*q['steps']*dt)*1000)
        rms=float(np.linalg.norm(obs['predicted_hz'][i]-mean[i])/max(np.linalg.norm(mean[i]),np.sqrt(B)))
        assert abs(mean_rate-row['MC_cycle_mean_hz'])<1e-10
        assert abs(rms-row['normalized_waveform_RMSE'])<1e-12
    W=check['wave_samples'][1];wave=resample(z['wave'],W,axis=2)
    assert np.min(wave[:,1:])>=0
    refined=simulate(z['pars'],wave,R,T,dt,q['burn_steps'],q['steps'],c['seed'],B,device)
    refined_rate=(refined/exposure[None,None,:]*1000).mean(1)
    prediction=resample(z['prediction_hz'],W,axis=0)
    pos=phases*W;lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;a=pos-np.floor(pos)
    sampled=(1-a[:,None])*prediction[lo]+a[:,None]*prediction[hi]
    predicted=np.array([np.bincount(indices,weights=sampled[:,j],minlength=B)/(exposure/dt) for j in range(4)])
    rows=[]
    for i in range(4):
        denom=max(np.linalg.norm(mean[i]),np.sqrt(B))
        change=float(np.linalg.norm(refined_rate[i]-mean[i])/denom)
        oldmean=counts[i].mean(0).sum()/(q['steps']*dt)*1000
        newmean=refined[i].mean(0).sum()/(q['steps']*dt)*1000
        meanchange=float(abs(newmean-oldmean)/max(oldmean,1.))
        predchange=float(np.linalg.norm(predicted[i]-obs['predicted_hz'][i])/max(np.linalg.norm(obs['predicted_hz'][i]),np.sqrt(B)))
        rows.append(dict(label=q['rows'][i]['label'],MC_relative_wave_difference=change,
            MC_relative_mean_difference=meanchange,model_relative_wave_difference=predchange))
    passed=all(r['MC_relative_wave_difference']<check['acceptance']['MC_relative_wave_difference_max']
        and r['MC_relative_mean_difference']<check['acceptance']['MC_relative_mean_difference_max']
        and r['model_relative_wave_difference']<check['acceptance']['model_relative_wave_difference_max'] for r in rows)
    np.savez_compressed(DEST/'refined_sampling_response.npz',counts=refined,measured_hz=refined_rate,predicted_hz=predicted)
    audit=dict(status='PASS' if passed else 'NEEDS_REFINEMENT',aggregation_reproduced=True,rows=rows,
        source_samples=check['wave_samples'],scope=check['scope'],original_verdict=q['verdict'])
    write(DEST/'independent_audit.json',audit);log('WAVEFORM INDEPENDENT AUDIT',audit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    main(p.parse_args().device)
