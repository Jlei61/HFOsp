"""Actual-state Gaussian-flux diagnostic with paired path-block uncertainty."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr

OUT = Path(__file__).resolve().parents[2] / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def flux(stats, denominator, dt):
    n = stats[:, 0]
    mean = np.divide(stats[:, 1], n, out=np.zeros_like(n), where=n>0)
    second = np.divide(stats[:, 2], n, out=np.zeros_like(n), where=n>0)
    variance = second-mean**2
    scale = np.maximum.reduce([abs(second), mean**2, np.ones_like(n)])
    assert np.min(variance/scale) > -1e-9
    nondegenerate = variance > 1e-13*scale
    probability = (mean >= 0).astype(float)
    probability[nondegenerate] = ndtr(mean[nondegenerate]/np.sqrt(variance[nondegenerate]))
    gaussian = np.sum(n*probability, axis=-1)/denominator*1000/dt
    actual = np.sum(stats[:, 5], axis=-1)/denominator*1000/dt
    return gaussian, actual, n, mean, variance, probability


def main():
    read = lambda p: json.loads(p.read_text())
    dest = OUT / 'conditional_gaussian_flux_diagnostic'; contract = read(dest/'contract.json')
    assert read(dest/'execution.json')['status'] == 'COMPLETE'
    R = contract['replicates']; dt=contract['dt_ms']; steps=round(contract['duration_ms']/dt)
    rows=[]; rng=np.random.default_rng(920034); groups=32; draws=300
    weights=rng.multinomial(groups,np.ones(groups)/groups,size=draws)
    for case in contract['cases']:
        pop=case['workpoint']['pop']; raw=np.load(dest/f'{pop}.npy',mmap_mode='r')
        reference=np.load(dest/f'{pop}_reference.npz'); original=read(dest/f'{pop}.json')
        assert raw.shape[0:2] == (2,6) and raw.shape[-1] == R
        count=raw[:,5].sum(axis=1)
        assert np.array_equal(count[0],reference['raw'][:,2]) and np.array_equal(count[1],reference['raw'][:,3])
        assert np.array_equal(.5*(count[0]-count[1]),reference['raw'][:,0])
        totals=raw.sum(axis=-1)
        assert np.max(totals[:,0].sum(axis=-1)) <= R*steps
        gaussian, actual, n, mean, variance, probability=flux(totals,R*steps,dt)
        amplitude=case['absolute_amplitude']
        gauss_gain=(gaussian[0]-gaussian[1])/(2*amplitude)
        true_gain=(actual[0]-actual[1])/(2*amplitude)
        empirical_per_path=(count[0]-count[1])*1000/contract['duration_ms']/(2*amplitude)
        assert abs(empirical_per_path.mean()-true_gain)<1e-10
        # Cluster whole time histories by independent noise paths. The same
        # blocks are resampled for the plus/minus conditions.
        blocks=raw.reshape(2,6,raw.shape[2],groups,R//groups).sum(axis=-1)
        bootstrap=[]
        for weight in weights:
            sample=np.tensordot(blocks,weight,axes=([-1],[0]))
            g,a,*_=flux(sample,R*steps,dt)
            bootstrap.append([(g[0]-g[1])/(2*amplitude),(a[0]-a[1])/(2*amplitude)])
        bootstrap=np.array(bootstrap); differences=bootstrap[:,0]-bootstrap[:,1]
        third=np.divide(totals[:,3],n,out=np.zeros_like(n),where=n>0)-3*mean*np.divide(totals[:,2],n,out=np.zeros_like(n),where=n>0)+2*mean**3
        fourth=np.divide(totals[:,4],n,out=np.zeros_like(n),where=n>0)-4*mean*np.divide(totals[:,3],n,out=np.zeros_like(n),where=n>0)+6*mean**2*np.divide(totals[:,2],n,out=np.zeros_like(n),where=n>0)-3*mean**4
        valid=(n>=1000)&(variance>1e-10)
        skew=np.full_like(n,np.nan); excess=np.full_like(n,np.nan)
        skew[valid]=third[valid]/variance[valid]**1.5
        excess[valid]=fourth[valid]/variance[valid]**2-3
        row=dict(population=pop,source_case_id=case['id'],amplitude_mv2=amplitude,
                 observed_rate_hz=actual.tolist(),gaussian_flux_rate_hz=gaussian.tolist(),
                 observed_gain_hz_per_mv2=float(true_gain),gaussian_flux_gain_hz_per_mv2=float(gauss_gain),
                 gain_relative_difference=float(abs(gauss_gain-true_gain)/max(abs(true_gain),1e-12)),
                 observed_gain_path_SEM=float(empirical_per_path.std(ddof=1)/np.sqrt(R)),
                 gaussian_minus_actual_gain=float(gauss_gain-true_gain),
                 paired_block_bootstrap_difference_95percent=np.quantile(differences,[.025,.975]).tolist(),
                 paired_block_bootstrap_difference_sd=float(differences.std(ddof=1)),
                 exact_count_parity=True,
                 activity_weighted_abs_skew=[float(np.sum(totals[s,5,valid[s]]*abs(skew[s,valid[s]]))/max(totals[s,5,valid[s]].sum(),1)) for s in range(2)],
                 activity_weighted_abs_excess_kurtosis=[float(np.sum(totals[s,5,valid[s]]*abs(excess[s,valid[s]]))/max(totals[s,5,valid[s]].sum(),1)) for s in range(2)])
        np.savez_compressed(dest/f'{pop}_readout.npz',totals=totals,edges=reference['edges'],
                            gaussian_probability=probability,conditional_skew=skew,
                            conditional_excess_kurtosis=excess,bootstrap_gains=bootstrap,
                            eligible_counts=n,actual_spikes=totals[:,5])
        rows.append(row)
    result=dict(status='ACTUAL_STATE_GAUSSIAN_FLUX_DIAGNOSIS_COMPLETE',rows=rows,
                model_promoted=False,statistical_unit='8192independent noise paths per workpoint, paired across+/-; bootstrap32pathblocks,300draws. Time steps are not independent samples.',
                scope='Gaussian tail reconstruction using measured LIF moments, not an autonomous model or newly blind validation. Conditional distributions are pooled across the fixed4s recording, including the original modulation-on transient. This cannot by itself certify an instantaneous closure at every time or a network bifurcation.')
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
