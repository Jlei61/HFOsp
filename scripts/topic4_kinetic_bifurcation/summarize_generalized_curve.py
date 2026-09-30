"""Extract plot observables from an actually replayed corrected return point.

Never assigns stability from the appearance of the trajectory. Native-rate
extrema and a 50-ms spatial peak are kept as different observables.
"""
from analyze_qualification import *
from scipy.ndimage import uniform_filter1d
import argparse


def run(a):
    source=a.spectrum;cfg=json.load(open(source/'config.json'))
    root=Path(cfg['corrected_source']);result=json.load(open(root/'result.json'))
    assert result['status']=='GENERALIZED_RETURN_CORRECTED'
    with np.load(source/'reference_trajectory.npz') as z:
        rates=z['rate_0p1ms'];fields=z['field_1ms'];slow=z['M_1ms'];count=z['count_e'];T=float(z['effective_return_time_ms'])
    h=.1;n=int(np.floor(T/h));fraction=T/h-n
    mean=(rates[:n].sum(0)+fraction*rates[n])/ (T/h)
    rate_1ms=rates[:len(fields)*10].reshape(-1,10,4).mean(1)
    spatial_readout=fields@count/count.sum()
    mismatch=float(np.max(abs(spatial_readout-rate_1ms[:,0])))
    assert mismatch<1e-10
    smooth=uniform_filter1d(rate_1ms,50,axis=0,mode='nearest')
    valid=np.arange(25,min(len(fields)-25,int(T)-25))
    assert len(valid)>0
    peak=int(valid[np.argmax(smooth[valid,0])])
    image=fields[peak-25:peak+25].mean(0)
    region_mean=rate_1ms[peak-25:peak+25].mean(0)
    assert abs(image@count/count.sum()-region_mean[0])<1e-10
    folder=OUT/'curve_point_readouts'/a.label;folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'observables.npz',spatial_peak_50ms=image,count_e=count,
        mean_rate_hz=mean,minimum_native_rate_hz=rates[:n+1].min(0),maximum_native_rate_hz=rates[:n+1].max(0))
    report=dict(status='CORRECTED_RETURN_POINT_READOUT_COMPLETE',source=str(source.resolve()),D=cfg['D'],
        corrected_source=str(root.resolve()),effective_return_time_ms=T,
        observable_order=['global_E','coreA_E','coreB_E','other_E'],mean_rate_hz=mean,
        minimum_native_rate_hz=rates[:n+1].min(0),maximum_native_rate_hz=rates[:n+1].max(0),
        mean_M_approx=slow[:int(T)].mean(0),native_rate_definition='Spike probability divided by the native 0.1-ms step, in Hz per neuron',
        mean_definition='Native step spike count sum with partial final bin, divided by effective return duration',
        spatial_peak_window_phase_ms=[peak-25,peak+25],spatial_peak_regional_mean_hz=region_mean,
        spatial_rate_agreement_max_hz=mismatch,
        weighted_return_residual=result['best_weighted_return_residual'],stability='NOT_ASSIGNED',
        scope='Readouts for the corrected generalized-return point. Stability, interpolation and model correspondence remain separate.',
        native_integer_periodicity='NOT_CLAIMED',bifurcation_type='NOT_CLASSIFIED')
    (folder/'readout.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    print(json.dumps(safe(report),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spectrum',type=Path,required=True);ap.add_argument('--label',required=True)
    run(ap.parse_args())
