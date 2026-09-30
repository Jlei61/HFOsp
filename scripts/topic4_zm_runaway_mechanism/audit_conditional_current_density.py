"""Read saved threshold flux and independently reconstruct validation metrics."""
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    read=lambda p:json.loads(p.read_text())
    dest=OUT/'conditional_current_density';contract=read(OUT/'conditional_current_density_contract.json')
    results=[]
    for path in sorted(dest.glob('*_grid*.json')):
        q=read(path);z=np.load(path.with_suffix('.npz'))
        dataset=q.get('dataset','factorial_waveform');j=q['source_index']
        original=np.load(OUT/dataset/'response.npz');params=np.load(OUT/dataset/'prepared.npz')
        assert np.array_equal(z['pars'],params['pars'][j])
        dt=float(z['dt_ms']);T=float(z['T_ms']);n=len(z['step_rate_hz']);b=len(z['predicted_hz'])
        bins=np.minimum(((((np.arange(n)+1)*dt/T)%1)*b).astype(int),b-1)
        exposure=np.bincount(bins,minlength=b)*dt
        own=np.bincount(bins,weights=z['step_rate_hz']*dt,minlength=b)/exposure
        assert np.max(abs(own-z['predicted_hz']))<1e-8
        counts=original['counts'][j]
        target=(counts/(exposure[None,:]/1000)).mean(0)
        assert np.max(abs(target-z['measured_hz']))<1e-8
        error=float(np.linalg.norm(own-target)/max(np.linalg.norm(target),np.sqrt(b)))
        mean=float(np.average(own,weights=exposure));target_mean=float(counts.sum(1).mean()/(n*dt)*1000)
        bias=abs(mean-target_mean)/max(target_mean,1.)
        assert abs(error-q['waveform_L2'])<1e-10 and abs(bias-q['relative_mean_error'])<1e-10
        min_eigen=0.;minimum_mass=1.
        total=np.zeros((5,5))
        for block in [z['final_free'],z['final_refractory']]:
            total+=block.sum(0);minimum_mass=min(minimum_mass,float(block[:,0,0].min()))
            for raw in block:
                if raw[0,0] > 1e-12:
                    mu=raw[0,1:]/raw[0,0]
                    covariance=raw[1:,1:]/raw[0,0]-np.outer(mu,mu)
                    relative=float(np.linalg.eigvalsh((covariance+covariance.T)/2).min()/max(np.trace(covariance),1.))
                    min_eigen=min(min_eigen,relative)
        assert minimum_mass>=-1e-12 and min_eigen>=-1e-8
        assert abs(total[0,0]-1)<1e-8
        results.append(dict(dataset=dataset,index=j,grid=q['requested_grid_nodes'],
            waveform_L2=error,relative_mean_error=bias,independent_readout_pass=True,
            final_minimum_mass=minimum_mass,minimum_relative_covariance_eigenvalue=min_eigen,
            waveform_pass=error<=.15 and bias<=.1,numerical_pass=q['numerical_pass']))
    pairs=[]
    for index in contract['full_indices']:
        a=np.load(dest/f'index{index:02d}_grid128.npz')['predicted_hz']
        b=np.load(dest/f'index{index:02d}_grid256.npz')['predicted_hz']
        difference=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.))
        pairs.append(dict(index=index,relative_grid_difference=difference,passed=difference<=.02))
    full=[r for r in results if r['dataset']=='factorial_waveform' and r['index'] in contract['full_indices'] and r['grid'] in [128,256]]
    assert len(full)==8
    q=dict(status='DENSITY_RESPONSE_INDEPENDENT_AUDIT_PASS',rows=results,grid_pairs=pairs,
        full_waveform_gate_pass=all(r['waveform_pass'] and r['numerical_pass'] for r in full) and all(p['passed'] for p in pairs),
        model_promoted=False,scope='Local imposed inputs; predictions use their own computed threshold flux. No model fitting, autonomous spatial correspondence, weak-response validation, or bifurcation acceptance.')
    (dest/'independent_audit.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps({k:v for k,v in q.items() if k!='rows'}))


if __name__=='__main__':main()
