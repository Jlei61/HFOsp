"""Independent conservation and count audit of recorded ensemble impulses."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import lfilter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    read=lambda p:json.loads(p.read_text())
    dest=OUT/'membrane_balance_diagnostic';c=read(OUT/'membrane_balance_diagnostic_contract.json')
    z=np.load(dest/'balance.npz');old=np.load(OUT/c['source']/'response.npz')
    assert np.array_equal(z['counts'],old['counts'][c['indices']])
    raw=z['block_sums'];R=c['inherited']['replicates'];errors=[];rows=[]
    for j in range(len(c['indices'])):
        a=z['pars'][j,18];gap=z['pars'][j,1]-z['pars'][j,21]
        block_res=raw[j,:,:,0]-a*raw[j,:,:,4]-(1-a)*raw[j,:,:,1]+raw[j,:,:,2]+raw[j,:,:,3]
        assert np.max(abs(block_res))<1e-8
        m=np.add.reduce(raw[j],axis=0)/R
        assert np.array_equal(m,z['ensemble_mean'][j])
        inp=(1-a)*m[:,1]-m[:,2]-m[:,3]
        reconstructed,_=lfilter([1.],[1.,-a],inp,zi=[a*m[0,4]])
        error=float(np.max(abs(reconstructed-m[:,0])));assert error<1e-8;errors.append(error)
        assert np.min(raw[j,:,:,2])>=0
        reset_charge=float(m[:,2].sum()*R);spikes=int(z['counts'][j].sum())
        assert reset_charge>=spikes*gap
        rows.append(dict(condition=['full','mean_only'][j],
            independent_lfilter_error_mv=error,max_block_conservation_error=float(np.max(abs(block_res))),
            mean_reset_jump_per_spike_mv=reset_charge/spikes,threshold_reset_gap_mv=float(gap),
            overshoot_fraction_of_reset_charge=float(1-spikes*gap/reset_charge)))
    q=dict(status='MEMBRANE_BALANCE_INDEPENDENT_AUDIT_PASS',rows=rows,
        spike_counts_bitwise=True,max_ensemble_reconstruction_error_mv=max(errors),
        scope=c['scope'])
    (dest/'independent_audit.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))


if __name__=='__main__':main()
