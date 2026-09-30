"""Restore measured receiver E/I strength heterogeneity; no fitted coefficient."""
from shared import *
import time

def main():
    start=time.time();s,*_=runtime.setup(J,1.,848101);totals=[]
    for kind in ('ampa','gaba'):
        total=sum((np.asarray(m.sum(1)).ravel() for m in s.net[kind+'_by_delay']),start=np.zeros(s.n_e+s.n_i))
        totals.append(total)
    totals=np.array(totals).T
    for partition in ['adaptive','grid20','adaptive1']:
        suffix='' if partition=='adaptive' else '_'+partition;m=np.load(OUT/f'model{suffix}.npz');group=m['group'];count=m['count'];P=len(count)
        gains=[]
        for j in range(2):
            mean=np.bincount(group,weights=totals[:,j],minlength=P)/count
            gain=np.divide(totals[:,j],mean[group],out=np.ones(len(group)),where=mean[group]>0)
            assert np.allclose(np.bincount(group,weights=gain,minlength=P)/count,1.,atol=1e-12)
            gains.append(gain)
        gains=np.array(gains).T
        np.savez_compressed(OUT/f'node_gain{suffix}.npz',gain_sorted=gains[m['order']],incoming_totals=totals)
    write(OUT/'node_gain_protocol.json',dict(status='COMPLETE',seconds=time.time()-start,J=J,
        definition='Each neuron recurrent E/I current is multiplied by its actual summed incoming gating jump divided by its spatial-group mean; group mean gain exactly 1.',
        external_drive='unchanged; deterministic external component is removed before recurrent scaling and then added back',
        motivation='Block mean plus equal outside thresholds and deterministic external input collapses within-block neurons onto identical states.',
        approximations='Only static receiver strength heterogeneity restored; source-specific and delay-specific heterogeneity and private recurrent fluctuations still averaged.',
        fitted_parameters=0))
    print('node gains ready',time.time()-start,flush=True)

if __name__=='__main__':main()
