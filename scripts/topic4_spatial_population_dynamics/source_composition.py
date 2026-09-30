"""Measured source-region composition, preserving each target-block mean."""
from shared import *
import time

def main():
    t=time.time();s,*_=runtime.setup(J,1.,848101)
    m=np.load(OUT/'model_adaptive1.npz');region=m['region'];group=m['group'];count=m['count'];P=len(count);N=len(group)
    total=np.zeros((N,6))
    for kind,offset in [('ampa',0),('gaba',s.n_e)]:
        for mat in s.net[kind+'_by_delay']:
            if not mat.nnz:continue
            c=mat.tocoo();index=c.row.astype(np.int64)*6+region[c.col+offset]
            total+=np.bincount(index,weights=c.data,minlength=N*6).reshape(N,6)
    means=np.stack([np.bincount(group,weights=total[:,j],minlength=P)/count for j in range(6)],axis=1)
    gains=np.divide(total,means[group],out=np.ones_like(total),where=means[group]>0)
    assert np.allclose(np.stack([np.bincount(group,weights=gains[:,j],minlength=P)/count for j in range(6)],axis=1),1.,atol=1e-12)
    np.savez_compressed(OUT/'source_composition_adaptive1.npz',gain_sorted=gains[m['order']],incoming_totals=total,group_means=means)
    rows=[]
    for r in range(6):
        rows.append(dict(target_region=r,source_gain_std=np.std(gains[region==r],axis=0)))
    write(OUT/'source_composition_protocol.json',dict(status='COMPLETE',seconds=time.time()-t,J=J,
        definition='kappa_iq = total original incoming jump from source region q / target population mean of that total; q = AE,BE,SE,AI,BI,SI',
        invariant='Population mean of each gain is 1; external currents unchanged; all physical delay bins retained in block input.',
        remaining_approximation='Within each source region, target-specific spatial and delay profiles and private recurrent fluctuations still averaged.',
        fitted_parameters=0,summary=rows))
    print(json.dumps(safe(rows)),flush=True)

if __name__=='__main__':main()
