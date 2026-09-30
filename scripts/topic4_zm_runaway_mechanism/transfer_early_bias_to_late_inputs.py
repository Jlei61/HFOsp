"""Test locked early-input offsets on existing near-onset local LIF references.

No new fitting or simulation. Original late group-mean-threshold case only.
"""
from common import OUT,np,read,write,log
from analyze_native_early_surround_inputs import expected_flux
from datetime import datetime

DEST=OUT/'early_response_bias_diagnostic/late_input_transfer'
SOURCE=OUT/'native_input_bridge'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    offset=read(DEST.parent/'offsets.json')['offsets']
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the fixed early-population bias damage the existing near-onset local response?',
        inputs='Existing g20 sixselectednativeinput profiles8-10.37s, projectedmean/privatevariance;1s preparation, primary9-10.35s,27fixed50ms bins. Reuses untouched inputs and localLIF reference; not a g40 autonomous test.',
        correction='Exactly alreadylocked E/Ioffsets; no latefit or nativeonsettarget.',
        reference='Original groupmeanthreshold independentGaussianLIF dt0.1and0.05, firstsixcases, no mixturecase substitution.',
        check='All6groups compared against bothsteps; waveformL2<=.15 andtotalcountrelativeerror<=.10, same localcriteria. This developmental transfer cannot waive earlyoffaxisFAIL or otheroriginalvalidationfails.',
        budget='Read-only response reconstruction/scoring only. No refit, newreference, network or branch launch.'))
    z=np.load(SOURCE/'fixed_readout.npz');info=np.load(SOURCE/'selected_input_history.npz')
    r=z['projected_private']/1000;ref=np.where(info['population']==0,2.,1.);occupied=np.zeros_like(r)
    for j in range(6):
        for lag in range(1,round(ref[j]/.1)):
            occupied[lag:,j]+=r[:-lag,j]*.1
    p=r*.1/(1-occupied);assert p.min()>0 and p.max()<1
    ell=np.log(p)-np.log1p(-p);original,w=expected_flux(ell,ref)
    err=float(np.max(abs(original-z['projected_private'])));assert err<1e-8
    b=np.where(info['population']==0,offset['E'],offset['I']);new,occupancy=expected_flux(ell+b,ref)
    assert occupancy<=1+1e-12
    starts=z['bin_start_ms'];counts=np.stack([new[(z['time_ms']>=lo)&(z['time_ms']<lo+50)].sum(0)*.1/1000*info['group_size'] for lo in starts])
    rows=[];refs={}
    for dt in [.1,.05]:
        mc=np.load(SOURCE/f'local_lif/dt{dt:g}.npz');truth=[]
        for j,g in enumerate(info['groups']):
            assert mc['groups'][j]==g and mc['group_sizes'][j]==info['group_size'][j]
            nr=int(mc['replicates'][j]);target=mc['counts'][j,:nr].sum(0,dtype=np.uint64)/float(nr)*info['group_size'][j]
            truth.append(target);pred=counts[:,j];parent=z['counts_projected_private'][:,j]
            l2=float(np.linalg.norm(pred-target)/max(np.linalg.norm(target),1.));bias=float(abs(pred.sum()/target.sum()-1))
            rows.append(dict(group=int(g),pop='E' if info['population'][j]==0 else 'I',dt_ms=dt,
                L2=l2,count_error=bias,parent_L2=float(np.linalg.norm(parent-target)/max(np.linalg.norm(target),1.)),
                parent_count_error=float(abs(parent.sum()/target.sum()-1)),passed=l2<=.15 and bias<=.10))
        refs[f'MC_dt{dt:g}']=np.array(truth).T
    np.savez_compressed(DEST/'readout.npz',groups=info['groups'],rate_hz=new,counts=counts,bin_start_ms=starts,**refs)
    write(DEST/'result.json',dict(status='LATE_TRANSFER_PASS' if all(r['passed'] for r in rows) else 'LATE_TRANSFER_FAIL',
        rows=rows,passed=sum(r['passed'] for r in rows),total=len(rows),flux_reconstruction_error_hz=err,
        model_promoted=False,earlier_failures_retained=True,bifurcation_type='NOT_ESTABLISHED'))
    log('LOCKED EARLY BIAS LATE TRANSFER',rows)


if __name__=='__main__':main()
