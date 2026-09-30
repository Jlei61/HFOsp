"""One fixed-graph sensitivity to a stationary variance-partition correction."""
from common import *
from scipy import sparse
from shared_variance_delay_audit import kernels
from closure_network_sensitivity import setup
import closure_stochastic_sensitivity as protocol
from runner import summarize
from native_readouts import readouts
import argparse


def split(s,dt):
    out={};checks=[]
    for k,kind in enumerate(['ampa','gaba']):
        row,col,a=s.raw[k];rq,cq,q=s.raw[k+2]
        assert np.array_equal(row,rq) and np.array_equal(col,cq)
        kc,kd,c0d,meta=kernels(s.rise[k],s.decay[k],dt,len(s.delays))
        dd=abs(np.arange(len(kc))[:,None]-np.arange(len(kc))[None,:])
        shared=np.asarray(a.multiply(a@kc[dd]).sum(1)).ravel()/s.sizes[col]
        shared_discrete=np.asarray(a.multiply(a@kd[dd]).sum(1)).ravel()/s.sizes[col]
        total=np.asarray(q.sum(1)).ravel()
        ratio=meta['Heun_to_continuous_C0_ratio'];f=shared/total
        assert np.isfinite(f).all() and f.min()>=-1e-12 and f.max()<=1+1e-12
        roundoff=(f<0)|(f>1);roundoff_count=int(roundoff.sum())
        roundoff_size=float(np.max(np.maximum(-f,f-1),initial=0))
        # Only cleanup of roundoff at the analytically bounded endpoints.
        f=np.minimum(1,np.maximum(0,f))
        private=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
        r=np.repeat(np.arange(s.P),np.diff(private.indptr));keys=r*s.P+private.indices%s.P
        basekeys=row*s.P+col;where=np.searchsorted(basekeys,keys)
        assert np.array_equal(basekeys[where],keys)
        original=private.data.copy();private.data*=1-f[where]
        remainder=np.bincount(where,weights=private.data,minlength=len(f))
        residual=float(np.max(abs(remainder+shared-total)/np.maximum(total,1e-30)))
        assert residual<1e-12,residual
        out[kind]=private
        checks.append(dict(synapse=kind,stationary_pairwise_relative_error=residual,
            removed_fraction_range=[float(f.min()),float(f.max())],
            private_operator_minimum=float(private.data.min()),
            fraction_roundoff_cleanup_count=roundoff_count,fraction_roundoff_max=roundoff_size,
            finite_Heun_remaining_variance_relative_max=float(np.max(abs(remainder+shared_discrete*ratio-total)/total)),
            previous_Heun_fraction_max=float(np.max(shared_discrete*ratio/total)),
            all_pair_fractions_physical=True,operator_nnz=private.nnz))
    return out,checks


def main(device):
    c=read(OUT/'shared_variance_network_sensitivity_contract.json')
    assert read(OUT/'shared_variance_delay_audit/result.json')['status']=='STATIONARY_POISSON_SPLIT_DIAGNOSTIC_COMPLETE'
    dest=OUT/'shared_variance_network_sensitivity';dest.mkdir(exist_ok=True)
    folder=dest/'units_history_delay_covariance';folder.mkdir(exist_ok=True)
    assert not (folder/'result.json').exists(),'One allowed run; preserve prior result'
    s,e=setup(c['response_variant'],c['dt_ms'],device,stochastic=True)
    initial=np.load(OUT/'closure_stochastic_sensitivity/initial.npz')
    assert np.array_equal(e.y.get(),initial['state'])
    e.history[:]=e.cp.asarray(initial['history'])
    operators,qa=split(s,c['dt_ms'])
    for k,kind in enumerate(['ampa','gaba']):
        a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=operators[kind]
        assert np.array_equal(a.indices,q.indices) and np.array_equal(a.indptr,q.indptr)
        e.ops[4*k+3]=e.cp.asarray(q.data)
        sparse.save_npz(folder/f'private_{kind}.npz',q)
    actual=protocol.actual_input_rhs_audit(s,e)
    assert np.array_equal(e.y.get(),initial['state']) and np.array_equal(e.history.get(),initial['history'])
    write(folder/'implementation_audit.json',dict(status='PASS',stationary_split=qa,
          actual_input_RHS=actual,same_initial_state_and_history=True))
    protocol.DEST=dest
    R,z,m=protocol.record(s,e,c['duration_ms'],'units_history_delay_covariance')
    _,field,whole,count=summarize(R,s)
    t=np.arange(len(R))+1.;ts=(np.arange(len(z))+1)*10.;d=1-z[:,s.E]@s.mean_weights
    events,summary,_,_=readouts(t,field,count,'units_history_delay_covariance')
    np.savez_compressed(folder/'trajectory.npz',time_ms=t,group_rate_hz=R.astype('float32'),
        field_E_hz=field.astype('float32'),global_E_hz=whole,cell_counts=count,
        Z=z.astype('float32'),M_current=m.astype('float32'),state_time_ms=ts,D=d,
        final_state=e.y.get(),final_history=e.history.get(),final_tick=e.tick,dt_ms=c['dt_ms'])
    summary.update(status='COMPLETE',Z_and_M_dynamic=True,initial_bitwise=True,
        D8000=float(d[np.flatnonzero(ts==8000)[0]]),D9870=float(d[np.flatnonzero(ts==9870)[0]]),
        D_final=float(d[-1]),replacement_promoted=False,scope=c['scope'])
    summary['events']=[{k:v for k,v in event.items() if k!='onset'} for event in events]
    write(folder/'result.json',summary);log('DELAY VARIANCE NETWORK RESULT',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    main(p.parse_args().device)
