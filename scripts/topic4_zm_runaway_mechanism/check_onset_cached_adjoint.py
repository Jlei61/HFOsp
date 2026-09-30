"""Check the full delayed adjoint by its defining bilinear identity."""
from common import OUT,np,write,read,log
from onset_state_continuation import build
from onset_cached_tangent import CachedTangent
from onset_cached_adjoint import CachedAdjoint
from onset_variational_return import Coordinates
from fine_rate_frozen_Z_fields import capture,restore
import argparse,os,time


def main(device):
    out=OUT/'core_a_bifurcation_type_20260924/numerical_checks/full_cached_adjoint'
    out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    source=OUT/'core_a_bifurcation_type_20260924/near_returns/above70s_A_sustained_B_cycle/exact_seed/node00.npz'
    write(out/'contract.json',dict(source=str(source),dt_ms=.05,interval_ms=10,
        method='Transpose the entire exact cached tangent step in reverse order: reset/write of delayed history and M, refractory response, input-memory cascades, covariance, synaptic filters, and original spatial delay graph. Uses independent adjoint buffers/clock. Atomic transpose summation is checked by full-state bilinear products.',
        acceptance='Three independent full-state vector/covector pairs, normalized coordinate bilinear relative defect<1e-9; all original nominal model arrays remain bitwise unchanged by adjoint evaluation. This is an implementation gate, no bifurcation/chaos/basin certificate.',model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()));begin=time.time()
    e=build(device);base=dict(np.load(source));restore(e,base);c=Coordinates(base,e.s)
    forward=CachedTangent(e,round(10/e.dt));forward.graph()
    reverse=CachedAdjoint(forward);reverse.graph();rows=[]
    rng=np.random.default_rng(92498)
    for j in range(3):
        v=rng.normal(size=c.size);w=rng.normal(size=c.size)
        v.reshape(-1,c.P)[4,~e.s.E]=0;w.reshape(-1,c.P)[4,~e.s.E]=0
        v/=np.linalg.norm(v);w/=np.linalg.norm(w)
        restore(e,base);c.set_tangent(forward,v);forward.chunk();Jv=c.tangent(forward)
        before=capture(e);end_tick=int(before['clock'][0]);assert end_tick==c.tick+round(10/e.dt)
        reverse.set_covector(c,w,end_tick);reverse.chunk();JT_w=reverse.covector(c)
        assert int(reverse.clock.get()[0])==c.tick
        after=capture(e);assert all(np.array_equal(a,after[k]) for k,a in before.items())
        lhs=float(w@Jv);rhs=float(v@JT_w);error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
        row=dict(direction=j,forward_inner_product=lhs,adjoint_inner_product=rhs,
            relative_bilinear_error=error,adjoint_nominal_state_bitwise_unchanged=True,
            forward_product_norm=float(np.linalg.norm(Jv)),adjoint_product_norm=float(np.linalg.norm(JT_w)))
        rows.append(row);write(out/'progress.json',rows);log('FULL CACHED ADJOINT CHECK',row)
        assert error<1e-9,row
    write(out/'result.json',dict(status='PASS',rows=rows,seconds=time.time()-begin,
        scope='Full-state transpose derivative implementation at actual A-sustained/B-intermittent state. Does not provide any critical multiplier, manifold, bifurcation, or native-SNN mechanism evidence.',model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args();main(a.device)
