"""Validate the new implementation separately from its scientific adequacy."""
from common import *
from rate import phi,table_build,sparse_delay,sparse_mv,sparse_window
from scipy import integrate,sparse,special
import math

def main():
    table,lo,step=table_build();rng=np.random.default_rng(20260916);errors=[]
    for _ in range(300):
        tm=rng.choice([10.,20.]);ref=tm/10;theta=rng.uniform(13.,20.);mu=rng.uniform(5.,60.)
        ve=10**rng.uniform(-2,2.3);vi=10**rng.uniform(-2,2.3);sig=math.sqrt(ve+vi)
        m=mu-1.0325*math.sqrt((4.2*ve+19*vi)/tm);a=(11-m)/sig;b=(theta-m)/sig
        expected=0. if b>12 else 1/(ref+tm*math.sqrt(math.pi)*integrate.quad(lambda u:special.erfcx(-u),a,b,epsabs=1e-10)[0])
        actual=phi(np.array([mu]),np.array([ve]),np.array([vi]),np.array([tm]),np.array([ref]),
            np.array([[theta]]),np.ones((1,1)),table,lo,step,11.)[0]
        errors.append(1000*abs(actual-expected))
    assert max(errors)<.001,max(errors)
    ops=[]
    old=np.load(ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/projected_graph.npz')
    six_count=old['count'];scale=np.ones((6,6));scale[0,0]=scale[1,1]=J
    projection=[]
    for grid in (10,20):
        model=np.load(OUT/f'grid{grid}/model.npz');reg=model['region'];count=model['count']
        assert np.array_equal(reg[model['group']],old['region'])
        P=len(count);D=read(OUT/f'grid{grid}/prepared.json')['delay_bins']
        hist=rng.uniform(0,.2,(D,P));head=17;ordered=hist[(head-np.arange(D))%D].ravel()
        for kind in ('ampa','gaba'):
            M=sparse.load_npz(OUT/f'grid{grid}/{kind}_delay.npz')
            out=sparse_delay(M.indptr,M.indices,M.data,hist,head,P,D)
            err=float(abs(out-M@ordered).max());assert err<1e-9
            coo=M.tocoo();col=(D-1-coo.col//P)*P+coo.col%P
            reverse=sparse.coo_matrix((coo.data,(coo.row,col)),shape=M.shape).tocsr()
            dup=np.tile(hist,(2,1)).ravel()
            fast=sparse_window(reverse.indptr,reverse.indices,reverse.data,dup,(head+1)*P)
            ferr=float(abs(fast-out).max());assert ferr<1e-9
            Q=sparse.load_npz(OUT/f'grid{grid}/{kind}_variance.npz');x=hist[0]
            qerr=float(abs(sparse_mv(Q.indptr,Q.indices,Q.data,x)-Q@x).max());assert qerr<1e-9
            ops.append(dict(grid=grid,kind=kind,delay_operator_error=err,variance_operator_error=qerr,duplicate_ring_error=ferr))
            rise=.7 if kind=='ampa' else 1.;is_source=np.arange(6)<3 if kind=='ampa' else np.arange(6)>=3
            mc=M.tocoo();a=reg[mc.row];b=reg[mc.col%P];lag=mc.col//P
            weights=mc.data*rise/np.where(a<3,20.,10.)*count[mc.row]/six_count[a]
            coarse=np.bincount((lag*6+a)*6+b,weights=weights,minlength=D*36).reshape(D,6,6)
            expected=old['W']*scale*is_source[None,None,:]
            pe=float(abs(coarse-expected).max());assert pe<1e-10
            qc=Q.tocoo();a=reg[qc.row];b=reg[qc.col]
            coarseQ=np.bincount(a*6+b,weights=qc.data*count[qc.row]/six_count[a],minlength=36).reshape(6,6)
            expectedQ=old['Q'].sum(0)*scale**2*is_source[None,:]
            qe=float(abs(coarseQ-expectedQ).max());assert qe<1e-10
            projection.append(dict(grid=grid,kind=kind,per_delay_W_six_population_error=pe,Q_six_population_error=qe))
    # Exact expectation of one native event: same gate and current update.
    impulse=[]
    for rise,decay in [(.7,3.5),(1.,18.)]:
        native_g=native_c=rate_g=rate_c=0.;err=0.
        for k in range(1000):
            spike=1. if k==3 else 0.
            native_g=native_g*np.exp(-.1/rise)+spike*2.7
            native_c=native_g+(native_c-native_g)*np.exp(-.1/decay)
            rate_g=rate_g*np.exp(-.1/rise)+.1*(spike/.1)*2.7
            rate_c=rate_g+(rate_c-rate_g)*np.exp(-.1/decay)
            err=max(err,abs(native_c-rate_c))
        assert err<1e-12;impulse.append(err)
    write(OUT/'numerical_validation.json',dict(status='PASS_IMPLEMENTATION_ONLY',transfer_max_absolute_error_hz=max(errors),
        operators=ops,native_synaptic_impulse_error=impulse,original_graph_projection=projection,scientific_correspondence='not implied'))
    print('PASS',max(errors),flush=True)

if __name__=='__main__':main()
