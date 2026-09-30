"""Locate a secondary -1 crossing using the actual doubled-branch coordinate.

Near its parent PD, the child is ill-conditioned under fixed-J Newton solves.
Fixing the established antiperiodic-mode amplitude leaves J as a BVP unknown.
Only the continuation coordinate changes; every spatial group/delay remains.
"""
from rate_antiperiodic import *
from scipy.optimize import brentq


def main():
    from cupyx.scipy.sparse.linalg import LinearOperator as CL,gmres
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second')
    p.add_argument('--parent',required=True);p.add_argument('--parent-mode',required=True)
    p.add_argument('--seed',required=True);p.add_argument('--N',type=int,default=2048)
    p.add_argument('--device',type=int,default=1);p.add_argument('--label',default='PD_upper_child_next_amplitude')
    p.add_argument('--cache-glob');p.add_argument('--scan',type=float,nargs='+')
    p.add_argument('--stream-harmonics',action='store_true')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--amplitude-bracket',type=float,nargs=2)
    p.add_argument('--tol',type=float,default=1e-9,
        help='Periodic BVP residual tolerance in Hz; stricter values do not change the model or root coordinate')
    a=p.parse_args();s=RateField();crit=read(a.parent);base=np.load(crit['orbit'])
    assert 0<a.tol<=1e-9
    parent=resample(base['r'],a.N//2,axis=0);base_r=np.r_[parent,parent]
    u=np.load(a.parent_mode)['u'];u=resample(np.r_[u,-u],a.N,axis=0);u/=abs(u).max()
    c=np.r_[u.ravel()/np.sum(u*u),0.,0.];cache=[]
    def add(path):
        z=np.load(path);r=resample(z['r'],a.N,axis=0)
        coord=float(c@np.r_[((r-base_r)*1000).ravel(),0.,0.])
        cache.append((coord,r,float(z['T']),float(z['J'])))
    add(a.first);add(a.second);bracket=sorted([cache[0][0],cache[1][0]])
    if a.amplitude_bracket:bracket=sorted(a.amplitude_bracket)
    if a.cache_glob:
        for path in sorted((PERIODIC_OUT/'orbits').glob(a.cache_glob)):
            if float(np.load(path)['residual'])<2e-8:add(path)
    q=np.load(a.seed)['u'].real
    q=resample(np.r_[q,-q],2*a.N,axis=0)[:a.N].ravel();q/=np.linalg.norm(q)
    rows=[];last=None
    def evaluate(coord):
        nonlocal last
        old,r,T,J=min(cache,key=lambda x:abs(x[0]-coord))
        r=r+(coord-old)*u/1000
        pred=np.r_[(r*1000).ravel(),np.log(T),J*1000]
        target=coord+c@np.r_[(base_r*1000).ravel(),0.,0.]
        pred+=c*(target-c@pred)/(c@c)
        o=Periodic(s,a.N,a.device);o.low_memory=True;o.harmonic_chunk_size=64;o.derivative_chunk_size=64
        o.stream_harmonics=a.stream_harmonics;o.host_krylov=a.host_krylov
        o.normalize_linear_rhs=True;o.krylov_restart=180
        o.linear_target_aware=a.tol<1e-9
        r,T,J,err,hist=o.solve(pred[:-2].reshape(a.N,s.P)/1000,T,J,
            arc=(pred,c,np.ones_like(c)),maxiter=26,tol=a.tol)
        path=save_orbit(s,r,T,J,err,hist,f'{a.label}_eval_a{coord:.12f}_N{a.N}')
        assert err<2e-8,(coord,J,err)
        if a.tol<1e-9:
            assert err<=a.tol*1.01,('Child BVP did not meet requested tolerance',coord,J,err)
        half=float(np.linalg.norm(r-np.roll(r,a.N//2,axis=0))/np.linalg.norm(r-r.mean(0)))
        assert half>1e-5,('Repeated parent',half)
        cache.append((coord,r,T,J));cp=o.cp;o.cache=None;del o;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        anti=Antiperiodic(s,path,a.N,a.device,harmonic_chunk_size=64,stream_harmonics=a.stream_harmonics);qq=cp.asarray(q);dim=len(q)
        def mv(x):return cp.r_[anti.apply(x[:-1])+qq*x[-1],cp.vdot(qq,x[:-1]).real]
        op=CL((dim+1,dim+1),matvec=mv,dtype=float);rhs=cp.zeros(dim+1);rhs[-1]=1
        if a.host_krylov:
            from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres
            host_op=HostOperator(op.shape,matvec=lambda v:(op@cp.asarray(v)).get(),dtype=float)
            yy,info=host_gmres(host_op,rhs.get(),rtol=2e-10,atol=1e-12,restart=180,maxiter=20)
            y=cp.asarray(yy);del host_op,yy
        else:y,info=gmres(op,rhs,tol=2e-10,atol=1e-12,restart=180,maxiter=3600)
        error=float(cp.linalg.norm(op@y-rhs));assert error<1e-7,(info,error)
        eta=float(y[-1]);mode=y[:-1];defect=float(cp.linalg.norm(anti.apply(mode))/cp.linalg.norm(mode))
        row=dict(coordinate='parent_PD_mode_amplitude_Hz',coordinate_value=coord,J_EE_core=J,T_ms=T,
            orbit=str(path),orbit_residual_hz=err,half_period_relative_mismatch=half,
            streamed_harmonic_actions=a.stream_harmonics,host_krylov=a.host_krylov,
            border_scalar=eta,linear_residual=error,antiperiodic_relative_residual=defect)
        rows.append(row);write(PERIODIC_OUT/f'{a.label}_scan_N{a.N}.json',rows)
        last=(row,mode.get().reshape(a.N,s.P));print('AMPLITUDE PD BORDER',row,flush=True)
        del op,anti;gc.collect();cp.get_default_memory_pool().free_all_blocks();return eta
    if a.scan:
        for coord in a.scan:evaluate(coord)
        return
    coord=brentq(evaluate,*bracket,xtol=1e-8,rtol=1e-12)
    h=min(.005,(bracket[1]-bracket[0])*.0001)
    eplus=evaluate(coord+h);plus=last[0]
    eminus=evaluate(coord-h);minus=last[0]
    dJ=(plus['J_EE_core']-minus['J_EE_core'])/(2*h)
    slope=(eplus-eminus)/(2*h);assert abs(dJ)>1e-12 and abs(slope)>1e-8,(dJ,slope)
    evaluate(coord);row,mode=last;assert row['antiperiodic_relative_residual']<1e-7,row
    row.update(label=a.label,N=a.N,multiplier=-1.,dborder_dJ=slope/dJ,dborder_dcoordinate=slope,
        dJ_dcoordinate=dJ,type='secondary period-doubling candidate located on the doubled branch',
        criticality='NOT_COMPUTED',validation='Requires temporal-mesh, positive parent and independent full-state/history mode checks.')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row)
    save_periodic_array(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz',u=mode,J=row['J_EE_core'],T=row['T_ms'])
    print('AMPLITUDE PD ROOT',row,flush=True)


if __name__=='__main__':main()
