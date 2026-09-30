"""Stationary, tangent, and full-characteristic checks for the rate-only figure."""
from rate_field_spectrum import *
from nyquist import parity
from scipy.sparse.linalg import splu


def numerical(s):
    branch=np.load(OUT/'critical_revision/branch.npz');stationary=[]
    for i in np.linspace(0,len(branch['J'])-1,25,dtype=int):
        J=float(branch['J'][i]);r=branch['rates'][i];y=s.equilibrium_state(r,J);arr=np.array([a@r for a in s.matrices(J)])
        stationary.append(float(abs(s.rhs(y,arr)).max()))
    rows=[]
    for core in ['A','B']:
        z=np.load(RATE_OUT/f'hopf_{core}.npz');r=z['rates'];J=float(z['J']);lam=1j*float(z['omega']);v=z['vector']
        y=s.equilibrium_state(r,J);dy=s.eigenstate(r,J,lam,v);arr=np.array([a@r for a in s.matrices(J)])
        da=np.array([a@v for a in s.matrices(J,lam)]);scale=1e-5/np.max(abs(dy));fd=[]
        for component in [np.real,np.imag]:
            f=(s.rhs(y+scale*component(dy),arr+scale*component(da))-s.rhs(y-scale*component(dy),arr-scale*component(da)))/(2*scale)
            fd.append(f)
        derivative=fd[0]+1j*fd[1]
        error=float(np.linalg.norm(derivative-lam*dy)/np.linalg.norm(lam*dy))
        rows.append(dict(core=core,tangent_eigenstate_relative_error=error))
    # CPU/GPU identity at a nonstationary burst state as well as equilibrium.
    from pathlib import Path
    trajectories=list((RATE_OUT/'runs/pilot').glob('*/trajectory.npz'));gpu=[]
    for p in trajectories:
        J=float(p.parent.name[1:]);d=np.load(p);y=d['final_state'];r=s.output(y)
        e=RateIntegrator(s,J,initial=y);e.arrivals(0)
        e.k['rhs'](((s.P+127)//128,),(128,),(e.y,e.arr,e.pars,e.f))
        cpu=s.rhs(y,np.array([a@r for a in s.matrices(J)]));diff=e.f.get()-cpu
        gpu.append(dict(J=J,maximum_absolute_rhs_error=float(abs(diff).max()),relative_rhs_error=float(np.linalg.norm(diff)/max(np.linalg.norm(cpu),1e-15))))
    result=dict(maximum_equilibrium_rhs_residual=max(stationary),equilibria_checked=len(stationary),tangent=rows,cpu_gpu=gpu)
    assert max(stationary)<1e-7 and max(x['tangent_eigenstate_relative_error'] for x in rows)<1e-4
    assert all(x['maximum_absolute_rhs_error']<1e-7 for x in gpu)
    write(RATE_OUT/'numerical_checks.json',result);print('NUMERICAL',result,flush=True)


def root_counts(s):
    rows=[]
    for J in [.6,.934,.942,.95]:
        dest=RATE_OUT/'nyquist'/f'J{J:.6f}.json'
        if dest.exists():rows.append(read(dest));continue
        r,ok,_=s.solve(J);assert ok;cache={}
        def evaluate(f):
            f=float(f)
            if f in cache:return cache[f]
            lam=2j*np.pi*f/1000
            M=(sparse.diags(s.filter_response(lam))@s.characteristic(r,J,lam)).tocsc()
            lu=splu(M);phase=float(np.angle(np.exp(1j*(np.angle(lu.U.diagonal()).sum()+np.pi*(parity(lu.perm_r)+parity(lu.perm_c))))))
            bound=float(np.asarray(abs(sparse.eye(s.P)-M).sum(1)).max());cache[f]=(phase,bound);return cache[f]
        freq=np.unique(np.r_[0.,np.linspace(.02,15,101),np.linspace(15,60,31),np.geomspace(60,2000,35)])
        for j in range(2):
            freq=np.sort(np.r_[freq,(freq[:-1]+freq[1:])/2])
        for f in freq:evaluate(f)
        for j in range(10):
            freq=np.array(sorted(cache));phase=np.unwrap([evaluate(f)[0] for f in freq]);jumps=abs(np.diff(phase))
            if jumps.max()<.25:break
            for i in np.flatnonzero(jumps>=.25):evaluate((freq[i]+freq[i+1])/2)
        assert jumps.max()<.25
        count=-round((phase[-1]-phase[0])/np.pi)
        q=dict(J_EE_core=J,unstable_roots=count,points=len(freq),maximum_phase_step=float(jumps.max()),tail_loop_norm=evaluate(freq[-1])[1],
            normalization='Left multiplied by positive-mixture H(lambda); H has no poles/zeros in Re(lambda)>=0',
            limit='Numerically refined determinant count; not an analytic enumeration of every possible branch')
        write(dest,q);rows.append(q);print('COUNT',q,flush=True)
    write(RATE_OUT/'nyquist.json',dict(rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--counts',action='store_true');args=p.parse_args();s=RateField()
    if args.counts:root_counts(s)
    else:numerical(s)
