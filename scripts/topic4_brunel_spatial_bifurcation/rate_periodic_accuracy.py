"""Oversampled continuous Fourier defect before Floquet classification.

Evaluate the nonlinearity on a four-times finer grid without discarding the
complex Nyquist response of the delay/filter operators. This detects aliasing
that a tiny collocation residual alone cannot reveal.
"""
from rate_periodic import *
import gc
from audit_rate_filter_states import filter_state_minima


def defect(s,path,device=0,factor=4,harmonic_chunk_size=0,stream_harmonics=False):
    # Streamed callers request bounded storage even on smaller meshes.
    # Blocking retains every Fourier harmonic and the identical Phi; the
    # full-bank/blocked agreement is checked on smooth and sharp profiles.
    if stream_harmonics or len(np.load(path)['r']) >= 4096:
        return defect_bounded(s,path,device,factor,harmonic_chunk_size or 64)
    z=np.load(path);r=z['r'];T=float(z['T']);J=float(z['J']);N=len(r);M=factor*N
    o=Periodic(s,N,device);o.low_memory=True;cp=o.cp;K=o.K
    o.harmonic_chunk_size=harmonic_chunk_size;o.derivative_chunk_size=harmonic_chunk_size
    o.stream_harmonics=stream_harmonics
    kernels=o.kernels(T,J);cf=cp.fft.rfft(cp.asarray(r),axis=0)/N
    a,b,qa,qb=[(op@cf.ravel()).reshape(K,s.P) for op in kernels[0]]
    ha,hg,hva,hvg,hm=kernels[3];tm,ref,th,alpha,tf,ts,E=o.gpars
    mf=cp.stack([tm*(s.area[0]*ha*a-s.area[1]*hg*b)-.5*E*hm*cf,
        tm*s.area[0]**2*hva*qa,tm*s.area[1]**2*hvg*qb])
    mf[:,0]+=o.private
    pad=cp.zeros((3,M//2+1,s.P),complex);pad[:,:K]=mf*M;pad[:,K-1]*=.5
    mom=cp.fft.irfft(pad,n=M,axis=1)
    phi=cp.empty((M,s.P));o.phik(((M*s.P+127)//128,),(128,),
        (cp.ascontiguousarray(mom),o.pars,phi,np.int32(M)))
    rp=cp.zeros((M//2+1,s.P),complex);rp[:K]=cf*M;rp[K-1]*=.5
    rr=cp.fft.irfft(rp,n=M,axis=0);lam=2j*cp.pi*cp.arange(M//2+1)[:,None]/T
    H=alpha/(1+lam*tf)+(1-alpha)/(1+lam*ts)
    err=rr-cp.fft.irfft(cp.fft.rfft(phi,axis=0)*H,n=M,axis=0)
    region=[]
    for i in range(3):
        mask=s.E&(s.geo['group_region']==i);w=cp.asarray(s.geo['group_size'][mask],float);w/=w.sum()
        region.append(float(cp.max(cp.abs(err[:,mask]@w)))*1000)
    out=dict(orbit=str(path),N=N,check_N=M,J_EE_core=J,T_ms=T,
        maximum_group_defect_Hz=float(cp.max(cp.abs(err)))*1000,regional_defect_Hz=region,
        minimum_rate_Hz=float(cp.min(rr))*1000)
    print('CONTINUOUS DEFECT',out,flush=True)
    # kernels contain closures over their Periodic owner. Break the cache
    # cycle before returning so a refinement loop can release the GPU bank.
    o.cache=None
    return out


def defect_bounded(s,path,device=0,factor=4,harmonic_chunk_size=64,phase_chunk_size=512):
    """Same fourfold Fourier defect with bounded GPU storage.

    Keep every harmonic, edge, physical delay, and all 935 populations. Only
    linear actions are blocked in frequency; CPU FFTs reconstruct all phases.
    The identical GPU transfer function is evaluated in blocks of time. This
    avoids allocating sparse indices for every harmonic simultaneously.
    """
    from scipy import fft
    z=np.load(path);r=z['r'];T=float(z['T']);J=float(z['J']);N=len(r);M=factor*N
    block=harmonic_chunk_size;K=N//2+1
    o=Periodic(s,2*block,device);cp=o.cp
    cf=fft.rfft(r,axis=0,workers=4)/N
    mf=np.empty((3,K,s.P),complex)
    for first in range(0,K,block):
        last=min(K,first+block);count=last-first
        lam=2j*cp.pi*cp.arange(first,last)[:,None]/T
        phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0])
        v=cp.asarray(cf[first:last]);arrivals=[]
        for kind,(d,mask,index,ptr) in enumerate(o.raw):
            scale=cp.where(mask,J**(1 if kind==0 else 2),1.) if kind in (0,2) else 1.
            data=(d@phase).T*scale;edges=d.shape[0]
            op=o.cs.csr_matrix((data.ravel(),index[:count*edges],ptr[:count*s.P+1]),
                shape=(count*s.P,count*s.P))
            arrivals.append((op@v.ravel()).reshape(count,s.P))
        a,b,qa,qb=arrivals
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]))
        hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        hva=1/(1+lam*s.tau[0]/2);hvg=1/(1+lam*s.tau[1]/2);hm=1/(1+lam*1000)
        tm,ref,th,alpha,tf,ts,E=o.gpars
        mf[:,first:last]=cp.stack([tm*(s.area[0]*ha*a-s.area[1]*hg*b)-.5*E*hm*v,
            tm*s.area[0]**2*hva*qa,tm*s.area[1]**2*hvg*qb]).get()
    mf[:,0]+=o.private.get()
    # Zero-padding preserves the same real-series Nyquist convention used by
    # defect(); the imaginary filtered response at that harmonic is retained.
    mom=np.empty((3,M,s.P))
    pad=np.zeros((M//2+1,s.P),complex)
    for kind in range(3):
        pad.fill(0);pad[:K]=mf[kind]*M;pad[K-1]*=.5
        mom[kind]=fft.irfft(pad,n=M,axis=0,workers=4)
    del mf
    phi=np.empty((M,s.P))
    for first in range(0,M,phase_chunk_size):
        last=min(M,first+phase_chunk_size);count=last-first
        moments=cp.asarray(np.ascontiguousarray(mom[:,first:last]));target=cp.empty((count,s.P))
        o.phik(((count*s.P+127)//128,),(128,),
            (moments,o.pars,target,np.int32(count)))
        phi[first:last]=target.get()
    del mom
    pad.fill(0);pad[:K]=cf*M;pad[K-1]*=.5
    rr=fft.irfft(pad,n=M,axis=0,workers=4)
    del pad,cf
    filtered=fft.rfft(phi,axis=0,workers=4)
    del phi
    for first in range(0,len(filtered),block):
        last=min(len(filtered),first+block)
        lam=2j*np.pi*np.arange(first,last)[:,None]/T
        filtered[first:last]*=s.filter_response(lam)
    err=rr-fft.irfft(filtered,n=M,axis=0,workers=4)
    regional=[]
    for i in range(3):
        mask=s.E&(s.geo['group_region']==i);w=s.geo['group_size'][mask].astype(float);w/=w.sum()
        regional.append(float(np.max(np.abs(err[:,mask]@w)))*1000)
    out=dict(orbit=str(path),N=N,check_N=M,J_EE_core=J,T_ms=T,
        maximum_group_defect_Hz=float(np.max(np.abs(err)))*1000,regional_defect_Hz=regional,
        minimum_rate_Hz=float(np.min(rr))*1000,
        computation='Exact blocked harmonic actions, CPU FFT, identical GPU Phi in time blocks',
        harmonic_chunk_size=block,phase_chunk_size=phase_chunk_size)
    print('CONTINUOUS BOUNDED DEFECT',out,flush=True)
    return out


def prepare(path,device=0,max_N=4096,host_krylov=False,min_N=0,check_filter_states=False,
            harmonic_chunk_size=0,before_mesh=None,stream_harmonics=False,adaptive_memory=False):
    s=RateField();path=Path(path);origin=path;precision_retries=set()
    while True:
        if before_mesh:before_mesh(len(np.load(path)['r']),'DEFECT')
        current_N=len(np.load(path)['r'])
        use_stream=stream_harmonics or (adaptive_memory and current_N>=8192)
        q=defect(s,path,device,harmonic_chunk_size=harmonic_chunk_size,stream_harmonics=use_stream);N=q['N']
        filter_pass=True
        if check_filter_states:
            z=np.load(path);q['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
            filter_pass=q['filter_state_check']['positive']
        # These are declared numerical survey thresholds, not native-model
        # acceptance criteria or a bifurcation-location error bound.
        if (filter_pass and N>=min_N and q['maximum_group_defect_Hz']<.1 and max(q['regional_defect_Hz'])<.001
            and q['minimum_rate_Hz']>=-1e-9):
            return path,dict(status='RESOLUTION_CHECKED',source=str(origin),**q)
        # A mesh refinement can retain the same underconverged interpolant:
        # the default 2e-8 Hz Newton stop is looser than the 1e-9 Hz filter
        # positivity tolerance. If the off-grid defect is already explained
        # by the stored algebraic residual, first correct that residual at
        # the same N. Aliasing-dominated errors still require a finer mesh.
        solve_tol=2e-11 if check_filter_states else 2e-8
        stored_error=float(np.load(path)['residual'])
        precision_retry=(check_filter_states and not filter_pass and N not in precision_retries
                         and stored_error>solve_tol and
                         q['maximum_group_defect_Hz']<=5*stored_error and N>=min_N)
        target_N=N if precision_retry else max(2*N,min_N)
        if target_N>max_N:return path,dict(status='RESOLUTION_UNRESOLVED',source=str(origin),**q)
        if precision_retry:
            precision_retries.add(N)
            print('ALGEBRAIC PRECISION REFINEMENT',N,'stored Hz',stored_error,
                  'off-grid Hz',q['maximum_group_defect_Hz'],'target Hz',solve_tol,flush=True)
        import cupy as cp
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        if before_mesh:before_mesh(target_N,'NEWTON')
        z=np.load(path)
        use_stream=stream_harmonics or (adaptive_memory and target_N>=8192)
        index_check=PERIODIC_OUT/'bounded_harmonic_index_monodromy_check.json'
        bounded=use_stream and index_check.exists() and read(index_check).get('status')=='PASS'
        # Keep all harmonics while reusing local sparse-index blocks. The
        # operator, parameter-derivative and full-history flow equivalence
        # were checked before this lower-storage construction is enabled.
        capacity=max(64,harmonic_chunk_size) if bounded else None
        o=Periodic(s,target_N,device,harmonic_capacity=capacity);o.low_memory=True
        o.stream_harmonics=use_stream
        o.harmonic_chunk_size=harmonic_chunk_size;o.derivative_chunk_size=harmonic_chunk_size
        o.normalize_linear_rhs=True
        o.host_krylov=host_krylov or (adaptive_memory and target_N>=4096)
        o.linear_target_aware=check_filter_states
        r,T,J,err,history=o.solve(resample(z['r'],target_N,axis=0),float(z['T']),float(z['J']),
                                 maxiter=24,tol=solve_tol)
        name=f'{origin.stem}_accuracy_N{target_N}'
        # An unsuccessful older solve is evidence too. Keep it intact and
        # give a resumed numerical attempt its own artifact identity.
        if (PERIODIC_OUT/'orbits'/f'{name}.npz').exists():
            name+=f'_attempt_{time.time_ns()}'
        candidate=save_orbit(s,r,T,J,err,history,name)
        o.cache=None;del o;gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        if err>2e-8:return candidate,dict(status='REFINEMENT_FAILED',source=str(origin),residual_Hz=err)
        path=candidate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--device',type=int,default=0);p.add_argument('--refine',action='store_true')
    a=p.parse_args()
    if a.refine:print(prepare(a.path,a.device))
    else:print(defect(RateField(),a.path,a.device))
