"""Convert a periodic-family derivative into a physical DDE initial history.

The period derivative acts on every delay/filter and on physical history time.
This is a diagnostic trial direction, not an eigenvector or stability verdict.
"""
from native_path import *
from streaming_periodic import HarmonicOperator, StreamPeriodic
from periodic_v3 import TAU_M


def state_harmonics(o, sol, tangent):
    cp=o.cp;s=o.s;r=cp.asarray(sol['r']);N=len(r);K=N//2+1
    T=float(sol['T']);D=float(sol['D']);s.set_D(D);Z=cp.asarray(s.Z)
    v=cp.asarray(tangent);assert v.size==r.size+2
    ell=v[-2];dD=v[-1]*.001;dr=v[:r.size].reshape(r.shape)*.001
    dz=cp.asarray(path_Z_derivative(s,D))*dD
    ops,_,_,lam=o.kernels(T)
    rf=cp.fft.rfft(r,axis=0);drf=cp.fft.rfft(dr,axis=0)
    a=[(q@rf.ravel()).reshape(K,s.P) for q in ops]
    da=[(q@drf.ravel()).reshape(K,s.P) for q in ops]
    phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0])
    phase*=cp.asarray(s.delays)[:,None]*lam[:,0]*ell
    for j,raw in enumerate(o.raw):
        da[j]+=(HarmonicOperator(o,raw,phase,cache=False)@rf.ravel()).reshape(K,s.P)
    def filt(pair,tau):
        h,dh=pair;f=1/(1+lam*tau)
        return h*f,dh*f+h*f*(lam*tau/(1+lam*tau))*ell
    def scaled(pair,c):return pair[0]*c,pair[1]*c
    tm=cp.asarray(s.tm);E=cp.asarray(s.E)
    qa=filt(scaled((a[0],da[0]),tm*s.area[0]),s.rise[0])
    ia=filt(qa,s.decay[0])
    qg=filt(scaled((a[1],da[1]),tm*s.area[1]),s.rise[1])
    ig=filt(qg,s.decay[1])
    va=filt(scaled((a[2],da[2]),tm*s.area[0]**2),s.tau[0]/2)
    vg=filt(scaled((a[3],da[3]),tm*s.area[1]**2),s.tau[1]/2)
    m=filt(scaled((rf,drf),.5*E),TAU_M)
    mu=(ia[0]-Z*ig[0]-m[0],ia[1]-Z*ig[1]-dz*ig[0]-m[1])
    vi=(Z*Z*vg[0],Z*Z*vg[1]+2*Z*dz*vg[0])
    poles=[cp.asarray(q) for q in s.poles]
    pairs=[filt(mu,poles[0]),filt(mu,poles[1]),filt(va,poles[2]),
           filt(vi,poles[3]),qa,ia,qg,ig,va,vg,m,None,
           filt(va,poles[4]),filt(vi,poles[5])]
    # Return normalized-phase harmonics. Constant external input is affine and
    # has zero family derivative; the Z state is handled explicitly below.
    out=[]
    for j,pair in enumerate(pairs):
        if pair is None:
            h=cp.zeros_like(rf);dh=cp.zeros_like(rf);h[0]=N*Z;dh[0]=N*dz
        else:h,dh=pair
        if j in (0,1):h=h.copy();h[0]+=N*cp.asarray(s.private_mu)
        if j in (2,12):h=h.copy();h[0]+=N*cp.asarray(s.private_ve)
        out.append((h.get(),dh.get()))
    return out


def physical_history(sol,tangent,steps,depth):
    """Rate-history derivative at fixed negative physical time, population blocks."""
    from scipy.fft import rfft,irfft
    r=sol['r'];N,P=r.shape;T=float(sol['T']);v=np.asarray(tangent)
    dr=v[:r.size].reshape(r.shape)*.001;ell=float(v[-2])
    lag=-np.arange(1,depth+1)*T/steps;indices=(-np.arange(1,depth+1))%steps
    history=np.empty((depth,P));freq=2j*np.pi*np.arange(N//2+1)/T
    for lo in range(0,P,16):
        hi=min(lo+16,P);rf=rfft(r[:,lo:hi].T,axis=1);df=rfft(dr[:,lo:hi].T,axis=1)
        if N%2==0:rf[:,-1]=rf[:,-1].real*.5;df[:,-1]=df[:,-1].real*.5
        direct=irfft(df,n=steps,axis=1)[:,indices].T*(steps/N)
        time_derivative=irfft(rf*freq,n=steps,axis=1)[:,indices].T*(steps/N)
        history[:,lo:hi]=direct-lag[:,None]*ell*time_derivative
    return history


def initial_vector(o,sol,tangent,steps,depth,include_parameter=False):
    harmonics=state_harmonics(o,sol,tangent);N=len(sol['r'])
    weights=np.full(N//2+1,2.);weights[0]=1.
    if N%2==0:weights[-1]=1.
    state=np.array([weights@dh.real/N for h,dh in harmonics])
    dz=state[11].copy()
    if not include_parameter:state[11]=0. # homogeneous conditional flow holds Z
    history=physical_history(sol,tangent,steps,depth)
    q=dict(source_D=float(sol['D']),source_T_ms=float(sol['T']),N=N,
           dD_dlog_coordinate=float(tangent[-1])*.001,
           dT_dlog_coordinate=float(tangent[-2])*float(sol['T']),
           held_Z_parameter_direction_norm=float(np.linalg.norm(dz)),
           scope='Physical family tangent; at nonzero dD it also has parameter forcing and is not a homogeneous Floquet eigenvector.')
    return np.r_[state.ravel(),history.ravel()],q


def flow_diagnostic(o,sol,tangent,m,phase):
    """Test the family-tangent/Jordan relation using an independent time map.

    At an exact parameter fold, dD vanishes and M*x-x=-T'*phase.
    Away from it an omitted parameter-forcing term remains. Therefore this
    diagnostic never assigns a bifurcation type from a small residual alone.
    """
    x,q=initial_vector(o,sol,tangent,m.n,m.Dd)
    def project(v):return v-phase*(phase@v)/(phase@phase)
    mx=m.matvec(x);px=project(x);pmx=project(mx)
    q.update(projected_unit_defect=float(np.linalg.norm(pmx-px)/np.linalg.norm(px)),
             projected_Rayleigh_value=float(px@pmx/(px@px)),
             raw_Jordan_relative_residual=float(np.linalg.norm(mx-x+q['dT_dlog_coordinate']*phase)/
                 max(np.linalg.norm(x),np.linalg.norm(q['dT_dlog_coordinate']*phase))),
             projected_fraction=float(np.linalg.norm(px)/np.linalg.norm(x)),
             dt_ms=m.dt,status='DIAGNOSTIC_ONLY',
             caution='Nonzero parameter tangent causes inhomogeneous forcing; no automatic fold or eigenvalue acceptance.')
    return q


def check(device=0):
    from orbit_reconstruction import orbit_states
    from scipy.signal import resample
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz')
    r=resample(z['r'],129,axis=0);N=len(r);T=float(z['T']);D=.219
    # The historical seed is exactly at a piecewise-Z path knot. A centered
    # difference there averages two slopes and is not its right derivative.
    # This constitutive (non-root) check uses a segment interior instead.
    o=StreamPeriodic(s,N,device);o.cache_mean_operators=False
    phase=np.arange(N)/N
    dr=1e-4*np.sin(2*np.pi*phase)[:,None]*np.ones((1,s.P))
    tangent=np.r_[(dr/.001).ravel(),.03,.2]
    sol=dict(r=r,T=T,D=D)
    hs=state_harmonics(o,sol,tangent)
    base=np.stack([np.fft.irfft(h,n=N,axis=0) for h,dh in hs],axis=1)
    analytic=np.stack([np.fft.irfft(dh,n=N,axis=0) for h,dh in hs],axis=1)
    reference,_=orbit_states(o,sol,N)
    base_error=float(np.linalg.norm(base-reference[:-1])/np.linalg.norm(base))
    assert base_error<1e-12,base_error
    rows=[]
    for h in [1e-4,5e-5]:
        plus=dict(r=r+h*dr,T=T*np.exp(h*tangent[-2]),D=D+h*tangent[-1]*.001)
        minus=dict(r=r-h*dr,T=T*np.exp(-h*tangent[-2]),D=D-h*tangent[-1]*.001)
        yp,_=orbit_states(o,plus,N);ym,_=orbit_states(o,minus,N)
        fd=(yp[:-1]-ym[:-1])/(2*h)
        error=float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic))
        component_errors=(np.linalg.norm((fd-analytic).transpose(1,0,2).reshape(14,-1),axis=1)/
                          np.maximum(np.linalg.norm(analytic.transpose(1,0,2).reshape(14,-1),axis=1),1e-30))
        rows.append(dict(h=h,relative_error=error,component_errors=component_errors.tolist()))
        log('PHYSICAL TANGENT FINITE DIFFERENCE',rows[-1])
    assert all(q['relative_error']<1e-6 for q in rows),rows
    steps=512;depth=71;hist=physical_history(sol,tangent,steps,depth)
    lag=-np.arange(1,depth+1)*T/steps
    expected=1e-4*np.sin(2*np.pi*lag/T)[:,None]*np.ones((1,s.P))
    # Independently evaluate source rate Fourier derivative at unwrapped lags.
    rf=np.fft.rfft(r,axis=0);w=np.full(len(rf),2.);w[0]=1.
    freq=2j*np.pi*np.arange(len(rf))/T
    rt=(np.exp(lag[:,None]*freq)@(w[:,None]*freq[:,None]*rf)).real/N
    expected-=lag[:,None]*tangent[-2]*rt
    history_error=float(np.max(abs(hist-expected)));assert history_error<1e-11,history_error
    q=dict(status='PASS',base_state_relative_error=base_error,derivative_checks=rows,
           physical_history_max_error=history_error,
           scope='Reconstruction and chain-rule check only; no stability or onset claim.')
    write(OUT/'physical_cycle_tangent_check.json',q);log('PHYSICAL TANGENT CHECK',q)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    check(p.parse_args().device)
