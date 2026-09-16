#!/usr/bin/env python3
"""Exact linearized native-step map, with delay history; shift-invert Arnoldi."""
from analyze_topic4_corrected_bifurcation import *
from scipy.linalg import lu_factor, lu_solve
from scipy.signal import lfilter
from scipy.sparse.linalg import LinearOperator, eigs


class Tangent:
    def __init__(self,s,r,q):
        self.s=s;self.q=q;self.n=n=s.n;self.D=s.cfg['max_delay_steps'];self.size=n*(10+2*self.D)
        self.u,_,self.var=s.blocks(r,q);self.r=r
        self.ar=np.exp(-s.dt/np.array([s.ra,s.rg,s.ra,s.rg]))[:,None]
        self.ad=np.exp(-s.dt/np.array([s.ta,s.tau,s.ta,s.tau]))[:,None]
        self.B=s.dt*np.array([s.m.tau_mem_e_ms,s.m.tau_mem_e_ms,s.m.tau_mem_i_ms,s.m.tau_mem_i_ms])[:,None]/np.array([s.ra,s.rg,s.ra,s.rg])[:,None]
        self.alpha=s.dt/s.tr

    def unpack(self,x):
        n=self.n;return x[:2*n],x[2*n:6*n].reshape(4,n),x[6*n:10*n].reshape(4,n),x[10*n:].reshape(2,self.D,n)

    def drive(self,h):
        o=self.s.ops;q=self.q
        return np.array([o['ee']@h[0].ravel(),q*(o['ei']@h[1].ravel()),o['ie']@h[0].ravel(),q*(o['ii']@h[1].ravel())])

    def signed(self,c):return np.r_[c[0]-c[1],c[2]-c[3]]

    def action(self,x):
        r,g,c,h=self.unpack(x);ng=self.ar*g+self.B*self.drive(h);nc=self.ad*c+(1-self.ad)*ng
        nr=(1-self.alpha)*r+self.alpha*(self.var@r+self.u*self.signed(nc))
        nh=np.empty_like(h);nh[:,0]=r.reshape(2,self.n);nh[:,1:]=h[:,:-1]
        return np.r_[nr,ng.ravel(),nc.ravel(),nh.ravel()]

    def inverse(self,z):
        lu=lu_factor(self.s.characteristic(z,self.r,self.q));n=self.n;D=self.D
        weights=z**(-np.arange(1,D+1));Wr={k:(v@weights).reshape(n,n) for k,v in self.s.block_ops.items()}
        def solve(b):
            br,bg,bc,bh=self.unpack(b)
            hc=lfilter([1/z],[1,-1/z],bh,axis=1)
            dc=self.drive(hc);gc=(bg+self.B*dc)/(z-self.ar)
            cc=(bc+(1-self.ad)*(z*gc-bg))/(z-self.ad)
            r=lu_solve(lu,br/self.alpha+self.u*self.signed(z*cc-bc))
            h=hc+weights[None,:,None]*r.reshape(2,1,n)
            # Recover filters from separated constant and rate-dependent drives.
            dr=np.array([Wr['ee']@r[:n],self.q*(Wr['ei']@r[n:]),Wr['ie']@r[:n],self.q*(Wr['ii']@r[n:])])
            g=gc+self.B*dr/(z-self.ar);c=(bc+(1-self.ad)*(z*g-bg))/(z-self.ad)
            return np.r_[r,g.ravel(),c.ravel(),h.ravel()]
        return LinearOperator((self.size,self.size),matvec=solve,dtype=complex)


def spectrum(branch,q,tau=20.611550480127335,frequencies=(0,10,40,100),k=6):
    s=System(tau=tau);a=np.load(OUT/'fixed_points.npz')
    keys=[key for key in a.files if key.startswith(branch+'_')];key=min(keys,key=lambda x:abs(float(x.split('_')[1])-q))
    r,err,ok=s.solve(q,a[key]);assert ok,(q,err)
    t=Tangent(s,r,q);rng=np.random.default_rng(46);b=rng.normal(size=t.size);z=1.001+.003j;x=t.inverse(z)@b
    inverse_error=float(np.linalg.norm(z*x-t.action(x)-b)/np.linalg.norm(b));assert inverse_error<1e-8,inverse_error
    rows=[]
    for freq in frequencies:
        z0=np.exp((.00003+2j*np.pi*freq/1000)*s.dt);inv=t.inverse(z0)
        ev,V=eigs(inv,k=k,which='LM',tol=2e-9,maxiter=700,ncv=max(2*k+4,24),v0=b.astype(complex))
        for j,mu in enumerate(ev):
            zz=z0-1/mu;lam=np.log(zz)/s.dt*1000;v=V[:,j]
            residual=float(np.linalg.norm(t.action(v)-zz*v)/np.linalg.norm(v))
            if all(abs(complex(x['real_per_s'],x['imag_per_s'])-lam)>1e-3 for x in rows):
                rows.append({'real_per_s':float(lam.real),'imag_per_s':float(lam.imag),'frequency_hz':float(abs(lam.imag)/2/np.pi),'multiplier_abs':float(abs(zz)),'full_map_residual':residual,'seed_frequency_hz':freq})
        print('spectrum shift',branch,q,tau,freq,flush=True)
    rows.sort(key=lambda x:x['real_per_s'],reverse=True)
    result={'branch':branch,'q':q,'tau_gaba_ms':tau,'dimension':t.size,'fixed_point_residual':err,'inverse_identity_relative_error':inverse_error,'roots':rows,
            'method':'Shift-invert Arnoldi of exact native-step spatial rate tangent with all 358 delay bins; external filter perturbations decouple and decay.',
            'coverage':'Roots nearest specified shifts, not a certified count of all unstable roots. Each root checked in full tangent map.'}
    folder=OUT/'spectra';folder.mkdir(exist_ok=True);write(folder/f'{branch}_q{q:g}_tau{tau:g}.json',result)
    print('leading sampled',branch,q,rows[:4],flush=True);return result


def unstable_count(branch,q,tau=20.611550480127335):
    """Argument principle on exact map. det(M/M0) -> 1 at infinity; poles all inside unit disk.

    Therefore minus the unit-circle winding is the number of unstable map roots.
    Conjugacy permits the upper semicircle; repeat the angle grid to check phase resolution.
    """
    s=System(tau=tau);a=np.load(OUT/'fixed_points.npz');key=min([k for k in a.files if k.startswith(branch+'_')],key=lambda k:abs(float(k.split('_')[1])-q))
    r,err,ok=s.solve(q,a[key]);assert ok;pre=s.blocks(r,q);checks=[]
    for length in [800,1600]:
        theta=np.r_[0,np.geomspace(1e-7,np.pi,length)];angles=[]
        for th in theta:
            z=np.exp(1j*th);M=s.characteristic(z,r,q,pre=pre);sign,_=np.linalg.slogdet(M)
            reference=(z-1)*s.tr/s.dt+1
            angles.append(np.angle(sign*np.exp(-1j*np.angle(reference).sum())))
        phase=np.unwrap(angles);count=-(phase[-1]-phase[0])/np.pi
        checks.append({'grid_size':length,'unstable_count_raw':float(count),'max_phase_step_radians':float(np.max(abs(np.diff(phase))))})
    assert abs(checks[-1]['unstable_count_raw']-round(checks[-1]['unstable_count_raw']))<1e-6
    assert abs(checks[-1]['unstable_count_raw']-checks[0]['unstable_count_raw'])<1e-6
    assert checks[-1]['max_phase_step_radians']<np.pi/2,checks
    result={'branch':branch,'q':q,'tau_gaba_ms':tau,'unstable_roots':round(checks[-1]['unstable_count_raw']),'checks':checks,
        'method':'Numerical argument principle on det(M(z))/product((z-1)*tau_rate/dt+1); all poles inside unit circle, ratio tends to one at infinity. Upper semicircle doubled by real conjugacy. Grid-doubling convergence is numerical, not interval-arithmetic certification.'}
    folder=OUT/'spectra';folder.mkdir(exist_ok=True);write(folder/f'count_{branch}_q{q:g}_tau{tau:g}.json',result);print(result,flush=True)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['high','low'],default='low');p.add_argument('--q',type=float,default=.72);p.add_argument('--tau',type=float,default=20.611550480127335);p.add_argument('--frequencies',type=float,nargs='+',default=[0,10,40,100]);p.add_argument('--count',action='store_true');a=p.parse_args()
    if a.count:unstable_count(a.branch,a.q,a.tau)
    else:spectrum(a.branch,a.q,a.tau,a.frequencies)
