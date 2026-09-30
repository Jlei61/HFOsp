"""v3 Z/M spatial rate model: same 1 mm / 935-group network as v2, new colored-noise MC transfer.

Static (equilibrium) layer. Units: ms, mV, spikes/ms/cell. J_EE_core = 1 (physical graph weights).
Equilibrium equations per group g (E mask E_g, threshold theta_g, tau_m, refractory per population):
  mu_g  = tau_m [area_A (A r)_g + area_A J_ext nu_g] - Z_g tau_m area_G (B r)_g - m_g,   m_g = 0.5 E_g r_g  (eta_M M, mV)
  vE_g  = tau_m area_A^2 [(QA r)_g + J_ext^2 nu_g]
  vI_g  = Z_g^2 tau_m area_G^2 (QB r)_g
  r_g   = phi_pop(mu_g, vE_g, vI_g; theta_g)        (TransferSpline of the CRN Monte-Carlo table)
A,B = first-moment delayed operators summed over delays; QA,QB second-moment operators
(sum of individual squared physical weights). Z acts on E targets only (I groups Z=1).
D = 1 - <Z_E> weighted by original E cell counts. Prescribed path: power transform of the native
9.420 s per-cell Z field (as v2). Any per-cell or per-group Z field may also be set directly.
"""
from common_v3 import *
from transfer_spline import TransferSpline, device_block, CUDA_DEVICE
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve
from functools import lru_cache

THRESHOLD_Z=95.19851312666987   # mV, native per-cell rule 1[I_I < THRESHOLD_Z]
TAU_Z=5000.;TAU_M=1000.;ETA_M=0.0005
TABLE=DEST/'transfer_table'

class SpatialRateV3:
    def __init__(self,grid=20,table=TABLE,quiet=False):
        self.folder=OPERATORS/f'g{grid}';self.prep=read(self.folder/'prepared.json');self.p=self.prep['params'];self.grid=grid
        g=self.geo=dict(np.load(self.folder/'geometry.npz'));p=self.p
        self.P=len(g['group_size']);self.E=g['population']==0;self.sizes=g['group_size'].astype(float)
        self.tm=np.where(self.E,p['tau_m_E'],p['tau_m_I']);self.ref=np.where(self.E,p['tau_ref_E'],p['tau_ref_I'])
        self.theta=g['threshold_mv'].astype(float);self.nu=self.prep['nu_ext_per_ms']
        self.jext=np.where(self.E,p['J_ext_E'],p['J_ext_I']);self.rise=np.array([p['tau_r_AMPA'],p['tau_r_GABA']])
        self.decay=np.array([p['tau_d_AMPA'],p['tau_d_GABA']]);self.tau=self.rise+self.decay
        self.area=.1/(self.rise*(1-np.exp(-.1/self.rise)))
        self.delays=(np.arange(self.prep['max_delay_steps'])+1)*.1
        self.raw=[]
        for name in ('mean_ampa','mean_gaba','variance_ampa','variance_gaba'):
            a=sparse.load_npz(self.folder/f'{name}.npz').tocoo();row=a.row;col=a.col%self.P;di=a.col//self.P
            keys,inv=np.unique(row.astype(np.int64)*self.P+col,return_inverse=True)
            delay_matrix=sparse.coo_matrix((a.data,(inv,di)),shape=(len(keys),self.prep['max_delay_steps'])).tocsr()
            self.raw.append((keys//self.P,keys%self.P,delay_matrix))
        self.mean_weights=self.sizes[self.E]/self.sizes[self.E].sum()
        self.members=g['cell_group'][:32000]
        src=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints/t9420ms.npz'
        with np.load(src) as z:self.logz=np.log(z['slow__z'][:32000].astype(float))
        self.spline={pop:TransferSpline(Path(table)/f'table_{pop}.npz') for pop in 'EI'}
        self.pop_index=np.where(self.E,0,1)
        self.Z=np.ones(self.P);self.D=0.;self.z_source='baseline Z=1';self.z_override=None
        # external drive per group (may be overridden by time-varying drives in the integrator)
        self.set_external(np.full(self.P,self.nu))
        if not quiet:log('SpatialRateV3 ready: P=%d E groups=%d'%(self.P,self.E.sum()))

    # ---------- external drive ----------
    def set_external(self,nu_per_group):
        self.nu_group=np.asarray(nu_per_group,float)
        self.private_mu=self.tm*self.area[0]*self.jext*self.nu_group
        self.private_ve=self.tm*self.area[0]**2*self.jext**2*self.nu_group

    # ---------- delayed operators ----------
    @lru_cache(maxsize=8)
    def matrices(self,lam=0.):
        phase=np.exp(-lam*self.delays);out=[]
        for r,c,delay in self.raw:
            out.append(sparse.coo_matrix((delay@phase,(r,c)),shape=(self.P,self.P)).tocsr())
        return out

    # ---------- Z field ----------
    def project(self,x):
        return np.bincount(self.members,weights=x,minlength=self.P)/np.maximum(self.sizes,1)
    def set_D(self,D):
        if getattr(self,'z_override',None) is not None:   # conditional system on an explicit Z field (stage B): D is a label only
            self.set_Z(self.z_override,source=getattr(self,'z_override_source','explicit override'));return
        if not 0<=D<=1:raise ValueError('D outside physical domain')
        self.D=float(D);self.Z=np.ones(self.P)
        if D==1:self.Z[self.E]=0.
        elif D>0:
            a=brentq(lambda a:np.exp(a*self.logz).mean()-(1-D),0,1e6,xtol=1e-13)
            self.Z[self.E]=self.project(np.exp(a*self.logz))[self.E]
        assert abs(self.Z[self.E]@self.mean_weights-(1-D))<1e-11
        self.z_source='prescribed native 9.420 s power path'
    def set_Z(self,Z,source='explicit'):
        Z=np.asarray(Z,float);assert Z.shape==(self.P,) and Z.min()>=0 and Z.max()<=1
        self.Z=Z.copy();self.Z[~self.E]=1.;self.D=float(1-self.Z[self.E]@self.mean_weights);self.z_source=source
    def set_Z_cells(self,zcell,source='per-cell field'):
        Z=np.ones(self.P);Z[self.E]=self.project(np.asarray(zcell,float))[self.E];self.set_Z(Z,source)

    # ---------- transfer ----------
    def phi(self,mu,ve,vi,order=1):
        out={k:np.zeros(self.P) for k in ('rate','d_mu','d_ve','d_vi')}
        if order>=2:out['hessian']=np.zeros((3,3,self.P))
        for k,pop in enumerate('EI'):
            m=self.pop_index==k;ev=self.spline[pop].evaluate(mu[m],ve[m],vi[m],self.theta[m],order=order)
            for key in out:out[key][...,m]=ev[key]
        return out

    # ---------- equilibrium ----------
    def moments(self,r,m=None):
        a,b,qa,qb=self.matrices()
        m=.5*self.E*r if m is None else m
        mu=self.tm*self.area[0]*(a@r)-self.Z*self.tm*self.area[1]*(b@r)-m+self.private_mu
        ve=self.tm*self.area[0]**2*(qa@r)+self.private_ve
        vi=self.Z**2*self.tm*self.area[1]**2*(qb@r)
        return mu,ve,vi
    def residual(self,r):
        return self.phi(*self.moments(r))['rate']-r
    def jacobian(self,r):
        a,b,qa,qb=self.matrices();g=self.phi(*self.moments(r))
        K=sparse.diags(g['d_mu']*self.tm)@(self.area[0]*a-sparse.diags(self.Z*self.area[1])@b)
        K=K+sparse.diags(g['d_ve']*self.tm*self.area[0]**2)@qa+sparse.diags(g['d_vi']*self.Z**2*self.tm*self.area[1]**2)@qb
        return (K-sparse.diags(1+.5*g['d_mu']*self.E)).tocsc()
    def solve(self,r=None,tol=1e-11,maxiter=80,verbose=False):
        if r is None:
            r=self.phi(*self.moments(np.zeros(self.P)))['rate']
        r=np.maximum(np.asarray(r,float).copy(),0.);trace=[]
        for it in range(maxiter):
            f=self.residual(r);err=float(abs(f).max());trace.append(err)
            if verbose:log('newton',it,err)
            if err<tol:return r,True,trace
            step=spsolve(self.jacobian(r),-f);alpha=1.
            for back in range(30):
                trial=r+alpha*step
                if trial.min()>=-tol and np.all(trial<1/self.ref):
                    trial=np.maximum(trial,0.)
                    if abs(self.residual(trial)).max()<err:r=trial;break
                alpha*=.5
            else:return r,False,trace
        return r,False,trace
    def global_rate(self,r):return float(r[self.E]@self.mean_weights*1000)
    def regional_rates(self,r):
        region=self.geo['group_region']
        return [float(np.average(r[self.E&(region==i)],weights=self.sizes[self.E&(region==i)])*1000) for i in range(3)]
    def cell_field(self,r):
        """E rate per 20x20 cell (Hz), cell-count weighted."""
        cell=self.geo['group_cell'];f=np.zeros(self.grid*self.grid);n=np.zeros(self.grid*self.grid)
        np.add.at(f,cell[self.E],r[self.E]*self.sizes[self.E]);np.add.at(n,cell[self.E],self.sizes[self.E]);return f/np.maximum(n,1)*1000
    def identity(self):
        return dict(operators=str(self.folder),graph_identity=self.prep['graph_identity'],tables={p:sha256(TABLE/f'table_{p}.npz') for p in 'EI'},
                    z_path_source=str(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints/t9420ms.npz'))

if __name__=='__main__':
    s=SpatialRateV3()
    for D in [0.,.1,.2]:
        s.set_D(D);r,ok,tr=s.solve();log('D',D,'ok',ok,'iters',len(tr),'global',s.global_rate(r),'regional',s.regional_rates(r))
    # Jacobian check by finite differences on random directions
    s.set_D(.1);r,ok,_=s.solve();J=s.jacobian(r);rng=np.random.default_rng(0)
    for k in range(3):
        v=rng.standard_normal(s.P)*np.maximum(r,1e-4)*.01;h=1e-4
        fd=(s.residual(r+h*v)-s.residual(r-h*v))/(2*h);print('jacobian FD rel err',np.linalg.norm(J@v-fd)/np.linalg.norm(fd))
