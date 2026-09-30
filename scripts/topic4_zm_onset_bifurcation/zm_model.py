"""Z/M conditional extension of a frozen spatial rate framework.

No change to graph, transfer, response, thresholds or core coupling. Time is
ms and rate is spikes/ms/cell. D only parameterizes a prescribed spatial Z
field. Dynamic M and frozen M have separate equations and characteristic
matrices.
"""
from pathlib import Path
import sys, importlib.util

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_rate'))
import common as frozen_common
# Relocate source paths without modifying the frozen mathematical source.
frozen_common.ROOT=HERE.parents[1]
frozen_common.OUT=frozen_common.ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917'
from common import *
from model import SpatialBrunel
import response as frozen_response
from stable_response import install
install(frozen_response)
from response import characteristic as parent_characteristic, susceptibility
from scipy.optimize import brentq

DEST=ROOT/'results/topic4_sef_hfo/fig5_zm_onset_bifurcation_20260917'
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'


class ZMRate(SpatialBrunel):
    def __init__(self,grid=20,quadrature=48,mode='dynamic_M',m_current=0.):
        super().__init__(grid,quadrature)
        if mode not in ('dynamic_M','frozen_M'):raise ValueError(mode)
        self.mode=mode;self.m_current=m_current
        src=SOURCE/'replay/runs/eta0.0005_s9108401/checkpoints/t9420ms.npz'
        with np.load(src) as f:
            self.z_native=f['slow__z'][:32000].astype(float)
            self.m_native=f['slow__m'][:32000].astype(float)
        self.members=self.geo['cell_group'][:32000]
        self.sizes=self.geo['group_size'];self.mean_weights=self.sizes[self.E]/32000
        self.logz=np.log(self.z_native)
        self.m_shape=self.project(self.m_native/self.m_native.mean())
        self.dZ=np.zeros(self.P);self.D=0.
        self.static_A,self.static_B,self.static_QA,self.static_QB=self.matrices(1.)

    def project(self,values):
        return np.bincount(self.members,weights=values,minlength=self.P)/np.maximum(self.sizes,1)

    def set_D(self,D):
        if not 0<=D<=1:raise ValueError('Physical resource domain is D in [0,1]')
        self.D=float(D);self.Z=np.ones(self.P)
        if D==1:
            self.Z[self.E]=0.;self.dZ.fill(np.nan);return
        exponent=0. if D==0 else brentq(lambda a:np.exp(a*self.logz).mean()-(1-D),0,1e6,xtol=1e-13)
        z=np.exp(exponent*self.logz);derivative=-z*self.logz/np.mean(z*self.logz)
        self.Z[self.E]=self.project(z)[self.E]
        self.dZ=self.project(derivative);self.dZ[~self.E]=0
        assert abs(self.Z[self.E]@self.mean_weights-(1-D))<1e-11

    def moments(self,r,J=1.):
        mu,ve,vi=super().moments(r,1.)
        if self.mode=='frozen_M':mu=mu+.5*self.E*r-self.m_current*self.m_shape
        return mu,ve,vi

    def residual(self,r,D):
        self.set_D(D);return self.phi(*self.moments(r))-r

    def jacobian(self,r,D):
        self.set_D(D)
        # The parent uses the overridden moments but always adds equilibrium M.
        J=super().jacobian(r,1.)
        if self.mode=='frozen_M':
            gm=self.gains(self.moments(r))[0];J+=sparse.diags(.5*self.E*gm)
        return J

    def parameter_derivative(self,r,D):
        self.set_D(D);gm,ge,gi=self.gains(self.moments(r))
        dmu=-self.tm*self.area[1]*self.dZ*(self.static_B@r)
        dvi=2*self.tm*self.area[1]**2*self.Z*self.dZ*(self.static_QB@r)
        return gm*dmu+gi*dvi

    def characteristic(self,r,D,lam,method='shifted_white'):
        self.set_D(D)
        C=parent_characteristic(self,r,1.,lam,method)
        if self.mode=='frozen_M':
            gm=susceptibility(self,r,1.,lam,method)[0]
            C-=sparse.diags(.5*self.E*gm/(1+1000*lam))
        return C

    def global_rate(self,r):return float(r[self.E]@self.mean_weights*1000)

    def solve_D(self,D,r=None,tol=1e-11):
        # Base solver dispatches residual/Jacobian to D methods.
        self.set_D(D)
        if r is None:r=self.phi(*self.moments(np.zeros(self.P)))
        return super().solve(D,r,tol)
