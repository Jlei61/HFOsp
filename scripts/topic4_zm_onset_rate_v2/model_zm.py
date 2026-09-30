"""Z/M extension of the frozen shared spatial rate DDE, J_EE_core=1.

Time: ms; rates: spikes/ms/cell; m: eta_M M in mV. The nine shared
states are unchanged; state 9 is inhibitory resource Z. Conditional runs
hold Z, autonomous runs use a Gaussian expectation of the native threshold
rule. This last expectation is an explicit, unvalidated population closure.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT/'results/topic4_sef_hfo/fig5_zm_rate_synchronized_20260917'
OLD = ROOT/'results/topic4_sef_hfo/fig5_zm_onset_bifurcation_20260917'
sys.path.insert(0, str(HERE/'frozen_shared'))
import common
common.ROOT = ROOT
common.OUT = DEST/'frozen_shared'
from common import *
import rate_field
rate_field.RATE_OUT = DEST/'frozen_shared'
from rate_field import RateField, RateIntegrator
from response import characteristic as brunel_characteristic
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.sparse.linalg import spsolve

THRESHOLD = 95.19851312666987
TAU_Z = 5000.


class ZMSpatialRate(RateField):
    def __init__(self, grid=20):
        super().__init__(grid)
        self.response_method = 'calibrated_full'
        self.response_fit = read(DEST/'frozen_shared/local_response_fit/result.json')['rows']
        self.variance_fit = read(DEST/'frozen_shared/local_response_fit/variance_result.json')['rows']
        self.sizes = self.geo['group_size']
        self.mean_weights = self.sizes[self.E]/self.sizes[self.E].sum()
        self.members = self.geo['cell_group'][:32000]
        src = ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints/t9420ms.npz'
        with np.load(src) as z:
            self.logz = np.log(z['slow__z'][:32000].astype(float))
        self.D = 0.

    def project(self, x):
        return np.bincount(self.members, weights=x, minlength=self.P)/np.maximum(self.sizes, 1)

    def set_D(self, D):
        if not 0 <= D <= 1:
            raise ValueError('D outside physical domain')
        self.D = float(D)
        self.Z = np.ones(self.P)
        if D == 1:
            self.Z[self.E] = 0.
        else:
            a = 0. if D == 0 else brentq(lambda a: np.exp(a*self.logz).mean()-(1-D), 0, 1e6, xtol=1e-13)
            self.Z[self.E] = self.project(np.exp(a*self.logz))[self.E]
        assert abs(self.Z[self.E]@self.mean_weights-(1-D)) < 1e-11

    def moments(self, r, J=1.):
        return super().moments(r, 1.)

    def residual(self, r, D):
        self.set_D(D)
        return self.phi(*self.moments(r))-r

    def jacobian(self, r, D):
        self.set_D(D)
        return super().jacobian(r, 1.)

    def solve_D(self, D, r=None, tol=1e-11):
        self.set_D(D)
        return super().solve(D, r, tol)

    def global_rate(self, r):
        return float(r[self.E]@self.mean_weights*1000)

    def state(self, r=None):
        if r is None:
            y = np.zeros((10, self.P))
        else:
            y = np.vstack([super().equilibrium_state(r, 1.), np.zeros(self.P)])
        y[9] = self.Z
        return y

    def rhs(self, y, arrivals, dynamic_z=False):
        xf, xs, qa, ia, qg, ig, va, vg, m, z = y
        a, b, aa, bb = arrivals
        r = self.output(y)
        target = self.phi(ia-z*ig-m+self.private_mu, va+self.private_ve, z*z*vg)
        # vg is the *unscaled* inhibitory diffusion variance. Its associated
        # instantaneous filtered-current variance is tm*vg/(2*(rise+decay)).
        sd = np.sqrt(np.maximum(self.tm*vg/(2*self.tau[1]), 1e-20))
        z_inf = ndtr((THRESHOLD-ig)/sd)
        return np.array([(target-xf)/self.tf, (target-xs)/self.ts,
            (self.tm*self.area[0]*a-qa)/self.rise[0], (qa-ia)/self.decay[0],
            (self.tm*self.area[1]*b-qg)/self.rise[1], (qg-ig)/self.decay[1],
            (self.tm*self.area[0]**2*aa-va)/(self.tau[0]/2),
            (self.tm*self.area[1]**2*bb-vg)/(self.tau[1]/2),
            (.5*self.E*r-m)/1000,
            self.E*(z_inf-z)/TAU_Z if dynamic_z else np.zeros(self.P)])

    def characteristic(self, r, D, lam, method='rate_dde'):
        self.set_D(D)
        if method == 'rate_dde':
            return super().characteristic(r, 1., lam)
        return brunel_characteristic(self, r, 1., lam, method)

    def eigenstate(self, r, D, lam, v):
        self.set_D(D)
        return np.vstack([super().eigenstate(r, 1., lam, v), np.zeros(self.P)])


def cuda_code(s):
    """Extend the snapshotted shared integrator without changing its filters."""
    code = rate_field.cuda_code(s)
    replacements = {
        'double mu=y[3*P+g]-y[5*P+g]-y[8*P+g]+pm;':
        'double z=y[9*P+g]; double mu=y[3*P+g]-z*y[5*P+g]-y[8*P+g]+pm;',
        'vi=y[7*P+g],var=': 'vi=z*z*y[7*P+g],var=',
        'out[8*P+g]=(.5*E*r-y[8*P+g])/1000.;':
        '''out[8*P+g]=(.5*E*r-y[8*P+g])/1000.;
 double sd=sqrt(fmax(tm*y[7*P+g]/(2*(1+dg)),1e-20));
 double zi=.5*erfc((y[5*P+g]-pars[13*P+g])/(sqrt(2.)*sd));
 out[9*P+g]=pars[12*P+g]*E*(zi-z)/pars[14*P+g];''',
        'if(i<9*P)': 'if(i<10*P)',
        'for(int j=0;j<9;j++)': 'for(int j=0;j<10;j++)'
    }
    for old, new in replacements.items():
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    return code


class ZMIntegrator(RateIntegrator):
    def __init__(self, s, dt=.1, initial=None, history=None, dynamic_z=False, device=0):
        initial = s.state() if initial is None else initial
        super().__init__(s, 1., dt, initial=initial, history=history, device=device)
        cp = self.cp
        self.pars = cp.vstack([self.pars, cp.full((1,s.P),float(dynamic_z)),
                              cp.full((1,s.P),THRESHOLD), cp.full((1,s.P),TAU_Z)])
        self.module = cp.RawModule(code=cuda_code(s), options=('--fmad=false',),
            name_expressions=['delayed','rhs','predictor','finish'])
        self.k = {n:self.module.get_function(n) for n in ['delayed','rhs','predictor','finish']}

    def step(self):
        n = (self.s.P+127)//128
        self.arrivals(self.tick)
        self.k['rhs']((n,),(128,),(self.y,self.arr,self.pars,self.f))
        self.k['predictor'](((10*self.s.P+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.arrivals(self.tick+1)
        self.k['rhs']((n,),(128,),(self.pred,self.arr,self.pars,self.f2))
        self.tick += 1
        self.k['finish']((n,),(128,),(self.y,self.f,self.f2,self.pars,self.history,self.dt,np.int32(self.tick),np.int32(self.depth)))
        return self.history[self.tick%self.depth]
