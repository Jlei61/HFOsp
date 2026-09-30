"""Two real states per rate population, with native local transfer calibration.

Schaffer et al. (2013) motivates the complex relaxation. The low-rate damping
regularization and original colored-input calibration are project hypotheses,
not an exact derivation of the present LIF network.
"""
from common import *
from scipy.interpolate import PchipInterpolator


class RateTransfer:
    def __init__(self, folder=None):
        folder=folder or OUT/'local_response/native_colored_grid_v1'
        a=np.load(Path(folder)/'table.npz')
        self.x=a['currents_mv'];self.theta=a['thresholds_e']
        rate=a['rate_hz'].reshape(9,-1)/1000
        cv=a['cv'].reshape(9,-1)
        # Interval sampling near zero rate is right-censored. Treat that region
        # as nonresonant pending the independent transient check.
        cv=np.where((rate<.005)|(~np.isfinite(cv)),1.,cv)
        cv=np.clip(cv,0.,1.5)
        self.f=PchipInterpolator(self.x,np.maximum.accumulate(rate,axis=1),axis=1)
        self.c=PchipInterpolator(self.x,cv,axis=1)

    def row(self, u, theta=18., population=0):
        u=np.clip(np.asarray(u),self.x[0],self.x[-1])
        if population==1:return self.f(u)[8],np.clip(self.c(u)[8],0.,1.5)
        hi=int(np.clip(np.searchsorted(self.theta,theta),1,7));lo=hi-1
        w=(theta-self.theta[lo])/(self.theta[hi]-self.theta[lo])
        return (1-w)*self.f(u)[lo]+w*self.f(u)[hi],np.clip((1-w)*self.c(u)[lo]+w*self.c(u)[hi],0.,1.5)


def coefficients(f,cv,tau,damping_scale=1.,omega_scale=1.,low_damping=1.):
    # Rates in /ms. The constant in the oscillatory damping will be qualified
    # against native transients, rather than inferred from network labels.
    alpha=damping_scale*2*np.pi**2*f*cv**2+low_damping*np.exp(-f*tau)/tau
    omega=omega_scale*2*np.pi*f
    return alpha,omega


def positive_rate(x,epsilon=.00001):
    # C-infinity nonnegative readout; epsilon=.01 Hz, independently checked.
    return .5*(x+np.hypot(x,epsilon))


def trajectory(u,theta,population,dt=.1,pars=(1.,1.,1.),initial=None):
    t=RateTransfer();f,cv=t.row(u,theta,population);tau=20. if population==0 else 10.
    alpha,omega=coefficients(f,cv,tau,*pars)
    z=complex(f[0]) if initial is None else complex(initial)
    out=np.empty((len(u),2))
    for k in range(len(u)):
        z=f[k]+(z-f[k])*np.exp((-alpha[k]-1j*omega[k])*dt)
        out[k]=z.real,z.imag
    return out
