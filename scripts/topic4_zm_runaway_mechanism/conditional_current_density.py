"""Deterministic voltage-density / conditional-current moment response.

One voltage grid, with mass and first/second conditional moments of the four
AMPA/GABA current states. Each voltage component assumes Gaussian currents.
Threshold flux enters a refractory queue carrying those current moments;
currents continue evolving throughout refractoriness. No sampled neurons.

This is an exploratory local response, not the accepted spatial model.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def normal_integrals(lo, hi):
    """Unnormalised moments 0..3 of a standard normal on [lo,hi]."""
    if lo >= 0:
        p = .5*(math.erfc(lo/math.sqrt(2))-math.erfc(hi/math.sqrt(2)))
    elif hi <= 0:
        p = .5*(math.erfc(-hi/math.sqrt(2))-math.erfc(-lo/math.sqrt(2)))
    else:
        p = 1-.5*math.erfc(-lo/math.sqrt(2))-.5*math.erfc(hi/math.sqrt(2))
    a = math.exp(-.5*lo*lo)/math.sqrt(2*math.pi) if abs(lo)<38 else 0.
    b = math.exp(-.5*hi*hi)/math.sqrt(2*math.pi) if abs(hi)<38 else 0.
    return p, a-b, p+lo*a-hi*b, (lo*lo+2)*a-(hi*hi+2)*b


@njit(cache=True)
def advance(raw, pars, ve, vi):
    """Exact discrete synaptic evolution of unnormalised raw moments."""
    left = np.empty((5,5)); out = np.empty((5,5))
    for j in range(5):
        left[0,j] = raw[0,j]
        left[1,j] = pars[6]*raw[1,j]
        left[2,j] = pars[7]*raw[1,j]+pars[8]*raw[2,j]
        left[3,j] = pars[17]*raw[3,j]
        left[4,j] = pars[9]*raw[3,j]+pars[10]*raw[4,j]
    for i in range(5):
        out[i,0] = left[i,0]
        out[i,1] = pars[6]*left[i,1]
        out[i,2] = pars[7]*left[i,1]+pars[8]*left[i,2]
        out[i,3] = pars[17]*left[i,3]
        out[i,4] = pars[9]*left[i,3]+pars[10]*left[i,4]
    mass = raw[0,0]
    out[1,1] += mass*ve*pars[11]**2
    out[1,2] += mass*ve*pars[11]*pars[12]
    out[2,1] += mass*ve*pars[11]*pars[12]
    out[2,2] += mass*ve*(pars[12]**2+pars[13]**2)
    out[3,3] += mass*vi*pars[14]**2
    out[3,4] += mass*vi*pars[14]*pars[15]
    out[4,3] += mass*vi*pars[14]*pars[15]
    out[4,4] += mass*vi*(pars[15]**2+pars[16]**2)
    return out


@njit(cache=True)
def weighted_raw(raw, mean, direction, moments, intercept, slope):
    """Gaussian raw moments after a linear-in-standardised-voltage weight."""
    m0,m1,m2,m3 = moments
    w0 = intercept*m0+slope*m1
    w1 = intercept*m1+slope*m2
    w2 = intercept*(m2-m0)+slope*(m3-m1)
    out = raw*w0
    mass=raw[0,0]
    for i in range(4):
        out[0,i+1] += mass*direction[i]*w1
        out[i+1,0] = out[0,i+1]
        for j in range(4):
            out[i+1,j+1] += mass*((mean[i]*direction[j]+direction[i]*mean[j])*w1+
                                          direction[i]*direction[j]*w2)
    return out


@njit(cache=True)
def deposit_point(raw, value, grid, output):
    if value <= grid[0]:
        output[0] += raw
    elif value >= grid[-1]:
        output[-1] += raw
    else:
        k = np.searchsorted(grid,value)-1
        frac=(value-grid[k])/(grid[k+1]-grid[k])
        output[k] += raw*(1-frac); output[k+1] += raw*frac


@njit(cache=True)
def membrane_remap(raw, voltage, mu, pars, grid, output, spikes):
    mass=raw[0,0]
    if mass < 1e-24:
        return mass, 0., 0.
    mean=raw[0,1:]/mass
    cov=raw[1:,1:]/mass-np.outer(mean,mean)
    a=pars[18]; decay=1-a; threshold=pars[1]
    center=a*voltage+decay*(mu+mean[1]-mean[3])
    variance=decay**2*(cov[1,1]+cov[3,3]-2*cov[1,3])
    scale=decay**2*max(abs(cov[1,1])+abs(cov[3,3]),1.)
    if variance < -1e-8*scale:
        raise ValueError('Nonpositive conditional current covariance')
    if variance <= 1e-24:
        if center >= threshold:
            spikes += raw
        else:
            deposit_point(raw,center,grid,output)
        return 0., mass if center < grid[0] else 0., min(variance/scale,0.)
    sigma=math.sqrt(variance)
    direction=decay*(cov[:,1]-cov[:,3])/sigma
    threshold_z=(threshold-center)/sigma
    if threshold_z < 10:
        m=normal_integrals(max(threshold_z,-40.),40.)
        spikes += weighted_raw(raw,mean,direction,m,1.,0.)
    # Grid below threshold is remapped with positive linear hats. This
    # preserves the first voltage moment away from the two edge intervals.
    low_z=(grid[0]-center)/sigma
    low_mass=0.
    if low_z > -10:
        m=normal_integrals(-40.,min(low_z,40.))
        output[0] += weighted_raw(raw,mean,direction,m,1.,0.)
        low_mass=mass*m[0]
    k0=max(0,np.searchsorted(grid,center-10*sigma)-1)
    k1=min(len(grid)-1,np.searchsorted(grid,center+10*sigma))
    for k in range(k0,k1):
        lo=(grid[k]-center)/sigma; hi=(grid[k+1]-center)/sigma
        m=normal_integrals(lo,hi); width=grid[k+1]-grid[k]
        left=weighted_raw(raw,mean,direction,m,(grid[k+1]-center)/width,-sigma/width)
        total=weighted_raw(raw,mean,direction,m,1.,0.)
        output[k] += left; output[k+1] += total-left
    top_z=(grid[-1]-center)/sigma
    if top_z<10 and threshold_z>-10:
        m=normal_integrals(max(top_z,-40.),min(threshold_z,40.))
        output[-1] += weighted_raw(raw,mean,direction,m,1.,0.)
    return 0., low_mass, min(variance/scale,0.)


def voltage_grid(theta, reset, n, minimum=-500.):
    """Fixed sinh grid, refined near threshold, with reset exactly a node."""
    u=np.linspace(np.arcsinh(.01/5),np.arcsinh((theta-minimum)/5),n)
    grid=np.sort(theta-5*np.sinh(u))
    return np.sort(np.unique(np.r_[grid,reset]))


@njit(cache=True)
def simulate(pars,wave,grid,dt,period,burn_steps,record_steps,bins,
             baseline_mu=np.nan,baseline_ve=0.,baseline_vi=0.,phase_offset_ms=0.):
    """Return own rate counts, occupancy, voltage, and conservation evidence."""
    n=len(grid); nref=int(pars[19]); w=wave.shape[1]
    free=np.zeros((n,5,5)); free[np.searchsorted(grid,pars[21]),0,0]=1.
    refractory=np.zeros((nref,5,5)); head=0
    expected=np.zeros((5,5));expected[0,0]=1.
    rate_counts=np.zeros(bins); exposure=np.zeros(bins)
    voltage_sum=np.zeros(bins); ref_sum=np.zeros(bins)
    max_mass_error=0.; max_current_error=0.; removed=0.; low_flow=0.; worst_cov=0.
    step_rates=np.zeros(record_steps)
    for t in range(-burn_steps,record_steps):
        phase=(((t+1)*dt+phase_offset_ms)/period)%1.; pos=phase*w
        lo=int(pos)%w; hi=(lo+1)%w; f=pos-math.floor(pos)
        mu=(1-f)*wave[0,lo]+f*wave[0,hi]
        ve=(1-f)*wave[1,lo]+f*wave[1,hi]
        vi=(1-f)*wave[2,lo]+f*wave[2,hi]
        if t<0 and not math.isnan(baseline_mu):
            mu=baseline_mu;ve=baseline_ve;vi=baseline_vi
        for j in range(n):
            if free[j,0,0]>=1e-24:
                free[j]=advance(free[j],pars,ve,vi)
        for j in range(nref):
            if refractory[j,0,0]>=1e-24:
                refractory[j]=advance(refractory[j],pars,ve,vi)
        free[np.searchsorted(grid,pars[21])] += refractory[head]
        refractory[head] = 0.
        new=np.zeros((n,5,5)); spikes=np.zeros((5,5))
        for j in range(n):
            cut,under,negative=membrane_remap(free[j],grid[j],mu,pars,grid,new,spikes)
            removed+=cut; low_flow+=under; worst_cov=min(worst_cov,negative)
        refractory[head]=spikes; head=(head+1)%nref; free=new
        expected=advance(expected,pars,ve,vi)
        if t>=0:
            b=min(int(phase*bins),bins-1)
            rate_counts[b]+=spikes[0,0]; exposure[b]+=dt
            vm=0.;refp=0.
            for j in range(n):vm+=free[j,0,0]*grid[j]
            for j in range(nref):refp+=refractory[j,0,0]
            voltage_sum[b]+=vm+refp*pars[21]; ref_sum[b]+=refp
            step_rates[t]=spikes[0,0]/dt*1000
        if t%100==0:
            total=np.zeros((5,5))
            for j in range(n):total+=free[j]
            for j in range(nref):total+=refractory[j]
            max_mass_error=max(max_mass_error,abs(total[0,0]-1.))
            for i in range(5):
                for j in range(5):
                    max_current_error=max(max_current_error,abs(total[i,j]-expected[i,j])/max(1.,abs(expected[i,j])))
    return (rate_counts/exposure*1000, voltage_sum/(exposure/dt), ref_sum/(exposure/dt),
            step_rates, free, refractory,
            np.array([max_mass_error,max_current_error,removed,low_flow,worst_cov]))
