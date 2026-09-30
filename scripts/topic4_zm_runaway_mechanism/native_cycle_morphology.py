"""Compare solved spatial cycles without assuming they lie on one branch.

One overall phase shift is allowed in the comparison. Descriptive core peak
offsets do not establish propagation direction, speed or bifurcation type.
"""
from common import *
from scipy.signal import resample
from scipy.optimize import minimize_scalar


def main():
    s=model();base=OUT/'periodic'
    paths=[base/'native_Z219_stable_side_G8505_M65536/point0000.npz',
           base/'native_long_period_bridge_G8505_M65536/point0000.npz',
           base/'native_Z78_refinement_G8193_M65536/point0000.npz',
           base/'native_Z78_second_G8505_M65536/point0000.npz']
    extra=base/'native_unstable_period_extension_G8505_M65536/point0000.npz'
    if extra.exists():paths.append(extra)
    N=8505;rows=[];reference=None
    region=np.asarray(s.geo['group_region'])
    for p in paths:
        z=np.load(p);assert float(z['residual'])<2e-8
        r=resample(z['r'],N,axis=0);T=float(z['T'])
        f=np.fft.rfft(r[:,s.E],axis=0)/N
        if reference is None:reference=f.copy()
        factor=np.r_[1.,np.full(len(f)-1,2.)];freq=np.arange(len(f))
        cross=np.sum(reference*np.conj(f)*s.mean_weights,axis=1)
        nx=float(np.sum(factor[:,None]*abs(reference)**2*s.mean_weights))
        ny=float(np.sum(factor[:,None]*abs(f)**2*s.mean_weights))
        def distance(shift):return nx+ny-2*float(np.sum(factor*cross*np.exp(2j*np.pi*freq*shift)).real)
        corr=np.fft.irfft(cross,n=8*N);start=np.argmax(corr)/(8*N)
        fit=minimize_scalar(distance,bounds=(start-2/N,start+2/N),method='bounded',options={'xatol':1e-14})
        core=[]
        for j in range(3):
            mask=s.E&(region==j);weights=s.sizes[mask]/s.sizes[mask].sum()
            rates=r[:,mask]@weights*1000;k=int(rates.argmax())
            denom=rates[(k-1)%N]-2*rates[k]+rates[(k+1)%N]
            delta=.5*(rates[(k-1)%N]-rates[(k+1)%N])/denom if denom else 0.
            core.append(dict(region=['A','B','surround'][j],peak_phase=float(((k+delta)/N)%1),
                peak_hz=float(rates.max()),mean_hz=float(rates.mean())))
        lag=(core[1]['peak_phase']-core[0]['peak_phase'])%1
        if lag>.5:lag-=1
        rows.append(dict(source=str(p),D=float(z['D']),T_ms=T,
            phase_aligned_E_rate_relative_L2_from_first=float(np.sqrt(max(0,fit.fun)/nx)),
            global_phase_shift=fit.x,core_B_minus_A_peak_offset_ms=lag*T,regions=core))
    q=dict(status='DESCRIPTIVE_COMPLETE',rows=rows,
        observable='Neuron-weighted spatial E rate L2 after one global phase shift; regional primary peak offset on each solved periodic waveform',
        scope='Different Z fields; no same-parameter coexistence or branch connectivity inferred from similarity. Core peak offset is not a propagation-speed estimate.')
    write(OUT/'periodic/native_cycle_morphology.json',q)
    log('NATIVE CYCLE MORPHOLOGY',q)


if __name__=='__main__':main()
