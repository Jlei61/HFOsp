"""Phase-aligned rate-profile distances; a diagnostic, not a global connection proof."""
from rate_periodic import *
from scipy.optimize import minimize_scalar, brentq


def distances(x,c,weights):
    """x[fast,slow,group], c[fast,group]; one global fast phase per slow slice."""
    N=len(x);fx=np.fft.fft(x,axis=0);fc=np.fft.fft(c,axis=0)
    cross=np.sum(fx*fc.conj()[:,None,:]*weights[None,None,:],axis=-1)
    corr=np.fft.ifft(cross,axis=0).real/N
    peak=np.argmax(corr,axis=0);cols=np.arange(x.shape[1]);cm=corr[(peak-1)%N,cols];c0=corr[peak,cols];cp=corr[(peak+1)%N,cols]
    denom=cm-2*c0+cp;delta=np.divide(.5*(cm-cp),denom,out=np.zeros_like(c0),where=abs(denom)>1e-20)
    phases=(peak+delta)/N;dist=[]
    # Compute the final residual directly after a Fourier phase shift. The
    # parabolic correlation estimate can exceed the true maximum and is not
    # an admissible nonnegative-distance estimator near a close approach.
    frequencies=np.fft.fftfreq(N)*N
    for j,shift in enumerate(phases):
        # Refine the common phase using the exact Fourier correlation, so a
        # near-saddle distance does not bottom out at parabolic peak error.
        fit=minimize_scalar(lambda v:-np.real(np.sum(cross[:,j]*np.exp(2j*np.pi*frequencies*v)))/N**2,
            bounds=((peak[j]-1)/N,(peak[j]+1)/N),method='bounded',options={'xatol':1e-13})
        # Bounded minimization has a sqrt(eps)*abs(phase) stopping term.
        # At large temporal meshes this can miss even an exact integer
        # shift. Refine the Fourier stationary condition where bracketed,
        # and choose by the direct residual, avoiding correlation cancellation.
        derivative=lambda v:np.real(np.sum(cross[:,j]*(2j*np.pi*frequencies)*
            np.exp(2j*np.pi*frequencies*v)))/N**2
        left,right=(peak[j]-1)/N,(peak[j]+1)/N
        candidates=[float(fit.x),float(peak[j]/N)]
        if derivative(left)>0 and derivative(right)<0:
            candidates.append(brentq(derivative,left,right,xtol=5e-16,rtol=1e-14))
        residuals=[]
        for shift in candidates:
            shifted=np.fft.ifft(fc*np.exp(-2j*np.pi*frequencies*shift)[:,None],axis=0).real
            residuals.append(float(np.sqrt(np.mean(np.sum((x[:,j]-shifted)**2*weights,axis=-1)))))
        best=int(np.argmin(residuals));phases[j]=candidates[best];dist.append(residuals[best])
    return np.array(dist),phases


def main():
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();rows=[]
    paths=[PERIODIC_OUT/f'tori/arcTR2_{i:04d}_N64x32.npz' for i in [0,4,8,11]]
    N=256;M=128
    for path in paths:
        z=np.load(path);r=resample(resample(z['r']*1000,N,axis=0),M,axis=1);targets=[]
        for name in ['upper','middle','lower']:
            t=np.load(PERIODIC_OUT/f'orbits/TR2_endpoint_{name}_N128.npz');c=resample(t['r']*1000,N,axis=0)
            d,shift=distances(r,c,weights);targets.append(dict(target=name,J_EE_core=float(t['J']),T_ms=float(t['T']),
                full_rate_RMS_distance_range_Hz=[float(min(d)),float(max(d))],
                closest_slow_angle_rad=float(2*np.pi*np.argmin(d)/M),distances_Hz=d,fast_phase_shifts_cycles=shift))
        rows.append(dict(torus=str(path),J_EE_core=float(z['J']),T_ms=float(z['T']),
            slow_period_s=float(2*np.pi/z['nu']/1000),targets=targets))
    # An identical shifted waveform must align to numerical precision.
    test,_=distances(np.roll(c,35,axis=0)[:,None,:],c,weights);assert test[0]<1e-7
    out=dict(rows=rows,identity_shift_check_Hz=float(test[0]),observable='Neuron-weighted RMS Hz across all 935 rate groups and the fast angle, minimized over a single fast phase for each slow angle.',
        endpoint_status='NOT_CLASSIFIED',scope='Similarity to periodic fast waveforms does not prove a homoclinic/heteroclinic torus termination, stability, or an exact global connection. Targets are solved at one common nearby J; torus temporal resolution remains separately checked.')
    write(PERIODIC_OUT/'TR2_periodic_target_distances.json',out)
    for row in rows:print('TORUS TARGET DISTANCE',row['slow_period_s'],[(q['target'],q['full_rate_RMS_distance_range_Hz']) for q in row['targets']],flush=True)


if __name__=='__main__':main()
