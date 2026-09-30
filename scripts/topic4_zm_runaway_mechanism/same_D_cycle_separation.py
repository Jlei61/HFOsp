"""Exclude a pure phase duplicate in the two accepted same-Z cycles."""
from native_path import *
from scipy.optimize import minimize_scalar


def main():
    s=model();attach_rate_entry_path(s);folder=OUT/'periodic/rate_same_D_pair_G8193'
    pair=read(folder/'result.json')
    assert pair['status']=='TWO_DISTINCT_CONVERGED_CYCLES_AT_IDENTICAL_Z'
    low=np.load(pair['lower_source']);high=np.load(pair['upper']['path'])
    assert np.array_equal(low['Z'],high['Z']) and float(low['D'])==float(high['D'])
    x=low['r'][:,s.E];y=high['r'][:,s.E];N=len(x)
    assert N==len(y) and N%2==1
    a=np.fft.rfft(x,axis=0)/N;b=np.fft.rfft(y,axis=0)/N
    weights=s.mean_weights;freq=np.arange(len(a));factor=np.r_[1.,np.full(len(a)-1,2.)]
    cross=np.sum(a*np.conj(b)*weights,axis=1)
    nx=float(np.sum(factor[:,None]*abs(a)**2*weights))
    ny=float(np.sum(factor[:,None]*abs(b)**2*weights))
    def distance(shift):
        return nx+ny-2*float(np.sum(factor*cross*np.exp(2j*np.pi*freq*shift)).real)
    corr=np.fft.irfft(cross,n=8*N)
    start=int(np.argmax(corr))/(8*N)
    fit=minimize_scalar(distance,bounds=(start-2/N,start+2/N),method='bounded',options={'xatol':1e-14})
    delta=fit.x;relative=float(np.sqrt(max(0,fit.fun)/nx))
    assert relative>1e-3,relative
    q=dict(status='DISTINCT_AFTER_OPTIMAL_GLOBAL_PHASE_ALIGNMENT',D=float(low['D']),
           Z_identical=True,periods_ms=[float(low['T']),float(high['T'])],
           shift_fraction_of_cycle=float(delta),relative_E_neuron_weighted_rate_L2=relative,
           definition='All E groups, original neuron counts, Fourier-exact normalized-phase shift',
           stability='Not inferred from geometric separation')
    write(folder/'phase_separation.json',q);log('SAME D GEOMETRY',q)


if __name__=='__main__':main()
