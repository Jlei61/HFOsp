"""Small cycles from the high-rate Hopf; verify normal-form direction by BVP."""
from periodic_zm import *


def main():
    s=ZMSpatialRate();z=np.load(PERIODIC_OUT/'normal_form_high.npz')
    nf=read(PERIODIC_OUT/'normal_form_high.json');r=z['r'];D=float(z['D']);w=float(z['w']);q=z['q']
    N=64;p=ZMPeriodic(s,N,device=0);phase=2*np.pi*np.arange(N)/N;rows=[]
    for a in [1.,2.,4.,8.,16.]:
        guess=r[None,:]+2*a*np.real(np.exp(1j*phase[:,None])*q)
        guess+=a*a*np.real(z['h11'][None,:]+np.exp(2j*phase[:,None])*z['h20'][None,:])
        rr,T,DD,err,hist=p.solve(guess,2*np.pi/(w+nf['omega_shift_per_amplitude_squared']*a*a),
            D+nf['D_shift_per_amplitude_squared']*a*a,amplitude=(q,a),tol=2e-10)
        assert err<2e-9
        save_orbit(s,rr,T,DD,err,hist,f'highHopf_A{a:g}_N{N}')
        rows.append(dict(amplitude=a,D=DD,T_ms=T,residual_hz=err,D_shift_per_amplitude_squared=(DD-D)/a**2,
                         normal_form_relative_difference=(DD-D)/(a*a*nf['D_shift_per_amplitude_squared'])-1))
        write(PERIODIC_OUT/'high_hopf_cycles.json',dict(rows=rows))


if __name__=='__main__':main()
