"""Verify static equations against the current response and independent DC steps.

No root search, continuation, stability classification or model promotion.
"""
from common import OUT, np, write, log
from current_rate_equilibrium import CurrentRateEquilibrium, torch, SCALE, DT0
from nonlinear_rate_response import normalized_input
from refractory_rate_response import covariance_matrices
from fine_rate_frozen_Z_fields import native_field
from scipy.special import expit
from pathlib import Path
import hashlib

DEST=OUT/'current_rate_analysis_interface'


def main():
    DEST.mkdir(exist_ok=True)
    s=CurrentRateEquilibrium(40); rng=np.random.default_rng(920091); rows=[]
    for tm,level in [(9000,.0002),(9420,.02),(9870,.2)]:
        s.set_Z(native_field(s,tm),source=f'native {tm}ms entire field')
        r=level*rng.uniform(.8,1.2,s.P)
        mu,ve,vi=s.moments(r); operating=s.local_operating(mu,ve,vi)
        explicit=np.empty(s.P); covariance_error=0.; discrete_flux_error=0.
        for pop,mask in [('E',s.E),('I',~s.E)]:
            p=np.column_stack([mu[mask],ve[mask],vi[mask]]); theta=s.theta[mask]
            f=np.zeros((mask.sum(),39));f[:,:3]=normalized_input(p,theta)/SCALE
            b=s.bases[pop].evaluate(p,theta)
            with torch.no_grad():
                explicit[mask]=s.nets[pop].stationary(torch.tensor(f),torch.tensor(b)).numpy()/1000
            for dt in [.05,.025]:
                A,B,C,_,_=covariance_matrices(pop,dt)
                for ch in range(2):
                    cov=np.linalg.solve(np.eye(3)-A[ch],B[ch])
                    covariance_error=max(covariance_error,abs(C[ch]*cov[2]-1))
                # At constant firing each of the nref-1 previous bins has r*dt.
                flux=(1-(s.ref[mask]-dt)*explicit[mask])*expit(operating['log_hazard'][mask]+np.log(dt/DT0))/dt
                discrete_flux_error=max(discrete_flux_error,float(np.max(abs(flux-explicit[mask]))))
        rate_error=float(np.max(abs(explicit-operating['rate'])))
        assert rate_error<1e-12 and covariance_error<1e-10 and discrete_flux_error<1e-12
        J=s.jacobian(r); fd=[]
        for direction in range(3):
            v=r*rng.uniform(-1,1,s.P)
            for h in [1e-4,5e-5]:
                numerical=(s.residual(r+h*v)-s.residual(r-h*v))/(2*h)
                analytical=J@v
                error=float(np.linalg.norm(analytical-numerical)/max(np.linalg.norm(numerical),1e-14))
                fd.append(dict(direction=direction,h=h,relative_error=error))
                assert error<2e-5,(tm,direction,h,error)
        row=dict(native_Z_time_ms=tm,D=s.D,probe_rate_per_ms=level,
            stationary_readout_max_error_per_ms=rate_error,covariance_DC_error=covariance_error,
            implicit_flux_error_per_ms=discrete_flux_error,jacobian_directions=fd)
        rows.append(row); log('STATIC CURRENT RESPONSE CHECK',tm,'PASS')
    sources=[Path(__file__),Path(__file__).with_name('current_rate_equilibrium.py')]
    write(DEST/'static_implementation_check.json',dict(status='PASS',grid=40,groups=s.P,
        rows=rows,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        object='Current conditioned39 deterministic full-diffusion expected-rate static equations; M at its rate-dependent equilibrium, prescribed full Z field and constant original mean external input.',
        limits='Probes are not equilibria. Static Jacobian is not the temporal generator. No stability or bifurcation assignment and no local/native acceptance waiver.',
        root_searches=0,model_promoted=False))


if __name__=='__main__':main()
