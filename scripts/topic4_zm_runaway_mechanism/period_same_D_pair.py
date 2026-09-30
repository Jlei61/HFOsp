"""Correct the two near-turn cycles onto exactly the same spatial Z field.

Only the upper-period cycle is corrected. The lower accepted cycle defines D;
no network parameter other than the already selected spatial Z path is changed.
Existence of two cycles is kept separate from their stability or basin role.
"""
from compact_periodic import *
from native_cycles import save
import argparse
from scipy.signal import resample


def main(a):
    s=model();attach_rate_entry_path(s)
    low,high=[np.load(f) for f in [a.lower,a.upper]]
    assert len(low['r'])==len(high['r'])
    assert all(float(z['residual'])<2e-8 for z in [low,high])
    assert read(OUT/'compact_periodic_check.json')['status']=='PASS'
    CompactExactGalerkin.harmonic_block=33
    o=CompactExactGalerkin(s,len(high['r']),a.M,a.device)
    o.cache_mean_operators=False;o.cp.fft.config.get_plan_cache().set_size(0)
    o.linear_tolerance_floor=.001;o.log_linear_progress=True
    out=OUT/'periodic'/a.label;out.mkdir(parents=True,exist_ok=True)
    o.iteration_checkpoint=out/'current_iterate.npz'
    D=float(low['D'])
    guess_r=high['r'];guess_T=float(high['T']);prediction=None
    if a.tangent_seed:
        t=np.load(a.tangent_seed);v=t['tangent'];nr=t['r'].size
        assert t['r'].shape[1]==s.P and len(v)==nr+2
        slope=.001*v[-1]
        increment=(D-float(high['D']))/slope
        assert abs(increment)<.001
        dr=.001*v[:nr].reshape(t['r'].shape)
        guess_r=guess_r+increment*resample(dr,len(guess_r),axis=0)
        guess_T*=np.exp(increment)
        prediction=dict(source=a.tangent_seed,delta_log_period=float(increment),
                        predicted_period_ms=guess_T,dD_dlogT=float(slope))
        log('TANGENT PAIR PREDICTOR',prediction)
    sol=o.solve(guess_r,guess_T,D,maxiter=12,tol=2e-8,restart=80,maxit_lin=480)
    assert sol['residual']<2e-8,sol['residual']
    assert abs(sol['D']-D)<1e-15
    q=save(s,sol,out,'upper_same_D')
    assert abs(sol['T']-float(low['T']))>.01,'Corrector returned to the lower cycle'
    assert abs(s.Z-low['Z']).max()<1e-13
    write(out/'result.json',dict(status='TWO_DISTINCT_CONVERGED_CYCLES_AT_IDENTICAL_Z',
        lower_source=a.lower,upper_seed=a.upper,upper=q,D=D,predictor=prediction,
        period_difference_ms=sol['T']-float(low['T']),
        Z='same held spatial field',M='dynamic in both periodic boundary-value problems',
        stability='NOT_INFERRED_FROM_THE_PAIR',bifurcation_type='REQUIRES_TURN_AND_STABILITY_ANALYSIS'))
    log('SAME D CYCLES',D,float(low['T']),sol['T'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lower');p.add_argument('upper')
    p.add_argument('--label',default='rate_same_D_pair_G8193');p.add_argument('--M',type=int,default=32768)
    p.add_argument('--tangent-seed',help='Accepted nearby branch tangent, used only as a corrector initial guess')
    p.add_argument('--device',type=int,default=1);main(p.parse_args())
