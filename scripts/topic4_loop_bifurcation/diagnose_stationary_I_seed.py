#!/usr/bin/env python3
"""One solver diagnostic: equilibrate unobserved local I rates before full Newton."""
import time
import numpy as np
from scipy.sparse.linalg import spsolve
from campaign import read,write,sha
from stationary_native_diagnostic import Stationary,OUT


def main():
    path=OUT/'I_seed_diagnostic_contract.json';assert not path.exists()
    write(path,dict(status='FROZEN_SINGLE_SOLVER_DIAGNOSTIC',source_sha256=sha(__file__),
        reason='FailedK9 solve had maximumI residual733Hz,E211Hz; initialization only suppliedglobalImean,not localI distribution. Smallfielddifference ofthatunconvergediterate isnot correspondence.',
        budget='Only existingZ.21/K9high point. HoldEtemporarily tosolve localI equations<=40steps, then releaseallrates foroneoriginal80stepNewton solve. No parameterchange orfit to nativeoutput.',
        interpretation='Initialization diagnostic of approximate stationaryequations only; no continuation,stability orbifurcation. Failure doesnot establish no equilibrium.'))
    e=Stationary();e.set_fields(.21,9.);r=np.load(OUT/'exit_z0.21_k9_high.npz')['rate_per_ms'].copy();mask=~e.s.E;trace=[];start=time.time()
    for k in range(40):
        f,A,u,v,*_=e.evaluate(r,True);assert np.all(v[mask]==0.)
        error=float(abs(f[mask]).max());trace.append(error)
        if error<1e-9:break
        step=spsolve(A[mask][:,mask],-f[mask]);alpha=1.
        for back in range(30):
            trial=r.copy();trial[mask]+=alpha*step
            if trial[mask].min()>=-1e-10 and np.all(trial[mask]<1/e.s.ref[mask]):
                trial[mask]=np.maximum(trial[mask],0.)
                if abs(e.evaluate(trial)[mask]).max()<error:r=trial;break
            alpha*=.5
        else:break
    seed=r.copy();r,ok,full=e.solve(r);f=e.evaluate(r)
    np.savez_compressed(OUT/'I_seed_diagnostic_arrays.npz',I_equilibrated_seed=seed,final_candidate=r,residual=f)
    result=dict(status='DIAGNOSTIC_COMPLETE_NO_STABILITY',I_only_converged=bool(trace[-1]<1e-9),I_trace=trace,
        full_converged=ok,full_trace=full,wholeE_Hz=float(e.s.global_rate(r)),core_Hz=(e.regional@r*1000)[:2].tolist(),
        maximum_residual_Hz=float(abs(f).max()*1000),elapsed_s=time.time()-start,formal_bifurcation_allowed=False)
    write(OUT/'I_seed_diagnostic_result.json',result);print({k:v for k,v in result.items() if k not in ['I_trace','full_trace']},flush=True)


if __name__=='__main__':main()
