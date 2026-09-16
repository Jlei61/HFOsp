"""Parameter continuation of nonconstant periodic solutions toward the fold."""
from model import System,OUT
from periodic import Orbit
import numpy as np,json
from scipy.interpolate import CubicSpline

def continue_branch(sequence,initial):
    s=System();f=json.loads((OUT/'fold.json').read_text());gc=f['g'];z=np.load(initial);r=z['r'];T=float(z['T']);oldg=float(z['g']);rows=[]
    for g in sequence:
        predT=150+(T-150)*np.sqrt((oldg-gc)/(g-gc))
        N=max(1024,2**int(np.ceil(np.log2(predT/.65))))
        # Preserve the fast burst segment in physical time while extending the
        # quiet bottleneck; this is an initial guess, never the solved waveform.
        pad=min(120.,T*.2,predT*.2);tn=np.arange(N)*predT/N
        told=np.interp(tn,[0,pad,predT-pad,predT],[0,pad,T-pad,T])
        tt=np.arange(len(r)+1)*T/len(r);rr=np.r_[r,r[:1]]
        guess=CubicSpline(tt,rr,bc_type='periodic')(told)
        o=Orbit(s,g,N);rnew,Tnew,err,history=o.solve(guess,predT,maxiter=32)
        row=dict(g=g,T_ms=Tnew,N=N,residual=err,min_hz=(rnew.min(0)*1000).tolist(),max_hz=(rnew.max(0)*1000).tolist(),mean_hz=(rnew.mean(0)*1000).tolist())
        rows.append(row);(OUT/'periodic'/'continuation_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        np.savez_compressed(OUT/'periodic'/f'g{g:.8f}_N{N}.npz',r=rnew,T=Tnew,g=g,residual=err,history=history)
        if err>1e-8:raise RuntimeError(('Orbit continuation failed; checkpoint saved but excluded',row))
        print('ACCEPTED ORBIT',json.dumps(row),flush=True)
        r,T,oldg=rnew,Tnew,g
    return rows

if __name__=='__main__':
    start=OUT/'periodic/g1.15000000_N1024.npz'
    rows=continue_branch([1.145,1.14,1.135,1.132,1.13,1.128,1.127,1.1263],start)
    (OUT/'periodic_branch.json').write_text(json.dumps(rows,indent=2)+'\n')
