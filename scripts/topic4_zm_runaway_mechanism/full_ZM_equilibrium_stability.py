"""Delayed characteristic spectrum at autonomous Z/M stationary solutions."""
from common import *
from root_count_v3 import outer_bound, count, refine_root, det_sign_zero


def main():
    s=model();destination=OUT/'full_ZM_equilibria'
    rows=[]
    for label in ['low','released_high']:
        z=np.load(destination/f'{label}.npz');r=z['r'];s.set_Z(z['Z'])
        assert np.max(abs(s.residual(r)))<1e-10
        bound=outer_bound(s,r,0.,True)
        row=dict(label=label,global_E_hz=s.global_rate(r),D=float(z['D']),
                 zero_frequency_absolute_gain_bound=bound,Z='dynamic',M='dynamic')
        log('FULL ZM STABILITY',label,'absolute gain',bound)
        # This absolute, nonnegative gain matrix bounds the characteristic
        # feedback for every Re(lambda)>=0 at R=0. All eliminated filters
        # have strictly stable poles. rho(K)<1 excludes singular I-A there.
        if bound<1-1e-6:
            row.update(status='STABLE_BY_ABSOLUTE_GAIN_BOUND',unstable_roots=0)
        else:
            row.update(count(s,r,N=128,dynamic_z=True))
            row['status']=('UNSTABLE' if row['unstable_roots']>0 else 'STABLE_BY_CONTOUR') if row['unstable_roots'] is not None else 'UNRESOLVED'
            if row.get('unstable_roots',0):
                found=[]
                for guess in [.001,.005,.01,.02,.01+.03j,.01+.08j,.02+.15j]:
                    try:q=refine_root(s,r,complex(guess),dynamic_z=True)
                    except Exception as e:
                        log('ROOT REFINE',label,guess,type(e).__name__);continue
                    if q is None:continue
                    lam,v,res=q
                    if lam.real<=0 or any(abs(lam-complex(*old['lambda_per_ms']))<1e-7 for old in found):continue
                    found.append(dict(lambda_per_ms=[lam.real,lam.imag],residual=res))
                    log('ROOT',label,lam,res)
                row['positive_roots_found']=found
        rows.append(row);write(destination/'stability.json',dict(status='RUNNING',rows=rows))
    write(destination/'stability.json',dict(status='COMPLETE',rows=rows,
        scope='Local stability in the frozen spatial rate equations with both Z and M autonomous. Does not establish native SNN equivalence or a basin boundary.'))


if __name__=='__main__':main()
