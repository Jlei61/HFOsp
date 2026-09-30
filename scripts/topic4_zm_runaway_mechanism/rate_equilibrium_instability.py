"""Track a verified positive characteristic root on the actual rate Z branch.

One positive-real-part root is sufficient to prove instability. This does not
count all unstable roots and does not label a static fold as onset.
"""
from equilibrium_spectrum import *
from root_count_v3 import det_sign_zero


def main():
    s=model();original=s.characteristic;rows=[]
    output=OUT/'rate_equilibrium_branch_stability.json'
    old={(q['branch'],q['index']):q for q in read(output)['rows']} if output.exists() else {}
    source=OUT/'equilibrium_spectra/target_mode2.npz';z=np.load(source)
    lam0=complex(z['lambda_per_ms']);v0=z['v'];assert lam0.real>0
    for label in ['rate_up','rate_down']:
        info=read(OUT/'equilibria'/label/'result.json');lam=lam0;v=v0
        for point in info['rows']:
            prior=old.get((label,point['index']))
            if prior and prior['status']=='UNSTABLE':
                rows.append(prior);continue
            z=np.load(point['path']);r=z['r'];s.set_Z(z['Z']);s.characteristic=original
            assert abs(s.residual(r)).max()<1e-9;cache_characteristic(s,r)
            ans=refine_root(s,r,lam,v)
            if ans is None:ans=refine_root(s,r,lam0,v0)
            q=dict(branch=label,index=point['index'],D=float(z['D']),source=point['path'])
            if ans is None:q.update(status='UNRESOLVED')
            else:
                lam,v,err=ans;q.update(status='UNSTABLE' if lam.real>1e-8 else 'TRACKED_ROOT_NOT_POSITIVE',
                    lambda_per_ms=[lam.real,lam.imag],residual=err)
            if q['status']!='UNSTABLE':
                q['det_zero_sign']=det_sign_zero(s,r)
                if q['det_zero_sign']<0:
                    q['status']='UNSTABLE';q['evidence']='Negative zero-frequency characteristic determinant implies a positive real root'
                else:
                    q['tracked_nonpositive_root']=q.get('lambda_per_ms')
                    attempts=[]
                    for guess in [.005+.2j,.04+.1j,.05+.3j,.1+.05j,.02+.5j]:
                        found=refine_root(s,r,guess)
                        if found is None:continue
                        lv,vv,err=found;attempts.append(dict(lambda_per_ms=[lv.real,lv.imag],residual=err))
                        if lv.real>1e-8:
                            q.update(status='UNSTABLE',lambda_per_ms=[lv.real,lv.imag],residual=err);break
                    q['additional_root_search']=attempts
            rows.append(q)
            if point['index']%10==0:log('RATE EQUILIBRIUM ROOT',q)
            write(output,dict(status='RUNNING',rows=rows))
    write(output,dict(status='COMPLETE',rows=rows,
        all_sampled_points_unstable=all(q['status']=='UNSTABLE' for q in rows),
        claim='A positive full-delay characteristic root at each sampled equilibrium; no root-count completeness inferred'))


if __name__=='__main__':main()
