"""Classify only computed spectra; polynomial filtering is not a root ranking.

Unreturned multipliers are bounded using the smallest returned transformed
modulus. A tiny returned multiplier must not be called the dominant multiplier.
"""
from floquet_zm import *


def classify(q):
    raw=np.asarray(q['multipliers'])
    vals=raw.astype(complex) if raw.ndim==1 else raw[:,0]+1j*raw[:,1]
    res=np.asarray(q['residuals'])
    neutral=q.get('identified_neutral_index');good=res<1e-4*np.maximum(1,abs(vals))
    considered=np.ones(len(vals),bool)
    if neutral is not None:considered[neutral]=False
    growing=considered&good&(abs(vals)>1.01)
    if growing.any():
        return dict(status='UNSTABLE',verified_growing_multiplier=vals[np.flatnonzero(growing)[0]],
                    complete_unstable_count=False,source=q['orbit'],D=q['D'])
    rho=q.get('polynomial_filter_rho');tau=q.get('smallest_returned_transformed_modulus')
    if rho is None or tau is None:
        bound=max(abs(vals[considered]),default=1.)
        margin=max(1e-4,10*q['phase_multiplier_error'])
        ok=(len(vals)>=2 and neutral is not None and good.all() and bound<1-margin)
        return dict(status='STABLE' if ok else 'UNRESOLVED',source=q['orbit'],D=q['D'],
                    nontrivial_modulus_upper_bound=float(bound),phase_error=q['phase_multiplier_error'],
                    note='Numerical direct Arnoldi largest-modulus spectrum; autonomous phase mode identified and excluded',
                    stability_numerical_margin=margin)
    missing_bound=(rho+np.sqrt(rho*rho+4*tau))/2
    returned=max(abs(vals[considered]),default=0.)
    bound=max(missing_bound,returned)
    margin=max(1e-4,10*q['phase_multiplier_error'])
    ok=(neutral is not None and good.all() and q['phase_multiplier_error']<.01 and bound<1-margin)
    return dict(status='STABLE' if ok else 'UNRESOLVED',nontrivial_modulus_upper_bound=float(bound),
                missing_multiplier_modulus_bound=float(missing_bound),phase_error=q['phase_multiplier_error'],
                neutral_index=neutral,source=q['orbit'],D=q['D'],stability_numerical_margin=margin,
                note='Numerical largest-modulus Arnoldi coverage of M(M-rho I); excludes identified autonomous phase mode')


def refresh():
    rows=[]
    for path in sorted((PERIODIC_OUT/'floquet').glob('*.json')):
        q=read(path);row=classify(q);row.update(floquet_file=str(path),dt_ms=q['dt_ms']);rows.append(row)
    write(PERIODIC_OUT/'stability_by_orbit.json',dict(rows=rows))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--orbits',nargs='*');p.add_argument('--device',type=int,default=0)
    p.add_argument('--dt',type=float,default=.1);a=p.parse_args()
    for name in a.orbits or []:
        path=PERIODIC_OUT/'orbits'/f'{name}.npz'
        result=PERIODIC_OUT/'floquet'/f'{name}_dt{a.dt:g}.json'
        q=read(result) if result.exists() else clean(compute(path,a.dt,2,a.device,True))
        if classify(q)['status']=='UNRESOLVED':q=compute(path,a.dt,4,a.device,True)
        print('CLASSIFICATION',clean(classify(q)),flush=True);refresh()
    refresh()
