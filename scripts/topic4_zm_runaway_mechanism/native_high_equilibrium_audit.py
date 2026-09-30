"""Local folds and conditional delayed stability of the high-rate branch."""
from native_high_equilibrium_branch import parameter_column
from native_path import *
from refine_native_folds import characteristic_zero
from equilibrium_spectrum import cache_characteristic
from root_count_v3 import outer_bound, count
import equilibria_v3 as arc


def main():
    src=OUT/'equilibria/native_high_descent';data=read(src/'result.json')
    assert data['status']!='RUNNING'
    out=OUT/'equilibria/native_high_descent_audit';out.mkdir(parents=True,exist_ok=True)
    s=model();attach_native_path(s);arc.RS=.1;arc.DS=.1
    arc.param_derivative=parameter_column
    folds=[]
    for bracket in data['turn_brackets']:
        if bracket['path_knots']:continue
        i,j=bracket['indices']
        L=dict(np.load(src/f'point{i:04d}.npz'));R=dict(np.load(src/f'point{j:04d}.npz'))
        L['D']=float(L['D']);R['D']=float(R['D'])
        q=arc.refine_fold(s,L,R)
        if q is None:q=dict(status='REFINEMENT_FAILED',bracket=[i,j])
        else:
            r,v,w=q.pop('r'),q.pop('v'),q.pop('w');s.set_D(q['D'])
            q.update(bracket=[i,j],equilibrium_residual_hz=float(abs(s.residual(r)).max()*1000),
                characteristic_zero=characteristic_zero(s,r,v,w),
                distance_to_path_knot=float(min(abs(s.path_D_knots-q['D']))),
                interpretation='Conditional high-equilibrium fold on the explicitly extended Z path; no identification with observed onset')
            np.savez_compressed(out/f'fold_{i:04d}_{j:04d}.npz',r=r,v=v,w=w,D=q['D'],Z=s.Z)
        folds.append(q);write(out/'folds.json',dict(status='RUNNING',rows=folds));log('HIGH FOLD',q)
    write(out/'folds.json',dict(status='COMPLETE',rows=folds))
    i,j=data['turn_brackets'][0]['indices']
    indices=[0,min(range(i),key=lambda k:abs(data['rows'][k]['D']-.4)),i,j]
    rows=[]
    for index in indices:
        s=model();attach_native_path(s)
        p=src/f'point{index:04d}.npz';z=np.load(p);r=z['r'];s.set_Z(z['Z'])
        residual=float(abs(s.residual(r)).max()*1000);assert residual<2e-8
        checks=cache_characteristic(s,r);bound=outer_bound(s,r,0.,False)
        q=dict(index=index,source=str(p),D=s.D,global_E_hz=s.global_rate(r),
            Z='held',M='dynamic',residual_hz=residual,characteristic_cache_checks=checks,
            zero_frequency_absolute_gain_bound=bound)
        if bound<1-1e-6:q.update(status='STABLE_BY_ABSOLUTE_GAIN_BOUND',unstable_roots=0)
        else:
            c=count(s,r,N=128);q['contour']=c;q['unstable_roots']=c['unstable_roots']
            q['status']=('UNSTABLE' if c['unstable_roots']>0 else 'STABLE_BY_CONTOUR') if c['unstable_roots'] is not None else 'UNRESOLVED'
        rows.append(q);write(out/'stability.json',dict(status='RUNNING',rows=rows));log('HIGH STABILITY',q)
    write(out/'stability.json',dict(status='COMPLETE',rows=rows,
        limitation='Selected equilibrium points only. No stability label is interpolated across unsampled intervals; this path extension is not the observed onset Z trajectory.'))


if __name__=='__main__':main()
