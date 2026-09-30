"""Refine saved equilibrium turns; do not equate them with attractor loss."""
from native_path import *
import equilibria_v3 as arc
import argparse


def main(a):
    s=model();path=(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s);arc.RS=.1;arc.DS=.1
    src=OUT/'equilibria'/a.branch;data=read(src/'result.json')
    out=OUT/'equilibria'/f'{a.branch}_fold_audit';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,j in data['fold_brackets']:
        name=f'turn_{i:04d}_{j:04d}'
        if (out/f'{name}.json').exists():
            q=read(out/f'{name}.json')
            if 'characteristic_zero' in q:rows.append(q);continue
            if q.get('status')!='REFINED':rows.append(q);continue
            old=np.load(out/f'{name}.npz');r,v,w=old['r'],old['v'],old['w'];s.set_D(float(old['D']))
            q['characteristic_zero']=characteristic_zero(s,r,v,w)
            write(out/f'{name}.json',q);rows.append(q);continue
        L=dict(np.load(src/f'point{i:04d}.npz'));R=dict(np.load(src/f'point{j:04d}.npz'))
        L['D']=float(L['D']);R['D']=float(R['D']);q=arc.refine_fold(s,L,R)
        if q is None:q=dict(status='REFINEMENT_FAILED',bracket=[i,j])
        else:
            r,v,w=q.pop('r'),q.pop('v'),q.pop('w');s.set_D(q['D'])
            energy=s.sizes*s.E*abs(v)**2;energy/=energy.sum()
            q.update(status='REFINED',bracket=[i,j],
                     equilibrium_residual_hz=float(abs(s.residual(r)).max()*1000),
                     distance_to_spatial_path_knot=float(min(abs(q['D']-np.array(path['D'])))),
                     effective_E_groups=float(1/np.sum(energy**2)),
                     largest_E_group=int(np.argmax(energy)),largest_E_group_fraction=float(energy.max()),
                     characteristic_zero=characteristic_zero(s,r,v,w),
                     attractor_relation='NOT_ESTABLISHED; equilibrium fold is not proof of runaway threshold')
            np.savez_compressed(out/f'{name}.npz',r=r,v=v,w=w,D=q['D'],Z=s.Z,energy=energy)
        rows.append(q);write(out/f'{name}.json',q);write(out/'summary.json',dict(status='RUNNING',rows=rows));log(name,q)
    write(out/'summary.json',dict(status='COMPLETE',rows=rows))


def characteristic_zero(s,r,v,w):
    C=s.characteristic(r,0.);derivatives=[]
    for h in [2e-6,1e-6,5e-7]:
        derivative=(s.characteristic(r,h)-s.characteristic(r,-h))/(2*h)
        derivatives.append(float(np.real(w@(derivative@v))))
    scale=max(abs(np.mean(derivatives)),1e-30)
    return dict(right_residual=float(np.linalg.norm(C@v)/np.linalg.norm(v)),
        left_residual=float(np.linalg.norm(C.T@w)/np.linalg.norm(w)),
        w_dC_dlambda_v_ms=derivatives,
        temporal_simple_zero=bool(abs(np.mean(derivatives))>1e-5 and np.ptp(derivatives)/scale<.02))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch')
    p.add_argument('--family',choices=['native','rate'],default='native');main(p.parse_args())
