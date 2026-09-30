"""Classify selected low-family folds and check their neighboring stability.

Select the first two and the last accepted turn of the completed first batch,
without selecting by modal shape or strength. This is not an all-fold census.
"""
from native_high_equilibrium_branch import parameter_column
from refine_native_folds import characteristic_zero
from equilibrium_spectrum import cache_characteristic
from root_count_v3 import refine_root
from native_path import *
import equilibria_v3 as arc


def main():
    src=OUT/'equilibria/native_low_continued01'
    data=read(src/'result.json');assert data['status']!='RUNNING'
    selected=[data['turn_brackets'][k] for k in [0,1,-1]]
    dest=OUT/'equilibria/native_low_selected_fold_audit';dest.mkdir(exist_ok=True)
    write(dest/'contract.json',dict(source=str(src/'result.json'),selected=selected,
        selection='First two and last recorded turn before the original 40-turn budget stop',
        question='Are these smooth equilibrium saddle-nodes, where are their zero modes, and are their adjacent equilibria already unstable?',
        scope='Selected local validation only; no onset label or all-fold classification'))
    rows=[];arc.RS=.1;arc.DS=.1;arc.param_derivative=parameter_column
    for b in selected:
        assert not b['path_knots']
        i,j=b['indices'];s=model();path=attach_native_path(s)
        L=dict(np.load(src/f'point{i:04d}.npz'));R=dict(np.load(src/f'point{j:04d}.npz'))
        L['D']=float(L['D']);R['D']=float(R['D'])
        refined=arc.refine_fold(s,L,R)
        if refined is None:
            rows.append(dict(bracket=[i,j],status='REFINEMENT_FAILED'))
            continue
        r,v,w=refined.pop('r'),refined.pop('v'),refined.pop('w');s.set_D(refined['D'])
        energy=s.E*s.sizes*abs(v)**2;energy/=energy.sum()
        cells=np.bincount(s.geo['group_cell'],weights=energy,minlength=400)
        cell=int(np.argmax(cells))
        refined.update(bracket=[i,j],characteristic_zero=characteristic_zero(s,r,v,w),
            equilibrium_residual_hz=float(abs(s.residual(r)).max()*1000),
            distance_to_path_knot=float(min(abs(s.path_D_knots-refined['D']))),
            mode_energy_A_B_surround=[float(energy[s.geo['group_region']==k].sum()) for k in range(3)],
            maximal_E_energy_cell_mm=[cell%20+.5,cell//20+.5],effective_E_cells=float(1/sum(cells*cells)))
        np.savez_compressed(dest/f'fold_{i:04d}_{j:04d}.npz',r=r,v=v,w=w,Z=s.Z,D=s.D,energy=energy)
        adjacent=[]
        for index in [i,j]:
            z=np.load(src/f'point{index:04d}.npz');ss=model();attach_native_path(ss);ss.set_D(float(z['D']))
            rr=z['r'];assert abs(ss.residual(rr)).max()*1000<2e-8
            cache_characteristic(ss,rr);roots=[]
            for omega in [.03,.1,.2]:
                try:ans=refine_root(ss,rr,.01+omega*1j,tol=1e-10)
                except Exception as exc:
                    log('LOW FOLD ROOT FAILURE',index,omega,repr(exc));continue
                if ans is None:continue
                lam,vector,error=ans
                roots.append(dict(lambda_per_ms=[lam.real,lam.imag],residual=error))
            adjacent.append(dict(index=index,D=ss.D,roots=roots,
                status='UNSTABLE_BY_POSITIVE_ROOT' if any(q['lambda_per_ms'][0]>1e-7 for q in roots) else 'STABILITY_NOT_ESTABLISHED'))
        refined['adjacent_equilibria']=adjacent
        refined['onset_relation']='NOT_ESTABLISHED; a saddle-node on an unstable equilibrium family is not itself loss of the observed bursting attractor'
        rows.append(refined);write(dest/'result.json',dict(status='RUNNING',rows=rows));log('LOW SELECTED FOLD',refined)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,Z='held native path',M='dynamic',
        scope='Three prespecified representative turns of this connected low-root family. Other turns remain unclassified.'))


if __name__=='__main__':main()
