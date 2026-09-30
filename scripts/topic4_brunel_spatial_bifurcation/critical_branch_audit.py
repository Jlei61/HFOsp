"""Join the actual continued branch and audit repeated stationary folds."""
from common import *
from model import SpatialBrunel
from scipy.sparse.linalg import eigs
BASE=OUT/'critical_revision'

def inventory():
    s=SpatialBrunel(response='calibrated_full');rows=[];states=[];r=None
    values=np.unique(np.r_[np.arange(.4,.94001,.01),np.arange(.940,.96501,.0005),np.arange(.97,1.00001,.005)])
    for J in values:
        r,ok,tr=s.solve(float(J),r);assert ok
        rows.append(dict(J_EE_core=J,rates_hz=s.regional_rates(r),source='new direct equilibrium solve',source_index=None,residual=tr[-1]))
        states.append(r.copy())
    paths=[('g20/arclength_v2',None),('expanded/low_arclength',782),('expanded/low_arclength_v2',None)]
    for name,limit in paths:
        data=read(OUT/name/'result.json')['rows'];data=data if limit is None else data[:limit]
        for row in data[1:]:
            file=OUT/name/f'step{row["step"]:04d}.npz';z=np.load(file)
            rows.append(dict(J_EE_core=float(z['J']),rates_hz=s.regional_rates(z['rates']),source=str(file),source_index=row['step'],residual=row['residual']))
            states.append(z['rates'])
    last_J=rows[-1]['J_EE_core']
    data=read(OUT/'expanded/high_arclength_v2/result.json')['rows']
    # Only the initial monotonic upper segment extends the joined branch.
    data=[q for q in data[:int(np.argmin([q['J_EE_core'] for q in data]))+1] if q['J_EE_core']>last_J]
    for row in data[::-1]:
        file=OUT/'expanded/high_arclength_v2'/f'step{row["step"]:04d}.npz';z=np.load(file)
        rows.append(dict(J_EE_core=float(z['J']),rates_hz=s.regional_rates(z['rates']),source=str(file),source_index=row['step'],residual=row['residual']))
        states.append(z['rates'])
    rates=np.array(states);J=np.array([q['J_EE_core'] for q in rows]);regional=np.array([q['rates_hz'] for q in rows])
    joins=[i for i in range(1,len(rows)) if str(Path(rows[i]['source']).parent)!=str(Path(rows[i-1]['source']).parent)]
    checks=[dict(index=i,J_before=J[i-1],J_after=J[i],rate_norm_difference=float(np.linalg.norm(rates[i]-rates[i-1])),regional_difference_hz=regional[i]-regional[i-1]) for i in joins]
    reversals=np.flatnonzero(np.diff(J)[:-1]*np.diff(J)[1:]<0)+1
    np.savez_compressed(BASE/'branch.npz',rates=rates,J=J,regional=regional,reversal_indices=reversals)
    write(BASE/'branch.json',dict(status='COMPLETE',rows=rows,points=len(rows),join_checks=checks,reversal_indices=reversals,
        meaning='Single connected stationary path from low activity to the upper branch; J reverses direction at folds. Not a parameter sweep trajectory or all possible branches.'))
    print('BRANCH',len(rows),'reversals',len(reversals),'joins',checks,flush=True)

def fold_audit():
    s=SpatialBrunel(response='calibrated_full');finequad=SpatialBrunel(quadrature=96,response='calibrated_full')
    folders=[OUT/'g20/fold']+sorted((OUT/'expanded/folds').glob('*/'))
    vv=[];rr=[];rows=[];size=s.geo['group_size'];pos=s.geo['positions'];reg=s.geo['group_region']
    for folder in folders:
        q=read(folder/'result.json');z=np.load(folder/'fold.npz');r=z['rates'];v=z['right'];J=float(z['J'])
        normed=np.sqrt(size)*v;normed/=np.linalg.norm(normed);vv.append(normed);rr.append(r)
        A=finequad.jacobian(r,J);quad_res=float(np.linalg.norm(A@v)/np.linalg.norm(v));fres=float(abs(finequad.residual(r,J)).max())
        en=size*abs(v)**2;en/=en.sum();centroid=(en[:,None]*pos).sum(0)
        h=5e-7;H=(s.jacobian(r+h*v,J)-s.jacobian(r-h*v,J))/(2*h)
        coefficient=float(z['left']@(H@v)/2)
        rows.append(dict(label=folder.name,source=str(folder),J_EE_core=J,rates_hz=q['rates_hz'],regional_energy=q['regional_energy'],inhibitory_energy=q['inhibitory_energy'],
            mode_centroid_mm=centroid,quadrature96_fixedpoint_residual=fres,quadrature96_zero_mode_residual=quad_res,
            quadratic_coefficient=q['quadratic_coefficient'],quadratic_half_step=coefficient,
            transversality=q['transversality'],core_family='A' if q['regional_energy'][0]>q['regional_energy'][1] else 'B'))
    vv=np.array(vv);overlap=abs(vv@vv.T);families=[]
    # Same local fold can appear at several backgrounds of the other core.
    # Group by null-mode overlap, not by close parameter coordinates alone.
    for i,q in enumerate(rows):
        match=next((j for j,f in enumerate(families) if all(overlap[i,k]>.995 for k in f)),None)
        if match is None:families.append([i])
        else:families[match].append(i)
    for k,f in enumerate(families):
        for i in f:rows[i]['mode_family']=k+1
    np.savez_compressed(BASE/'fold_modes.npz',weighted_vectors=vv,rates=np.array(rr),overlap=overlap)
    write(BASE/'fold_audit.json',dict(rows=rows,mode_overlap=overlap,mode_families=families,
        family_definition='Pairwise neuron-weighted null-vector cosine above .995; this is a similarity grouping, not a new bifurcation type',
        meaning='Stationary folds visited along an arclength path, often at different background states of the other core. They do not imply repeated transitions of a stable burst trajectory.'))
    print('FOLD FAMILIES',families,flush=True)
    for family in families:print([(rows[i]['label'],rows[i]['J_EE_core'],rows[i]['rates_hz']) for i in family],flush=True)

if __name__=='__main__':
    BASE.mkdir(exist_ok=True);inventory();fold_audit()
