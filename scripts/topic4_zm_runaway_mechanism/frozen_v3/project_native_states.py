"""Project native checkpoints to the 935-group level: Z (mean, mean z^2), M, rates, currents, delay history.

Used for stage B (actual Z field family along the entry trajectory) and A4 initial states.
Rates: E per group from spikes in a window ending at the checkpoint (chunks raster is only a
sample; use field_0p1ms per cell for E, population_0p1ms for I split by cell size for I groups).
"""
from common_v3 import *
import glob
geo=dict(np.load(OPERATORS/'g20/geometry.npz'));E=geo['population']==0;P=len(E);cg=geo['cell_group'];sizes=geo['group_size']
def project(x,members):return np.bincount(members,weights=x,minlength=P)/np.maximum(np.bincount(members,minlength=P),1)
out={}
for cp in sorted(glob.glob(str(NATIVE/'checkpoints/t*ms.npz'))):
    t=int(Path(cp).stem[1:-2]);z=np.load(cp);zc=z['slow__z'][:32000];mc=z['slow__m'][:32000]
    row=dict(time_ms=t,Z_group=project(zc,cg[:32000]),Z2_group=project(zc*zc,cg[:32000]),M_group=project(mc,cg[:32000]),
        IE_group=project(z['I_E'],cg),II_group=project(z['I_I'],cg),sE_group=project(z['s_E'],cg),sI_group=project(z['s_I'],cg),
        V_group=project(z['V'],cg),D=float(1-np.average(project(zc,cg[:32000])[E],weights=sizes[E])),
        frac_II_above=project((z['I_I'][:32000]>=95.19851312666987).astype(float),cg[:32000]))
    # ring history projected to groups: ring_sE (359,40000) pending arrivals
    ring=z['ring_sE'];rowE=np.zeros((ring.shape[0],P))
    for k in range(ring.shape[0]):rowE[k]=project(ring[k],cg)
    ringI=z['ring_sI'];rowI=np.zeros((ringI.shape[0],P))
    for k in range(ringI.shape[0]):rowI[k]=project(ringI[k],cg)
    row['ring_sE_group']=rowE;row['ring_sI_group']=rowI
    out[t]=row;log('checkpoint',t,'D=%.4f'%row['D'],'mean M %.2f'%mc.mean(),'frac above %.3f'%(z['I_I'][:32000]>=95.2).mean())
np.savez_compressed(DEST/'native_reference/checkpoint_projections.npz',times=np.array(sorted(out)),**{f'{k}_{t}':v for t,row in out.items() for k,v in row.items() if k!='time_ms'})
write(DEST/'native_reference/checkpoint_projections.json',{str(t):dict(D=row['D'],mean_M=float(row['M_group'][E]@sizes[E]/sizes[E].sum())) for t,row in out.items()})
