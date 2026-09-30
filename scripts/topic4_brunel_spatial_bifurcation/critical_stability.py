"""Temporal eigenpair certificates along the joined spatial equilibrium path."""
from common import *
from model import SpatialBrunel
from contour_roots import refine,extract
BASE=OUT/'critical_revision'

def main():
    s=SpatialBrunel(response='calibrated_full');z=np.load(BASE/'branch.npz');J=z['J'];rates=z['rates']
    dest=BASE/'stability';dest.mkdir(exist_ok=True);rows=[]
    seeds=[]
    for core in ['A','B']:
        q=np.load(OUT/f'g20/hopf_{core}_calibrated_full/critical.npz');seeds.append((1j*float(q['omega']),q['vector']))
    upper=np.load(OUT/'expanded/modes/upper_J1.3/modes.npz');ix=int(np.argmax(upper['roots'].real));highseed=(upper['roots'][ix],upper['vectors'][ix])
    ids=set(np.flatnonzero((J>=.94)&(np.arange(len(J))<113)).tolist())
    ids.update(range(113,len(J),4));ids.add(len(J)-1)
    for i in z['reversal_indices']:
        ids.update(range(max(0,int(i)-3),min(len(J),int(i)+4)))
    primary=None
    for index in sorted(ids):
        j=float(J[index]);r=rates[index];path=dest/f'point{index:04d}.npz';found=[]
        if path.exists():
            old=np.load(path);found=list(zip(old['roots'],old['vectors'],old['residuals']))
        elif index<113:
            updated=[]
            for lam,v in seeds:
                q=refine(s,r,j,lam,v)
                if q is None:raise RuntimeError(('onset mode failed',index,j))
                found.append(q);updated.append(q[:2])
            seeds=updated
        else:
            candidates=([primary] if primary is not None else [])+seeds+[highseed]
            for lam,v in candidates:
                q=refine(s,r,j,complex(lam),v)
                if q is not None and q[0].real>1e-7:
                    found=[q];break
            if not found:
                ev,vec,_=extract(s,r,j,n=24,m=24,corners=[-.02+.0001j,.18+.0001j,.18+.32j,-.02+.32j])
                for k,lam in enumerate(ev):
                    if not (-.03<lam.real<.2 and 0<lam.imag<.35):continue
                    q=refine(s,r,j,lam,vec[:,k])
                    if q is not None and q[0].real>1e-7:found=[q];break
        if found:
            primary=max(found,key=lambda q:q[0].real)[:2]
            np.savez_compressed(path,J=j,rates=r,roots=np.array([q[0] for q in found]),vectors=np.array([q[1] for q in found]),residuals=np.array([q[2] for q in found]))
        roots=[]
        for lam,v,err in found:
            en=s.geo['group_size']*abs(v)**2;en/=en.sum();reg=s.geo['group_region']
            roots.append(dict(lambda_per_ms=lam,frequency_hz=abs(lam.imag)*1000/(2*np.pi),residual=err,
                regional_energy=[float(en[s.E&(reg==k)].sum()) for k in range(3)],inhibitory_energy=float(en[~s.E].sum())))
        row=dict(index=index,J_EE_core=j,roots=roots,positive_mode_found=any(q['lambda_per_ms'].real>1e-7 for q in roots),
            rates_hz=s.regional_rates(r),source=str(path) if found else None)
        rows.append(row)
        if len(rows)%5==0:
            print('point',index,'/',len(J),j,'positive',row['positive_mode_found'],'roots',[q['lambda_per_ms'] for q in roots],flush=True)
            write(dest/'result.json',dict(status='RUNNING',rows=rows))
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,branch_points=len(J),spectral_points=len(rows),
        unresolved_positive_search=[q['index'] for q in rows if q['index']>=113 and not q['positive_mode_found']],
        meaning='At least one numerically refined positive-growth mode establishes instability at each certified point. Onset spectra follow two modes and use separate full-determinant counts. Not a Floquet computation.'))

if __name__=='__main__':main()
