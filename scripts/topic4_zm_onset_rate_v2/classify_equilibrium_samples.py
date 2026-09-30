"""Reclassify legacy equilibrium geometry with the synchronized rate DDE.

Negative det C(0) proves a positive real characteristic root since
det[I-H(lambda)K(lambda)] tends to +1 as real lambda tends to infinity.
Positive det C(0) does NOT establish stability; then seek an actual RHP root.
"""
from audit_and_spectrum import refine
from equilibrium_root_count import parity
from model_zm import *
from scipy.sparse.linalg import splu

BRANCHES=['D_arclength_lower','D_arclength_upper_focused','D_arclength_upper_onset_range',
 'D_arclength_upper_onset_range_v2','D_arclength_upper_onset_range_v3','D_arclength_upper_onset_range_v4',
 'D_gap_lower_guarded','D_gap_middle_down','D_gap_middle_up','D_gap_middle_to_low']


def main():
    out=DEST/'periodic_completion/equilibrium_classification';out.mkdir(exist_ok=True)
    s=ZMSpatialRate()
    for branch in BRANCHES:
        allrows=read(OLD/'g20'/branch/'result.json')['rows'];step=max(1,len(allrows)//60)
        indices=sorted(set(list(range(0,len(allrows),step))+[len(allrows)-1]));rows=[];previous=None
        for j in indices:
            old=allrows[j];path=OLD/'g20'/branch/f'point{old["index"]:04d}.npz';z=np.load(path);r=z['r'];D=float(z['D'])
            C=s.characteristic(r,D,0.);lu=splu(C.tocsc());diag=lu.U.diagonal()
            sign=float(np.prod(np.sign(diag.real))*(-1)**(parity(lu.perm_r)+parity(lu.perm_c)))
            q=dict(index=old['index'],source=str(path),D=D,global_E_hz=s.global_rate(r),determinant_C0_sign=sign,status='UNRESOLVED')
            if sign<0:q.update(status='UNSTABLE',evidence='Negative zero-frequency determinant and positive determinant at real lambda infinity imply a real RHP root')
            else:
                seeds=[]
                if previous is not None:seeds.append(previous)
                seeds.append((.017+.03j if q['global_E_hz']<5 else .015+.14j,None))
                if q['global_E_hz']>=5:seeds.append((.01+.18j,None))
                for lam,v in seeds:
                    try:found=refine(s,r,D,lam,v)
                    except (ValueError,RuntimeError,FloatingPointError):found=None
                    if found is None:continue
                    lam,v,err=found
                    if lam.imag<0:lam=lam.conjugate();v=v.conjugate()
                    if lam.real>1e-7:
                        previous=(lam,v);q.update(status='UNSTABLE',lambda_per_ms=lam,characteristic_residual=err,evidence='Positive real part of a refined rate-DDE characteristic root');break
            rows.append(q)
            if len(rows)%10==0:print(branch,len(rows),'/',len(indices),'unstable',sum(a['status']=='UNSTABLE' for a in rows),flush=True)
            write(out/f'{branch}.json',dict(branch=branch,rows=rows,total_geometry_points=len(allrows),sample_spacing=step))


if __name__=='__main__':main()
