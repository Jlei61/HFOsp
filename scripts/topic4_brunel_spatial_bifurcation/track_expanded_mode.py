"""Track one confirmed unstable mode along the upper spatial branch."""
from common import *
from model import SpatialBrunel
from contour_roots import refine

def main():
    s=SpatialBrunel(response='calibrated_full');base=OUT/'expanded';dest=base/'persistent_mode';dest.mkdir(exist_ok=True)
    q=np.load(base/'modes/upper_J1.3/modes.npz');idx=np.argmax(q['roots'].real)
    lam=q['roots'][idx];v=q['vectors'][idx];r=q['rates'];rows=[]
    arc=read(base/'high_arclength_v2/result.json')['rows']
    # The already continued upper arc supplies a predictor when fixed-J
    # Newton cannot cross a sharply curved portion in one parameter step.
    arc=[x for x in arc if x['J_EE_core']>=1.3]
    for J in np.round(np.arange(1.3,2.001,.025),8):
        previous_v=v.copy()
        cached=dest/f'J{J:.6f}.npz'
        if cached.exists():
            old=np.load(cached);r=old['rates'];lam=complex(old['root']);v=old['vector']
        r,ok,tr=s.solve(float(J),r)
        if not ok:
            near=min(arc,key=lambda x:abs(x['J_EE_core']-J))
            seed=np.load(base/'high_arclength_v2'/f'step{near["step"]:04d}.npz')['rates']
            r,ok,tr=s.solve(float(J),seed)
        if not ok:
            write(dest/'result.json',dict(status='BOUNDED_STOP',stop_reason='stationary Newton failed',failed_J=J,rows=rows))
            return
        result=refine(s,r,float(J),lam,v);assert result is not None,J
        lam,newv,err=result
        overlap=abs(np.vdot(previous_v,newv))/(np.linalg.norm(previous_v)*np.linalg.norm(newv));v=newv
        row=dict(J_EE_core=J,lambda_per_ms=lam,frequency_hz=lam.imag*1000/(2*np.pi),residual=err,rates_hz=s.regional_rates(r),previous_vector_overlap=overlap)
        rows.append(row);print(row,flush=True)
        np.savez_compressed(dest/f'J{J:.6f}.npz',rates=r,J=J,vector=v,root=lam)
        write(dest/'result.json',dict(status='RUNNING',rows=rows))
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,
        positive_at_all_sampled_points=all(x['lambda_per_ms'].real>0 for x in rows),
        meaning='A continuously tracked positive-growth spatial mode of the response approximation; high-rate local response calibration remains unvalidated'))

if __name__=='__main__':main()
