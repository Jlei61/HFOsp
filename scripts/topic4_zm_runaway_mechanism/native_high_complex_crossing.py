"""Track one independently verified high-branch complex instability.

Reports a spectral crossing only. No nonlinear Hopf criticality, unique first
instability, native onset, or small periodic branch is inferred from it.
"""
from native_path import *
from equilibrium_spectrum import cache_characteristic
from root_count_v3 import refine_root


def root_at(D,r,lam,v):
    s=model();attach_native_path(s);s.set_D(D)
    r,ok,tr=s.solve(r);assert ok and tr[-1]*1000<2e-8,tr[-1]
    cache_characteristic(s,r);ans=refine_root(s,r,lam,v=v,tol=1e-11)
    assert ans is not None
    value,vector,error=ans
    overlap=float(abs(np.vdot(v,vector))/np.linalg.norm(v)/np.linalg.norm(vector))
    assert overlap>.8 and abs(value-lam)<.05,(D,overlap,value,lam)
    assert value.imag>1e-4
    return s,r,value,vector,error,overlap


def main():
    src=OUT/'equilibria/native_high_descent';dest=OUT/'equilibria/native_high_descent_audit'
    seed=np.load(dest/'point0118_root2.npz');r=seed['r'];D=float(seed['D'])
    lam=complex(seed['lambda_per_ms']);v=seed['v'];assert lam.real>0
    rows=[dict(index=118,D=D,lambda_per_ms=[lam.real,lam.imag])];left=(D,r,lam,v)
    for index in range(117,95,-1):
        z=np.load(src/f'point{index:04d}.npz')
        s,r,lam,v,error,overlap=root_at(float(z['D']),z['r'],lam,v);D=s.D
        rows.append(dict(index=index,D=D,lambda_per_ms=[lam.real,lam.imag],residual=error,mode_overlap=overlap))
        write(dest/'complex_crossing.json',dict(status='TRACKING',rows=rows))
        log('COMPLEX ROOT TRACK',rows[-1])
        right=(D,r,lam,v)
        if lam.real*left[2].real<0:break
        left=right
    else:
        write(dest/'complex_crossing.json',dict(status='NO_CROSSING_IN_SEGMENT',rows=rows));return
    original_bracket=[left[0],right[0]];refinement=[]
    for k in range(40):
        mid=(left[0]+right[0])/2
        near=min([left,right],key=lambda x:abs(x[0]-mid))
        s,r,lam,v,error,overlap=root_at(mid,(left[1]+right[1])/2,near[2],near[3])
        refinement.append(dict(D=mid,lambda_per_ms=[lam.real,lam.imag],residual=error,mode_overlap=overlap))
        if abs(lam.real)<1e-10:break
        if lam.real*left[2].real>0:left=(mid,r,lam,v)
        else:right=(mid,r,lam,v)
    assert abs(lam.real)<1e-10
    center=(mid,r,lam,v);slopes=[];step=min(1e-5,abs(original_bracket[1]-original_bracket[0])*.05)
    for h in [step,step/2]:
        plus=root_at(mid+h,r,lam,v);minus=root_at(mid-h,r,lam,v)
        slopes.append(float((plus[2].real-minus[2].real)/(2*h)))
    from scipy.sparse.linalg import eigs
    C=s.characteristic(r,lam);ev=eigs(C,k=4,sigma=0,return_eigenvectors=False,tol=1e-11)
    distances=sorted(float(abs(x)) for x in ev)
    energy=s.E*s.sizes*abs(v)**2;energy/=energy.sum()
    q=dict(status='COMPLEX_ROOT_CROSSING_REFINED',D=mid,global_Z=1-mid,
        global_E_hz=s.global_rate(r),lambda_per_ms=[lam.real,lam.imag],frequency_hz=lam.imag*1000/(2*np.pi),
        equilibrium_residual_hz=float(abs(s.residual(r)).max()*1000),characteristic_residual=error,
        crossing_slopes=slopes,crossing_step_sizes=[step,step/2],zero_matrix_eigenvalue_distances=distances,
        mode_energy_A_B_surround=[float(energy[s.geo['group_region']==i].sum()) for i in range(3)],
        rows=rows,original_bracket=original_bracket,refinement=refinement,
        nonlinear_Hopf_criticality='NOT_ESTABLISHED',other_critical_modes='NOT_EXCLUDED',
        scope='One complex mode crossing on the extended nativeZ high-equilibrium branch, heldZ and dynamicM. Not a native-onset bifurcation; not a claim that this is the first equilibrium instability.')
    np.savez_compressed(dest/'complex_crossing.npz',r=r,D=mid,Z=s.Z,lambda_per_ms=lam,v=v)
    write(dest/'complex_crossing.json',q);log('COMPLEX CROSSING',q)


if __name__=='__main__':main()
