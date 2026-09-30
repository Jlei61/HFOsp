"""Check synchronized equations and compare selected roots, without relabeling old spectra."""
from model_zm import *
import argparse
from scipy.sparse.linalg import eigs


def states():
    result = []
    for folder, nums, labels in [
        ('D_arclength_lower/refined_folds', [1,2], ['SN1','SN2']),
        ('D_arclength_upper_focused/refined_folds_stable_response',[1,2,3,4],['SN3','SN4','SN5','SN6']),
        ('D_gap_lower_guarded/verified_outer_fold',[31],['SN7'])]:
        for num, label in zip(nums,labels):
            result.append((label,OLD/'g20'/folder/f'fold{num}.npz'))
    result += [('middle',OLD/'g20/D_gap_middle_down/point0000.npz'),
               ('upper_D0228',OLD/'g20/D_arclength_upper_onset_range_v4/point0078.npz')]
    return result


def refine(s,r,D,lam,v=None,method='rate_dde',tol=1e-9):
    if v is None:
        ev,vec=eigs(s.characteristic(r,D,lam,method),k=4,sigma=0,tol=1e-10)
        v=vec[:,np.argmin(abs(ev))]
    pivot=np.argmax(abs(v));v=v/v[pivot]
    c=sparse.csr_matrix((np.ones(1),([0],[pivot])),shape=(1,s.P))
    for it in range(24):
        A=s.characteristic(r,D,lam,method);f=A@v;err=np.linalg.norm(f)/np.linalg.norm(v)
        if err<tol:return lam,v/np.linalg.norm(v),float(err)
        h=1e-6;d=(s.characteristic(r,D,lam+h,method)-s.characteristic(r,D,lam-h,method))/(2*h)
        B=sparse.bmat([[A,sparse.csr_matrix((d@v)[:,None])],[c,sparse.csr_matrix((1,1))]],format='csc')
        change=spsolve(B,np.r_[-f,0j])
        for step in 2.**-np.arange(12):
            nl=lam+step*change[-1];nv=v+step*change[:-1]
            if abs(nl)>.6 or abs(nl+.001)<1e-7:continue
            if np.linalg.norm(s.characteristic(r,D,nl,method)@nv)/np.linalg.norm(nv)<err:
                lam,v=nl,nv;break
        else:return None
    return None


def audit(s):
    rows=[]
    for label,path in states():
        z=np.load(path);r=z['r'];D=float(z['D']);s.set_D(D)
        y=s.state(r);arr=np.array([a@r for a in s.matrices(1.)]);C=s.characteristic(r,D,0.)
        delta=C+s.jacobian(r,D)
        q=dict(label=label,source=str(path),D=D,global_E_hz=s.global_rate(r),
            static_residual_hz=float(abs(s.residual(r,D)).max()*1000),
            frozen_Z_rhs_residual=float(abs(s.rhs(y,arr)).max()),
            dc_characteristic_jacobian_error=float(abs(delta.data).max()) if delta.nnz else 0.)
        if 'v' in z and 'w' in z:
            h=1e-6;d=(s.characteristic(r,D,h)-s.characteristic(r,D,-h))/(2*h)
            q['fold_temporal_zero_derivative']=float(np.real(z['w']@(d@z['v'])))
        rows.append(q)
    # D=0 must retain the shared nine-state vector field exactly.
    base=RateField();r,ok,_=base.solve(1.);assert ok
    s.set_D(0);y=s.state(r);arr=np.array([a@r for a in s.matrices(1.)])
    baseline=float(abs(base.rhs(y[:9],arr)-s.rhs(y,arr)[:9]).max())
    gpu=[]
    for D,dynamic in [(0.,False),(.2,False),(.2,True)]:
        s.set_D(D);y=s.state(r);y[3]*=1.2;y[5]*=.9
        e=ZMIntegrator(s,initial=y,dynamic_z=dynamic);e.arrivals(0)
        e.k['rhs'](((s.P+127)//128,),(128,),(e.y,e.arr,e.pars,e.f))
        cpu=s.rhs(y,e.arr.get(),dynamic_z=dynamic)
        gpu.append(dict(D=D,dynamic_Z=dynamic,maximum_rhs_error=float(abs(cpu-e.f.get()).max())))
        for _ in range(100):e.step()
        end=e.y.get();assert np.isfinite(end).all() and end[9].min()>=0 and end[9].max()<=1
    result=dict(status='PASS',rows=rows,Z1_shared_rhs_error=baseline,cpu_gpu=gpu,
        scope='RHS identity, unchanged equilibrium equations and zero-frequency characteristic, physical Z, GPU implementation; not SNN validation')
    assert max(q['static_residual_hz'] for q in rows)<1e-6
    assert max(q['frozen_Z_rhs_residual'] for q in rows)<1e-7
    assert max(q['dc_characteristic_jacobian_error'] for q in rows)<1e-9
    assert baseline<1e-12 and max(q['maximum_rhs_error'] for q in gpu)<1e-7
    write(DEST/'equation_checks.json',result);print('CHECKS PASS',flush=True)


def spectrum(s):
    # Same special-function equation, faster stable evaluator already audited
    # in v1. Preserve the frozen source and declare the numerical substitution.
    sys.path.insert(0,str(ROOT/'scripts/topic4_zm_onset_bifurcation'))
    from response_transport import white_transport
    import response
    response.white=white_transport
    folder=DEST/'critical_spectra';folder.mkdir(exist_ok=True)
    rows=read(folder/'result.json')['rows'] if (folder/'result.json').exists() else []
    for label,path in states():
        if any(q['label']==label for q in rows):continue
        z=np.load(path);r=z['r'];D=float(z['D']);q=dict(label=label,D=D,global_E_hz=s.global_rate(r),models={})
        for method in ['shifted_white','calibrated_full','rate_dde']:
            roots=[]
            for initial in ([.01+.028j,.015+.04j] if label in ['SN1','SN2'] else [.015+.13j,.015+.19j,.01+.04j]):
                try:found=refine(s,r,D,initial,method=method)
                except (RuntimeError,ValueError,FloatingPointError):found=None
                if found is None:continue
                lam,v,err=found
                if lam.imag<0:lam,v=lam.conjugate(),v.conjugate()
                if any(abs(lam-complex(*a['lambda_per_ms']))<1e-6 for a in roots):continue
                energy=s.sizes*abs(v)**2;energy/=energy.sum()
                item=dict(lambda_per_ms=[lam.real,lam.imag],frequency_hz=lam.imag*1000/(2*np.pi),residual=err,
                    E_energy=[float(energy[s.E&(s.geo['group_region']==k)].sum()) for k in range(3)],I_energy=float(energy[~s.E].sum()))
                if method=='rate_dde':
                    dy=s.eigenstate(r,D,lam,v);y=s.state(r);arr=np.array([a@r for a in s.matrices(1.)]);da=np.array([a@v for a in s.matrices(1.,lam)])
                    eps=1e-5/max(abs(dy).max(),1e-12);fd=[]
                    for part in [np.real,np.imag]:fd.append((s.rhs(y+eps*part(dy),arr+eps*part(da))-s.rhs(y-eps*part(dy),arr-eps*part(da)))/(2*eps))
                    defect=np.linalg.norm(fd[0]+1j*fd[1]-lam*dy)
                    if abs(lam)>1e-7:
                        error=defect/np.linalg.norm(lam*dy)
                        item['rhs_tangent_relative_error']=float(error)
                        assert error<1e-4,error
                    else:
                        # At a fold lambda is zero: a relative error divided
                        # by lambda is undefined. Check the generator defect.
                        error=defect/np.linalg.norm(dy)
                        item['zero_mode_generator_defect_per_ms']=float(error)
                        assert error<1e-6,error
                np.savez_compressed(folder/f'{label}_{method}_{len(roots)}.npz',r=r,D=D,lam=lam,vector=v)
                roots.append(item)
            q['models'][method]=dict(roots=roots,stability='UNSTABLE' if any(a['lambda_per_ms'][0]>1e-7 for a in roots) else 'UNRESOLVED',full_spectrum=False)
            print(label,method,[(a['lambda_per_ms'],a['frequency_hz']) for a in roots],flush=True)
        rows.append(q);write(folder/'result.json',dict(status='RUNNING',rows=rows))
    write(folder/'result.json',dict(status='COMPLETE',rows=rows,numerical_evaluator='v1 response_transport: same parabolic-cylinder equation',
        scope='Selected modes; methods are separate dynamic closures, not interchangeable stability evidence'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spectrum',action='store_true');a=p.parse_args();s=ZMSpatialRate()
    spectrum(s) if a.spectrum else audit(s)
