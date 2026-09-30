"""Reuse exact saved Arnoldi products to estimate a transverse return operator.

J(F^n)Q=Qplus H was already measured in the full native model. Only a few
additional native/tangent steps are needed for each polynomial return. The
result is a projected candidate operator at the old reference, never a
substitute for a corrected orbit and convergence checks.
"""
from generalized_return_spectrum import *
from scipy.linalg import null_space


def run(a):
    spec=read(a.spectrum/'config.json');source=Path(spec['source']);cfg=read(source/'config.json')
    assert read(a.endpoint/'config.json')['resumed_from']==str(source.resolve())
    folder=OUT/'cached_normal_projections'/a.label;folder.mkdir(parents=True,exist_ok=False)
    storage=Path('/data/hfosp/topic4_sef_hfo/kinetic_population_bifurcation_20260916/cached_normal_projections')/a.label
    storage.mkdir(parents=True,exist_ok=False)
    H=np.load(a.spectrum/'arnoldi.npz')['H'];dim=H.shape[1];qfiles=[Path(spec['krylov_storage'])/f'q{i:03d}.npy' for i in range(dim+1)]
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);x=coords.pack(m);initial_step=m.step_index
    m.advance_step();first=coords.pack(m);m.advance_step();second=coords.pack(m)
    normal=(-3*x+4*first-second)/(2*DT);normal/=cp.linalg.norm(normal);del first,second
    nq=[]
    for p in qfiles[:dim]:
        q=cp.asarray(np.load(p,mmap_mode='r'));nq.append(float(cp.dot(normal,q).get()));del q
    C=null_space(np.asarray(nq)[None,:]);assert np.max(abs(np.asarray(nq)@C))<1e-12
    m.restore(a.endpoint);assert m.step_index-initial_step==spec['steps'];end=capture(m)
    frames=[];values=[]
    for step in range(a.order+1):
        if step:m.advance_step()
        d=coords.pack(m)-x;frames.append(d);values.append(float(cp.dot(normal,d).get()))
    alpha=crossing(values,a.order);c=coefficients(alpha,a.order);dc=coefficient_derivatives(alpha,a.order)
    residual=sum(float(ci)*d for ci,d in zip(c,frames));chord=sum(float(ci)*d for ci,d in zip(dc,frames))
    closure=float(cp.linalg.norm(residual).get());denominator=float(cp.dot(normal,chord).get());assert denominator>0
    del frames,residual
    t=NetworkTangent(m);B=np.zeros((dim,dim));yy=np.zeros((dim,dim));yfiles=[];rows=[];started=time.time()
    write(folder/'config.json',dict(spectrum=str(a.spectrum),source=str(source),endpoint=str(a.endpoint),
        D=cfg['D'],order=a.order,phase_fraction=alpha,full_native_steps=spec['steps'],storage=str(storage),
        scope='Projected normal-return operator at uncorrected source; no Floquet/stability/bifurcation acceptance'))
    for j in range(dim):
        for k,v in end.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=v
        m.step_index=end['step_index'];image=cp.zeros(coords.size)
        for i,p in enumerate(qfiles):
            if H[i,j]:
                q=cp.asarray(np.load(p,mmap_mode='r'));image+=float(H[i,j])*q;del q
        coords.unpack(image,t,project=True);y=float(c[0])*coords.pack(t);del image
        for k in range(1,a.order+1):t.advance();y+=float(c[k])*coords.pack(t)
        y-=chord*cp.dot(normal,y)/denominator
        for i,p in enumerate(qfiles[:dim]):
            q=cp.asarray(np.load(p,mmap_mode='r'));B[i,j]=float(cp.dot(q,y).get());del q
        for i,p in enumerate(yfiles):
            old=cp.asarray(np.load(p,mmap_mode='r'));yy[i,j]=yy[j,i]=float(cp.dot(old,y).get());del old
        yy[j,j]=float(cp.dot(y,y).get());target=storage/f'Gq{j:03d}.npy';np.save(target,cp.asnumpy(y));yfiles.append(target)
        row=dict(column=j,image_norm=float(cp.linalg.norm(y).get()),section_residual=float(cp.dot(normal,y).get()))
        rows.append(row);write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),columns=rows,wall_s=time.time()-started));print(a.label,row,flush=True);del y
    matrix=C.T@B@C;vals,vec=np.linalg.eig(matrix);order=np.argsort(-abs(vals));vals=vals[order];vec=vec[:,order]
    eigrows=[]
    for k,value in enumerate(vals):
        coeff=C@vec[:,k];norm=coeff.conj()@coeff
        r2=(coeff.conj()@yy@coeff-value*(coeff.conj()@B.T@coeff)-value.conjugate()*(coeff.conj()@B@coeff)+abs(value)**2*norm).real
        eigrows.append(dict(eigenvalue=[float(value.real),float(value.imag)],modulus=float(abs(value)),
            full_operator_residual_from_Gram=float(np.sqrt(max(0.,r2/norm.real)))))
    np.savez_compressed(folder/'operator.npz',H=H,Q_dot_GQ=B,GQ_dot_GQ=yy,section_null_basis=C,normal_matrix=matrix,
        eigenvalues=vals,eigenvectors_in_Q=C@vec,normal_dot_Q=nq)
    write(folder/'result.json',dict(status='PROJECTED_NORMAL_OPERATOR_COMPLETE',source=str(source),D=cfg['D'],
        source_full_transverse_return_residual=closure,phase_fraction=alpha,rows=eigrows,projection_dimension=C.shape[1],
        input_section_constraint_residual=float(np.max(abs(np.asarray(nq)@C))),wall_s=time.time()-started,
        scope='Candidate normal operator from measured full-map products, plus differentiated original partial steps. Does not certify the source curve or its stability.'))
    print(eigrows,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spectrum',type=Path,required=True);ap.add_argument('--endpoint',type=Path,required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--order',type=int,choices=[3,5],default=5)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
