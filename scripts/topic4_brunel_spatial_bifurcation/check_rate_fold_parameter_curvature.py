"""Check cycle-fold curvature from separately solved neighbouring cycles.

Differentiate J along the mean-rate coordinate, rather than subtracting two
nearly zero tangent components. Keep the original tangent-derived estimate
and require both step-size and temporal-grid agreement before using this
independent result. This diagnostic alone never certifies a cycle fold.
"""
from rate_periodic import *
import gc


def check(label, resolutions, steps, device):
    s=RateField(); folder=PERIODIC_OUT/'curvature_rechecks';folder.mkdir(exist_ok=True)
    output=folder/(label+'.json'); rows=[]
    for N in resolutions:
        root=read(PERIODIC_OUT/f'{label}_N{N}.json')
        assert root['coordinate'] in ['core_A_mean_Hz','core_B_mean_Hz']
        z=np.load(root['orbit']); r=z['r'];T=float(z['T']);J=float(z['J'])
        core='AB'.index(root['coordinate'][5]); mask=s.E&(s.geo['group_region']==core)
        weights=s.geo['group_size']*mask;weights=weights/weights.sum()
        c=np.r_[np.tile(weights/N,N),0.,0.]
        y=np.r_[(r*1000).ravel(),np.log(T),J*1000]
        tangent=np.load(PERIODIC_OUT/f'{label}_tangent_N{N}.npz')['tangent']
        assert abs(c@tangent-1)<1e-6
        o=Periodic(s,N,device);cp=o.cp
        o.low_memory=True
        bank=(N//2+1)*sum(len(v[0]) for v in s.raw)*16
        free=int(cp.cuda.runtime.memGetInfo()[0])
        o.stream_harmonics=free<2*bank+int(2.5*1024**3)
        print('CURVATURE OPERATOR STORAGE',dict(N=N,streamed=o.stream_harmonics,
            free_bytes=free,bank_bytes=bank,equations_changed=False),flush=True)
        o.harmonic_chunk_size=32;o.derivative_chunk_size=32
        o.normalize_linear_rhs=True;o.host_krylov=True;o.linear_target_aware=True
        o.krylov_restart=160
        for h in steps:
            neighbours=[]
            for sign in [-1,1]:
                name=f'{label}_curvature_N{N}_h{h:g}_{"minus" if sign<0 else "plus"}'
                meta=PERIODIC_OUT/'orbits'/(name+'.json')
                if meta.exists():
                    saved=read(meta); rr=np.load(saved['path'])['r']
                    TT=saved['T_ms'];JJ=saved['J_EE_core'];err=saved['residual_hz']
                else:
                    pred=y+sign*h*tangent
                    seed_kind='root tangent'
                    # Reuse already solved neighbours on the preceding
                    # temporal grid; the new full BVP residual still decides
                    # acceptance on this grid and at this exact coordinate.
                    lower=[n for n in resolutions if n<N]
                    coarse=(PERIODIC_OUT/'orbits'/f'{label}_curvature_N{max(lower)}_h{h:g}_{"minus" if sign<0 else "plus"}.npz') if lower else None
                    if coarse is not None and coarse.exists():
                        prior=np.load(coarse)
                        pred=np.r_[(resample(prior['r'],N,axis=0)*1000).ravel(),
                            np.log(float(prior['T'])),float(prior['J'])*1000]
                        lower_N=max(lower)
                        lower_root=read(PERIODIC_OUT/f'{label}_N{lower_N}.json')
                        lower_z=np.load(lower_root['orbit'])
                        lower_y=np.r_[(resample(lower_z['r'],N,axis=0)*1000).ravel(),
                            np.log(float(lower_z['T'])),float(lower_z['J'])*1000]
                        lower_tangent=np.load(PERIODIC_OUT/f'{label}_tangent_N{lower_N}.npz')['tangent']
                        lower_tangent=np.r_[resample(lower_tangent[:-2].reshape(lower_N,s.P),N,axis=0).ravel(),lower_tangent[-2:]]
                        # Correct the known mesh error at the root and its
                        # first derivative before the unchanged BVP solve.
                        pred+=y-lower_y+sign*h*(tangent-lower_tangent)
                        seed_kind='preceding grid with root and tangent mesh correction'
                    else:
                        for larger in sorted(v for v in steps if v>h):
                            pair=[PERIODIC_OUT/'orbits'/f'{label}_curvature_N{N}_h{larger:g}_{side}.npz'
                                  for side in ['minus','plus']]
                            if not all(f.exists() for f in pair):continue
                            ym,yp=[np.r_[(v['r']*1000).ravel(),np.log(float(v['T'])),float(v['J'])*1000]
                                   for v in [np.load(f) for f in pair]]
                            pred=y+sign*h*(yp-ym)/(2*larger)+h*h*(yp+ym-2*y)/(2*larger**2)
                            seed_kind='solved symmetric neighbours';break
                    pred+=c*((c@y+sign*h)-c@pred)/(c@c)
                    print('CURVATURE SEED',N,h,sign,seed_kind,flush=True)
                    arc=(pred,c,np.ones_like(c))
                    rr,TT,JJ,err,history=o.solve(pred[:-2].reshape(N,s.P)/1000,
                        np.exp(pred[-2]),pred[-1]/1000,arc=arc,tol=5e-12,maxiter=16)
                    assert err<5e-12,dict(N=N,h=h,sign=sign,residual=err)
                    path=save_orbit(s,rr,TT,JJ,err,history,name);saved=read(path.with_suffix('.json'))
                coordinate=float(rr.mean(0)@weights*1000)
                assert abs(coordinate-(c@y+sign*h))<1e-10
                neighbours.append(dict(sign=sign,orbit=saved['path'],J_EE_core=JJ,T_ms=TT,
                    coordinate=coordinate,residual_Hz=err))
            jm,jp=[q['J_EE_core'] for q in neighbours]
            row=dict(N=N,root_source=str(PERIODIC_OUT/f'{label}_N{N}.json'),
                root_orbit=root['orbit'],J_EE_core=J,coordinate=root['coordinate'],h_Hz=h,
                curvature=(jp+jm-2*J)/h**2,central_slope=(jp-jm)/(2*h),
                tangent_curvature=root['d2J_dcoordinate2'],neighbours=neighbours)
            rows.append(row);print('PARAMETER CURVATURE',row,flush=True)
            write(output,dict(status='RUNNING',label=label,pid=os.getpid(),rows=rows))
        del o;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    fine=[q for q in rows if q['N']==resolutions[-1]]
    coarse=[q for q in rows if q['N']==resolutions[-2]]
    mesh=abs(fine[-1]['curvature']-coarse[-1]['curvature'])/abs(fine[-1]['curvature'])
    step=max(abs(v[-1]['curvature']-v[-2]['curvature'])/abs(v[-1]['curvature'])
             for v in [coarse,fine])
    passed=mesh<.01 and step<.01 and min(abs(q['curvature']) for q in rows)>1e-8
    result=dict(status='CURVATURE_CHECKED' if passed else 'CURVATURE_UNRESOLVED',
        label=label,rows=rows,mesh_relative_change=mesh,step_relative_change=step,
        curvature=fine[-1]['curvature'],criteria=dict(mesh_relative_change=.01,
        step_relative_change=.01,nonlinear_residual_Hz=5e-12),
        method='Central second difference of J on independently solved full-space cycles; same core-mean coordinate, two temporal meshes, three coordinate steps.',
        scope='Curvature diagnostic only. Root, physical waveform, non-phase null tangent and independent variational checks are still required. Original tangent-derived estimates are preserved.')
    write(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label')
    p.add_argument('--N',type=int,nargs='+',default=[1024,2048])
    p.add_argument('--h',type=float,nargs='+',default=[.0002,.0001,.00005])
    p.add_argument('--device',type=int,default=0);a=p.parse_args()
    assert len(a.N)>=2 and a.N==sorted(set(a.N))
    assert len(a.h)>=3 and a.h==sorted(set(a.h),reverse=True) and min(a.h)>0
    check(a.label,a.N,a.h,a.device)
