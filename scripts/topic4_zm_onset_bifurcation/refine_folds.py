"""Refine turning brackets on the same equilibrium curve and test SN conditions."""
from continue_D import *
import argparse


def main(a):
    s=ZMRate(a.grid,mode=a.mode,m_current=a.m_current)
    parent=DEST/f'g{a.grid}'/a.branch;info=read(parent/'result.json')
    dest=parent/a.output;dest.mkdir(exist_ok=True);rows=[]
    for number,(i,j) in enumerate(info['fold_brackets']):
        if a.fold_index is not None and number!=a.fold_index:continue
        L=dict(np.load(parent/f'point{i:04d}.npz'));R=dict(np.load(parent/f'point{j:04d}.npz'))
        xl=np.r_[L['r']/RS,float(L['D'])/DS];xr=np.r_[R['r']/RS,float(R['D'])/DS]
        tl=L['tangent'];tr=R['tangent']
        for k in range(36):
            chord=xr-xl;chord/=np.linalg.norm(chord)
            xm,ok,nit=correct(s,(xl+xr)/2,chord)
            if not ok:raise RuntimeError(('fold refinement',number,k))
            r=xm[:-1]*RS;D=float(xm[-1]*DS);tm=tangent(s,r,D,tl)
            if abs(tm[-1])<1e-10 or np.linalg.norm(xr-xl)<1e-9:break
            if tm[-1]*tl[-1]>0:xl=xm;tl=tm
            else:xr=xm;tr=tm
        J=s.jacobian(r,D)
        ev,V=eigs(J,k=3,sigma=0,tol=1e-12);select=np.argmin(abs(ev));v=V[:,select].real;v/=np.linalg.norm(v)
        ew,W=eigs(J.T,k=3,sigma=0,tol=1e-12);w=W[:,np.argmin(abs(ew))].real;w/=w@v
        fd=s.parameter_derivative(r,D);trans=float(w@fd);quadratic=[];timescale=[]
        for h in [2e-5,1e-5,5e-6]:
            quadratic.append(float(w@((s.jacobian(r+h*v,D)-s.jacobian(r-h*v,D))@v)/(2*h)))
        for h in [1e-5,3e-6,1e-6]:
            Cp=s.characteristic(r,D,h);Cm=s.characteristic(r,D,-h)
            timescale.append(float(np.real(w@((Cp-Cm)@v)/(2*h))))
        total=s.sizes*abs(v)**2;total*=s.E
        participation=[float(total[s.geo['group_region']==region].sum()/total.sum()) for region in range(3)]
        second=float(np.sort(abs(ev))[1]);res=float(np.max(abs(s.residual(r,D))))
        q=np.array(quadratic);tc=np.array(timescale)
        passed=(res<1e-9 and abs(ev[select])<1e-7 and second>1e-5 and abs(trans)>1e-8
            and abs(np.mean(q))>1e-6 and np.ptp(q)<.02*abs(np.mean(q))
            and abs(np.mean(tc))>1e-5 and np.ptp(tc)<.05*abs(np.mean(tc)))
        row=dict(number=number+1,D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
            equilibrium_residual_hz=res*1000,zero_eigenvalue=ev[select],next_static_eigenvalue_distance=second,
            null_residual=float(np.linalg.norm(J@v)),transversality_w_FD=trans,quadratic_w_Frr_vv=quadratic,
            temporal_zero_derivative_w_Clambda_v=timescale,mode_E_energy_A_B_surround=participation,
            critical_type='SN' if passed else 'TURNING_POINT_NOT_YET_VERIFIED',
            scope='Conditional prescribed-Z equilibrium with '+a.mode+'; not established as onset of global bursting')
        np.savez_compressed(dest/f'fold{number+1}.npz',r=r,D=D,Z=s.Z,v=v,w=w,eigenvalues=ev)
        rows.append(row);write(dest/'result.json',dict(status='COMPLETE' if a.fold_index is not None or number==len(info['fold_brackets'])-1 else 'RUNNING',rows=rows))
        print(row,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--branch',default='D_arclength_lower')
    p.add_argument('--mode',choices=['dynamic_M','frozen_M'],default='dynamic_M');p.add_argument('--m-current',type=float,default=0.)
    p.add_argument('--output',default='refined_folds_stable_response')
    p.add_argument('--fold-index',type=int)
    main(p.parse_args())
