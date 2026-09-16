#!/usr/bin/env python3
"""Pseudo-arclength continuation with a moving phase condition, exact v1 map."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator,gmres
from topic4_fig5_z_frozen_v1 import Orbit,OUT as PRIOR,BASE
from topic4_fig5_z_cycle_preconditioner import bordered_inverse

OUT=BASE.parent/'fig5_z_branch_extension_20260915'


class Arc:
    def __init__(self,N,gpu=None):
        self.N=N;self.o=Orbit(-.08,N);m=self.o.m
        self.gpu=gpu
        weights=np.r_[m.unit_count,m.count_i];weights=weights/weights.sum()/N/100
        self.weight=np.r_[np.tile(weights,N),1.,100.]

    def norm(self,x):return float(np.sqrt(np.dot(x*x,self.weight)))
    def pack(self,r,T,s):return np.r_[(r*1000).ravel(),np.log(T),s]
    def unpack(self,y):return y[:-2].reshape(self.N,-1)/1000,float(np.exp(y[-2])),float(y[-1])

    def correct(self,pred,tangent,ref,maxiter=10):
        o=self.o;rr,_,_=self.unpack(ref)
        dref=np.fft.irfft(2j*np.pi*o.k[:,None]*np.fft.rfft(rr*1000,axis=0),n=self.N,axis=0)
        pp=dref/np.sum(dref*dref);ap=(self.weight[:-2]*tangent[:-2]).reshape(self.N,-1)
        at,ass=self.weight[-2:]*tangent[-2:];row=self.weight*tangent;y=pred.copy();history=[]
        for it in range(maxiter):
            r,T,s=self.unpack(y);o.s=s;o.z,o.z2=o.eq.z(s)
            f,jr=o.evaluate_fixed(r,T,True);phase=np.sum((r-rr)*1000*pp);arc=np.dot(y-pred,row)
            F=np.r_[(f*1000).ravel(),phase,arc];error=float(abs(F).max());history.append(error)
            print('ARC ITER',it,'s',s,'T',T,'res',error,flush=True)
            if getattr(self,'checkpoint',None):
                np.savez_compressed(self.checkpoint,r=r,T=T,s=s,residual_hz=error,N=self.N,history=history)
            if error<1e-6:return y,error,history
            eps=1e-5;o.z,o.z2=o.eq.z(s+eps);fp=o.evaluate_fixed(r,T);o.z,o.z2=o.eq.z(s-eps);fm=o.evaluate_fixed(r,T);o.z,o.z2=o.eq.z(s)
            fs=(fp-fm)*1000/(2*eps);ft=jr.period_derivative*1000
            dlin=jr;gpu_linear=None
            if self.gpu is not None:
                from topic4_fig5_z_cycle_gpu import GpuLinear
                gpu_linear=GpuLinear(o,jr,self.gpu);dlin=gpu_linear
            def mv(v):
                dv=v[:-2].reshape(self.N,-1)
                return np.r_[(dlin(dv/1000)*1000+ft*v[-2]+fs*v[-1]).ravel(),np.sum(dv*pp),np.dot(v,row)]
            J=LinearOperator((len(y),len(y)),matvec=mv,dtype=float);M=None;inv=None
            try:
                inv=bordered_inverse(o,jr,ft,fs,pp,ap,at,ass) if gpu_linear is None else gpu_linear.bordered(jr,ft,fs,pp,ap,at,ass)
                M=LinearOperator(J.shape,matvec=inv,dtype=float)
            except ValueError:pass
            baseM=None;deflated=None
            if getattr(self,'deflate_phase',False) and M is not None and not getattr(self,'gpu_krylov',False):
                # Resolve weak relative-phase directions explicitly. This is a
                # low-rank inverse correction, not an added model constraint.
                geo=np.load(o.m.folder/'geometry.npz');pos=[]
                for arr,group,count in [(geo['positions_e'],o.m.unit_of,o.m.unit_count),(geo['positions_i'],o.m.cell_i,o.m.count_i)]:
                    pos.append(np.c_[np.bincount(group,weights=arr[:,0],minlength=len(count))/count,np.bincount(group,weights=arr[:,1],minlength=len(count))/count])
                pos=np.vstack(pos);centers=np.array([[4.19921431597,9.12890135365],[16.47920304044,3.965511533]])
                dist=((pos[:,None,:]-centers[None,:,:])**2).sum(2);mask=(dist[:,0]<dist[:,1]).astype(float)[None,:]
                oscill=(r-r.mean(0))*1000;mean=np.broadcast_to(r.mean(0)*1000,r.shape)
                basis=[dref,dref*mask,oscill*mask,oscill*(1-mask),mean*mask,mean*(1-mask)]
                V=np.column_stack([np.r_[v.ravel(),0.,0.] for v in basis]);V,R=np.linalg.qr(V,mode='reduced');keep=abs(np.diag(R))>1e-10;V=V[:,keep]
                baseM=M;K=np.column_stack([baseM@(J@v) for v in V.T]);D=V-K;C=V.T@K
                condition=float(np.linalg.cond(C));print('PHASE DEFLATION CONDITION',condition,flush=True)
                if np.isfinite(condition) and condition<1e13:
                    def deflated(x):
                        px=baseM@x
                        return px+D@np.linalg.solve(C,V.T@px)
                    M=LinearOperator(J.shape,matvec=deflated,dtype=float)
            counts=[0]
            def count(_):counts[0]+=1
            if getattr(self,'gpu_krylov',False) and gpu_linear is not None:
                import torch
                from topic4_fig5_z_gpu_krylov import solve as gpu_solve
                tft,tfs,tpp,trow=[gpu_linear.tensor(x) for x in [ft,fs,pp,row]]
                def tj(v):
                    dv=v[:-2].reshape(self.N,-1)
                    return torch.cat([(gpu_linear.derivative_tensor(dv/1000)*1000+tft*v[-2]+tfs*v[-1]).flatten(),(dv*tpp).sum().reshape(1),(v*trow).sum().reshape(1)])
                if self.N>=1024:
                    def right_product(v):return tj(inv.tensor(gpu_linear.tensor(v))).cpu().numpy()
                    right_operator=LinearOperator(J.shape,matvec=right_product,dtype=float)
                    def counter(value):
                        counts[0]+=1
                        if counts[0]%25==0:print('HOST ARNOLDI',counts[0],value,flush=True)
                    zz,info=gmres(right_operator,-F,restart=200,maxiter=8,rtol=1e-7,atol=1e-12,callback=counter,callback_type='pr_norm')
                    delta=inv(zz);zz=None;right_operator=None;right_product=None
                else:
                    dd,linear_history=gpu_solve(tj,gpu_linear.tensor(-F),inv.tensor,restart=200,cycles=8,rtol=1e-7)
                    delta=dd.cpu().numpy();info=0 if linear_history[-1]<1e-7*np.linalg.norm(F) else 1;counts[0]=len(linear_history);dd=None
                tj=None
            else:
                delta,info=gmres(J,-F,M=M,rtol=min(1e-4,max(1e-7,error*.003)),atol=1e-10,restart=40,maxiter=6,callback=count,callback_type='pr_norm')
            print('ARC GMRES',info,counts[0],float(np.linalg.norm(J@delta+F)),flush=True)
            # Release this Newton step's GPU matrices before building kernels
            # at another period or a finer phase grid.
            M=None;J=None;inv=None;dlin=None;gpu_linear=None;baseM=None;deflated=None
            if self.gpu is not None:
                import torch
                torch.cuda.empty_cache()
            print('ARC NEWTON STEP',float(delta[-2]),float(delta[-1]),self.norm(delta),flush=True)
            for back in range(8):
                trial=y+2.**(-back)*delta
                if abs(trial[-2]-y[-2])>.25 or abs(trial[-1]-y[-1])>.08:continue
                rt,tt,st=self.unpack(trial);o.s=st;o.z,o.z2=o.eq.z(st);ff=o.evaluate_fixed(rt,tt)
                rhs=np.r_[(ff*1000).ravel(),np.sum((rt-rr)*1000*pp),np.dot(trial-pred,row)]
                print('ARC LINE SEARCH',back,float(abs(rhs).max()),float(np.linalg.norm(rhs)),flush=True)
                if np.linalg.norm(rhs)<np.linalg.norm(F):y=trial;break
            else:return y,error,history
        return y,float(abs(rhs).max()),history


def main():
    p=argparse.ArgumentParser();p.add_argument('--core',choices=['a','b'],required=True);p.add_argument('--N',type=int,default=64);p.add_argument('--points',type=int,default=35);p.add_argument('--step',type=float);p.add_argument('--resume',action='store_true');p.add_argument('--gpu',type=int);p.add_argument('--seeds',nargs=2);p.add_argument('--family');p.add_argument('--minimum-tolerance',type=float,default=.01);a=p.parse_args()
    dest=OUT/f'arc_{a.family or a.core}_N{a.N}';dest.mkdir(parents=True,exist_ok=True);solver=Arc(a.N,a.gpu)
    if a.family=='physical' and a.gpu is not None:
        from topic4_fig5_z_frequency_parallel import install
        install();solver.o.preconditioner_modes=129;solver.gpu_krylov=True
    candidates={}
    for folder in [PRIOR/('periodic_from_crossing_b' if a.core=='b' else 'periodic_from_crossing'),OUT/f'periodic_{a.core}_N{a.N}']:
        for f in folder.glob('amp*_N*.npz'):
            z=np.load(f);amp=float(z['control_amplitude_hz'])
            if float(z['residual_hz'])<1e-5:candidates[amp]={k:z[k] for k in z.files}
    seeds=[candidates[k] for k in sorted(candidates)[-2:]]
    if a.seeds:seeds=[dict(np.load(f)) for f in a.seeds]
    existing=sorted(dest.glob('point[0-9]*.npz')) if a.resume else []
    if len(existing)>=2:seeds=[dict(np.load(f)) for f in existing[-2:]]
    elif len(existing)==1:seeds=[seeds[-1],dict(np.load(existing[-1]))]
    yy=[solver.pack(resample(z['r'],a.N,axis=0),float(z['T']),float(z['s'])) for z in seeds]
    tangent=yy[-1]-yy[-2];distance=solver.norm(tangent);tangent/=distance
    step=a.step or min(distance,.05);rows=[];start=len(existing);t0=time.time()
    for index in range(start,start+a.points):
        accepted=False
        for retry in range(7):
            pred=yy[-1]+step*tangent;result,error,hist=solver.correct(pred,tangent,yy[-1]);correction=solver.norm(result-pred)
            if error<1e-5 and correction<.8*step:
                accepted=True;break
            print('ARC RETRY',index,retry,error,correction,step,flush=True);step*=.5
            if step<1e-4:break
        if not accepted:break
        r,T,s=solver.unpack(result);newtan=result-yy[-1];actual=solver.norm(newtan);newtan/=actual;angle=float(np.dot(newtan*tangent,solver.weight))
        dense=resample(r,4*a.N,axis=0);minimum=float(dense.min()*1000);resolution_needed=minimum < -a.minimum_tolerance
        row=dict(index=index,s=s,mean_z=1-s,period_ms=T,residual_hz=error,N=a.N,step=step,correction=correction,tangent_s=float(newtan[-1]),alignment=angle,iterations=len(hist),seconds=time.time()-t0,unit_minimum_hz=minimum,resolution_needed=resolution_needed)
        rows.append(row);tmp=dest/'point.writing.npz';np.savez_compressed(tmp,r=r,T=T,s=s,residual_hz=error,N=a.N,tangent=newtan,history=hist);os.replace(tmp,dest/f'point{index:04d}.npz')
        (dest/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('ARC ACCEPTED',row,flush=True)
        yy=[yy[-1],result];tangent=newtan
        if resolution_needed or s>=solver.o.eq.ss[-1] or s<-.5 or T>3000:break
        factor=1.25 if len(hist)<=4 and correction<.3*step and angle>.98 else (.7 if len(hist)>6 or angle<.9 else 1.)
        step=min(.09,max(1e-4,step*factor))
    status='RESOLUTION_INCREASE_REQUIRED' if accepted and resolution_needed else ('BOUNDED_ARC_COMPLETE' if accepted else 'NUMERICAL_STOP_UNCLASSIFIED')
    (dest/'status.json').write_text(json.dumps(dict(status=status,accepted=len(rows),last_s=solver.unpack(yy[-1])[-1]),indent=2)+'\n')


if __name__=='__main__':main()
