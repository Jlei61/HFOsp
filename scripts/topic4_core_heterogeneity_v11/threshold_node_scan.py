"""Bounded mean-threshold critical-point scan, retaining empirical variance."""
from common import *
from branches import fold
from spectral import leading
from scipy.optimize import root
import argparse,time

DEST=OUT/'threshold_nodes';DEST.mkdir(exist_ok=True)
BASE=System();MEAN=BASE.mean_A;STD=BASE.original_std_A

class MeanSystem(FrozenSystem):
    def __init__(self,mean=MEAN,groups=32):
        super().__init__(groups=groups)
        self.threshold[0]+=float(mean)-MEAN
        self.mean_A=float(mean);self.original_std_A=STD

def folds():
    seed=read(V2/'fold.json');initial=np.r_[np.array(seed['r_hz'])/1000,seed['g'],seed['v']]
    rows=[]
    for mean in np.linspace(MEAN,16.5,25):
        s=MeanSystem(mean);f=fold(s,initial)
        initial=np.r_[np.array(f['r_hz'])/1000,f['g'],f['v']]
        assert min(f['r_hz'])>-1e-5
        f.update(mean_A_mV=float(mean),sigma_A_mV=STD,critical_core='AB'[int(np.argmax(abs(np.array(f['v'])[:2])))])
        if len(rows)%6==0:
            ev=leading(s,initial[:6],f['g'],40);ev2=leading(s,initial[:6],f['g'],64)
            assert min(abs(ev))<1e-5 and min(abs(ev2))<1e-5
            assert abs(max(ev.real)-max(ev2.real))<1e-4
            f['leading_real_per_s']=float(max(ev2.real));f['spectrum_N40_N64_checked']=True
            r,err,ok=s.solve(f['g']-.0001,initial[:6]*.96)
            assert ok and max(leading(s,r,f['g']-.0001,40).real)<0
        rows.append(f);write(DEST/'mean_fold.json',rows)
        print('MEAN FOLD',mean,f['g'],flush=True)
    # Refine the actual A/B switch rather than locating it by eye.
    s=System(.964)
    def fn(y):
        r=y[:6]/1000;g,h=y[-2:]
        s.threshold[0]=s.mean_A+h*(BASE.threshold[0]-s.mean_A)
        J=s.jac(r,g)
        return np.r_[s.F(r,g)*1000,J[0,0],J[1,1]]
    q=root(fn,np.r_[.43,.413,0,0,0,0,1.17455,.9633],tol=1e-10)
    err=float(abs(fn(q.x)).max());assert err<1e-7
    vals=np.linalg.eigvals(s.jac(q.x[:6]/1000,q.x[-2]));small=sorted(abs(vals))[:2]
    assert max(small)<1e-7
    ev=leading(s,q.x[:6]/1000,q.x[-2],64)
    write(DEST/'AB_onset_switch.json',dict(g=q.x[-2],h=q.x[-1],sigma_A_mV=q.x[-1]*STD,
        r_hz=q.x[:6].tolist(),residual=err,static_eigenvalues=vals.real.tolist(),
        nearest_dynamic_roots_per_s=[[float(x.real),float(x.imag)] for x in sorted(ev,key=abs)[:4]],
        interpretation='A/B onset switch with two numerically zero modes; no Bogdanov-Takens classification'))
    print('ONSET SWITCH',q.x[-2:],err,flush=True)

def periodic(label,validate=False,resume=False):
    import periodic_boundaries as p
    import pd_joint as pd
    import fold_joint as lp
    p.System=pd.System=lp.System=MeanSystem
    if validate:
        old=read(DEST/f'{label}.json')['points'][-1];N=1024
        z,t=p.load_seed(ROOT/old['source'],N)
        if label.startswith('PD'):q,t,row=pd.correct(old['mean_A_mV'],z,N)
        else:q,t,row=lp.correct(old['mean_A_mV'],z,t,N)
        row['difference_g']=row['g']-old['g'];row['difference_T_ms']=row['T_ms']-old['T_ms']
        row['mean_A_mV']=old['mean_A_mV'];row.pop('h',None)
        assert abs(row['difference_g'])<1e-6
        write(DEST/f'{label}_grid_validation.json',row)
        return
    N=512;z,t=p.load_seed(p.seeds()[label],N)
    rows=[];shifts=[0.,-.15,-.35,-.6];prior=None
    if resume:
        prior=read(DEST/f'{label}.json');write(DEST/f'{label}_large_step_attempt.json',prior)
        rows=prior['points'];z,t=p.load_seed(ROOT/rows[-1]['source'],N)
        shifts=[-.015,-.030] if label=='LP1' else [-.4,-.45,-.5,-.55,-.6]
    for shift in shifts:
        mean=MEAN+shift
        try:
            if label.startswith('PD'):q,t,row=pd.correct(mean,z,N)
            else:q,t,row=lp.correct(mean,z,t,N)
            row.pop('h',None);row.update(mean_A_mV=mean,sigma_A_mV=STD,label=label,status='REFINED')
            path=DEST/f'{label}_mean{mean:.6f}_N{N}.npz'
            np.savez_compressed(path,r=q[:-2].reshape(N,6)*.01,T=np.exp(q[-2]),g=q[-1]*.01,mean_A=mean,N=N,tangent=t)
            row['source']=str(path.relative_to(ROOT));rows.append(row);z=q
            write(DEST/f'{label}.json',dict(status='RUNNING',points=rows))
        except Exception as e:
            write(DEST/f'{label}.json',dict(status='PARTIAL',points=rows,error=repr(e),failed_mean=mean));raise
    write(DEST/f'{label}.json',dict(status='DONE',points=rows))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('task');a.add_argument('--validate',action='store_true');a.add_argument('--resume-small',action='store_true');x=a.parse_args()
    if x.task=='fold':folds()
    else:periodic(x.task,x.validate,x.resume_small)
