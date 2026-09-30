"""Fresh prescribed-input assay of the locked transient correction.

Predictions at both steps are saved before acquiring fresh independent LIF
targets. Native output, onset timing and Z-path targets are never fitted.
"""
from transient_response_bias import DEST,inputs,np,read,write,log
from conditioned_refractory_rate import load_models
from refractory_rate_response import covariance_matrices,implicit_flux
from nonlinear_rate_response import physical_from_features
from native_input_local_lif import simulate,condition,static_run
from numba import njit
from datetime import datetime
import argparse,hashlib,os,time,torch

VDIR=DEST/'fresh_validation'


def prepare():
    assert read(DEST/'locked.json')['status']=='LOCKED_BEFORE_TRANSFER_AND_NEW_REFERENCES'
    VDIR.mkdir(exist_ok=True);assert not (VDIR/'profiles.json').exists();rows=[]
    for kind,groups in [('early',[531,185,945,2613,2237]),('late',[279,594])]:
        source=inputs(kind);waves=[];meta=[]
        for group in groups:
            j=int(np.flatnonzero(source['groups']==group)[0]);wave=source['raw_wave'][j]
            for variant,warp,mean_scale,var_scale in [(0,.8,.9,1.1),(1,1.25,1.1,.9)]:
                idx=np.minimum(np.floor(np.arange(wave.shape[1])*warp).astype(int),wave.shape[1]-1)
                x=wave[:,idx].copy();x[0]=11+mean_scale*(x[0]-11);x[1:3]*=var_scale
                assert np.isfinite(x).all() and x[1:3].min()>=0
                info=dict(id=len(rows),kind=kind,index=len(waves),group=int(group),pop=int(source['pop'][j]),N=int(source['N'][j]),
                    theta=float(source['theta'][j]),variant=variant,time_warp=warp,mean_scale=mean_scale,variance_scale=var_scale,
                    burn_ms=500. if kind=='early' else 1000.,bins=50 if kind=='early' else 27,bin_ms=50.,steps=int(x.shape[1]))
                rows.append(info);meta.append(info);waves.append(x)
        np.savez_compressed(VDIR/f'{kind}_inputs.npz',wave=waves,theta=[r['theta'] for r in meta],population=[r['pop'] for r in meta],N=[r['N'] for r in meta])
    assert len(rows)==14
    write(VDIR/'profiles.json',dict(status='STIMULI_LOCKED_BEFORE_PREDICTION_AND_TARGETS',created_local=datetime.now().astimezone().isoformat(),rows=rows,
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in VDIR.glob('*_inputs.npz')},
        parameter_hash=hashlib.sha256((DEST/'locked.json').read_bytes()).hexdigest()))
    log('FRESH TRANSIENT INPUTS',len(rows))


@njit
def raw_features(wave,theta,A,B,C,dt):
    G,_,W=wave.shape;factor=round(.1/dt);T=W*factor
    cov=np.zeros((G,2,3));h=np.zeros((G,3,4,3));out=np.empty((T,G,39));taus=np.array([1.,4.,16.,64.])
    for k in range(T):
        ix=k//factor
        for g in range(G):
            v=np.zeros(2)
            for c in range(2):
                a,b,d=cov[g,c]
                for q in range(3):cov[g,c,q]=A[g,c,q,0]*a+A[g,c,q,1]*b+A[g,c,q,2]*d+B[g,c,q]*wave[g,c+1,ix]
                v[c]=C[g,c]*cov[g,c,2]
            v[1]*=wave[g,3,ix]**2;scale=theta[g]-11
            u=np.array([np.arcsinh((wave[g,0,ix]-11)/scale)/3,np.log1p(v[0]/scale**2)/2,np.log1p(v[1]/scale**2)/2])
            out[k,g,:3]=u
            for c in range(3):
                for j in range(4):
                    b=dt/taus[j];e=np.exp(-b);h1,h2,h3=h[g,c,j]
                    h[g,c,j,0]=e*h1+(1-e)*u[c]
                    h[g,c,j,1]=e*(h2+b*h1)+(1-e*(1+b))*u[c]
                    h[g,c,j,2]=e*(h3+b*h2+.5*b*b*h1)+(1-e*(1+b+.5*b*b))*u[c]
                    for q in range(3):out[k,g,3+c*12+j*3+q]=h[g,c,j,q]-u[c]
    return out


def predict():
    meta=read(VDIR/'profiles.json');assert not (VDIR/'predictions_locked.json').exists()
    assert hashlib.sha256((DEST/'locked.json').read_bytes()).hexdigest()==meta['parameter_hash']
    nets,bases,parent=load_models();parameters=read(DEST/'locked.json')['parameters'];summary=[]
    for kind in ['early','late']:
        z=np.load(VDIR/f'{kind}_inputs.npz');wave=z['wave'];pop=z['population'];theta=z['theta'];rows=[r for r in meta['rows'] if r['kind']==kind]
        for dt in [.1,.05]:
            coefficients=[covariance_matrices('E' if p==0 else 'I',dt) for p in pop]
            f=raw_features(wave,theta,np.array([a[0] for a in coefficients]),np.array([a[1] for a in coefficients]),np.array([a[2] for a in coefficients]),dt)
            ell=np.empty(f.shape[:2]);correction=np.empty_like(ell)
            for p,key in [(0,'E'),(1,'I')]:
                mask=pop==p;flat=f[:,mask].reshape(-1,39);ths=np.broadcast_to(theta[mask],(len(f),mask.sum())).ravel();values=[]
                for start in range(0,len(flat),32768):
                    sl=slice(start,start+32768);base=bases[key].evaluate(physical_from_features(flat[sl],ths[sl]),ths[sl])
                    with torch.no_grad():values.append(nets[key].logits(torch.tensor(flat[sl]),torch.tensor(base)).numpy())
                ell[:,mask]=np.concatenate(values).reshape(len(f),mask.sum())
                H=np.mean(f[:,mask,3:]**2,axis=2);q=parameters[key];correction[:,mask]=q['b']*H/(H+q['h'])
            predicted=[];baseline=[];rate=[];parent_rate=[]
            for j,row in enumerate(rows):
                r,minimum=implicit_flux(ell[:,j]+correction[:,j],dt,2. if pop[j]==0 else 1.)
                old,oldminimum=implicit_flux(ell[:,j],dt,2. if pop[j]==0 else 1.)
                assert minimum>=-1e-10 and oldminimum>=-1e-10
                start=round(row['burn_ms']/dt);end=start+round(row['bins']*50/dt);width=round(50/dt)
                predicted.append(r[start:end].reshape(row['bins'],width).sum(1)*dt/1000*row['N'])
                baseline.append(old[start:end].reshape(row['bins'],width).sum(1)*dt/1000*row['N'])
                rate.append(r);parent_rate.append(old)
            path=VDIR/f'{kind}_prediction_dt{dt:g}.npz'
            np.savez_compressed(path,counts=np.array(predicted),parent_counts=np.array(baseline),
                rate_hz=np.array(rate).T,parent_rate_hz=np.array(parent_rate).T,dt_ms=dt)
            summary.append(dict(kind=kind,dt_ms=dt,conditions=len(rows),steps=len(f),file=path.name));log('FRESH TRANSIENT PREDICTION',kind,dt)
    write(VDIR/'predictions_locked.json',dict(status='PREDICTIONS_LOCKED_BEFORE_NEW_LIF_TARGETS',created_local=datetime.now().astimezone().isoformat(),
        parameter_hash=meta['parameter_hash'],rows=summary,prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in VDIR.glob('*prediction*.npz')},
        reference_acquired=False,parent_weights_unchanged=True))


def acquire(device):
    locked=read(VDIR/'predictions_locked.json');assert locked['status']=='PREDICTIONS_LOCKED_BEFORE_NEW_LIF_TARGETS'
    assert not (VDIR/'jobs.json').exists();jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(VDIR/'jobs.json',jobs)
    # Same original LIF integrator and waveform indexing, independently checked.
    pars=np.array([condition(20.,18.,1.,1.,'E'),condition(16.,14.5,1.,1.,'I')]);R=256
    wave=np.ones((2,4,5000));wave[:,0]=np.array([20.,16.])[:,None];theta=np.broadcast_to(pars[:,1,None],(2,R)).copy()
    original=static_run(pars,R,400.,100.,7918,device=device)[:,:,2]
    observed=simulate(pars,wave,theta,np.array([R,R]),.1,100.,8,50.,7918,device).sum(2)
    assert np.array_equal(original,observed) and observed.sum()>0
    write(VDIR/'lif_implementation_check.json',dict(status='PASS',constant_input_counts_bitwise=True))
    rows=read(VDIR/'profiles.json')['rows']
    for kind in ['early','late']:
        z=np.load(VDIR/f'{kind}_inputs.npz');meta=[r for r in rows if r['kind']==kind];N=z['N'];nr=np.ceil(8192/N).astype(int)*N;R=int(nr.max())
        theta=np.broadcast_to(z['theta'][:,None],(len(N),R)).copy()
        for dt in [.1,.05]:
            pars=np.array([condition(0.,row['theta'],1.,1.,'E' if row['pop']==0 else 'I',dt=dt) for row in meta])
            start=time.time();count=simulate(pars,z['wave'],theta,nr,dt,meta[0]['burn_ms'],meta[0]['bins'],50.,920141,device)
            np.savez_compressed(VDIR/f'{kind}_lif_dt{dt:g}.npz',counts=count,replicates=nr,N=N,dt_ms=dt)
            jobs['completed'].append([kind,dt]);write(VDIR/'jobs.json',jobs);log('FRESH TRANSIENT LIF',kind,dt,'seconds',round(time.time()-start,2))
    jobs['status']='COMPLETE';write(VDIR/'jobs.json',jobs)


def score():
    assert read(VDIR/'jobs.json')['status']=='COMPLETE';locked=read(VDIR/'predictions_locked.json')
    for name,h in locked['prediction_hashes'].items():assert hashlib.sha256((VDIR/name).read_bytes()).hexdigest()==h
    rows=read(VDIR/'profiles.json')['rows'];result=[]
    for kind in ['early','late']:
        infos=[r for r in rows if r['kind']==kind]
        mc0=np.load(VDIR/f'{kind}_lif_dt0.1.npz');mc1=np.load(VDIR/f'{kind}_lif_dt0.05.npz')
        p0=np.load(VDIR/f'{kind}_prediction_dt0.1.npz');p1=np.load(VDIR/f'{kind}_prediction_dt0.05.npz')
        for j,info in enumerate(infos):
            nr=int(mc0['replicates'][j]);N=info['N'];x=mc0['counts'][j,:nr].astype(float);y=mc1['counts'][j,:nr].astype(float)
            means=[x.mean(0)*N,y.mean(0)*N];split=(y[:nr//2].mean(0)-y[nr//2:].mean(0))*N
            denominator=max(np.linalg.norm(means[1]),1.);numerical=float(np.linalg.norm(means[0]-means[1])/denominator);noise=float(np.linalg.norm(split)/denominator)
            step=float(np.linalg.norm(p0['counts'][j]-p1['counts'][j])/denominator)
            comparisons=[]
            for k,pred in enumerate([p0,p1]):
                truth=means[k];den=max(np.linalg.norm(truth),1.)
                r=pred['counts'][j];old=pred['parent_counts'][j];l2=float(np.linalg.norm(r-truth)/den);bias=float(abs(r.sum()/truth.sum()-1))
                comparisons.append(dict(dt_ms=.1 if k==0 else .05,L2=l2,count_error=bias,parent_L2=float(np.linalg.norm(old-truth)/den),
                    parent_count_error=float(abs(old.sum()/truth.sum()-1)),passed=l2<=.15 and bias<=.10))
            result.append(dict(**info,comparisons=comparisons,LIF_two_step_L2=numerical,MC_split_L2=noise,LIF_numerical_pass=numerical<=max(.05,3*noise),
                rate_two_step_L2=step,rate_numerical_pass=step<=.02,scientific_pass=all(r['passed'] for r in comparisons)))
    passed=all(r['scientific_pass'] and r['LIF_numerical_pass'] and r['rate_numerical_pass'] for r in result)
    write(VDIR/'result.json',dict(status='FRESH_LOCAL_PASS' if passed else 'FRESH_LOCAL_FAIL',rows=result,
        scientific_passed=sum(r['scientific_pass'] for r in result),total=len(result),
        LIF_numerical_failures=sum(not r['LIF_numerical_pass'] for r in result),rate_numerical_failures=sum(not r['rate_numerical_pass'] for r in result),
        model_promoted=False,developmental_failures_retained=True,bifurcation_type='NOT_ESTABLISHED'))
    log('FRESH TRANSIENT RESULT',sum(r['scientific_pass'] for r in result),'/',len(result),
        'LIFstepfail',sum(not r['LIF_numerical_pass'] for r in result),'ratestepfail',sum(not r['rate_numerical_pass'] for r in result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','predict','acquire','score']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'prepare':prepare,'predict':predict,'acquire':lambda:acquire(a.device),'score':score}[a.command]()
