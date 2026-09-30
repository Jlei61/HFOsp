"""Actual local GPU flow and separate raw-count audit for transient correction."""
from transient_response_bias import DEST,np,read,write,log
from refractory_rate_cuda import LocalGPUResponse,source
from validate_transient_response_correction import VDIR
import argparse,gc,hashlib


def implementation(device):
    assert read(VDIR/'predictions_locked.json')['status']=='PREDICTIONS_LOCKED_BEFORE_NEW_LIF_TARGETS'
    parameters=read(DEST/'locked.json')['parameters'];rows=[]
    for kind in ['early','late']:
        z=np.load(VDIR/f'{kind}_inputs.npz');G=len(z['theta']);wave=z['wave']
        for dt in [.1,.05]:
            factor=round(.1/dt);physical=wave[:,:3].transpose(2,1,0).repeat(factor,axis=0).copy()
            zp=wave[:,3].T.repeat(factor,axis=0).copy();T=len(physical)
            e=LocalGPUResponse(z['population'],z['theta'],dt,T+1,physical,device=device);cp=e.cp
            text=source(G);needle='  double occupied=0.;int nref=(int)llround(refractory[g]/dt);'
            assert text.count(needle)==1
            term='  double H=0.;for(int j=3;j<39;j++)H+=f[j]*f[j];H/=36.;\n'
            term+=f'  double b=p==0?{parameters["E"]["b"]:.17g}:{parameters["I"]["b"]:.17g};\n'
            term+=f'  double h=p==0?{parameters["E"]["h"]:.17g}:{parameters["I"]["h"]:.17g};ell+=b*H/(H+h);\n'
            text=text.replace(needle,term+needle)
            module=cp.RawModule(code=text,options=('--fmad=false',),name_expressions=['local_rate'])
            e.kernel=module.get_function('local_rate');Z=cp.asarray(zp)
            loadz=cp.RawKernel(r'''
extern "C" __global__ void loadz(double* z,const double* path,const int* clock,int G){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g<G)z[g]=path[(long long)clock[0]*G+g];
}''','loadz',options=('--fmad=false',))
            def step():
                loadz(((G+127)//128,),(128,),(e.Z,Z,e.clock,np.int32(G)));e.step()
            step();cp.cuda.get_current_stream().synchronize()
            e.state.fill(0);e.history.fill(0);e.rate.fill(0);e.clock.fill(0);e.Z.fill(1)
            cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                for _ in range(100):step()
                graph=stream.end_capture()
            for _ in range(T//100):graph.launch(stream)
            stream.synchronize();observed=e.history.get()[1:]*1000
            predicted=np.load(VDIR/f'{kind}_prediction_dt{dt:g}.npz')['rate_hz']
            error=float(np.max(abs(observed-predicted)));relative=float(np.linalg.norm(observed-predicted)/max(np.linalg.norm(predicted),1.))
            assert error<1e-4 and relative<1e-8,(kind,dt,error,relative)
            assert int(e.clock.get()[0])==T and np.array_equal(e.Z.get(),zp[-1])
            rows.append(dict(kind=kind,dt_ms=dt,groups=G,steps=T,maximum_error_hz=error,relative_L2=relative))
            log('TRANSIENT GPU FLOW PARITY',rows[-1])
            del e,Z,module,graph;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(VDIR/'implementation_audit.json',dict(status='PASS',rows=rows,
        scope='Same nonlinearGPU localflow andindependentCPU features/renewal atbothsteps. Not a spatialmodel or bifurcation acceptance.'))


def score():
    report=read(VDIR/'result.json');meta=read(VDIR/'profiles.json');locked=read(VDIR/'predictions_locked.json')
    for name,h in meta['input_hashes'].items():assert hashlib.sha256((VDIR/name).read_bytes()).hexdigest()==h
    for name,h in locked['prediction_hashes'].items():assert hashlib.sha256((VDIR/name).read_bytes()).hexdigest()==h
    checked=[]
    for kind in ['early','late']:
        z=np.load(VDIR/f'{kind}_inputs.npz');rows=[r for r in meta['rows'] if r['kind']==kind]
        counts=[];predictions=[];allmc=[]
        for dt in [.1,.05]:
            mc=np.load(VDIR/f'{kind}_lif_dt{dt:g}.npz');p=np.load(VDIR/f'{kind}_prediction_dt{dt:g}.npz');means=[]
            for j,info in enumerate(rows):
                n=int(mc['replicates'][j]);assert n>=8192 and n%info['N']==0
                a=mc['counts'][j,:n];assert a.shape==(n,info['bins'])
                maximum=26 if info['pop']==0 else 51;assert a.max()<=maximum
                means.append(a.sum(0,dtype=np.uint64)/float(n)*info['N'])
                start=round(info['burn_ms']/dt);end=start+round(info['bins']*50/dt)
                prediction=p['rate_hz'][start:end,j].reshape(info['bins'],round(50/dt)).sum(1)*dt/1000*info['N']
                assert np.max(abs(prediction-p['counts'][j]))<1e-8
            counts.append(np.array(means));predictions.append(p['counts']);allmc.append(mc)
        for j,info in enumerate(rows):
            rr=next(r for r in report['rows'] if r['id']==info['id'])
            for k in range(2):
                target=counts[k][j];pred=predictions[k][j];den=max(np.sqrt(np.sum(target**2)),1.)
                error=float(np.sqrt(np.sum((target-pred)**2))/den);bias=float(abs(pred.sum()-target.sum())/target.sum())
                expected=rr['comparisons'][k]
                assert abs(error-expected['L2'])<1e-10 and abs(bias-expected['count_error'])<1e-10
                assert expected['passed']==(error<=.15 and bias<=.10)
            den=max(np.linalg.norm(counts[1][j]),1.)
            step=float(np.linalg.norm(predictions[0][j]-predictions[1][j])/den)
            lifstep=float(np.linalg.norm(counts[0][j]-counts[1][j])/den)
            assert abs(step-rr['rate_two_step_L2'])<1e-10 and abs(lifstep-rr['LIF_two_step_L2'])<1e-10
            checked.append(info['id'])
    write(VDIR/'independent_audit.json',dict(status='PASS',conditions=checked,raw_counts_checked=True,rate_bins_checked=True,
        scores_and_step_errors_reproduced=True,scientific_status=report['status'],model_promoted=False))
    log('TRANSIENT INDEPENDENT SCORE AUDIT PASS',report['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['implementation','score']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    implementation(a.device) if a.command=='implementation' else score()
