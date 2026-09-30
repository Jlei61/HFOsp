"""Only restore external mean AMPA filtering; separate paired diagnostic."""
from common import OUT,ROOT,np,read,write,log
import refractory_spatial_diagnostic as original
from runner import summarize
from native_readouts import readouts
from datetime import datetime
import argparse,time,os,hashlib

DEST=OUT/'conditioned_refractory_external_filter_pair'

def corrected_code(P):
    text=original.code(P)
    before='force=tm*(c==0?areaA:areaG)*arr[c*P+g];'
    assert text.count(before)==1
    text=text.replace(before,'force=tm*(c==0?areaA:areaG)*arr[c*P+g]+(c==0?pm:0.);')
    before='physical[g]=syn[P+g]-syn[5*P+g]*syn[3*P+g]-syn[4*P+g]+pm;'
    assert text.count(before)==1
    return text.replace(before,'physical[g]=syn[P+g]-syn[5*P+g]*syn[3*P+g]-syn[4*P+g];')

class FilteredEngine(original.SpatialEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.filtered_module=self.cp.RawModule(code=corrected_code(self.s.P),options=('--fmad=false',),name_expressions=['physical_step'])
        self.k['physical_step']=self.filtered_module.get_function('physical_step')

def register():
    assert read(OUT/'external_mean_filter_audit/result.json')['status']=='FORCING_DIAGNOSTIC_COMPLETE_NO_NETWORK_CHANGE'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    c=read(original.DEST/'contract.json');c.update(created_local=datetime.now().astimezone().isoformat(),
        status='REGISTERED_BEFORE_PAIRED_NETWORK_RUNS',
        question='Does restoring the native externalAMPA mean path alter autonomous events,propagation,Z/M andentry?',
        only_change='Addexternalmeanforcing to theexistingAMPAq/I two-pole equations,remove directpm frommu. ExternalvariancealreadyusesAMPAcovariance. Noaddedstate or fittedparameter; equilibriumDCgain unchanged.',
        reason='Native src/topic4_raster_protocol_engine.py:430-435 routesexternalPoisson arrivals throughAMPA; inheritedrate meanbypassed it. Recorded1msinput auditfoundE RMSdifference.33mV,max1.76mV.',
        baseline=str(original.DEST),
        runs=[dict(label='recorded_drive_expected',drive='seed9108401',noise=False,seed=1),dict(label='recorded_drive_poisson_seed1',drive='seed9108401',noise=True,seed=1)],
        limitations='Externaldrive retainsrecorded1ms averages; this restores thephysicalmeanfilter withinthatsampling convention, not exactsub-msforcing. Alllocalresponse failuresremain. Noautomaticpromotion or bifurcation.',
        native_source_sha256=hashlib.sha256((ROOT/'src/topic4_raster_protocol_engine.py').read_bytes()).hexdigest())
    write(DEST/'contract.json',c)

def check(device):
    rows=[];rng=np.random.default_rng(920081)
    for dt in [.05,.025]:
        e=FilteredEngine(dt=dt,device=device);s=e.s;cp=e.cp;t=e.transport
        for tick in [0,round(1/dt)-1,round(9420/dt)]:
            old=rng.uniform(0,20,(6,s.P));old[4]*=.01;old[5]=rng.uniform(.5,1.,s.P)
            arr=rng.uniform(0,.01,(4,s.P));e.syn[:]=cp.asarray(old);t.arr[:]=cp.asarray(arr);e.local.clock.fill(tick)
            nu=t.drive[t.drive_index(tick)].get();pm=s.tm*s.area[0]*s.jext*nu;pv=s.tm*s.area[0]**2*s.jext**2*nu
            expected=old.copy();coeff=e.coefficients.get()
            for c in range(2):
                a,b,d=coeff[3*c:3*c+3];force=s.tm*s.area[c]*arr[c]+(pm if c==0 else 0.)
                expected[2*c]=a*old[2*c]+(1-a)*force
                expected[2*c+1]=b*old[2*c]+d*old[2*c+1]+(1-d-b)*force
            physical=np.array([expected[1]-old[5]*expected[3]-old[4],s.tm*s.area[0]**2*arr[2]+pv,s.tm*s.area[1]**2*arr[3]])
            e.k['physical_step'](((s.P+127)//128,),(128,),(e.syn,t.arr,t.pars,e.coefficients,t.drive,np.int32(t.drive_on),np.int32(t.n_drive),e.local.clock,dt,e.local.physical))
            error=float(max(abs(e.syn.get()-expected).max(),abs(e.local.physical.get()-physical).max()));assert error<1e-10,error
            rows.append(dict(dt=dt,tick=tick,maximum_error=error))
        # Constantforcing fixedpoint: bothsyn states equalforce, sameDCmu asprior.
        force=s.tm*s.area[0]*arr[0]+pm;e.syn[0]=cp.asarray(force);e.syn[1]=cp.asarray(force)
        e.k['physical_step'](((s.P+127)//128,),(128,),(e.syn,t.arr,t.pars,e.coefficients,t.drive,np.int32(t.drive_on),np.int32(t.n_drive),e.local.clock,dt,e.local.physical))
        error=float(abs(e.syn.get()[:2]-force).max());assert error<1e-10
        e.reset();e.graph();prefix=e.chunk();assert np.isfinite(prefix).all()
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,DC_same=True,finite_prefix_ms=10,
        scope='Onlyexternalmeanfilterchange independentlychecked at bothsteps,heterogeneousgroups andthreeclocks. LocalrateCUDA andtransportchecks inheritedunchanged.'))
    log('EXTERNAL FILTER PAIRED IMPLEMENTATION PASS',rows)

def run(device):
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    progress=dict(status='RUNNING',expected=len(c['runs']),completed=[],pid=os.getpid());assert not (DEST/'jobs.json').exists();write(DEST/'jobs.json',progress)
    for row in c['runs']:
        folder=DEST/row['label'];folder.mkdir();e=FilteredEngine(dt=c['dt_ms'],drive=row['drive'],noise=row['noise'],seed=row['seed'],device=device);s=e.s;e.graph()
        R=[];Z=[];M=[];start=time.time()
        write(folder/'identity.json',dict(graph_identity=s.prep['graph_identity'],response=str(original.LOCAL/'fit/locked_weights.json'),groups=s.P,spatial_grid=s.grid,shared_split_qa=e.split_qa))
        for k in range(round(c['duration_ms']/10)):
            output=e.chunk();assert np.isfinite(output).all(),(row['label'],k,'NONFINITE_RATE')
            R.append(output);Z.append(e.syn[5].get());M.append(e.syn[4].get());assert Z[-1].min()>=-1e-10 and Z[-1].max()<=1+1e-10
            if (k+1)%100==0:
                log('FILTERED EXTERNAL',row['label'],(k+1)*10,'ms','seconds',round(time.time()-start,1));write(folder/'progress.json',dict(status='RUNNING',time_ms=(k+1)*10))
        rates=np.concatenate(R);z=np.array(Z);m=np.array(M);_,field,whole,count=summarize(rates[:,0],s);tms=np.arange(len(rates))+1.;ts=(np.arange(len(z))+1)*10.;D=1-z[:,s.E]@s.mean_weights
        events,summary,_,_=readouts(tms,field,count,row['label'])
        np.savez_compressed(folder/'trajectory.npz',time_ms=tms,group_rate_hz=rates[:,0].astype('f4'),group_expected_rate_hz=rates[:,1].astype('f4'),field_E_hz=field.astype('f4'),global_E_hz=whole,cell_counts=count,Z=z.astype('f4'),M_current=m.astype('f4'),D=D,state_time_ms=ts,
            final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),final_own_history=e.local.history.get(),final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
        summary.update(status='COMPLETE',label=row['label'],duration_ms=c['duration_ms'],Z_and_M_dynamic=True,D8000=float(D[799]),D9870=float(D[986]),D_final=float(D[-1]),seconds=time.time()-start,model_promoted=False)
        summary['events']=[{k:v for k,v in ev.items() if k!='onset'} for ev in events];write(folder/'result.json',summary);write(folder/'progress.json',dict(status='COMPLETE',time_ms=c['duration_ms']))
        progress['completed'].append(row['label']);write(DEST/'jobs.json',progress);log('FILTERED EXTERNAL COMPLETE',row['label'],summary['high_onset_ms'],summary['D9870']);del e
    progress['status']='COMPLETE';write(DEST/'jobs.json',progress)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=1);a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
