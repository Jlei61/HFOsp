"""Bounded autonomous network sensitivity to two already diagnosed closure changes.

This is not promotion of a replacement model. Graph, input, Z/M and cell
parameters are fixed. Frozen baseline, voltage units, and units+history-reading
are compared from the same physical zero state at the same .05ms step.
"""
from common import *
import dynamics_v3 as original
from response_tables import CUDA_RESP
from response_voltage_units import VoltageScaledResponseTable, CUDA_RESP_VOLTAGE_UNITS
from runner import GraphChunk, summarize
from native_readouts import readouts, window_stats
import argparse

DEST=OUT/'closure_network_sensitivity'


class SensitivityModel(DynamicModel):
    def effective(self, y, pm=None, pv=None):
        _,mus,vEf,vIf,_,ia,_,ig,va,vg,m,z,vEv,vIv=y
        pm=self.private_mu if pm is None else pm
        pv=self.private_ve if pv is None else pv
        mu=ia-z*ig-m+pm; ve=va+pv; vi=z*z*vg
        moments=(mus,vEv,vIv) if self.history_weights else (mu,ve,vi)
        w,_=self.resp.weights(self,*moments); al,ae,ai,ee,ei=w
        return (al*mu+(1-al)*mus+ee*(ve-vEf)+ei*(vi-vIf),
                np.maximum(ae*ve+(1-ae)*vEv,0),np.maximum(ai*vi+(1-ai)*vIv,0),(mu,ve,vi))


def setup(label, dt, device, stochastic=False):
    s=model(); s.set_Z(np.ones(s.P),source='common physical zero state')
    s.__class__=SensitivityModel; s.history_weights=label=='units_history'
    if label!='frozen':
        s.resp.tables={p:VoltageScaledResponseTable(t) for p,t in s.resp.tables.items()}
    base=original.cuda_code
    def code(s):
        result=base(s)
        if label!='frozen':
            assert result.count(CUDA_RESP)==1
            result=result.replace(CUDA_RESP,CUDA_RESP_VOLTAGE_UNITS)
        if s.history_weights:
            old='weights_at(pars,consts,WE,WI,g,mu,vE,vI,w,gr);'
            assert result.count(old)==1
            result=result.replace(old,'weights_at(pars,consts,WE,WI,g,mus,vEv,vIv,w,gr);')
        return result
    # Each engine retains its compiled kernels. The module-level factory is
    # restored immediately, and no frozen source file or pickle is changed.
    extra={}
    if stochastic:
        from run_network import group_drive
        extra=dict(drive=group_drive(s,'seed9108401'),noise=True,seed=1,
                   shared_split=original.shared_fraction_operators(s))
    original.cuda_code=code
    try:e=Integrator(s,dt=dt,dynamic_z=True,dynamic_m=True,device=device,**extra)
    finally:original.cuda_code=base
    return s,e


def audit(s,e,label):
    cp=e.cp
    point=np.load(BASE/'runs/A4_det_meandrive/checkpoints/t8000ms.npz')['state']
    # Compare full native 14-state derivatives at a genuinely active state.
    state0=e.y.copy(); hist0=e.history.copy()
    e.y[:]=cp.asarray(point)
    r=s.output(point)
    e.history[:]=cp.asarray(np.broadcast_to(r,e.history.shape).copy())
    e.arrivals(0)
    e.k['rhs'](((s.P+127)//128,),(128,),
        (e.y,e.arr,e.pars,e.consts,e.SE,e.SI,e.WE,e.WI,e.drive,np.int32(0),np.int32(0),e.f,e.rate))
    cpu,rate=s.rhs(point,e.arr.get(),dynamic_z=True)
    rhsdiff=float(np.max(abs(cpu-e.f.get())))
    ratediff=float(np.max(abs(rate-e.rate.get())))
    assert np.allclose(cpu,e.f.get(),rtol=1e-9,atol=1e-10),(label,rhsdiff)
    assert np.allclose(rate,e.rate.get(),rtol=1e-9,atol=1e-12),(label,ratediff)
    e.y[:]=state0;e.history[:]=hist0;e.tick=0
    return dict(CPU_GPU_rhs_max_error=rhsdiff,CPU_GPU_rate_max_error=ratediff,
                initial_state=state0.get(),initial_history=hist0.get())


def main(device):
    c=read(OUT/'closure_network_sensitivity_contract.json');DEST.mkdir(exist_ok=True)
    rows=[];initial=None
    baseline_initial=DEST/'frozen/initial.npz'
    if baseline_initial.exists():
        ref=np.load(baseline_initial)
        initial=(ref['state'].copy(),ref['history'].copy())
    for label in c['variants']:
        folder=DEST/label;folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            rows.append(read(folder/'result.json'));continue
        s,e=setup(label,c['dt_ms'],device)
        if initial is not None:
            # Use precisely the baseline prehistory. Recomputing the tiny
            # transfer-floor rate separately needlessly changes its last bits.
            e.history[:]=e.cp.asarray(initial[1])
        qa=audit(s,e,label)
        if initial is None:initial=(qa['initial_state'],qa['initial_history'])
        assert np.array_equal(qa['initial_state'],initial[0])
        # The physical zero state's rate history should also be unchanged.
        assert np.array_equal(qa['initial_history'],initial[1])
        np.savez_compressed(folder/'initial.npz',state=qa.pop('initial_state'),history=qa.pop('initial_history'))
        graph=GraphChunk(e,milliseconds=10);rates=[];Z=[];M=[];start=time.time()
        for chunk in range(round(c['duration_ms']/10)):
            rates.append(graph.run());Z.append(e.y[11].get());M.append(e.y[10].get())
            if (chunk+1)%100==0:
                assert bool(e.cp.isfinite(e.y).all())
                log('CLOSURE NETWORK',label,'t_ms',(chunk+1)*10,'seconds',round(time.time()-start,1))
                write(folder/'progress.json',dict(status='RUNNING',time_ms=(chunk+1)*10))
        R=np.concatenate(rates);_,field,global_rate,count=summarize(R,s)
        z=np.array(Z);m=np.array(M);t=np.arange(len(R))+1.
        event,summary,_,_=readouts(t,field,count,label)
        summary['windows']={f'{a}-{b}':window_stats(event,a,b) for a,b in c['event_windows_ms']}
        summary.update(status='COMPLETE',label=label,scope=c['scope'],qa=qa,dt_ms=c['dt_ms'],
                       D_final=float(1-z[-1,s.E]@s.mean_weights),seconds=time.time()-start)
        summary['events']=[{k:v for k,v in ev.items() if k!='onset'} for ev in event]
        np.savez_compressed(folder/'trajectory.npz',time_ms=t,group_rate_hz=R.astype('float32'),
             field_E_hz=field.astype('float32'),global_E_hz=global_rate,cell_counts=count,
             Z=z.astype('float32'),M_current=m.astype('float32'),state_time_ms=(np.arange(len(z))+1)*10.,
             D=1-z[:,s.E]@s.mean_weights,final_state=e.y.get(),final_history=e.history.get(),final_tick=0,dt_ms=c['dt_ms'])
        write(folder/'result.json',summary);rows.append(summary)
        write(DEST/'result.json',dict(status='RUNNING',completed=len(rows),rows=rows,scope=c['scope']))
        log('CLOSURE NETWORK COMPLETE',label,summary['high_onset_ms'],summary['D_final'])
        del graph,e,s
    write(DEST/'result.json',dict(status='DIAGNOSTIC_COMPLETE',completed=len(rows),rows=rows,scope=c['scope'],replacement_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
