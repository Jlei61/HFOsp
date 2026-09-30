"""Bounded autonomous rate-network pilot; native output is never an input."""
from common import *
from network import SpatialRate
import cupy as cp
import argparse
import os
import time


def main(a):
    folder=OUT/'runs'/a.label;folder.mkdir(parents=True,exist_ok=False)
    write(folder/'config.json',dict(**vars(a),dt_ms=DT,Z='dynamic' if a.D is None else 'fixed spatial field',M='dynamic',
        native_future_output_forcing=False,rate_units='spikes/ms/neuron internally',
        classification_scope='Short pilot only; no claimed bifurcation or correspondence acceptance'))
    write(folder/'status.json',dict(status='INITIALIZING',pid=os.getpid()))
    model=SpatialRate(a.kind,a.device,a.D,a.resource_closure)
    source=None
    if a.shared:
        with np.load(OUT/f'reference/input_s{a.seed}.npz') as z:
            source=np.c_[z['E_rate_per_ms'],np.repeat(z['I_rate_per_ms'][:,None],400,axis=1)].astype('float64')
        assert a.duration<=len(source)
    N=int(a.duration);fields=np.zeros((N,800));slow=np.zeros((N//10+1,2));n=cp.asarray(model.geo['count'][:model.NE]/32000.)
    block=cp.zeros(800);started=time.time();error=None
    for t in range(N*10):
        x=model.step(source[t//10] if source is not None and t%10==0 else None)
        block+=x
        if (t+1)%10==0:fields[t//10]=cp.asnumpy(block)*100.;block.fill(0.)
        if (t+1)%100==0:
            slow[(t+1)//100]=float((model.z@n).get()),float((model.m@n).get())
        if (t+1)%1000==0:
            recent=fields[t//10-99:t//10+1]
            if not np.isfinite(recent).all() or recent.min() < -1e-5:
                error='Nonfinite or negative interval-integrated rate';break
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time_ms=(t+1)*DT,wall_s=time.time()-started,
                mean_E_hz=float(recent[:,:400].mean(0)@model.geo['cell_count_e']/32000)))
            if (t+1)%5000==0:print('pilot',a.label,(t+1)*DT,'ms',time.time()-started,flush=True)
    duration=(t+1)//10
    fields=fields[:duration]
    global_rates=np.c_[fields[:,:400]@model.geo['cell_count_e']/32000,fields[:,400:]@model.geo['cell_count_i']/8000]
    slow[0]=1. if a.D is None else 1-a.D,0.
    np.savez_compressed(folder/'trajectory.npz',field_rate_1ms_hz=fields,global_rate_1ms_hz=global_rates,
        slow_10ms=slow[:duration//10+1],count_e=model.geo['cell_count_e'],count_i=model.geo['cell_count_i'])
    np.savez_compressed(folder/'end_state.npz',**{k:cp.asnumpy(getattr(model,k)) for k in ('r','aux','z','m','qa','ia','qg','ig','history')},tick=model.tick)
    tail=global_rates[max(500,duration//2):,0]
    r10=tail[:len(tail)//10*10].reshape(-1,10).mean(1) if len(tail)>=10 else np.array([])
    status=dict(status='NUMERICAL_FAILURE' if error else 'COMPLETE',error=error,duration_ms=duration,wall_s=time.time()-started,
        tail_mean_E_hz=float(tail.mean()) if len(tail) else None,tail_quiet_fraction=float(np.mean(r10<5)) if len(r10) else None,
        model_states=read(OUT/'operators/definition.json')['total_continuous_states'],
        acceptance='NOT_ASSIGNED')
    write(folder/'status.json',status);print(status,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--kind',choices=['linear','nonlinear'],default='linear')
    ap.add_argument('--D',type=float);ap.add_argument('--duration',type=int,default=1500);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--shared',action='store_true');ap.add_argument('--seed',type=int,default=9108401)
    ap.add_argument('--resource-closure',choices=['mean','lognormal'],default='mean');main(ap.parse_args())
