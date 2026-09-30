"""Finite-time separation of nearby full DDE states, including delay history.

This is a deterministic finite-window diagnostic, not by itself a proof of a
chaotic attractor. Z is fixed and M evolves. Repeat in epsilon and time step
before promoting a positive finite-time exponent.
"""
from endpoint_runs import EndpointIntegrator
from runner import GraphChunk
from common import *
import argparse


def main(a):
    s=model();state,history=checkpoint_initial(a.source);z=np.load(a.source)
    dt=float(z['dt_ms']) if 'dt_ms' in z else float(read(Path(a.source).parent/'contract.json')['dt_ms'])
    s.set_Z(state[11]);assert len(history)==round(s.delays[-1]/dt)+1
    engines=[EndpointIntegrator(s,dt=dt,initial=state,history=history,dynamic_z=False,
                                dynamic_m=True,device=a.device) for _ in range(2)]
    cp=engines[0].cp;scale=cp.asarray([10,10,100,100,10,10,10,10,100,100,.1,1,100,100])[:,None]
    graphs=[GraphChunk(e) for e in engines]
    y0=engines[0].y.copy();h0=engines[0].history.copy()
    a0=graphs[0].run();a1=graphs[1].run()
    assert np.array_equal(a0,a1) and bool(cp.array_equal(engines[0].y,engines[1].y))
    assert bool(cp.array_equal(engines[0].history,engines[1].history))
    for e in engines:e.y[:]=y0;e.history[:]=h0
    rng=np.random.default_rng(4491);perturb=cp.zeros_like(engines[0].y)
    perturb[5]=cp.asarray(rng.normal(size=s.P))*10
    norm=float(cp.sqrt(cp.mean((perturb/scale)**2)));engines[1].y+=perturb*(a.epsilon/norm)
    def endpoint(e):
        e.eval_rhs(e.y,0,e.f,e.rate2);e.history[0]=e.rate2
    endpoint(engines[1])
    e0,e1=engines;dim=e0.y.size+e0.history.size
    def distance():
        return float(cp.sqrt((cp.sum(((e1.y-e0.y)/scale)**2)+cp.sum(((e1.history-e0.history)/.1)**2))/dim))
    initial_norm=distance();previous=initial_norm;logs=[];rates=[];start=time.time()
    dest=OUT/'lyapunov'/a.label;dest.mkdir(parents=True,exist_ok=True)
    for k in range(a.duration//50):
        rates.append(graphs[0].run());graphs[1].run();current=distance()
        assert current>1e-30 and np.isfinite(current),('Unresolved perturbation',k,current)
        growth=np.log(current/previous);logs.append(growth)
        factor=a.epsilon/current;e1.y[:]=e0.y+factor*(e1.y-e0.y)
        e1.history[:]=e0.history+factor*(e1.history-e0.history);endpoint(e1)
        assert float(cp.max(abs(e1.y[11]-e0.y[11])))==0.
        previous=distance()
        if (k+1)%20==0:
            elapsed=(k+1)*50;discard=min(a.discard,elapsed-50)//50
            exponent=float(np.sum(logs[discard:])/((len(logs)-discard)*.05))
            log(a.label,'t_ms',elapsed,'finite_time_per_s',exponent,'wall_s',time.time()-start)
            write(dest/'progress.json',dict(status='RUNNING',time_ms=elapsed,finite_time_per_s=exponent))
    discard=a.discard//50;values=np.array(logs);R=np.concatenate(rates)
    np.savez_compressed(dest/'trajectory.npz',group_rate_hz=R.astype('float32'),
        global_E_hz=R[:,s.E]@s.mean_weights,log_growth=values,time_ms=(np.arange(len(values))+1)*50,
        final_state=e0.y.get(),final_history=e0.history.get(),final_tick=0,dt_ms=dt)
    row=dict(status='COMPLETE',source=a.source,D=s.D,Z='held',M='dynamic',dt_ms=dt,
        epsilon=a.epsilon,renormalization_ms=50,discard_ms=a.discard,duration_ms=a.duration,
        twin_replay_bitwise=True,
        exponent_per_s=float(values[discard:].sum()/((len(values)-discard)*.05)),
        block_exponents_per_s=(values[discard:len(values)-(len(values)-discard)%20].reshape(-1,20).sum(1)).tolist(),
        state_scales=cp.asnumpy(scale[:,0]).tolist(),history_scale_rate_per_ms=.1,
        interpretation='Finite-time largest-direction estimate; epsilon/step/window convergence required for chaos claim')
    write(dest/'result.json',row);log('PAIR RESULT',row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--label',required=True)
    p.add_argument('--epsilon',type=float,default=1e-7);p.add_argument('--duration',type=int,default=12000)
    p.add_argument('--discard',type=int,default=2000);p.add_argument('--device',type=int,default=0);main(p.parse_args())
