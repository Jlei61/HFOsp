"""One prespecified timestep check of the nonlinear history-reading diagnostic."""
from common import *
from closure_network_sensitivity import setup, audit
from runner import GraphChunk, summarize
from scipy.ndimage import uniform_filter1d
from closure_network_sensitivity_audit import high_entry
import argparse


def main(device):
    c=read(OUT/'closure_history_step_check_contract.json')
    dest=OUT/'closure_network_sensitivity/units_history_dt025';dest.mkdir(exist_ok=True)
    assert not (dest/'result.json').exists(), 'Retain the completed result'
    s,e=setup('units_history',c['dt_ms'],device)
    init=np.load(OUT/'closure_network_sensitivity/frozen/initial.npz')
    assert np.array_equal(init['history'],np.broadcast_to(init['history'][0],init['history'].shape))
    e.y[:]=e.cp.asarray(init['state'])
    e.history[:]=e.cp.asarray(np.broadcast_to(init['history'][0],e.history.shape).copy())
    qa=audit(s,e,'units_history_dt025')
    assert np.array_equal(qa['initial_state'],init['state'])
    assert np.array_equal(qa['initial_history'][::2],init['history'])
    np.savez_compressed(dest/'initial.npz',state=qa.pop('initial_state'),history=qa.pop('initial_history'))
    graph=GraphChunk(e,milliseconds=10);rates=[];Z=[];M=[];start=time.time()
    for chunk in range(round(c['duration_ms']/10)):
        rates.append(graph.run());Z.append(e.y[11].get());M.append(e.y[10].get())
        if (chunk+1)%100==0:
            assert bool(e.cp.isfinite(e.y).all())
            log('HISTORY STEP',10*(chunk+1),'ms',round(time.time()-start,1),'seconds')
            write(dest/'progress.json',dict(status='RUNNING',time_ms=10*(chunk+1)))
    R=np.concatenate(rates);_,field,whole,count=summarize(R,s)
    z=np.array(Z);m=np.array(M);t=np.arange(len(R))+1.;ts=(np.arange(len(z))+1)*10.
    d=1-z[:,s.E]@s.mean_weights
    fine_entry=high_entry(t,uniform_filter1d(whole,10,mode='nearest'))
    coarse=np.load(OUT/'closure_network_sensitivity/units_history/trajectory.npz')
    old_entry=high_entry(coarse['time_ms'],uniform_filter1d(coarse['global_E_hz'],10,mode='nearest'))
    np.savez_compressed(dest/'trajectory.npz',time_ms=t,group_rate_hz=R.astype('float32'),
        field_E_hz=field.astype('float32'),global_E_hz=whole,cell_counts=count,
        Z=z.astype('float32'),M_current=m.astype('float32'),state_time_ms=ts,D=d,
        final_state=e.y.get(),final_history=e.history.get(),final_tick=0,dt_ms=c['dt_ms'])
    ediff=abs(fine_entry-old_entry) if fine_entry is not None and old_entry is not None else None
    ddiff=float(np.max(abs(d-coarse['D'])))
    passed=ediff is not None and ediff<=c['entry_difference_ms_max'] and ddiff<=c['D_difference_max']
    result=dict(status='COMPARISON_COMPLETE',comparison_pass=passed,coarse_entry_ms=old_entry,
        fine_entry_ms=fine_entry,entry_difference_ms=ediff,D_max_difference=ddiff,
        fine_D9870=float(d[np.flatnonzero(ts==9870)[0]]),fine_D_final=float(d[-1]),qa=qa,
        initial_common_samples_bitwise=True,Z_and_M_dynamic=True,seconds=time.time()-start,
        scope=c['scope'],replacement_promoted=False)
    write(dest/'result.json',result);log('HISTORY STEP COMPLETE',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)
