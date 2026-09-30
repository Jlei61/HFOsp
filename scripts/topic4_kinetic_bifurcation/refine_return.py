"""FP64 full-state returns at native 0.1 ms resolution.

This checks a candidate trajectory return, not a periodic-orbit shooting solve.
Each lag is compared in all density coefficients, dynamic M, synaptic currents
and ordered delay memory. Nearby steps resolve coarse section-time error.
"""
from search_recurrence import *
import signal


def run(args):
    source=Path(args.source);config=read(source/'config.json')
    folder=OUT/'return_refinement'/args.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(config['D'],config['degree'],config['voltage_dv'],args.device,
                        basis_mode=config.get('basis_mode','legacy'))
    m.restore(source);initial=m.step_index*DT;start=capture(m)
    cfg=dict(config,initial_ms=initial,resumed_from=str(source.resolve()),precision='FP64',lags_ms=args.lags,radius_ms=args.radius)
    write(folder/'config.json',cfg)
    stop=[False]
    signal.signal(signal.SIGTERM,lambda signum,frame:stop.__setitem__(0,True))
    weights=cp.asarray(m.geo['group_size']/40000.)
    windows=[(max(1,round((lag-args.radius)/DT)),round((lag+args.radius)/DT)) for lag in args.lags]
    end=max(hi for lo,hi in windows);records=[];best={};traces=[];started=time.time();last=started;status='COMPLETE'
    for step in range(1,end+1):
        rate=m.advance_step()*1000/DT
        traces.append(cp.asnumpy(cp.r_[m.e_weights@rate,m.region_weights@rate]))
        for j,(lo,hi) in enumerate(windows):
            if lo<=step<=hi:
                state=capture(m);sep=separation(start,state,weights)
                row=dict(window=j,target_lag_ms=args.lags[j],actual_lag_ms=step*DT,**sep)
                records.append(row)
                if j not in best or row['score']<best[j][0]['score']:best[j]=(row,state)
            if step==hi:
                save_state(folder/f'best_window_{j}',best[j][1],cfg)
                write(folder/'completed_windows.json',dict(best=[x[0] for x in best.values()],all_steps=records))
        if time.time()-last>20:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),elapsed_ms=step*DT,
                wall_s=time.time()-started,best=[x[0] for x in best.values()]))
            print('return refinement',args.label,step*DT,[x[0]['score'] for x in best.values()],flush=True);last=time.time()
        if stop[0]:
            status='INTERRUPTED_SIGTERM';break
    for j,(row,state) in best.items():save_state(folder/f'best_window_{j}',state,cfg)
    np.savez_compressed(folder/'trace.npz',rate_0p1ms=np.asarray(traces))
    if stop[0]:
        save_state(folder/'interrupted_state',capture(m),cfg)
    write(folder/'result.json',dict(status=status,best=[x[0] for x in best.values()],all_steps=records,
        diagnostics=m.diagnostics(),wall_s=time.time()-started,
        interpretation='Uncorrected trajectory return. A small projected or full-state error alone does not prove periodicity or Floquet stability.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--lags',type=float,nargs='+',required=True)
    ap.add_argument('--radius',type=float,default=2.);ap.add_argument('--device',type=int,default=0)
    run(ap.parse_args())
