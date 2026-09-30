"""Continue an already saved failed return window without repeating its prefix.

The original section and metric are preserved. Both crossing directions and
actual rates are recorded; a search-window failure is never a bifurcation.
"""
from generalized_return_audit import *


def run(a):
    failed=a.failed;rcfg=read(failed/'config.json');source=Path(rcfg['source'])
    cfg=read(source/'config.json');folder=OUT/'return_window_diagnostics'/a.label
    folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);x=coords.pack(m);initial_step=m.step_index
    normal=cp.asarray(np.load(failed/'section_normal.npy'))
    m.advance_step();d1=coords.pack(m)-x;m.advance_step();d2=coords.pack(m)-x
    geometry=dict(one_step_displacement=float(cp.linalg.norm(d1).get()),
        one_step_section_value=float(cp.dot(normal,d1).get()),
        two_step_displacement=float(cp.linalg.norm(d2).get()),
        two_step_section_value=float(cp.dot(normal,d2).get()))
    del d1,d2
    m.restore(failed/'failed_window_endpoint');begin_step=m.step_index
    write(folder/'config.json',dict(failed_correction=str(failed.resolve()),source=str(source.resolve()),
        D=cfg['D'],start_elapsed_ms=(begin_step-initial_step)*DT,maximum_elapsed_ms=a.maximum_time,
        section='Exact same section normal and source metric as failed correction',geometry=geometry,
        object='First actual forward return after a failed search window, not a corrected root'))
    start=time.time();last=start;rows=[];crossings=[];deltas=[];values=[];anchor=None
    previous=coords.pack(m)-x;pv=float(cp.dot(normal,previous).get())
    block=cp.zeros(m.P);rates=[]
    for absolute_step in range(begin_step+1,initial_step+round(a.maximum_time/DT)+6):
        m.advance_step();block+=m.activity;d=coords.pack(m)-x;v=float(cp.dot(normal,d).get())
        elapsed=(m.step_index-initial_step)*DT
        rows.append([elapsed,v])
        if pv*v<0:
            direction='forward' if pv<0 else 'backward'
            crossings.append(dict(left_time_ms=elapsed-DT,right_time_ms=elapsed,direction=direction))
            print('crossing',crossings[-1],flush=True)
            if direction=='forward' and anchor is None:
                anchor=m.step_index-initial_step-1;deltas=[previous,d];values=[pv,v]
        elif anchor is not None:
            deltas.append(d);values.append(v)
        if (m.step_index-begin_step)%10==0:
            rate=block*1000.;rates.append(cp.asnumpy(cp.r_[elapsed,m.e_weights@rate,m.region_weights@rate]));block.fill(0.)
        if time.time()-last>20:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),elapsed_ms=elapsed,section_value=v,
                crossings=crossings,wall_s=time.time()-start));last=time.time()
            print(a.label,elapsed,v,flush=True)
        if len(deltas)==6:break
        previous=d;pv=v
    np.savez_compressed(folder/'trajectory.npz',section=np.asarray(rows),rate_1ms=np.asarray(rates))
    endpoint=folder/'endpoint';endpoint.mkdir();m.save(endpoint)
    write(endpoint/'config.json',dict(cfg,initial_ms=m.step_index*DT,source=str(source.resolve()),
        numerical_initial_state='Original native endpoint of return-window diagnostic'))
    estimates=[]
    if len(deltas)==6:
        for order in (3,5):
            alpha=crossing(values,order);defect=cp.zeros_like(x)
            for c,d in zip(coefficients(alpha,order),deltas):defect+=float(c)*d
            estimates.append(dict(order=order,anchor_steps=anchor,return_fraction=alpha,
                return_time_ms=(anchor+alpha)*DT,weighted_full_state_defect=float(cp.linalg.norm(defect).get())))
    write(folder/'result.json',dict(status='FORWARD_RETURN_FOUND' if estimates else 'NO_FORWARD_RETURN_IN_EXTENDED_WINDOW',
        crossings=crossings,estimates=estimates,diagnostics=m.diagnostics(),wall_s=time.time()-start,
        interpretation='Observed return timing and state defect only; no existence, stability or bifurcation-type acceptance'))
    print(estimates,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--failed',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--maximum-time',type=float,default=800.);ap.add_argument('--device',type=int,default=1)
    run(ap.parse_args())
