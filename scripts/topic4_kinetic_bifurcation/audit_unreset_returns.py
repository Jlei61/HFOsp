"""Native forward orbit from a corrected generalized-return point, without reset.

Interpolation is used only to read where each forward crossing intersects the
original section. It is never fed into the dynamics. This checks whether a
corrected numerical return represents the observed attracting native orbit.
"""
from generalized_return_audit import *


def run(a):
    rcfg=read(a.corrected/'config.json');source=Path(rcfg['source']);cfg=read(source/'config.json')
    folder=OUT/'unreset_return_audits'/a.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);normal=cp.asarray(np.load(a.corrected/'section_normal.npy'))
    m.restore(a.corrected/'best_state');reference=coords.pack(m);initial_step=m.step_index
    count=np.bincount(m.geo['group_cell'],weights=np.where(m.geo['population']==0,m.geo['group_size'],0),minlength=1600)
    write(folder/'config.json',dict(corrected_source=str(a.corrected.resolve()),D=cfg['D'],duration_ms=a.duration,
        native_step_ms=DT,method='Unchanged native forward iterations; no shooting correction, fractional initial states, or return resets',
        section='Fixed original normal through corrected point, using original source metric',
        Z='Frozen prescribed spatial field',M='Dynamic'))
    started=time.time();last=started;previous=cp.zeros_like(reference);pv=0.;deltas=[];values=[];anchor=None
    returns=[];crossings=[];rates=[];fields=[];M=[];block=cp.zeros(m.P)
    for step in range(1,round(a.duration/DT)+1):
        m.advance_step();block+=m.activity;d=coords.pack(m)-reference;v=float(cp.dot(normal,d).get())
        if pv*v<0:
            direction='forward' if pv<0 else 'backward'
            crossings.append(dict(left_ms=(step-1)*DT,right_ms=step*DT,direction=direction))
            if direction=='forward' and anchor is None:
                anchor=step-1;deltas=[previous,d];values=[pv,v]
        elif anchor is not None:
            deltas.append(d);values.append(v)
        if len(deltas)==6:
            estimates=[]
            for order in (3,5):
                alpha=crossing(values,order);defect=cp.zeros_like(reference)
                for c,q in zip(coefficients(alpha,order),deltas):defect+=float(c)*q
                estimates.append(dict(order=order,time_ms=(anchor+alpha)*DT,
                    weighted_distance_from_initial_root=float(cp.linalg.norm(defect).get()),
                    block_distances={k:float(cp.linalg.norm(defect[s]).get()) for k,s in coords.slices.items()}))
                del defect
            row=dict(index=len(returns)+1,estimates=estimates,diagnostics=m.diagnostics())
            returns.append(row);write(folder/'returns.json',returns);print('unreset return',row,flush=True)
            deltas=[];values=[];anchor=None
        if step%10==0:
            rate=block*1000.;rates.append(cp.asnumpy(cp.r_[m.e_weights@rate,m.region_weights@rate]))
            fields.append(cp.asnumpy(cp.bincount(m.observable_cell,weights=rate*m.e_sizes,minlength=1600))/np.maximum(count,1))
            M.append(cp.asnumpy(cp.r_[m.e_weights@m.M,m.region_weights@m.M]));block.fill(0.)
        previous=d;pv=v
        if time.time()-last>20:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),elapsed_ms=step*DT,
                completed_forward_returns=len(returns),section_value=v,wall_s=time.time()-started));last=time.time()
            print(a.label,step*DT,len(returns),flush=True)
    np.savez_compressed(folder/'trajectory.npz',rate_1ms=np.asarray(rates),field_1ms=np.asarray(fields),M_1ms=np.asarray(M),
        bin_centers_ms=np.arange(len(rates))+.5,count_e=count)
    endpoint=folder/'endpoint';endpoint.mkdir();m.save(endpoint)
    write(endpoint/'config.json',dict(cfg,initial_ms=m.step_index*DT,resumed_from=str((a.corrected/'best_state').resolve()),
        numerical_initial_state='Unreset original native continuation from corrected generalized-return point'))
    write(folder/'result.json',dict(status='UNRESET_NATIVE_RETURN_AUDIT_COMPLETE',returns=returns,crossings=crossings,
        diagnostics=m.diagnostics(),wall_s=time.time()-started,
        acceptance='Measured finite native orbit persistence; examine state distances and times, not merely rate similarity'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrected',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--duration',type=float,default=1500.);ap.add_argument('--device',type=int,default=0)
    run(ap.parse_args())
