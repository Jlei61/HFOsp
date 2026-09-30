"""Full delay-state fold-mode matching on every subinterval of a period.

The same variational DDE is integrated with physical delays. Reinitializing
from the independently reconstructed BVP tangent at each section prevents
roundoff from being multiplied through all strongly unstable directions over
an entire period. Keep whole-period monodromy evidence separately.
"""
from rate_floquet import *
from rate_fold_tangent_exact import harmonic_tangent,at_time
import gc


def flow_check(device):
    s=RateField();path=PERIODIC_OUT/'orbits/LPC_A_global_turn1_N1024.npz'
    m=Monodromy(s,path,.1,device)
    x=m.phase_vector();whole=m.matvec(x)
    edges=np.linspace(0,m.n,7,dtype=int);part=x.copy()
    for a,b in zip(edges[:-1],edges[1:]):part=m.interval_matvec(part,int(a),int(b))
    error=float(np.linalg.norm(part-whole)/np.linalg.norm(whole))
    assert error<1e-10,error
    q=dict(status='PASS',relative_partition_vs_whole_error=error,sections=6,
        orbit=str(path),dt_ms=m.dt,scope='Partitioned flow with carried complete history agrees with the existing whole-period GPU variational flow.')
    write(PERIODIC_OUT/'segmented_variational_flow_check.json',q);print(q,flush=True)


def check(label,dts,segments,device):
    assert read(PERIODIC_OUT/'segmented_variational_flow_check.json')['status']=='PASS'
    q=max([read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')],key=lambda q:q['N'])
    z=np.load(q['orbit']);s=RateField();r=z['r'];T=float(z['T']);J=float(z['J'])
    tangent=np.load(PERIODIC_OUT/f'{label}_tangent_N{len(r)}.npz')['tangent']
    data=harmonic_tangent(s,r,T,J,tangent)
    phase=dict(data,derivatives=data['states']*data['lambdas'][None,:,None],
        rate_derivative=data['rate']*data['lambdas'][:,None],dT=0.,dJ=0.)
    print('HARMONIC TANGENT READY',label,flush=True);rows=[]
    for dt in dts:
        m=Monodromy(s,Path(q['orbit']),dt,device,capture=False);cp=m.cp
        v0=at_time(data,m.D,m.dt,xp=cp);f0=at_time(phase,m.D,m.dt,xp=cp)
        vT=at_time(data,m.D,m.dt,T,xp=cp)
        phase_error=float(np.linalg.norm(f0-m.phase_vector())/np.linalg.norm(f0))
        boundary=float(np.linalg.norm(vT-v0+data['dT']*f0)/(np.linalg.norm(v0)+abs(data['dT'])*np.linalg.norm(f0)))
        assert phase_error<1e-10 and boundary<1e-10,(phase_error,boundary)
        edges=np.linspace(0,m.n,segments+1,dtype=int);matches=[]
        for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            a,b=int(a),int(b);start=a*m.dt;end=b*m.dt
            initial=at_time(data,m.D,m.dt,start,xp=cp)
            expected=at_time(data,m.D,m.dt,end,xp=cp)
            output=m.interval_matvec(initial,a,b)
            error=float(np.linalg.norm(output-expected)/(np.linalg.norm(initial)+np.linalg.norm(expected)))
            fi=at_time(phase,m.D,m.dt,start,xp=cp);fe=at_time(phase,m.D,m.dt,end,xp=cp)
            fp=m.interval_matvec(fi,a,b)
            pe=float(np.linalg.norm(fp-fe)/(np.linalg.norm(fi)+np.linalg.norm(fe)))
            row=dict(segment=i,start_ms=start,end_ms=end,fold_relative_defect=error,phase_relative_defect=pe)
            if i in [0,segments//2]:
                bad=np.roll(initial.reshape(-1,s.P),1,axis=1).ravel()
                result=m.interval_matvec(bad,a,b)
                row['permuted_group_control_defect']=float(np.linalg.norm(result-expected)/(np.linalg.norm(initial)+np.linalg.norm(expected)))
            matches.append(row)
        row=dict(dt_ms=m.dt,segments=segments,matches=matches,
            maximum_fold_relative_defect=max(v['fold_relative_defect'] for v in matches),
            maximum_phase_relative_defect=max(v['phase_relative_defect'] for v in matches),
            minimum_negative_control_defect=min(v['permuted_group_control_defect'] for v in matches if 'permuted_group_control_defect' in v),
            phase_reconstruction_relative_error=phase_error,generalized_boundary_relative_error=boundary)
        rows.append(row);print('SEGMENTED FOLD',label,{k:v for k,v in row.items() if k!='matches'},flush=True)
        del m;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        write(PERIODIC_OUT/f'{label}_segmented_mode_checks.json',dict(status='DIAGNOSTIC_COMPLETE' if len(rows)==len(dts) else 'RUNNING',
            label=label,orbit=q['orbit'],N=len(r),J_EE_core=J,T_ms=T,dT=data['dT'],dJ=data['dJ'],checks=rows,
            equation='w(t)=dY(t/T)/ds - (t/T)*T_prime*y_dot(t); w_dot=A(t)w+A_delay(t)w_delayed; w(T)-w(0)+T_prime*f(0)=0',
            scope='Independent physical-time variational matching of the periodic BVP tangent on all sections. All nine states and the full delayed rate history are retained. This diagnostic does not classify full Floquet stability or automatically promote a fold.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='*')
    p.add_argument('--dt',type=float,nargs='+',default=[.1,.05,.025])
    p.add_argument('--segments',type=int,default=16);p.add_argument('--device',type=int,default=0)
    p.add_argument('--flow-check',action='store_true');a=p.parse_args()
    if a.flow_check:flow_check(a.device)
    for label in a.labels:check(label,a.dt,a.segments,a.device)
