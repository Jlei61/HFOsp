#!/usr/bin/env python3
"""Frozen-path mathematics and conditional planes for prescribed native spatial Z."""
from topic4_spatial_boundary_common import ROOT, OUT as PREVIOUS, OLD, REFERENCE, read, write
from topic4_mixed_timescale_rate import MixedTimescaleSystem
from run_topic4_spatial_boundary_rate import Stepper
from plot_topic4_z_bifurcation_audit import bounded_mean
from topic4_e_only_z_tangent import sample_spectrum, EOnlyTangent
from audit_topic4_e_only_stability import count_unstable
import numpy as np
import time
import argparse

OUT=ROOT/'results/topic4_sef_hfo/prescribed_z_frozen_phase_v1'
TIMES=(8800,10150,10500)


def source():
    a=np.load(REFERENCE/'trajectory.npz');b=np.load(OLD/'external_input.npz');w=b['count_e']
    mapping=np.bincount(b['cell_e']*400+a['cell_e'],minlength=40000).reshape(100,400)/w[:,None]
    return a,b,a['z_field_10ms']@mapping.T


def field_at(fields,tm):
    k=int(tm//10);alpha=tm/10-k
    return (1-alpha)*fields[k]+alpha*fields[min(k+1,len(fields)-1)]


def protocol():
    write(OUT/'protocol.json',{'status':'DEFINED_BEFORE_RUNS','model':'Unfitted mixed-timescale corrected rate; same fixed C substrate, E-only postsynaptic Z, M off.',
        'source':str(PREVIOUS),'Z':'Actual native spatial Z, neuron-weighted 20x20-to-10x10 projection; no replacement by mean Z.',
        'frozen_path':'Native pre-refill path 0–10.68 s. Constant reference background input for autonomous fixed-point spectra; private input variance retained in Phi. OU variations are absent only from this frozen reference analysis.',
        'conditional_planes':'Three actual prescribed-Z/OU replay snapshots, 8.8, 10.15 and 10.5 s. Vary E/I rates by bounded common shifts. Hold all hidden states, Z and current external input fixed at the exact rate update.',
        'readouts':'Equilibrium branches and full delayed-map eigenmodes; true replay trajectory; conditional zero-drift contours and direction field. Static residual eigenvalues are not dynamical stability.',
        'scope':'Mathematics of this explicitly identified approximate model. Its known native-boundary mismatch remains; not a native SNN Hopf attribution or a closed 2D model.',
        'maximum_workers':2,'stop':'Deliver figures, numerical checks and interpretation; no biological tuning or official Fig5 replacement.'})


def branches():
    start=time.time();s=MixedTimescaleSystem(quadrature=33);_,_,fields=source();times=np.unique(np.r_[np.arange(0,10681,100),10680,TIMES])
    rows=[];arrays={}
    for name,ts,guess in [('low',times,np.full(200,.00005)),('high',times[::-1],np.full(200,.4))]:
        r=guess;segment=0;last_t=None
        for tm in ts:
            z=field_at(fields,tm);rr,err,ok=s.solve(z,r)
            if not ok:
                # Smaller parameter steps can distinguish a coarse solver jump
                # from a resolved continuation; a failure is not called a fold.
                if last_t is not None:
                    trial=r.copy();good=True
                    for tt in np.linspace(last_t,tm,11)[1:]:
                        trial,e,good=s.solve(field_at(fields,tt),trial)
                        if not good:break
                    if good:rr,err,ok=trial,e,True
                if not ok:
                    rows.append({'branch':name,'time_ms':float(tm),'valid':False,'residual':err,'segment':segment})
                    write(OUT/'branch_status.json',{'status':'RUNNING','rows':rows});last_t=None;segment+=1
                    continue
            r=rr;last_t=tm;key=f'{name}_{int(tm)}';arrays[key]=r.copy()
            rows.append({'branch':name,'time_ms':float(tm),'valid':True,'residual':err,'segment':segment,
                'mean_Z':float(np.average(z,weights=s.m.count_e)),
                'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),
                'I_mean_hz':float(np.average(r[100:],weights=s.m.count_i)*1000)})
            if len(rows)%10==0:write(OUT/'branch_status.json',{'status':'RUNNING','rows':rows,'seconds':time.time()-start})
    np.savez_compressed(OUT/'branches.npz',**arrays)
    write(OUT/'branch_status.json',{'status':'COMPLETE','rows':rows,'seconds':time.time()-start,
        'scope':'Direct continuation from low-rate and high-rate seeds; coincident branches are identified in subsequent analysis. Missing roots are unresolved, not proven folds.'})


def snapshot_planes(extra=False):
    start=time.time();s=MixedTimescaleSystem(quadrature=33);st=Stepper(system=s);m=s.m;n=s.n
    _,src,fields=source();expected=src['expected_rate_per_ms'];saved=np.load(PREVIOUS/'mixed_timescale/native_replay.npz')
    payload={};max_error=0.;checked=0
    times=(11100,12500) if extra else TIMES
    segments=[(9400,range(94000,125001))] if extra else [(8800,range(88000,88001)),(9400,range(94000,105001))]
    previous_rows=[]
    if extra:
        old=np.load(OUT/'exact_phase_snapshots.npz');payload={k:old[k].copy() for k in old.files}
        previous=read(OUT/'plane_status.json');previous_rows=previous['rows'];checked=previous['exact_replay_frames']
    for ck,steps in segments:
        st.restore(PREVIOUS/'mixed_timescale_checkpoints'/f'native_replay_t{ck}ms.npz')
        for step in steps:
            tm=step*s.dt
            # Preserve the original integer-step interpolation arithmetic so
            # floating-point differences cannot grow during replay.
            index=step//100;alpha=(step%100)/100
            z=(1-alpha)*fields[index]+alpha*fields[index+1]
            ne=expected[step,0].astype(float);ni=expected[step,1].astype(float)
            r=st.r.copy();nr=st.step(z,ne,ni)
            if step in [t*10 for t in times]:
                prefix=f't{int(tm)}'
                for key,val in [('r',r),('next_r',nr),('current',st.c),('z',z),('expected_e',ne),('expected_i',ni)]:payload[prefix+'_'+key]=val.copy()
            if (step+1)%10==0:
                observed=(nr.reshape(2,n)*1000).astype(np.float32)
                error=float(np.max(abs(observed-saved['fields_hz'][step//10])));max_error=max(max_error,error);checked+=1
                assert np.array_equal(observed,saved['fields_hz'][step//10]),(step,error)
            if step%5000==0:write(OUT/'plane_status.json',{'status':'REPLAYING','time_ms':tm,'seconds':time.time()-start})
    np.savez_compressed(OUT/'exact_phase_snapshots.npz',**payload)
    rows=previous_rows
    for tm in times:
        prefix=f't{tm}';r=payload[prefix+'_r'];c=payload[prefix+'_current'];z=payload[prefix+'_z'];ne=payload[prefix+'_expected_e'];ni=payload[prefix+'_expected_i']
        def drift(rr):
            e,i=rr[:n],rr[n:];te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms
            ex=np.r_[te*(m.v_ee@e+m.j_ext_e_mv**2*ne),ti*(m.v_ie@e+m.j_ext_i_mv**2*ni)]
            inh=np.r_[te*z*z*(m.v_ei@i),ti*(m.v_ii@i)]
            mu=np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]]
            d=(s.phi(mu,ex,inh)-rr)/s.tr
            return np.array([np.average(d[:n],weights=m.count_e),np.average(d[n:],weights=m.count_i)])*1e6
        center=np.array([np.average(r[:n],weights=m.count_e),np.average(r[n:],weights=m.count_i)])*1000
        actual=(payload[prefix+'_next_r']-r)/s.dt
        actual=np.array([np.average(actual[:n],weights=m.count_e),np.average(actual[n:],weights=m.count_i)])*1e6
        error=float(max(abs(actual-drift(r))));assert error<1e-6,error
        trail=saved['fields_hz'][tm-51:tm+50]
        trajectory=np.c_[np.average(trail[:,0],axis=1,weights=m.count_e),np.average(trail[:,1],axis=1,weights=m.count_i)]
        # Broad shared physical ranges retain the projected zero-drift curves.
        xs=np.linspace(0,500,41);ys=np.linspace(0,800,41);X,Y=np.meshgrid(xs,ys);U=np.zeros_like(X);V=U.copy()
        ers=[bounded_mean(r[:n],m.count_e,x/1000,.5) for x in xs]
        irs=[bounded_mean(r[n:],m.count_i,y/1000,1.) for y in ys]
        for j in range(len(ys)):
            for k in range(len(xs)):U[j,k],V[j,k]=drift(np.r_[ers[k],irs[j]])
        np.savez_compressed(OUT/f'plane_{tm}.npz',X=X,Y=Y,U=U,V=V,center_hz=center,trajectory_hz=trajectory,
            trajectory_time_ms=np.arange(tm-50,tm+51),mean_Z=np.average(z,weights=m.count_e),center_drift_hz_per_s=actual)
        rows.append({'time_ms':tm,'center_hz':center.tolist(),'center_drift_hz_per_s':actual.tolist(),'center_identity_error':error,
            'E_zero_contour':bool(U.min()<0<U.max()),'I_zero_contour':bool(V.min()<0<V.max())})
        write(OUT/'plane_status.json',{'status':'MAKING_PLANES','rows':rows,'seconds':time.time()-start})
    write(OUT/'plane_status.json',{'status':'COMPLETE','rows':rows,'exact_replay_frames':checked,'maximum_replay_error':max_error,
        'seconds':time.time()-start,'units':'Rates Hz, drift Hz/s. Nullclines are conditional projected zero-drift contours. Hidden states evolve along the real trajectory.'})


def spectra():
    s=MixedTimescaleSystem(quadrature=33);_,_,fields=source();a=np.load(OUT/'branches.npz');rows=[]
    # Actual replay windows plus earlier reference points, with both starting branches.
    for tm in (0,2000,2200,4000,6000,7600,8000,8700,8800,9000,10150,10500):
        seen=[]
        for branch in ('low','high'):
            key=f'{branch}_{tm}'
            if key not in a.files:continue
            r=a[key];z=field_at(fields,tm)
            if any(np.max(abs(r-v))<1e-7 for v in seen):continue
            seen.append(r);started=time.time()
            spec=sample_spectrum(s,r,z,frequencies=(0,3,15,40),k=4)
            assert max(x['full_map_residual'] for x in spec['roots'])<1e-7
            count=count_unstable(s,r,z)
            if count['status']!='PASS':count=count_unstable(s,r,z,grid_lengths=(1600,3200))
            row={'time_ms':tm,'branch':branch,'mean_Z':float(np.average(z,weights=s.m.count_e)),
                'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),'spectrum':spec,'root_count':count,'seconds':time.time()-started}
            rows.append(row);write(OUT/f'spectrum_{branch}_{tm}.json',row)
            write(OUT/'spectrum_status.json',{'status':'RUNNING','rows':rows})
            print(branch,tm,count['unstable_roots'],spec['roots'][0],flush=True)
    write(OUT/'spectrum_status.json',{'status':'COMPLETE','rows':rows})


def intersections():
    from scipy.optimize import root
    s=MixedTimescaleSystem(quadrature=33);m=s.m;n=s.n;a=np.load(OUT/'exact_phase_snapshots.npz');rows=[]
    for tm in TIMES:
        p=f't{tm}';r=a[p+'_r'];c=a[p+'_current'];z=a[p+'_z'];ne=a[p+'_expected_e'];ni=a[p+'_expected_i']
        def f(xy):
            rr=np.r_[bounded_mean(r[:n],m.count_e,xy[0]/1000,.5),bounded_mean(r[n:],m.count_i,xy[1]/1000,1.)]
            e,i=rr[:n],rr[n:];te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms
            ex=np.r_[te*(m.v_ee@e+m.j_ext_e_mv**2*ne),ti*(m.v_ie@e+m.j_ext_i_mv**2*ni)]
            inh=np.r_[te*z*z*(m.v_ei@i),ti*(m.v_ii@i)]
            mu=np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]];dr=(s.phi(mu,ex,inh)-rr)/s.tr
            return np.array([np.average(dr[:n],weights=m.count_e),np.average(dr[n:],weights=m.count_i)])*1e6
        xy=np.array([np.average(r[:n],weights=m.count_e),np.average(r[n:],weights=m.count_i)])*1000
        sol=root(f,xy,tol=1e-10);err=float(abs(f(sol.x)).max());assert err<1e-5,err
        h=.01;J=np.column_stack([(f(sol.x+np.eye(2)[i]*h)-f(sol.x-np.eye(2)[i]*h))/(2*h) for i in range(2)])
        ev=np.linalg.eigvals(J)
        rows.append({'time_ms':tm,'intersection_hz':sol.x.tolist(),'projected_zero_drift_residual_hz_per_s':err,
            'conditional_eigenvalues_per_s':[[float(x.real),float(x.imag)] for x in ev]})
    write(OUT/'conditional_zero_drift_intersections.json',{'rows':rows,
        'scope':'Zeros and derivative of frozen-hidden-state projected 2D field only. Not full delayed spatial model equilibria or eigenvalues.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['protocol','branches','planes','extra_planes','spectra','intersections']);args=p.parse_args()
    try:{'protocol':protocol,'branches':branches,'planes':snapshot_planes,'extra_planes':lambda:snapshot_planes(True),'spectra':spectra,'intersections':intersections}[args.part]()
    except Exception as exc:
        write(OUT/f'{args.part}_failure.json',{'error':repr(exc)});raise
