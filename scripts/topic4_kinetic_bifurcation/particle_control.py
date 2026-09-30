"""Finite-population control with exactly the candidate density's coupling.

Population replication changes sampling fluctuations, not coupling or Poisson
statistics per cell. This diagnoses the deterministic limit; it is not itself
a proof of correspondence with the accepted native SNN.
"""
from autonomous_density import *

CUDA = r'''
extern "C" __global__ void cells(
 const int* group,const double* theta,const double* tm,const double* ratio,
 const double* drive,const int* refs,const unsigned char* ext,
 const double* recurrentE,const double* recurrentI,const double* nativeTheta,const double* nativeZ,double* nativeM,
 const double* groupZ,const double* groupM,
 const int* momentGroup,const double* momentM,int* momentCounts,
 double* sa,double* ia,double* v,int* ref,int* counts,int N,int k,
 double ar,double ad,double jump,int microscopic){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int g=group[i];
 sa[i]=ar*sa[i]+jump*ratio[g]*(double)ext[(long long)k*N+i];
 ia[i]=ad*ia[i]+(1-ad)*sa[i];ref[i]=max(0,ref[i]-1);
 bool spike=false;double threshold=(microscopic&1)?nativeTheta[i]:theta[g];
 if(ref[i]==0){double decay=exp(-.1/tm[g]);
  double u=ia[i]+(microscopic?recurrentE[g]-((microscopic&2)?nativeZ[i]:groupZ[g])*recurrentI[g]-.0005*((microscopic&4)?nativeM[i]:((microscopic&8)?momentM[momentGroup[i]]:groupM[g])):drive[g]);
  v[i]=u+(v[i]-u)*decay;if(v[i]>=threshold){v[i]=11.;ref[i]=refs[g];atomicAdd(counts+g,1);spike=true;}}
 else v[i]=11.;
 if(spike && (microscopic&8))atomicAdd(momentCounts+momentGroup[i],1);
 if(i%40000<32000){nativeM[i]*=1.-.1/1000.;if(spike)nativeM[i]+=1.;}
}
extern "C" __global__ void populations(
 const int* counts,const double* sizes,const unsigned char* pop,
 double* M,double* history,double* activity,int P,int depth,int step){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double r=counts[g]/sizes[g];activity[g]=r;history[(long long)(step%depth)*P+g]=r;
 if(!pop[g])M[g]=M[g]*(1.-.1/1000.)+r;
}
extern "C" __global__ void moments(
 const int* counts,const double* sizes,const unsigned char* pop,double* M,int P){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 if(!pop[g])M[g]=M[g]*(1.-.1/1000.)+counts[g]/sizes[g];
}
'''


def run_control(args):
    if args.joint_strata:
        assert args.joint_strata>=1 and not any((args.microscopic,args.individual_theta,args.individual_z,args.individual_M,args.z_quadrature,args.z_permutation_seed is not None)), 'Joint strata use empirical threshold and Z means with dynamic stratum M; no mixed pilot flags'
    if args.z_permutation_seed is not None:
        assert args.individual_z and not any((args.microscopic,args.individual_theta,args.individual_M,args.z_quadrature)), 'Exchangeability control requires exact Z with identical parent thresholds and shared parent M'
    assert not args.relabel_private_with_z or args.z_permutation_seed is not None
    if args.z_quadrature:
        assert args.z_quadrature>=1 and not any((args.microscopic,args.individual_theta,args.individual_z,args.individual_M)), 'The resource-strata pilot varies only Z representation'
    theta_individual=args.microscopic or args.individual_theta
    z_individual=args.microscopic or args.individual_z or bool(args.z_quadrature)
    m_individual=args.microscopic or args.individual_M
    micro_flags=int(theta_individual)+2*int(z_individual)+4*int(m_individual)
    tag = '_microscopic' if args.microscopic else (f'_individual_flags{micro_flags}' if micro_flags else '')
    if args.z_quadrature:tag=f'_Zquadrature{args.z_quadrature}'
    if args.z_permutation_seed is not None:tag+=f'_Zperm{args.z_permutation_seed}'
    if args.relabel_private_with_z:tag+='_relabel_private'
    if args.joint_strata:tag=f'_joint_strata{args.joint_strata}'
    if args.shared_noise:tag+='_native_shared_OU'
    folder = OUT/'particle_controls'/'selected_g40'/f'D{args.D:.6f}_Nscale{args.scale}_seed{args.seed}_{args.duration:g}ms{tag}'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    model = AutonomousDensity(args.D, degree=6, device=args.device)
    del model.F, model.Q, model.flux
    cp.get_default_memory_pool().free_all_blocks()
    group = np.tile(model.geo['cell_group'], args.scale)
    N = len(group)
    dgroup = cp.asarray(group)
    native_geometry = np.load(ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916/coarse_40/geometry.npz')
    native_theta = cp.asarray(np.tile(np.r_[native_geometry['vtheta_e'], native_geometry['vtheta_i']], args.scale))
    z_field=field_at(args.D)[0];resource_definition=None;permutation_definition=None
    if args.z_permutation_seed is not None:
        permutation=np.arange(len(z_field));permutation_rng=np.random.default_rng(args.z_permutation_seed)
        parent=model.geo['cell_group']
        for g in np.flatnonzero(model.geo['population']==0):
            ids=np.flatnonzero(parent==g);permutation[ids]=permutation_rng.permutation(ids)
        assert np.array_equal(parent[permutation],parent)
        assert np.array_equal(np.sort(permutation),np.arange(len(permutation)))
        permuted=z_field[permutation]
        mean_error=float(np.max(abs(np.bincount(parent,weights=permuted)-np.bincount(parent,weights=z_field))/model.geo['group_size']))
        assert mean_error<1e-14
        np.savez_compressed(folder/'resource_permutation.npz',original_cell_source_index=permutation)
        permutation_definition=dict(seed=args.z_permutation_seed,
            private_inputs_relabelled_with_Z=bool(args.relabel_private_with_z),
            operation='Permute the exact empirical Z values only within each original E parent group; the same original-cell permutation is repeated across Nscale copies',
            invariants='Full within-parent Z multiset, parent threshold, dynamic shared M, communication and spatial-bin readout',
            within_parent_resource_mean_error=mean_error,
            law_invariance='Private Poisson histories are iid within each parent; all other dynamics and the reported spatial/core observables are exchangeable within that parent',
            scope='Law-preserving paired stochastic-history diagnostic for the Z-only grouped communication model; not a symmetry claim about the heterogeneous native graph')
        z_field=permuted
    if args.z_quadrature:
        from z_quadrature import resource_strata,quantized_field
        strata=resource_strata(model.geo,field_at(.225)[0],args.z_quadrature)
        represented,node_z=quantized_field(z_field,strata)
        parent_z=np.bincount(model.geo['cell_group'],weights=represented)/model.geo['group_size']
        mean_error=float(np.max(abs(parent_z-cp.asnumpy(model.Z))))
        assert mean_error<1e-14
        resource_definition=dict(levels=args.z_quadrature,
            membership='Stable resource-order quantiles within each original E group; the ordering is fixed by the reference Z path, independently of output activity',
            node_values='Exact empirical subgroup mean at the tested D',
            original_group_count=model.P,resource_strata=len(node_z),
            within_parent_mean_error=mean_error,
            E_resource_RMS_error=float(np.sqrt(np.mean((represented[:32000]-z_field[:32000])**2))),
            E_resource_max_error=float(np.max(abs(represented[:32000]-z_field[:32000]))),
            M='Original parent-group mean remains dynamic; only the represented Z dispersion changes')
        np.savez_compressed(folder/'resource_quadrature.npz',**strata,node_Z=node_z,represented_Z=represented)
        z_field=represented
    native_z = cp.asarray(np.tile(z_field, args.scale))
    native_m = cp.zeros(N)
    moment_group=dgroup;moment_m=model.M;moment_counts=cp.zeros(model.P,dtype=np.int32)
    moment_sizes=cp.asarray(model.geo['group_size']*args.scale,dtype=float);moment_pop=model.pop
    joint_definition=None
    if args.joint_strata:
        from z_quadrature import resource_strata,quantized_field
        strata=resource_strata(model.geo,field_at(.225)[0],args.joint_strata)
        represented,node_z=quantized_field(z_field,strata)
        original_theta=np.r_[native_geometry['vtheta_e'],native_geometry['vtheta_i']]
        node_theta=np.bincount(strata['cell_stratum'],weights=original_theta)/strata['stratum_size']
        represented_theta=node_theta[strata['cell_stratum']]
        native_theta=cp.asarray(np.tile(represented_theta,args.scale))
        native_z=cp.asarray(np.tile(represented,args.scale))
        moment_group=cp.asarray(np.tile(strata['cell_stratum'],args.scale))
        moment_sizes=cp.asarray(strata['stratum_size']*args.scale,dtype=float)
        moment_pop=cp.asarray(model.geo['population'][strata['parent_group']])
        moment_m=cp.zeros(len(node_z));moment_counts=cp.zeros(len(node_z),dtype=np.int32)
        micro_flags=1+2+8
        joint_definition=dict(levels=args.joint_strata,strata=len(node_z),
            membership='Fixed empirical Z ordering within original spatial/threshold groups; independent of activity',
            threshold='Exact empirical stratum mean of original per-cell thresholds',
            Z='Exact empirical stratum mean along the original physical Z(D) path',
            M='Dynamic stratum mean driven by its own spike counts; no parent-M substitution',
            E_Z_RMS_error=float(np.sqrt(np.mean((represented[:32000]-z_field[:32000])**2))),
            E_threshold_RMS_error_mv=float(np.sqrt(np.mean((represented_theta[:32000]-original_theta[:32000])**2))),
            communication='Original parent rates, original g40 operator and delays; only local conditional-state resolution changes',
            scope='Joint state-resolution candidate, not the earlier Z-only/shared-parent-M experiment')
        np.savez_compressed(folder/'joint_strata.npz',**strata,node_Z=node_z,node_threshold_mv=node_theta)
    sizes = cp.asarray(model.geo['group_size']*args.scale, dtype=float)
    theta = cp.asarray(model.geo['threshold_mv'])
    rng = cp.random.RandomState(args.seed)
    shared=None
    if args.shared_noise:
        from native_shared_drive import NativeSharedDrive
        native_prepared=read(ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916/coarse_40/prepared.json')
        shared=NativeSharedDrive(native_geometry['positions_e'],native_prepared['params'],
            native_prepared['spatial_ou'],model.nu,args.seed)
    sa = cp.zeros(N); ia = cp.zeros(N); v = cp.full(N, 11.); ref = cp.zeros(N, dtype=np.int32)
    count = cp.zeros(model.P, dtype=np.int32)
    ar, ad = model.synpars[:2]; jump = model.synpars[-1]
    # Analytic convolution of a 100-ms private Poisson prehistory: the discarded
    # AMPA tail is below 4e-13. Voltage remains at reset during preparation.
    for a in range(0, 1000, 100):
        ages = cp.arange(a, a+100, dtype=float)
        counts = rng.poisson(model.nu*DT, size=(100, N)).astype(float)
        sa += cp.power(ar, ages)@counts
        ia += ((1-ad)*(cp.power(ar, ages+1)-cp.power(ad, ages+1))/(ar-ad))@counts
    sa *= jump*model.dr[dgroup]; ia *= jump*model.dr[dgroup]
    if args.relabel_private_with_z:
        # Relabel all private prehistory and future innovations together with
        # Z. This must reproduce the original group observables exactly, and
        # validates the symmetry underlying the Z-only permutation diagnostic.
        full_permutation=cp.asarray(np.concatenate([permutation+k*40000 for k in range(args.scale)]))
        sa=sa[full_permutation].copy();ia=ia[full_permutation].copy()
    module = cp.RawModule(code=CUDA, options=('--fmad=false',), name_expressions=['cells', 'populations','moments'])
    cells, populations = [module.get_function(x) for x in ('cells', 'populations')]
    moments=module.get_function('moments')
    config = dict(D=args.D, duration_ms=args.duration, group_count=model.P,
        communication_operators=str(OPERATORS),
        particle_count=N, population_multiplier=args.scale, seed=args.seed,
        common_OU='zero', private_input='original filtered Poisson per neuron',
        Z='frozen individual field' if z_individual else 'frozen identical group field',
        M='dynamic individual states' if m_individual else 'dynamic population mean',
        thresholds='original individual values' if theta_individual else 'group empirical mean',
        individual_flags=dict(theta=bool(theta_individual),Z=bool(z_individual),M=bool(m_individual)),
        target='Independent finite-population realization of the density candidate; not native correspondence acceptance')
    if resource_definition is not None:
        config['Z']='frozen empirical resource-strata means'
        config['resource_quadrature']=resource_definition
    if permutation_definition is not None:config['resource_permutation']=permutation_definition
    if joint_definition is not None:
        config.update(joint_strata=joint_definition,Z='frozen empirical stratum means',
            M='dynamic stratum mean',thresholds='empirical stratum means',
            individual_flags=dict(theta=False,Z=False,M=False))
    if shared is not None:
        config.update(common_OU='Original global OU law, zero initial value',
            spatial_OU='Original local OU field and neuron-coordinate interpolation',
            shared_input_seeds=dict(global_seed=args.seed+700000,spatial_seed=args.seed+500000),
            shared_input_scope='Original noise laws with new independent streams; not replay of an original Fig5 realization. Baseline stationary private prehistory is identical across modulation conditions; comparison discards the first second.',
            spatial_OU_parameters=native_prepared['spatial_ou'])
    write(folder/'config.json', config)
    trace = []; fields = []; slow = []; block = cp.zeros(model.P)
    total = round(args.duration/DT); last = time.time()
    for start in range(0, total, 100):
        steps = min(100, total-start)
        intensity=model.nu*DT if shared is None else cp.asarray(shared.block(steps,args.scale)*DT)
        ext = rng.poisson(intensity, size=(steps, N))
        if args.relabel_private_with_z:ext=ext[:,full_permutation]
        assert int(ext.max().get()) < 256
        ext = ext.astype(cp.uint8)
        for k in range(steps):
            step = start+k
            model.recurrent((model.P,), (128,), (*model.operators, model.history, model.dp,
                model.tm, model.dr, model.qa, model.ia, model.qg, model.ig, model.qe, model.ie,
                model.Z, model.M, model.drive, np.int32(model.P), np.int32(model.D), np.int32(step), *model.synpars))
            count.fill(0)
            if args.joint_strata:moment_counts.fill(0)
            cells(((N+127)//128,), (128,), (dgroup, theta, model.tm, model.dr, model.drive,
                model.drefs, ext, model.ia, model.ig, native_theta, native_z, native_m,
                model.Z,model.M,moment_group,moment_m,moment_counts,sa, ia, v, ref, count, np.int32(N), np.int32(k), ar, ad, jump, np.int32(micro_flags)))
            if args.joint_strata:
                moments(((len(moment_m)+127)//128,),(128,),
                    (moment_counts,moment_sizes,moment_pop,moment_m,np.int32(len(moment_m))))
            populations(((model.P+127)//128,), (128,), (count, sizes, model.pop, model.M,
                model.history, model.activity, np.int32(model.P), np.int32(model.D), np.int32(step)))
            block += model.activity
            if (step+1) % 10 == 0:
                rate = block*1000.
                trace.append(cp.asnumpy(cp.r_[model.e_weights@rate, model.region_weights@rate]))
                fields.append(cp.asnumpy(cp.bincount(model.observable_cell,
                    weights=rate*model.e_sizes, minlength=1600)))
                block.fill(0.)
        slow.append([(start+steps)*DT, float((model.e_weights@model.M).get())])
        if time.time()-last > 20:
            write(folder/'status.json', dict(status='RUNNING', completed_ms=(start+steps)*DT,
                wall_s=time.time()-started, pid=os.getpid()))
            print('particle', args.D, 'scale', args.scale, 'ms', (start+steps)*DT, flush=True)
            last = time.time()
    sizes_e = np.bincount(model.geo['group_cell'], weights=np.where(model.geo['population'] == 0,
                           model.geo['group_size'], 0.), minlength=1600)
    field = np.asarray(fields)/np.maximum(sizes_e, 1)[None, :]
    rates = np.asarray(trace)
    assert np.allclose(field@sizes_e/32000., rates[:, 0], rtol=1e-12, atol=1e-12)
    assert bool(cp.isfinite(v).all().get()) and bool(cp.isfinite(ia).all().get())
    m_error = float(cp.max(abs(cp.bincount(dgroup, weights=native_m, minlength=model.P)/sizes-model.M)).get())
    assert m_error < 1e-9
    moment_error=None
    if args.joint_strata:
        moment_error=float(cp.max(abs(cp.bincount(moment_group,weights=native_m,minlength=len(moment_m))/moment_sizes-moment_m)).get())
        assert moment_error<1e-9
    np.savez_compressed(folder/'trajectory.npz', rate_1ms=rates, field_1ms=field,
                       count_e=sizes_e, slow_10ms=np.asarray(slow))
    if shared is not None:np.savez_compressed(folder/'shared_input.npz',**shared.arrays())
    write(folder/'status.json', dict(status='COMPLETE', completed_ms=total*DT,
        wall_s=time.time()-started, late_mean_hz=float(rates[len(rates)//2:, 0].mean()),
        individual_and_population_M_mean_error=m_error,
        individual_and_stratum_M_mean_error=moment_error,
        scope='Matched-coupling finite-population control only'))
    print(folder, 'COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=float, required=True)
    ap.add_argument('--scale', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1901)
    ap.add_argument('--duration', type=float, default=1000.)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--microscopic', action='store_true')
    ap.add_argument('--individual-theta',action='store_true')
    ap.add_argument('--individual-z',action='store_true')
    ap.add_argument('--individual-M',action='store_true')
    ap.add_argument('--z-quadrature',type=int,default=0,help='Resolve frozen Z by fixed empirical strata inside each original E group; keep original parent-group threshold and dynamic M')
    ap.add_argument('--joint-strata',type=int,default=0,help='Resolve threshold, frozen Z and dynamic M together in fixed empirical resource strata; original parent communication retained')
    ap.add_argument('--shared-noise',action='store_true',help='Use the original global and spatial OU input laws in the finite-population correspondence control')
    ap.add_argument('--z-permutation-seed',type=int,help='With --individual-z only, permute exact Z within exchangeable parent groups to test stochastic-history variation without changing its empirical law')
    ap.add_argument('--relabel-private-with-z',action='store_true',help='Numerical symmetry control: also relabel private prehistory and innovations, which must reproduce the original exact-Z trajectory')
    run_control(ap.parse_args())
