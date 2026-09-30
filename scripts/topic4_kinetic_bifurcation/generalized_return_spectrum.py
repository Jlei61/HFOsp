"""Normal spectrum of a corrected, approximate generalized return.

Differentiate the native map and the polynomial hyperplane intersection. The
transverse section is part of the defined operator; no native multiplier is
arbitrarily set to or removed as +1. This is not an exact discrete-period
Floquet spectrum. Interpolation-order and model-discretization checks remain.
"""
from generalized_return_audit import *


def coefficient_derivatives(alpha,order):
    result=[]
    for i in range(order+1):
        roots=[j for j in range(order+1) if j!=i]
        p=np.polynomial.Polynomial.fromroots(roots)/np.prod([i-j for j in roots])
        result.append(p.deriv()(alpha))
    return np.array(result)


def independent_section_derivative_check():
    # Known smooth dissipative circle map, evaluated without our GPU solver.
    angle=.1*np.sqrt(2);n=round(2*np.pi/angle);rho=.995;order=5
    direction=np.array([1.,0.]);normal=np.array([0.,1.])
    angles=(n+np.arange(order+1))*angle
    unit=np.c_[np.cos(angles),np.sin(angles)]
    def samples(radius):return unit*(1+(radius-1)*rho**(n+np.arange(order+1)))[:,None]
    frames=samples(1.);alpha=crossing(frames@normal,order)
    c=coefficients(alpha,order);dc=coefficient_derivatives(alpha,order)
    image=c@(unit*rho**(n+np.arange(order+1))[:,None])
    chord=dc@frames;analytic=image-chord*(normal@image)/(normal@chord)
    eps=1e-5;points=[]
    for radius in (1+eps,1-eps):
        y=samples(radius);a=crossing(y@normal,order);points.append(coefficients(a,order)@y)
    fd=(points[0]-points[1])/(2*eps)
    err=float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic));assert err<1e-7
    return dict(relative_derivative_error=err,analytic=analytic,finite_difference=fd,
        meaning='Independent known-circle finite difference checks the interpolated section derivative, not the SNN model')


def run(a):
    root=a.corrected;rcfg=read(root/'config.json');result=read(root/'result.json')
    assert result['status']=='GENERALIZED_RETURN_CORRECTED', 'First correct the defined generalized return'
    source=Path(rcfg['source']);cfg=read(source/'config.json');folder=OUT/'generalized_spectra'/a.label
    folder.mkdir(parents=True,exist_ok=False)
    storage=Path('/data/hfosp/topic4_sef_hfo/kinetic_population_bifurcation_20260916/generalized_krylov')/a.label
    storage.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);normal=cp.asarray(np.load(root/'section_normal.npy'))
    m.restore(root/'best_state');initial=capture(m);initial_step=m.step_index;x=coords.pack(m)
    t=NetworkTangent(m);n=rcfg['integer_steps'];order=a.order;comparison_order=3 if order==5 else 5
    config=dict(corrected_source=str(root.resolve()),metric_source=str(source),D=cfg['D'],integer_steps=n,
        interpolation_order=order,comparison_order=comparison_order,krylov_dimension=a.dimension,storage=str(storage),
        object='Derivative of generalized transverse return constructed from original FP64 native steps',
        scope='Normal-return spectrum candidate; no exact integer-period Floquet or critical-type claim',
        derivative_validation=independent_section_derivative_check())
    if a.initial_vector:
        seed_config=read(a.initial_vector.parent/'config.json')
        assert Path(seed_config['corrected_source']).resolve()==root.resolve()
        config['initial_vector']=str(a.initial_vector.resolve())
        config['initial_vector_definition']=seed_config
    write(folder/'config.json',config)
    rng=np.random.default_rng(6143)
    for k in ('qa','ia','qg','ig','M'):getattr(t,k)[:]=getattr(m,k)*cp.asarray(rng.normal(size=m.P))
    t.M[m.pop!=0]=0.;t.history[:]=m.history*cp.asarray(rng.normal(size=m.history.shape))
    q=cp.asarray(np.load(a.initial_vector)) if a.initial_vector else coords.pack(t)
    assert q.shape==(coords.size,)
    q-=normal*cp.dot(normal,q);q/=cp.linalg.norm(q)
    basis=[q];H=np.zeros((a.dimension+1,a.dimension));rows=[];started=time.time();last=started
    np.save(storage/'q000.npy',cp.asnumpy(q))
    def reset():
        for k,v in initial.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=v
        m.step_index=initial_step
        for k in ('masserror','maxneg','minflux','positive_emitted','negative_emitted','maximum_lower_mass'):
            getattr(m,k).fill(0.)
        m.minimum_drive_bound.fill(cp.inf);m.diagnostic_start_step=initial_step
    for j in range(a.dimension):
        reset();coords.unpack(basis[j],t,project=True);deltas=[];tangents=[];values=[]
        if j==0:
            rates=[];fields=[];slow=[];block=cp.zeros(m.P)
            ecount=np.bincount(m.geo['group_cell'],weights=np.where(m.geo['population']==0,m.geo['group_size'],0),minlength=1600)
        for step in range(1,n+max(order,comparison_order)+1):
            t.advance()
            if j==0:
                native_rate=m.activity*1000/DT
                rates.append(cp.asnumpy(cp.r_[m.e_weights@native_rate,m.region_weights@native_rate]))
                block+=m.activity
                if step%10==0:
                    rate=block*1000.
                    field=cp.bincount(m.observable_cell,weights=rate*m.e_sizes,minlength=1600)
                    fields.append(cp.asnumpy(field)/np.maximum(ecount,1))
                    slow.append(cp.asnumpy(cp.r_[m.e_weights@m.M,m.region_weights@m.M]))
                    block.fill(0.)
            if step>=n:
                delta=coords.pack(m)-x;deltas.append(delta);tangents.append(coords.pack(t));values.append(float(cp.dot(normal,delta).get()))
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),column=j,elapsed_ms=step*DT,
                    wall_s=time.time()-started,latest=rows[-1] if rows else None));last=time.time()
                print(a.label,j,step*DT,flush=True)
        alpha=crossing(values,order);c=coefficients(alpha,order);dc=coefficient_derivatives(alpha,order)
        if j==0:
            np.savez_compressed(folder/'reference_trajectory.npz',rate_0p1ms=np.asarray(rates),
                field_1ms=np.asarray(fields),M_1ms=np.asarray(slow),count_e=ecount,
                effective_return_time_ms=(n+alpha)*DT,
                rate_bin_centers_ms=(np.arange(len(rates))+.5)*DT,
                field_bin_centers_ms=np.arange(len(fields))+.5)
            write(folder/'reference_trajectory_definition.json',dict(D=cfg['D'],time_origin='Corrected generalized-return section, phase time zero',
                mean_rate_observable='Global E firing rate per neuron; not oscillation frequency',
                rate_order=['global_E','coreA_E','coreB_E','other_E'],private_noise='Autonomous density marginal',
                Z='Frozen spatial path',M='Dynamic',spatial_bin_mm=.5,
                scope='Unchanged native trajectory from a corrected generalized-return point; interpolation and normal stability remain separately qualified'))
        closure=sum(float(ck)*d for ck,d in zip(c,deltas))
        chord=sum(float(ck)*d for ck,d in zip(dc,deltas))
        v=sum(float(ck)*y for ck,y in zip(c,tangents))
        denominator=float(cp.dot(normal,chord).get());assert abs(denominator)>1e-10
        v-=chord*(cp.dot(normal,v)/denominator)
        # Compare the defined transverse operators using the same unchanged
        # native reference/tangent products; no extra physical replay needed.
        alt_alpha=crossing(values,comparison_order)
        alt_c=coefficients(alt_alpha,comparison_order);alt_dc=coefficient_derivatives(alt_alpha,comparison_order)
        alt_closure=sum(float(ck)*d for ck,d in zip(alt_c,deltas))
        alt_chord=sum(float(ck)*d for ck,d in zip(alt_dc,deltas))
        alt_v=sum(float(ck)*y for ck,y in zip(alt_c,tangents))
        alt_den=float(cp.dot(normal,alt_chord).get());assert abs(alt_den)>1e-10
        alt_v-=alt_chord*(cp.dot(normal,alt_v)/alt_den)
        operator_difference=float(cp.linalg.norm(alt_v-v).get())
        point_difference=float(cp.linalg.norm(alt_closure-closure).get())
        del alt_closure,alt_chord,alt_v
        section_error=float(abs(cp.dot(normal,v)).get());image_norm=float(cp.linalg.norm(v).get())
        for _ in range(2):
            for i in range(j+1):
                value=float(cp.dot(basis[i],v).get());H[i,j]+=value;v-=value*basis[i]
        H[j+1,j]=float(cp.linalg.norm(v).get());vals,vecs=np.linalg.eig(H[:j+1,:j+1]);ix=np.argsort(-abs(vals));vals=vals[ix];vecs=vecs[:,ix]
        errors=abs(H[j+1,j]*vecs[-1,:]);qa=m.diagnostics()
        valid=qa['finite'] and qa['maximum_mass_error']<1e-8 and qa['maximum_negative_voltage_probability']<5e-4 and qa['minimum_step_spike_probability']>-1e-6
        row=dict(dimension=j+1,ritz_real=vals.real,ritz_imag=vals.imag,ritz_absolute_residual=errors,
            phase_fraction=alpha,weighted_return_closure=float(cp.linalg.norm(closure).get()),
            comparison_order=comparison_order,comparison_order_return_difference=point_difference,
            comparison_order_derivative_absolute_difference=operator_difference,
            comparison_order_derivative_relative_difference=operator_difference/max(image_norm,1e-300),
            section_derivative_residual=section_error,image_norm=image_norm,diagnostics=qa,full_map_numerical_gate=bool(valid))
        rows.append(row);np.savez_compressed(folder/'arnoldi.npz',H=H[:j+2,:j+1]);write(folder/'spectrum_progress.json',dict(status='RUNNING',iterations=rows))
        if not valid:
            write(folder/'result.json',dict(status='NATIVE_MAP_NUMERICAL_GATE_FAILED',iterations=rows));return
        print('normal-return spectrum',a.label,j+1,list(zip(vals[:6],errors[:6])),flush=True)
        if H[j+1,j]<1e-13:break
        q=v/H[j+1,j];np.save(storage/f'q{j+1:03d}.npy',cp.asnumpy(q));basis.append(q)
        del deltas,tangents,closure,chord,v
    mode_files=[]
    for k in range(min(a.save_modes,len(vals))):
        for part,coeff in [('real',vecs[:,k].real),('imag',vecs[:,k].imag)]:
            if np.linalg.norm(coeff)<1e-12:continue
            vector=cp.zeros(coords.size)
            for i,value in enumerate(coeff):vector+=float(value)*basis[i]
            target=storage/f'mode_{k:02d}_{part}.npy';np.save(target,cp.asnumpy(vector))
            mode_files.append(dict(index=k,part=part,path=str(target),eigenvalue=[float(vals[k].real),float(vals[k].imag)],
                section_component=float(cp.dot(normal,vector).get()),coordinate_norm=float(cp.linalg.norm(vector).get())))
            del vector
    write(folder/'result.json',dict(status='GENERALIZED_NORMAL_SPECTRUM_COMPLETE',final=rows[-1],iterations=rows,mode_files=mode_files,
        wall_s=time.time()-started,acceptance='Requires Ritz, interpolation-order, nonlinear directional and discretization checks; no critical type inferred from a single spectrum'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrected',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--dimension',type=int,default=12);ap.add_argument('--order',type=int,choices=[3,5],default=5)
    ap.add_argument('--save-modes',type=int,default=3)
    ap.add_argument('--initial-vector',type=Path,help='A documented numerical Krylov start in the exact same original metric and section; physical return operator is unchanged')
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
