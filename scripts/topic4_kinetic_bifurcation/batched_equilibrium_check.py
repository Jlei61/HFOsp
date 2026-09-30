"""Verify one complete native network step without storing every PDF on GPU.

All delayed recurrent sums are evaluated together by the original CUDA kernel.
Only the subsequent, conditionally independent voltage transports are batched.
The original observe kernel updates dynamic M and the firing-history slot.
This is a fixed-point residual check, not a replacement evolution model.
"""
from equilibrium_predictor import *
from stationary_response import LocalStationaryDensity, CODE


def check(folder, device=0, batch_size=64):
    folder=Path(folder);cfg=read(folder/'config.json')
    cp.cuda.Device(device).use()
    p=EquilibriumProblem(Path(cfg['table']),'pchip')
    pars=p.prep['params'];P=p.P;depth=p.prep['max_delay_steps']+1
    with np.load(folder/'latest_rates.npz') as z:
        r=z['rate_hz'];D=float(z['D'])
    F=np.load(folder/'density_coefficients.npy',mmap_mode='r')
    Z,_=p.resource(D);M=r*p.e
    old=dict(qa=p.A@r,ia=p.A@r,qg=p.G@r,ig=p.G@r,M=M)
    state={k:cp.asarray(v) for k,v in old.items()}
    tm=cp.asarray(np.where(p.e,pars['tau_m_E'],pars['tau_m_I']))
    ratio=cp.asarray(np.where(p.e,1.,pars['tau_m_I']*pars['J_ext_I']/(pars['tau_m_E']*pars['J_ext_E'])))
    history=cp.asarray(np.broadcast_to(r*DT/1000.,(depth,P)).copy())
    operators=[]
    for name in ('ampa','gaba'):
        w=sparse.load_npz(OPERATORS/f'delay_{name}.npz').tocsr()
        operators.extend([cp.asarray(w.indptr,dtype=np.int64),cp.asarray(w.indices,dtype=np.int32),cp.asarray(w.data)])
    module=cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['recurrent','observe'])
    recurrent=module.get_function('recurrent');observe=module.get_function('observe')
    synpars=tuple(np.float64(v) for v in [np.exp(-DT/pars['tau_r_AMPA']),np.exp(-DT/pars['tau_d_AMPA']),
        np.exp(-DT/pars['tau_r_GABA']),np.exp(-DT/pars['tau_d_GABA']),pars['tau_r_AMPA'],pars['tau_r_GABA'],
        pars['tau_m_E']/pars['tau_r_AMPA']*pars['J_ext_E']])
    # The auxiliary mean external-current record is initialized at its own
    # stationary value. The colored private input itself remains in the PDF.
    nu=float(p.prep['nu_ext_per_ms']);qe=ratio*(synpars[-1]*nu*DT/(1-synpars[0]));ie=qe.copy()
    q0=qe.copy();i0=ie.copy();drive=cp.empty(P);z=cp.asarray(Z)
    recurrent((P,),(128,),(*operators,history,cp.full(P,nu),tm,ratio,state['qa'],state['ia'],state['qg'],state['ig'],
        qe,ie,z,state['M'],drive,np.int32(P),np.int32(depth),np.int32(0),*synpars))
    errors={k:float(cp.max(abs(state[k]-cp.asarray(old[k]))).get()) for k in ('qa','ia','qg','ig')}
    errors.update(qe=float(cp.max(abs(qe-q0)).get()),ie=float(cp.max(abs(ie-i0)).get()),F=0.,M=0.,history=0.,spike_rate_hz=0.)
    qa=dict(maximum_negative_voltage_probability=0.,minimum_step_spike_probability=0.,maximum_mass_error=0.,finite=True,
        min_Z=float(Z.min()),max_Z=float(Z.max()),maximum_lower_voltage_bin_mass=0.,minimum_total_drive_bound_mv=float('inf'))
    records=[];started=time.time()
    for left in range(0,P,batch_size):
        right=min(P,left+batch_size);s=slice(left,right);n=right-left
        m=LocalStationaryDensity(p.theta[s],p.geo['population'][s],cp.asnumpy(drive[s]),cfg['degree'],cfg['voltage_dv'],
            device,basis_mode=cfg.get('basis_mode','legacy'))
        m.F[:]=cp.asarray(F[s,:,:m.width]);rate=m.map()
        pdf_error=float(cp.max(abs(m.Q-m.F)).get());errors['F']=max(errors['F'],pdf_error)
        assert np.max(abs(F[s,:,m.width:]),initial=0.)==0., 'Nonzero inactive refractory padding'
        lm=cp.asarray(M[s]);lz=cp.asarray(Z[s]);lh=cp.asarray(np.broadcast_to(r[s]*DT/1000.,(depth,n)).copy())
        activity=cp.empty(n);neg=cp.zeros(n);minflux=cp.zeros(n);masserror=cp.zeros(n)
        observe((n,),(128,),(m.Q,m.flux,m.mass,cp.asarray(p.geo['population'][s],dtype=np.uint8),state['ig'][s],
            lz,lm,lh,activity,neg,minflux,masserror,np.int32(m.K),np.int32(m.nv),np.int32(m.width),
            np.int32(n),np.int32(depth),np.int32(0)))
        lz[:]=cp.asarray(Z[s])  # Original autonomous map clamps Z after observe.
        errors['M']=max(errors['M'],float(cp.max(abs(lm-cp.asarray(M[s]))).get()))
        errors['history']=max(errors['history'],float(cp.max(abs(lh[0]-cp.asarray(r[s]*DT/1000.))).get()))
        errors['spike_rate_hz']=max(errors['spike_rate_hz'],float(cp.max(abs(activity*1000/DT-cp.asarray(r[s]))).get()))
        qa['maximum_negative_voltage_probability']=max(qa['maximum_negative_voltage_probability'],float(neg.max().get()))
        qa['minimum_step_spike_probability']=min(qa['minimum_step_spike_probability'],float(minflux.min().get()))
        qa['maximum_mass_error']=max(qa['maximum_mass_error'],float(masserror.max().get()))
        qa['maximum_lower_voltage_bin_mass']=max(qa['maximum_lower_voltage_bin_mass'],float((m.Q[:,:,0]@m.mass).max().get()))
        qa['minimum_total_drive_bound_mv']=min(qa['minimum_total_drive_bound_mv'],float((m.drive+m.ratio*m.nodes.min()).min().get()))
        qa['finite']=qa['finite'] and bool(cp.isfinite(m.Q).all().get())
        records.append(dict(left=left,right=right,PDF_max_absolute_residual=pdf_error))
        del m,lm,lz,lh,activity,neg,minflux,masserror
        cp.get_default_memory_pool().free_all_blocks()
    result=dict(status='COMPLETE',D=D,mean_E_hz=float(p.eweights@r),batch_size=batch_size,
        one_step_absolute_residual=errors,density_diagnostics=qa,batches=records,wall_s=time.time()-started,
        method='Original all-group delayed-recurrence CUDA kernel; independent voltage batches; original observe/M/history kernel',
        scope='One complete native-map step at the stationary candidate, not a batched approximation of its coupling')
    write(folder/f'batched_check_{batch_size}.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--batch-size',type=int,default=64)
    a=ap.parse_args();print(check(a.source,a.device,a.batch_size),flush=True)
