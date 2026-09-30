"""Memory-bounded high-resolution local stationary density table.

Batches partition independent local conditions, not the recurrent network.
This is the same stationary density solve and changes no population coupling.
The assembled table remains a predictor until full-network root correction.
"""
from stationary_response import *


def run(args):
    folder=OUT/'stationary_response'/(args.label or f'degree{args.degree}_dv{args.dv:g}_low_dense_high_precision_batched')
    folder.mkdir(parents=True,exist_ok=args.resume);parts=folder/'per_batch';parts.mkdir(exist_ok=args.resume)
    currents=np.linspace(-2.,3.,101);thresholds=np.r_[np.linspace(14.,18.,17),18.]
    populations=np.r_[np.zeros(17,np.uint8),np.ones(1,np.uint8)]
    theta=np.repeat(thresholds,len(currents));pop=np.repeat(populations,len(currents));u=np.tile(currents,len(thresholds))
    cfg=dict(degree=args.degree,voltage_dv=args.dv,dt_ms=DT,basis_mode='high_precision',
        threshold_spacing_mv=.25,current_spacing_mv=.05,current_bounds_mv=[-2.,3.],batch_size=args.batch_size,
        intended_scope='High-resolution low-state stationary predictor; batches are independent local conditions; full network correction and stability required')
    if args.tolerance!=1e-10:cfg['stationary_predictor_tolerance']=args.tolerance
    if args.resume:
        assert read(folder/'config.json')==cfg, 'Resume must preserve the numerical table definition'
    else:write(folder/'config.json',cfg)
    started=time.time();rates=[];residuals=[];diagnostics=[];histories=[]
    Fdisk=None;edges=[];noise_mass=None
    if args.resume and (folder/'density_coefficients.npy').exists():
        Fdisk=np.lib.format.open_memmap(folder/'density_coefficients.npy',mode='r+')
    for b,left in enumerate(range(0,len(u),args.batch_size)):
        right=min(len(u),left+args.batch_size);batch=parts/f'batch{b:03d}';batch.mkdir(exist_ok=args.resume)
        m=LocalStationaryDensity(theta[left:right],pop[left:right],u[left:right],args.degree,args.dv,args.device,basis_mode='high_precision')
        if args.resume and (batch/'response.npz').exists():
            meta=read(batch/'status.json');assert meta['left']==left and meta['right']==right and meta['qa']['converged']
            with np.load(batch/'response.npz') as z:
                rates.append(z['rate_hz']);residuals.append(z['residual']);histories.append(z['iteration_history'])
            assert Fdisk is not None
            diagnostics.append(meta['qa']);edges.append(m.edges);noise_mass=cp.asnumpy(m.mass)
            del m;cp.get_default_memory_pool().free_all_blocks();continue
        def progress(it,residual,r):
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),batch=b,left=left,right=right,
                iteration=it,residual=float(residual.max()),wall_s=time.time()-started))
            print('batched table',args.degree,args.dv,b,it,residual.max(),flush=True)
        r,res,diag,hist=m.solve(10000,progress,tolerance=args.tolerance,acceleration=True)
        write(batch/'status.json',dict(qa=diag,left=left,right=right))
        if not diag['converged']:
            write(folder/'status.json',dict(status='INCOMPLETE_LOCAL_BATCH',batch=b,qa=diag));return
        if Fdisk is None:
            Fdisk=np.lib.format.open_memmap(folder/'density_coefficients.npy',mode='w+',dtype=np.float64,shape=(len(u),m.K,m.width))
        # The final batch contains only I groups with a shorter refractory
        # queue. The common table shape retains E padding, which is exactly
        # zero for I; the local map and all active queue entries are unchanged.
        assert m.width<=Fdisk.shape[2]
        Fdisk[left:right]=0.;Fdisk[left:right,:,:m.width]=cp.asnumpy(m.F)
        Fdisk.flush();edges.append(m.edges);noise_mass=cp.asnumpy(m.mass)
        np.savez_compressed(batch/'response.npz',rate_hz=r,residual=res,iteration_history=hist)
        rates.append(r);residuals.append(res);diagnostics.append(diag);histories.append(hist)
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),completed_conditions=right,total_conditions=len(u),wall_s=time.time()-started))
        del m;cp.get_default_memory_pool().free_all_blocks()
    shape=(len(thresholds),len(currents));r=np.concatenate(rates);res=np.concatenate(residuals)
    np.savez_compressed(folder/'response.npz',rate_hz=r.reshape(shape),residual=res.reshape(*shape,2),
        theta=thresholds,population=populations,current_mv=currents)
    np.savez_compressed(folder/'density_state.npz',F=Fdisk,theta=theta,population=pop,current_mv=u,
        edges=np.concatenate(edges),noise_mass=noise_mass)
    write(folder/'status.json',dict(status='STATIONARY_SOLVED',batches=diagnostics,wall_s=time.time()-started,
        maximum_stationary_residual=max(x['max_stationary_residual'] for x in diagnostics),
        maximum_mass_error=max(x['maximum_mass_error'] for x in diagnostics),network_equilibrium='NOT_YET_CORRECTED'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,required=True);ap.add_argument('--dv',type=float,required=True)
    ap.add_argument('--batch-size',type=int,default=256);ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--label')
    ap.add_argument('--tolerance',type=float,default=1e-10);run(ap.parse_args())
