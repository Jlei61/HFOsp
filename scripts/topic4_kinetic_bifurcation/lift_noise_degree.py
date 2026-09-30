"""Embed a corrected conditional density into a nested higher noise degree.

The physical voltage marginal and all recurrent/delay/M state are retained.
This is a numerical initial guess; a new trajectory/root must be computed.
"""
from autonomous_density import *


def run(a):
    cfg=read(a.source/'config.json');assert cfg.get('basis_mode','legacy')=='legacy'
    assert a.degree>cfg['degree']
    root=OUT/'noise_resolution_seeds'
    storage=Path('/data/hfosp/topic4_sef_hfo/kinetic_population_bifurcation_20260916/noise_resolution_seeds')
    storage.mkdir(parents=True,exist_ok=True)
    if not root.exists():root.symlink_to(storage,target_is_directory=True)
    assert root.resolve()==storage.resolve()
    folder=root/a.label;folder.mkdir(parents=True,exist_ok=False)
    nu=read(OPERATORS/'prepared.json')['nu_ext_per_ms']
    old=MovingBasis('E',cfg['degree'],nu,'stationary');Aold=old.advance(nu)
    new=MovingBasis('E',a.degree,nu,'stationary');Anew=new.advance(nu)
    k=len(old.frame['nodes']);K=len(new.frame['nodes'])
    embedding=new.frame['U'][:k,:].T@old.frame['U']
    frame_error=float(np.max(abs(new.frame['R'][:k,:k]-old.frame['R'])))
    mass_error=float(np.max(abs(new.frame['mass']@embedding-old.frame['mass'])))
    restricted_error=float(np.max(abs(embedding.T@Anew@embedding-Aold)))
    orthogonality=float(np.max(abs(embedding.T@embedding-np.eye(k))))
    assert frame_error<1e-8 and mass_error<1e-11 and restricted_error<1e-8 and orthogonality<1e-11
    with np.load(a.source/'checkpoint.npz') as z:state={k:z[k] for k in z.files}
    oldF=state['F'];assert oldF.shape[1]==k
    output=np.empty((len(oldF),K,oldF.shape[-1]))
    eig,vec=np.linalg.eig(Anew);at=np.argmin(abs(eig-1));marginal=vec[:,at].real
    marginal/=new.frame['mass']@marginal
    assert abs(eig[at]-1)<1e-10 and marginal.min()>0
    physical_error=0.;marginal_correction=0.;negative=0.
    for start in range(0,len(oldF),64):
        f=oldF[start:start+64];g=np.einsum('ab,gbv->gav',embedding,f,optimize=True)
        physical=np.einsum('k,gkv->gv',old.frame['mass'],f)
        err=marginal[None,:]-g.sum(2)
        marginal_correction=max(marginal_correction,float(np.max(abs(err))))
        distribution=np.maximum(physical,0.);distribution/=distribution.sum(1)[:,None]
        g+=err[:,:,None]*distribution[:,None,:]
        after=np.einsum('k,gkv->gv',new.frame['mass'],g)
        physical_error=max(physical_error,float(np.max(abs(after-physical))))
        negative=max(negative,float(np.maximum(-after,0.).sum(1).max()))
        output[start:start+len(g)]=g
    assert marginal_correction<1e-7 and physical_error<1e-8 and negative<5e-4
    state['F']=output
    np.savez_compressed(folder/'checkpoint.npz',**state)
    write(folder/'config.json',dict(cfg,degree=a.degree,density_states=int(output.size),
        source_degree=cfg['degree'],degree_lift_source=str(a.source.resolve()),
        numerical_initial_state='Nested orthonormal polynomial embedding, zero added coefficients, roundoff-only stationary marginal projection; all other complete state retained',
        scientific_acceptance='Higher-resolution initial guess only; not a corrected branch point'))
    report=dict(status='NESTED_DEGREE_INITIAL_GUESS_PREPARED',old_degree=cfg['degree'],new_degree=a.degree,
        old_nodes=k,new_nodes=K,nested_frame_error=frame_error,mass_functional_error=mass_error,
        restricted_noise_transition_error=restricted_error,embedding_orthogonality_error=orthogonality,
        stationary_marginal_roundoff_correction=marginal_correction,
        physical_voltage_marginal_max_error=physical_error,initial_negative_probability=negative,
        unchanged_states=[s for s in state if s!='F'],
        scope='Numerical resolution check within the same physical model. No change to D, spatial Z, dynamic M, communication, thresholds, delays or input law.')
    write(folder/'lift_qa.json',report);print(folder,report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--degree',type=int,required=True);ap.add_argument('--label',required=True)
    run(ap.parse_args())
