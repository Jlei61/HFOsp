"""Identify exact refractory phase sectors of a saturated density equilibrium.

When every represented private-current node makes a reset neuron fire on its
first available step, its refractory distribution rotates independently of any
small current/M perturbation. Nonzero Fourier harmonics have zero probability
mass and roots-of-unity eigenvalues. Their sector is autonomous, so its spectrum
belongs to the full triangular linearization even with outgoing connections.
Other network modes are not classified by this argument.
"""
from stationary_response import *
from network_tangent import CODE as TANGENT_CODE


def run(a):
    cfg=read(a.source/'config.json');assert read(a.source/'status.json')['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED'
    folder=OUT/'equilibrium_neutral_audits'/a.label;folder.mkdir(parents=True,exist_ok=False)
    geo=dict(np.load(OPERATORS/'geometry.npz'));prep=read(OPERATORS/'prepared.json');p=prep['params'];pop=geo['population']
    tm=np.where(pop==0,p['tau_m_E'],p['tau_m_I'])
    ratio=np.where(pop==0,1.,p['tau_m_I']*p['J_ext_I']/(p['tau_m_E']*p['J_ext_E']))
    basis=MovingBasis('E',cfg['degree'],prep['nu_ext_per_ms'],'stationary');A=basis.advance(prep['nu_ext_per_ms'])
    mass=basis.frame['mass'];nodes=basis.frame['nodes']
    with np.load(a.source/'checkpoint.npz') as z:
        F=z['F'];drive=z['ia']-z['Z']*z['ig']-.0005*z['M'];rates=z['M']
    decay=np.exp(-DT/tm);margin=decay*11+(1-decay)*(drive+nodes.min()*ratio)-geo['threshold_mv']
    ids=np.flatnonzero(margin>1e-6);assert len(ids)
    m=LocalStationaryDensity(geo['threshold_mv'][ids],pop[ids],drive[ids],cfg['degree'],cfg['voltage_dv'],a.device,
        basis_mode=cfg.get('basis_mode','legacy'));m.F[:]=cp.asarray(F[ids,:,:m.width]);reference=m.F.copy()
    support_error=np.sum(abs(F[ids,:,:m.nv]),axis=(1,2))
    assert support_error.max()<1e-12, 'A refractory-only equilibrium is required'
    module=cp.RawModule(code=TANGENT_CODE,options=('--fmad=false',),name_expressions=['voltage_tangent'])
    kernel=module.get_function('voltage_tangent');reference_moved=cp.ascontiguousarray(m.A@reference)
    delta=cp.zeros_like(m.F);out=cp.empty_like(delta);flux=cp.empty_like(m.flux);zero=cp.zeros(m.P)
    refs=cp.asnumpy(m.refs);rows=[]
    def apply(v,dd):
        moved=cp.ascontiguousarray(m.A@v)
        kernel((m.P*m.K,),(128,),(reference_moved,moved,out,flux,m.de,m.dc,m.dw,m.nodes,m.ratio,m.decay,
            m.drive,dd,m.refs,np.int32(m.K),np.int32(m.nv),np.int32(m.width)))
        return out.copy()
    current_response=apply(delta,cp.ones(m.P))
    for harmonic in (1,2):
        real=cp.zeros_like(delta);imag=cp.zeros_like(delta)
        for k,n in enumerate(refs):
            phase=2*np.pi*harmonic*np.arange(n)/n
            real[k,:,m.nv:m.nv+n]=m.stationary_noise_marginal[:,None]*cp.asarray(np.cos(phase))[None,:]
            imag[k,:,m.nv:m.nv+n]=m.stationary_noise_marginal[:,None]*cp.asarray(np.sin(phase))[None,:]
        jr=apply(real,zero);ji=apply(imag,zero)
        eigenvalues=np.exp(2j*np.pi*harmonic/refs)
        expected_r=cp.asarray(eigenvalues.real)[:,None,None]*real-cp.asarray(eigenvalues.imag)[:,None,None]*imag
        expected_i=cp.asarray(eigenvalues.imag)[:,None,None]*real+cp.asarray(eigenvalues.real)[:,None,None]*imag
        errors=cp.asnumpy(cp.sqrt(cp.sum((jr-expected_r)**2+(ji-expected_i)**2,axis=(1,2))/
                       cp.sum(real**2+imag**2,axis=(1,2))))
        eps=1e-6;m.F[:]=reference+eps*real;m.map();plus=m.Q.copy()
        m.F[:]=reference-eps*real;m.map();fd=(plus-m.Q)/(2*eps)
        fd_error=float((cp.linalg.norm(fd-jr)/cp.linalg.norm(jr)).get())
        rows.append(dict(harmonic=harmonic,eigenvalues=np.c_[eigenvalues.real,eigenvalues.imag],
            relative_eigenpair_residuals=errors,independent_native_difference_relative_error=fd_error))
    verified=(float(cp.linalg.norm(current_response).get())<1e-12 and
        max(float(np.max(row['relative_eigenpair_residuals'])) for row in rows)<1e-10 and
        max(row['independent_native_difference_relative_error'] for row in rows)<1e-7)
    report=dict(status='SATURATED_REFRACTORY_NEUTRAL_SECTORS_VERIFIED' if verified else 'NEUTRAL_SECTOR_CHECK_FAILED',source=str(a.source.resolve()),D=cfg['D'],
        group_indices=ids,populations=pop[ids],group_sizes=geo['group_size'][ids],group_cells=geo['group_cell'][ids],
        group_regions=geo['group_region'][ids],minimum_reset_to_threshold_margin_mv=margin[ids],
        equilibrium_rates_hz=rates[ids],refractory_steps=refs,voltage_support_l1=support_error,
        stationary_noise_residual=float(cp.max(abs(m.A@m.stationary_noise_marginal-m.stationary_noise_marginal)).get()),
        input_tangent_absolute_norm=float(cp.linalg.norm(current_response).get()),verified_harmonics=rows,
        phase_sector_dimension=int(np.sum(refs-1)),
        classification='Nonhyperbolic equilibrium with unit-modulus refractory phase modes; other modes may still be unstable',
        limitation='Current discrete density model and saturated support; not an onset critical point or a native-SNN synchrony mechanism',
        arnoldi_implication='A current/history-only starting vector does not excite these autonomous phase directions, so that single Krylov run cannot certify full asymptotic stability')
    write(folder/'result.json',report);print({k:v for k,v in report.items() if k!='verified_harmonics'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=1);run(ap.parse_args())
