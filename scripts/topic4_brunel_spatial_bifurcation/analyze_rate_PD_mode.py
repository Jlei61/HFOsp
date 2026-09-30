"""Read the spatial and burst-timing content of a full antiperiodic rate mode.

Mode amplitude is arbitrary. Normalize its largest E-group rate deviation to
one Hz and report first-order peak shifts per unit of that normalization.
These are derivatives, not simulated child events or SEEG voltages.
"""
from rate_periodic import *
from scipy.optimize import minimize_scalar


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='PD_double_upper')
    p.add_argument('--N',type=int,default=2048)
    p.add_argument('--accepted-parent',action='store_true')
    p.add_argument('--quiet-phase-origin',action='store_true',help='Use one common quiet phase for parent and antiperiodic mode')
    p.add_argument('--output-label');a=p.parse_args();s=RateField()
    if a.accepted_parent:
        validation=read(PERIODIC_OUT/f'{a.label}_validation.json')
        assert validation.get('full_acceptance',False)
        if 'accepted_mode' in validation:
            check=validation['continuous_orbit_check']
            assert check['filter_state_check']['positive']
            assert Path(check['orbit']).resolve()==Path(validation['accepted_parent_orbit']).resolve()
            root=dict(orbit=validation['accepted_parent_orbit'],J_EE_core=validation['J_EE_core'])
            mode_path=Path(validation['accepted_mode'])
            assert all(Path(v['orbit']).resolve()==Path(root['orbit']).resolve()
                       for v in validation['direct_monodromy_checks'])
        else:
            root=validation['filter_state_followup'];assert root['status']=='FILTER_AND_CRITICAL_MODE_RECHECKED'
            mode_path=Path(root['mode'])
    else:
        root=read(PERIODIC_OUT/f'{a.label}_N{a.N}.json')
        mode_path=PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz'
    z=np.load(root['orbit']);u=np.load(mode_path)['u']
    u=u/np.max(abs(u[:,s.E]));r=resample(z['r'],len(u),axis=0)*1000
    T=float(z['T']);N=len(u);t=np.arange(N)*T/N
    shift=0
    if a.quiet_phase_origin:
        core=s.E&(s.geo['group_region']<2);mass=s.geo['group_size']*core;mass/=mass.sum()
        shift=int(np.argmin(r@mass))
        r=np.roll(r,-shift,axis=0)
        # A wrapped piece of an antiperiodic mode changes sign. Rolling one
        # T-sized array would give an incorrect mode at the display seam.
        u=np.roll(np.r_[u,-u],-shift,axis=0)[:N]
    reg=np.array([s.regional_rates(v/1000) for v in r])
    mode=np.array([s.regional_rates(v/1000) for v in u])
    tt=np.arange(2*N+1)*T/N
    peaks={}
    for k in [0,1]:
        rate=CubicSpline(tt,np.r_[reg[:,k],reg[:,k],reg[:1,k]],bc_type='periodic')
        perturb=CubicSpline(tt,np.r_[mode[:,k],-mode[:,k],mode[:1,k]],bc_type='periodic')
        ids=find_peaks(np.tile(reg[:,k],3),height=20,distance=N//5)[0]
        ids=ids[(ids>=N)&(ids<2*N)]-N;rows=[]
        for i in ids:
            center=t[i]+T
            peak=minimize_scalar(lambda x:-float(rate(x)),
                bounds=(center-2*T/N,center+2*T/N),method='bounded').x
            at=peak%T
            curvature=float(rate(at,2));assert curvature<0
            rows.append(dict(peak_ms=at,peak_rate_Hz=float(rate(at)),
                delta_peak_height_Hz_per_unit=float(perturb(at)),
                delta_peak_time_ms_per_unit=-float(perturb(at,1))/curvature))
        peaks['AB'[k]]=rows
    w=s.geo['group_size']*s.E;power=np.mean(u*u,axis=0)
    baseline=np.array([w[s.geo['group_region']==k].sum()/w.sum() for k in range(3)])
    energy=np.array([(w*power)[s.geo['group_region']==k].sum()/(w*power).sum() for k in range(3)])
    cell=s.geo['group_cell'];num=np.bincount(cell[s.E],weights=(w*power)[s.E],minlength=400)
    den=np.bincount(cell[s.E],weights=w[s.E],minlength=400)
    field=np.sqrt(num/np.maximum(den,1));contact=u@s.geo['contact_rate_weights']
    result=dict(label=a.label,J_EE_core=root['J_EE_core'],T_ms=T,
        parent_orbit=root['orbit'],mode_source=str(mode_path),accepted_fine_parent=bool(a.accepted_parent),
        common_display_phase_shift_ms=shift*T/N,
        normalization='Largest absolute E-group rate mode component = 1 Hz; all derivatives per this unit.',
        E_mode_energy_A_B_surround=energy,E_population_fraction_A_B_surround=baseline,
        per_E_cell_mode_RMS_relative_to_network=np.sqrt(energy/baseline),
        core_peak_derivatives=peaks,
        maximum_regional_mode_Hz=np.max(abs(mode),axis=0),
        maximum_contact_mode_Hz=np.max(abs(contact),axis=0),
        meaning='The mode changes sign after one parent period, so adjacent parent-length windows receive opposite first-order changes. This does not establish child stability, a propagation-template switch, or irregular dynamics.')
    output_label=a.output_label or a.label
    write(PERIODIC_OUT/f'{output_label}_mode_readout.json',result)
    save_periodic_array(PERIODIC_OUT/f'{output_label}_mode_readout.npz',time_ms=t,
                        parent_regional_Hz=reg,regional_mode_Hz=mode,
                        E_cell_RMS_mode_Hz=field,contact_mode_Hz=contact)
    print(result,flush=True)


if __name__=='__main__':main()
