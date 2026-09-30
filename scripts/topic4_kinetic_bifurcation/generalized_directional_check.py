"""Independent nonlinear check of a measured generalized-return derivative.

Uses the first completed normal Arnoldi product, not a new tangent solve. The
two signed trajectories use the original FP64 physical map and the identical
fixed section. This tests a derivative; it does not certify an entire spectrum.
"""
from generalized_return_spectrum import *


def run(a):
    spectrum=a.spectrum;scfg=read(spectrum/'config.json')
    root=Path(scfg['corrected_source']);rcfg=read(root/'config.json')
    source=Path(rcfg['source']);cfg=read(source/'config.json')
    storage=Path(scfg['storage']);H=np.load(spectrum/'arnoldi.npz')['H']
    assert H.shape[1]>=1 and H.shape[0]>=2
    folder=OUT/'generalized_directional_checks'/a.label;folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);m.restore(root/'best_state')
    initial=capture(m);initial_step=m.step_index;x=coords.pack(m)
    normal=cp.asarray(np.load(root/'section_normal.npy'))
    direction=cp.asarray(np.load(storage/'q000.npy'))
    image=H[0,0]*direction+H[1,0]*cp.asarray(np.load(storage/'q001.npy'))
    assert abs(float(cp.dot(normal,direction).get()))<1e-10
    n=rcfg['integer_steps'];order=scfg['interpolation_order'];rows=[];started=time.time();last=started
    write(folder/'config.json',dict(spectrum=str(spectrum.resolve()),corrected_source=str(root.resolve()),
        epsilons=a.epsilons,relative_derivative_tolerance=a.tolerance,
        object='Central nonlinear finite difference of the same full-state generalized return',
        scope='A single measured normal direction; not complete spectral or bifurcation acceptance'))
    def reset(v):
        for k,value in initial.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=value
        m.step_index=initial_step;coords.unpack(v,m)
        for k in ('masserror','maxneg','minflux','positive_emitted','negative_emitted','maximum_lower_mass'):
            getattr(m,k).fill(0.)
        m.minimum_drive_bound.fill(cp.inf);m.diagnostic_start_step=initial_step
    for epsilon in a.epsilons:
        returns=[];qa=[];fractions=[]
        for sign in (-1,1):
            reset(x+sign*epsilon*direction);delta=[];values=[]
            for step in range(1,n+order+1):
                m.advance_step()
                if step>=n:
                    d=coords.pack(m)-x;delta.append(d);values.append(float(cp.dot(normal,d).get()))
                if time.time()-last>20:
                    write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),epsilon=epsilon,sign=sign,
                        elapsed_ms=step*DT,wall_s=time.time()-started,completed=len(rows)));last=time.time()
            alpha=crossing(values,order)
            returns.append(sum(float(c)*d for c,d in zip(coefficients(alpha,order),delta)))
            fractions.append(alpha);qa.append(m.diagnostics());del delta
        fd=(returns[1]-returns[0])/(2*epsilon);difference=fd-image
        relative=float((cp.linalg.norm(difference)/cp.linalg.norm(image)).get())
        residual_relative=float((cp.linalg.norm(difference)/cp.linalg.norm(image-direction)).get())
        valid=all(v['finite'] and v['maximum_mass_error']<1e-8 and v['maximum_negative_voltage_probability']<5e-4
                  and v['minimum_step_spike_probability']>-1e-6 for v in qa)
        row=dict(epsilon=epsilon,generalized_derivative_relative_error=relative,
            residual_derivative_relative_error=residual_relative,analytic_image_norm=float(cp.linalg.norm(image).get()),
            finite_difference_norm=float(cp.linalg.norm(fd).get()),return_fractions=fractions,
            section_derivative_residual=float(abs(cp.dot(normal,fd)).get()),full_map_numerical_gate=valid,
            pass_derivative=bool(valid and relative<a.tolerance),diagnostics=qa)
        rows.append(row);write(folder/'checks.json',dict(status='RUNNING',rows=rows));print(row,flush=True)
        del returns,fd,difference
    write(folder/'result.json',dict(status='DIRECTIONAL_CHECK_PASS' if all(v['pass_derivative'] for v in rows) else 'DIRECTIONAL_CHECK_FAILED',
        rows=rows,wall_s=time.time()-started,scope='Independent nonlinear derivative check in one section direction; stability and interpolation convergence remain separate'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spectrum',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--epsilons',type=float,nargs='+',default=[.0003,.00015]);ap.add_argument('--tolerance',type=float,default=.01)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
