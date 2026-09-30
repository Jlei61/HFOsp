"""Autonomous conditional-density candidate for continuation qualification.

The private colored Poisson process is integrated in a polynomial density basis.
Common OU is fixed at its mean, not replaced by a reused random realization.
Z follows the physical D path and is clamped; population-mean M stays dynamic.
This candidate must pass correspondence tests before any bifurcation is promoted.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'
sys.path.insert(0, str(OUT/'source_snapshot'))
sys.path.insert(1, str(ROOT/'scripts'))
from population_field_gpu import DensityFieldGPU, cp, MovingBasis, DT, read, write
from polynomial_density import transport
from topic4_kinetic_D_response import field_at
import numpy as np
import argparse
import time

OPERATORS = OUT/'operators/selected_g40_theta0.25'


class AutonomousDensity(DensityFieldGPU):
    def __init__(self, D, degree=6, dv=.125, device=1, basis_mode='legacy'):
        super().__init__(OPERATORS, degree=degree, dv=dv, device=device)
        self.parameter_D = D
        self.voltage_dv = dv
        self.nu = float(self.prep['nu_ext_per_ms'])
        self.basis_mode=basis_mode
        if basis_mode=='high_precision':
            from stable_stationary_basis import StationaryBasis
            self.basis=StationaryBasis('E',degree,self.nu)
        else:
            assert basis_mode=='legacy'
            self.basis = MovingBasis('E', degree, self.nu, 'stationary')
        oldmass = self.basis.frame['mass'].copy()
        self.transition = cp.asarray(self.basis.advance(self.nu))
        self.mass = cp.asarray(self.basis.frame['mass'])
        self.nodes = cp.asarray(self.basis.frame['nodes'])
        assert np.max(abs(cp.asnumpy(self.mass @ self.transition)-oldmass)) < 1e-10
        z, self.alpha = field_at(D)
        self.z_clamp = cp.asarray(np.bincount(self.geo['cell_group'], weights=z)/self.geo['group_size'])
        self.Z[:] = self.z_clamp
        e = self.geo['population'] == 0
        realized = 1-np.average(cp.asnumpy(self.Z)[e], weights=self.geo['group_size'][e])
        assert abs(realized-D) < 1e-12
        self.dp = cp.full(self.P, self.nu)
        self.step_index = 0
        # Stationary external synaptic noise and reset voltage are an initial
        # distribution, not an equilibrium. Recurrent currents begin at zero.
        F = np.zeros((self.P, self.K, self.width))
        at = (self.centers <= 11.).sum(1)-1
        ids = np.arange(self.P)
        fraction = (11-self.centers[ids, at])/(self.centers[ids, at+1]-self.centers[ids, at])
        mass = cp.asnumpy(self.mass)
        F[ids, :, at] = mass[None, :]*(1-fraction[:, None])
        F[ids, :, at+1] = mass[None, :]*fraction[:, None]
        self.F[:] = cp.asarray(F)
        self.qe[:] = self.dr*self.basis.mean[0]
        self.ie[:] = self.dr*self.basis.mean[1]
        self.e_weights = cp.asarray(np.where(e, self.geo['group_size'], 0.)/32000.)
        self.region_weights = cp.asarray(np.stack([
            np.where(e & (self.geo['group_region'] == j), self.geo['group_size'], 0.)
            for j in range(3)]).astype(float))
        self.region_weights /= self.region_weights.sum(1)[:, None]
        self.observable_cell = cp.asarray(self.geo['group_cell'])
        self.e_sizes = cp.asarray(np.where(e, self.geo['group_size'], 0.), dtype=float)
        self.positive_emitted=cp.zeros(self.P);self.negative_emitted=cp.zeros(self.P)
        self.maximum_lower_mass=cp.zeros(self.P)
        self.minimum_drive_bound=cp.full(self.P,cp.inf)
        self.noise_min=float(cp.min(self.nodes).get())
        self.diagnostic_start_step=0

    def advance_step(self,extra_drive=None):
        i32 = np.int32
        step = self.step_index
        self.recurrent((self.P,), (128,), (*self.operators, self.history, self.dp,
            self.tm, self.dr, self.qa, self.ia, self.qg, self.ig, self.qe, self.ie,
            self.Z, self.M, self.drive, i32(self.P), i32(self.D), i32(step), *self.synpars))
        if extra_drive is not None:self.drive+=cp.asarray(extra_drive)
        moved = self.mix_noise()
        self.voltage((self.P*self.K,), (128,), (moved, self.Q, self.flux,
            self.de, self.dc, self.dw, self.nodes, self.dr, self.decay, self.drive,
            self.drefs, i32(self.K), i32(self.nv), i32(self.width)))
        self.observe((self.P,), (128,), (self.Q, self.flux, self.mass, self.pop,
            self.ig, self.Z, self.M, self.history, self.activity, self.maxneg,
            self.minflux, self.masserror, i32(self.K), i32(self.nv), i32(self.width),
            i32(self.P), i32(self.D), i32(step)))
        self.Z[:] = self.z_clamp
        self.F, self.Q = self.Q, self.F
        self.positive_emitted+=cp.maximum(self.activity,0.)
        self.negative_emitted+=cp.maximum(-self.activity,0.)
        self.maximum_lower_mass=cp.maximum(self.maximum_lower_mass,self.F[:,:,0]@self.mass)
        self.minimum_drive_bound=cp.minimum(self.minimum_drive_bound,self.drive+self.dr*self.noise_min)
        self.step_index += 1
        return self.activity

    def mix_noise(self):
        return cp.ascontiguousarray(cp.matmul(self.transition,self.F))

    def save(self, folder):
        arrays = {name: cp.asnumpy(getattr(self, name)) for name in
                  ('F', 'history', 'qa', 'ia', 'qg', 'ig', 'qe', 'ie', 'Z', 'M',
                   'positive_emitted','negative_emitted','maximum_lower_mass','minimum_drive_bound')}
        np.savez_compressed(folder/'checkpoint.npz', **arrays, step_index=self.step_index,
                            diagnostic_start_step=self.diagnostic_start_step)

    def restore(self, folder, allow_D_change=False):
        config = read(Path(folder)/'config.json')
        assert config.get('basis_mode','legacy')==self.basis_mode, 'Checkpoint noise coordinates must match'
        assert (allow_D_change or config['D'] == self.parameter_D) and config['degree'] == self.degree
        assert config['voltage_dv']==self.voltage_dv and config['dt_ms']==DT
        assert config.get('communication_operators') == str(OPERATORS)
        with np.load(Path(folder)/'checkpoint.npz') as z:
            for name in ('F', 'history', 'qa', 'ia', 'qg', 'ig', 'qe', 'ie', 'Z', 'M'):
                assert getattr(self, name).shape == z[name].shape
                getattr(self, name)[:] = cp.asarray(z[name])
            self.step_index = int(z['step_index'])
            self.diagnostic_start_step=self.step_index
            for name in ('positive_emitted','negative_emitted','maximum_lower_mass','minimum_drive_bound'):
                if name in z:getattr(self,name)[:]=cp.asarray(z[name])
            if 'diagnostic_start_step' in z:self.diagnostic_start_step=int(z['diagnostic_start_step'])
        if allow_D_change:self.Z[:]=self.z_clamp
        assert float(cp.max(abs(self.Z-self.z_clamp)).get()) < 1e-14

    def diagnostics(self):
        report=super().diagnostics();size=cp.asarray(self.geo['group_size'],dtype=float)
        pos=float((self.positive_emitted@size).get());neg=float((self.negative_emitted@size).get())
        report.update(maximum_lower_voltage_bin_mass=float(self.maximum_lower_mass.max().get()),
            minimum_total_drive_bound_mv=float(self.minimum_drive_bound.min().get()),
            total_negative_output_fraction=neg/max(pos,1e-300),
            added_diagnostic_coverage_start_ms=self.diagnostic_start_step*DT)
        return report


def audit(model, folder, steps=50):
    """Compare CUDA transport with CPU and verify a delay-ring impulse."""
    from scipy.sparse import load_npz
    from scipy.sparse import csr_matrix
    g = model.geo
    ids = [int(np.argmin(g['threshold_mv'])),
           int(np.flatnonzero((g['population'] == 0) & (g['threshold_mv'] == 18))[0]),
           int(np.flatnonzero(g['population'] == 1)[0])]
    A, nodes, mass = [cp.asnumpy(x) for x in (model.transition, model.nodes, model.mass)]
    reference = cp.asnumpy(model.F[ids])
    max_state = 0.; max_flux = 0.
    for _ in range(steps):
        model.advance_step()
        drive = cp.asnumpy(model.drive[ids])
        flux = cp.asnumpy(model.flux[ids])
        for j, gid in enumerate(ids):
            reference[j], f = transport(A@reference[j], model.edges[gid], model.centers[gid],
                model.widths[gid], nodes*model.ratio[gid], g['threshold_mv'][gid], 11.,
                float(cp.asnumpy(model.decay[gid])), drive[j], int(model.refs[gid]))
            max_flux = max(max_flux, float(np.max(abs(f-flux[j]))))
        max_state = max(max_state, float(np.max(abs(reference-cp.asnumpy(model.F[ids])))))
    weights = [load_npz(OPERATORS/f'delay_{name}.npz').tocsr() for name in ('ampa', 'gaba')]
    original = {k: getattr(model, k).copy() for k in
                ('history', 'qa', 'ia', 'qg', 'ig', 'qe', 'ie', 'drive')}
    rng = np.random.default_rng(1909)
    history = rng.uniform(0., .001, (model.D, model.P))
    model.history[:] = cp.asarray(history)
    errors = []
    for step in (0, 1, model.D-1, model.D, model.D+1):
        for name in ('qa', 'ia', 'qg', 'ig'):
            getattr(model, name).fill(0.)
        model.recurrent((model.P,), (128,), (*model.operators, model.history, model.dp,
            model.tm, model.dr, model.qa, model.ia, model.qg, model.ig, model.qe, model.ie,
            model.Z, model.M, model.drive, np.int32(model.P), np.int32(model.D),
            np.int32(step), *model.synpars))
        ordered = history[(step-np.arange(1, model.D)) % model.D].reshape(-1)
        for W, name, rise in zip(weights, ('qa', 'qg'), ('tau_r_AMPA', 'tau_r_GABA')):
            expected = W@ordered*cp.asnumpy(model.tm)/model.prep['params'][rise]
            errors.append(float(np.max(abs(expected-cp.asnumpy(getattr(model, name))))))
    for name, val in original.items():
        getattr(model, name)[:] = val
    result = dict(cuda_cpu_max_state_error=max_state, cuda_cpu_max_flux_error=max_flux,
                  delayed_recurrence_max_error=max(errors), diagnostics=model.diagnostics(),
                  fixed_D_error=abs(1-float((model.e_weights@model.Z).get())-model.parameter_D),
                  M_dynamic=bool(float(model.M.max().get()) > 0), groups_tested=ids)
    result['pass'] = max_state < 1e-10 and max_flux < 1e-10 and max(errors) < 1e-9
    write(folder/'operator_audit.json', result)
    assert result['pass'], result
    print(result, flush=True)


def run(args):
    name=f'D{args.D:.6f}_degree{args.degree}_dv{args.dv:g}_{args.duration:g}ms'
    if args.label:name+='_'+args.label
    folder = OUT/'qualification'/'selected_g40'/name
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Refusing to overwrite prior output: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = AutonomousDensity(args.D, args.degree, args.dv, args.device)
    if args.resume:
        model.restore(args.resume)
    config = dict(D=args.D, alpha=model.alpha, degree=args.degree, voltage_dv=args.dv,
        communication_operators=str(OPERATORS),
        dt_ms=DT, duration_ms=args.duration, groups=model.P, density_states=int(model.F.size),
        Z='frozen physical spatial field', M='dynamic population mean',
        private_Poisson='native affine colored-noise moments', common_OU='fixed at zero',
        initial='reset voltage and stationary private external current; zero recurrent current',
        scientific_acceptance='UNTESTED; autonomous density candidate, not accepted SNN bifurcation')
    write(folder/'config.json', config)
    pulse=None
    if args.pulse_current:
        region={'A':0,'B':1}.get(args.pulse_region)
        mask=model.geo['population']==0
        if region is not None:mask&=model.geo['group_region']==region
        pulse=cp.asarray(mask.astype(float)*args.pulse_current)
        config.update(pulse_current_mv=args.pulse_current,pulse_duration_ms=args.pulse_ms,
                      pulse_region=args.pulse_region,pulse_scope='Initial finite perturbation; released after specified duration')
        write(folder/'config.json',config)
    if args.resume:
        config.update(initial='restored complete autonomous density and delay state',
                      resumed_from=str(Path(args.resume).resolve()), initial_ms=model.step_index*DT)
        write(folder/'config.json', config)
    if args.audit:
        audit(model, folder)
        # The audit evolves the distribution; start a fresh object for the run.
        del model
        cp.get_default_memory_pool().free_all_blocks()
        model = AutonomousDensity(args.D, args.degree, args.dv, args.device)
    trace = []; fields = []; slow = []; block = cp.zeros(model.P)
    last = time.time(); status = 'COMPLETE'
    for step in range(round(args.duration/DT)):
        extra=pulse if pulse is not None and step*DT<args.pulse_ms else None
        block += model.advance_step(extra_drive=extra)
        if (step+1) % 10 == 0:
            rate = block*1000.
            trace.append(cp.asnumpy(cp.r_[model.e_weights@rate, model.region_weights@rate]))
            fields.append(cp.asnumpy(cp.bincount(model.observable_cell,
                          weights=rate*model.e_sizes, minlength=1600)))
            block.fill(0.)
        if (step+1) % 100 == 0:
            slow.append([model.step_index*DT, float((model.e_weights@model.M).get()),
                         float((model.e_weights@model.Z).get())])
            diag = model.diagnostics()
            if not diag['finite'] or diag['maximum_mass_error'] > 1e-5 or diag['minimum_step_spike_probability'] < -1e-6:
                status = 'NUMERICAL_FAILURE'; break
        if time.time()-last > 20:
            write(folder/'status.json', dict(status='RUNNING', pid=os.getpid(),
                  completed_ms=model.step_index*DT, wall_s=time.time()-started, diagnostics=model.diagnostics()))
            print('D', args.D, 'ms', model.step_index*DT, 'seconds', round(time.time()-started), flush=True)
            last = time.time()
    cells = np.bincount(model.geo['group_cell'], weights=np.where(model.geo['population'] == 0,
                       model.geo['group_size'], 0.), minlength=1600)
    field = np.asarray(fields)/np.maximum(cells, 1)[None, :]
    rates = np.asarray(trace)
    assert np.allclose(field@cells/32000., rates[:, 0], rtol=1e-12, atol=1e-12)
    np.savez_compressed(folder/'trajectory.npz', rate_1ms=rates, field_1ms=field,
                        slow_10ms=np.asarray(slow), count_e=cells)
    model.save(folder)
    write(folder/'status.json', dict(status=status, completed_ms=model.step_index*DT,
        wall_s=time.time()-started, diagnostics=model.diagnostics(),
        late_mean_hz=float(rates[len(rates)//2:, 0].mean()), bifurcation_acceptance='NOT_ESTABLISHED'))
    print(folder, status, model.diagnostics(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=float, required=True)
    ap.add_argument('--degree', type=int, default=6)
    ap.add_argument('--dv', type=float, default=.125)
    ap.add_argument('--device', type=int, default=1)
    ap.add_argument('--duration', type=float, default=100.)
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--label');ap.add_argument('--pulse-current',type=float,default=0.)
    ap.add_argument('--pulse-ms',type=float,default=.1)
    ap.add_argument('--pulse-region',choices=['A','B','all_E'],default='A')
    run(ap.parse_args())
