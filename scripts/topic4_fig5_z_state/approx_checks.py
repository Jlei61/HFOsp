#!/usr/bin/env python3
"""Offline reduction diagnostics on the M0 replay records (design 6.1/6.2, no new SNN batches).

1. variance time-handling: stationary second moment with current rates vs. exact kernel-filtered second moment of
   the delayed rates (per 20x20 cell), on the real native E rates (0.1 ms) and regional I rates (1 ms).
2. within-cell joint (Z, threshold) structure: unit-level Z means vs cell means, Z spread inside cells.
3. static transfer check: native per-cell mean currents/Z/M at 5-ms samples -> Phi prediction vs native 10-ms rates.
"""
import numpy as np
from common import *  # noqa: F401,F403
from approx_system import ReducedModel, APPROX
import readouts as R

REPLAY_RUN = OUT / 'replay' / 'runs' / f'eta0.0005_s{MAIN_SEED}'


def load_replay_fields(run=REPLAY_RUN, lo_ms=0, hi_ms=12500):
    parts = dict(zm_step=[], z=[], m=[], ie=[], ii=[])
    for path in sorted((run / 'fields').glob('*.npz')):
        with np.load(path) as a:
            if int(a['end_step']) <= ms_to_step(lo_ms) or int(a['start_step']) >= ms_to_step(hi_ms):
                continue
            sel = (a['zm_step'] >= ms_to_step(lo_ms)) & (a['zm_step'] < ms_to_step(hi_ms))
            for k in parts:
                parts[k].append(a[k][sel])
    return {k: np.concatenate(v) for k, v in parts.items()}


def native_rates_0p1ms(run=REPLAY_RUN, lo_ms=0, hi_ms=12500):
    e = []; ireg = []
    for path in sorted((run / 'chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as a:
            s, t = int(a['start_step']), int(a['end_step'])
            if t <= ms_to_step(lo_ms) or s >= ms_to_step(hi_ms):
                continue
            e.append(a['field_0p1ms']); ireg.append(a['regions_1ms'][:, 3:])
    return np.concatenate(e), np.concatenate(ireg)


def variance_lag_check(model, lo_ms=8000, hi_ms=10400):
    """Compare stationary vs kernel-filtered second moment on real native rates (E: exact per cell; I: regional)."""
    e_counts, i_reg = native_rates_0p1ms(lo_ms=lo_ms, hi_ms=hi_ms)
    g = np.load(REF / 'geometry.npz'); rc = g['region_counts'].astype(float)
    re = e_counts.astype(float) / model.count_e[None, :] / model.dt                     # per ms per neuron, (T, 400)
    i_rate_reg = i_reg.astype(float) / rc[3:][None, :] / 1.                             # per ms per neuron, (T/10, 3)
    from approx_run import i_region_of_cells
    reg_of_cell = i_region_of_cells(model)
    ri = np.repeat(i_rate_reg[:, reg_of_cell], 10, axis=0)[:len(re)]                    # (T, 400)
    T = len(re); D = model.D; n = model.n
    stat_e = np.empty((T // 10, n)); filt_e = np.empty((T // 10, n)); stat_i = np.empty((T // 10, n)); filt_i = np.empty((T // 10, n))
    hE = np.zeros((D, n)); hI = np.zeros((D, n)); y = {k: np.zeros((3, n)) for k in ('ee', 'ei')}
    a_, b_ = model.arA, model.adA; ag, bg = model.arG, model.adG
    for t in range(T):
        xe_stat = model.v_ee @ re[t]; xi_stat = model.v_ei @ ri[t]
        xe_del = model.vops['ee'] @ hE.ravel(); xi_del = model.vops['ei'] @ hI.ravel()
        for j, c in enumerate((a_ * a_, b_ * b_, a_ * b_)):
            y['ee'][j] = c * (xe_del + y['ee'][j])
        for j, c in enumerate((ag * ag, bg * bg, ag * bg)):
            y['ei'][j] = c * (xi_del + y['ei'][j])
        normA = a_ * a_ / (1 - a_ * a_) + b_ * b_ / (1 - b_ * b_) - 2 * a_ * b_ / (1 - a_ * b_)
        normG = ag * ag / (1 - ag * ag) + bg * bg / (1 - bg * bg) - 2 * ag * bg / (1 - ag * bg)
        if t % 10 == 9:
            k = t // 10
            stat_e[k] = xe_stat; filt_e[k] = (y['ee'][0] + y['ee'][1] - 2 * y['ee'][2]) / normA
            stat_i[k] = xi_stat; filt_i[k] = (y['ei'][0] + y['ei'][1] - 2 * y['ei'][2]) / normG
        hE[1:] = hE[:-1]; hE[0] = re[t]; hI[1:] = hI[:-1]; hI[0] = ri[t]
    skip = 50                                        # discard filter warm-up (5 ms)
    w = model.count_e
    out = {}
    for name, s_, f_ in (('AMPA_to_E', stat_e[skip:], filt_e[skip:]), ('GABA_to_E', stat_i[skip:], filt_i[skip:])):
        active = s_ > np.percentile(s_, 90)
        diff = f_ - s_
        out[name] = dict(rms_relative_difference_all=float(np.sqrt(np.average((diff ** 2).mean(0), weights=w)) / np.sqrt(np.average((s_ ** 2).mean(0), weights=w))),
                         rms_relative_difference_active_top10pct=float(np.sqrt(np.mean(diff[active] ** 2)) / np.sqrt(np.mean(s_[active] ** 2))),
                         max_abs_relative_difference_active=float(np.max(np.abs(diff[active]) / np.maximum(s_[active], 1e-9))),
                         mean_signed_relative_difference_active=float(np.mean(diff[active] / np.maximum(s_[active], 1e-9))))
        # lag by cross-correlation of the network-mean series
        sm = np.average(s_, axis=1, weights=w); fm = np.average(f_, axis=1, weights=w); sm -= sm.mean(); fm -= fm.mean()
        lags = np.arange(-20, 21); cc = [np.sum(sm[max(0, l):len(sm) + min(0, l)] * fm[max(0, -l):len(fm) + min(0, -l)]) for l in lags]
        out[name]['lag_of_filtered_vs_stationary_ms'] = float(lags[int(np.argmax(cc))] * 1.)
    out['window_ms'] = [lo_ms, hi_ms]; out['stationary_formula'] = 'sum_j w_ij^2 r_j(t) with current rates'
    out['filtered_formula'] = 'sum_j w_ij^2 sum_tau k_tau^2 r_j(t-d_ij-tau) dt / sum_tau k_tau^2 dt (exact discrete double-exponential kernel), normalised to the same stationary limit'
    out['interpretation'] = 'Differences measure the lag/smoothing error of using current rates in the variance; independent-input assumption in both.'
    return out


def joint_z_check(model, times_ms=Z_TIMES_MS):
    out = {}
    for ms in times_ms:
        st = ckpt.load(REPLAY_RUN / 'checkpoints' / f't{ms}ms.npz'); z = st['slow']['z'][:NE]; m = st['slow']['m'][:NE]
        zu = model.unit_mean(z); zc = np.repeat(model.cell_mean(z, model.cell_e), model.K)
        within = np.sqrt(model.unit_mean(z ** 2) - zu ** 2)          # within-unit SD
        lowered = model.theta_u < 18.
        out[str(ms)] = dict(mean_Z=float(z.mean()), cell_mean_vs_unit_mean_max_abs=float(np.max(np.abs(zu - zc))),
                            cell_mean_vs_unit_mean_rms=float(np.sqrt(np.mean((zu - zc) ** 2))),
                            within_unit_sd_median=float(np.median(within)), within_unit_sd_max=float(within.max()),
                            lowered_units_mean_Z=float(np.average(zu[lowered], weights=model.w_u[lowered])) if lowered.any() else None,
                            equal_units_mean_Z=float(np.average(zu[~lowered], weights=model.w_u[~lowered])),
                            corr_unit_Z_vs_threshold_in_core_cells=float(np.corrcoef(zu[lowered], model.theta_u[lowered])[0, 1]) if lowered.sum() > 2 else None,
                            mean_M=float(m.mean()), lowered_units_mean_M=float(np.average(model.unit_mean(m)[lowered], weights=model.w_u[lowered])) if lowered.any() else None,
                            equal_units_mean_M=float(np.average(model.unit_mean(m)[~lowered], weights=model.w_u[~lowered])))
    return out


def static_transfer_check(model, lo_ms=8000, hi_ms=10400):
    """Native per-cell currents (5/10-ms snapshots) -> Phi prediction vs native cell rates (10-ms)."""
    f = load_replay_fields(lo_ms=lo_ms, hi_ms=hi_ms)
    e_counts, i_reg = native_rates_0p1ms(lo_ms=lo_ms, hi_ms=hi_ms)
    T = len(e_counts); base = ms_to_step(lo_ms)
    rows = []
    for j, step in enumerate(f['zm_step']):
        k = int(step) - base
        if k + 100 > T:
            break
        ie = f['ie'][j].astype(float); ii = f['ii'][j].astype(float); z = f['z'][j]; m = f['m'][j]
        # unit-level native currents: mean I_E, mean I_I per unit; variance across neurons within units as diagnostic
        mu_native_u = model.unit_mean(ie - z * ii - model.eta_M * m)
        ex_c = model.te * (model.v_ee @ (e_counts[k:k + 100].astype(float).sum(0) / model.count_e / 10.) + model.je ** 2 * model.nu_sig)
        rows.append(dict(step=int(step), mu_u=mu_native_u, ex_c=ex_c, rate_u=None,
                         rate_c=e_counts[k:k + 100].astype(float).sum(0) / model.count_e / 10.))
    # predicted rates with the mixed Phi (GABA variance from native I rates by region)
    g = np.load(REF / 'geometry.npz'); rc = g['region_counts'].astype(float)
    from approx_run import i_region_of_cells
    reg_of_cell = i_region_of_cells(model)
    pred = []; nat = []
    for r in rows:
        k10 = (r['step'] - base) // 10
        ri = (i_reg[k10:k10 + 10].astype(float).sum(0) / rc[3:] / 10.)[reg_of_cell]
        inh_u = model.te * model.z2_u * np.repeat(model.v_ei @ ri, model.K)
        model.z2_u = model.z2_u   # z2 not needed for the mean here; keep the model's current field
        p = model.phi_e(r['mu_u'], r['ex_c'], inh_u)
        pred.append((p * model.w_u).reshape(model.n, model.K).sum(1)); nat.append(r['rate_c'])
    pred = np.asarray(pred) * 1000.; nat = np.asarray(nat) * 1000.
    w = model.count_e
    err = pred - nat
    return dict(samples=int(len(pred)), window_ms=[lo_ms, hi_ms],
                network_mean_native_hz=float(np.average(nat, axis=1, weights=w).mean()), network_mean_predicted_hz=float(np.average(pred, axis=1, weights=w).mean()),
                rms_error_hz=float(np.sqrt(np.average((err ** 2).mean(0), weights=w))),
                relative_bias_over_20hz=float(np.mean(err[nat > 20] / nat[nat > 20])) if np.any(nat > 20) else None,
                relative_rms_over_20hz=float(np.sqrt(np.mean((err[nat > 20] / nat[nat > 20]) ** 2))) if np.any(nat > 20) else None,
                scope='Static check: native mean currents at snapshot times fed to the E transfer (with the model current z2 field); not a dynamic closure test; I variance from regional I rates.')


def main():
    model = ReducedModel(dict(grid=20, variance='filtered'))
    st = ckpt.load(REPLAY_RUN / 'checkpoints' / f't{ANCHOR_MS}ms.npz'); model.set_slow_fields(st['slow']['z'][:NE], st['slow']['m'][:NE])
    out = dict(variance_lag=variance_lag_check(model), joint_z=joint_z_check(model), static_transfer=static_transfer_check(model))
    import siegert_table
    out['siegert_table_qa'] = siegert_table.qa()
    write(APPROX / 'reduction_checks.json', out)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
