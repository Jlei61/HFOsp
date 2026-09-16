#!/usr/bin/env python3
"""Write execution_contract.json, the deterministic job table and status.json for this milestone."""
from common import *  # noqa: F401,F403


def job_table():
    jobs = []
    for seed in SEEDS:
        jobs.append(dict(stage='M0_replay', name=f'replay_s{seed}', seed=seed, device=DEVICE[seed], duration_ms=REPLAY_END_MS,
                         counts_as_independent_sample=False, status='PLANNED'))
    for z in Z_TIMES_MS:
        for h in HISTORY_MS:
            for w in ('W1', 'W2'):
                jobs.append(dict(stage='M1_frozen_Z_dynamic_M', name=f'z{z}_h{h}_{w}', z_source_ms=z, history_ms=h, future=w,
                                 anchor_ms=ANCHOR_MS, duration_ms=CONTINUATION_MS, seed_source=MAIN_SEED,
                                 counts_as_independent_sample=True, status='PLANNED'))
    jobs.append(dict(stage='M1_extension', name='extension_pair_TBD', count_max=8, duration_ms=EXTENSION_MS,
                     rule='section 5.3: earliest adjacent Z pair (low history, both futures SELF_LIMITED->PERSISTENT); else earliest inconsistent/UNRESOLVED pair; else none',
                     counts_as_independent_sample=False, status='CONDITIONAL'))
    jobs.append(dict(stage='M1_fixed_Z_M', name='fixed_zm_TBD', count_max=8, duration_ms=CONTINUATION_MS,
                     rule='section 5.3: only if the extended pair keeps consistent two sides; Z pair x M fields (8000/9420 ms) x W1/W2, 8000 ms fast history',
                     counts_as_independent_sample=True, status='CONDITIONAL'))
    jobs.append(dict(stage='M2_small_set', name='lif_small_set', count_max=60, cells_max=2048, duration_ms=2000,
                     seeds=[9108601, 9108602], status='CONDITIONAL_ON_DYNAMIC_RESPONSE_CHECK'))
    jobs.append(dict(stage='M2_approximation', name='approx_v1', budget_units=48, versions_max=2,
                     composition='24 main + 8 extension + 8 fixed M + 2 path replays + 4 deterministic + 2 sensitivity', status='PLANNED'))
    jobs.append(dict(stage='M3_bifurcation', name='branch_and_spectra', branch_points_max=400, spectrum_workpoints_max=6,
                     status='NOT_RUN_DEPENDENCY_UNTIL_M2_PASSES'))
    return jobs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = check_reference_sources()
    contract = dict(
        milestone='Current Fig.5 network: frozen-Z state map and approximation correspondence qualification',
        design='docs/archive/topic4/fig5_current_network_z_state_milestone_design_2026-09-15.md',
        design_sha256=sha(ROOT / 'docs/archive/topic4/fig5_current_network_z_state_milestone_design_2026-09-15.md'),
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        reference=dict(figure=str(REF / 'clean_panels_v2/review_square_complete_20260915/figures/fig5.pdf'),
                       figure_sha256=sha(REF / 'clean_panels_v2/review_square_complete_20260915/figures/fig5.pdf'),
                       metadata=str(REF / 'clean_panels_v2/review_square_complete_20260915/fig5_metadata.json'),
                       main_trajectory=str(REF / 'runs/eta0.0005_s9108401'), second_trajectory=str(REF / 'runs/eta0.0005_s9108402'),
                       protocol=str(REF / 'protocol.json'), protocol_sha256=sha(REF / 'protocol.json'),
                       identity=p['identity'], source_hashes=p['source_hashes'],
                       substrate=str(SUBSTRATE / 'substrate.json'), substrate_sha256=sha(SUBSTRATE / 'substrate.json')),
        fixed=dict(topology_seed=6101, NE=NE, NI=NI, dt_ms=DT_MS, eta_M=.0005, tau_M_s=1., tau_Z_s=5., I_th=old.THRESHOLD,
                   Vth_E_counts=dict(lowered=781, equal=31219, raised=0), core_radius_mm=1.5, readout_radius_mm=1.75),
        z_source_times_ms=Z_TIMES_MS, histories_ms=HISTORY_MS, anchor_ms=ANCHOR_MS, W2_seed=W2_SEED,
        W2_drive_seed=W2_SEED + W2_DRIVE_SEED_OFFSET, small_set_seeds=[9108601, 9108602],
        figure_marks_s=FIGURE_TIMES_S, high_onset_s=HIGH_ONSET_S, high_confirmation_s=HIGH_CONFIRM_S,
        budgets=dict(native_replay=2, native_frozen_Z_dynamic_M=24, native_extension_max=8, native_fixed_ZM_max=8,
                     small_set_max=60, approx_units_per_version=48, approx_versions_max=2, branch_points_max=400,
                     spectrum_workpoints_max=6),
        resources=dict(heavy_processes_max=4, minimum_available_memory_GiB=64, blas_threads=1, gpus=[0, 1],
                       storage='/data/hfosp/topic4_sef_hfo/fig5_current_network_z_state_v1 (symlinked from results/)'),
        statistical_unit='One complete continuation trajectory; cells, grid cells, events and windows are not independent networks.',
        stop='Complete the dispatched trajectories and closing analyses, deliver review packet, stop. No next milestone, no parameter search, no Fig.5 replacement.',
        preexisting_load='12 fig5_log_m_entry_extension workers + 8 three_observable_bo workers were running at start (not touched).')
    write(OUT / 'execution_contract.json', contract)
    write(OUT / 'jobs.json', job_table())
    write(OUT / 'status.json', dict(status='PREPARING', updated_at=time.time(), stage='M0', completed=0, failed=0, running={}))
    print('contract written', OUT)


if __name__ == '__main__':
    main()
