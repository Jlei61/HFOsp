"""Build new Fig4 C inputs from frozen native SNN and measured E1146 data.

Writes only this candidate source directory. No old producer is imported.
Run: python /absolute/path/to/c_build_data.py
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'src/group_event_analysis.py').exists())
NATIVE = ROOT / '.worktrees/topic4-continuous-core-state-r1'
FIG4 = ROOT / 'results/paper-ready-figure/fig4'
RUN = Path('/data/hfosp/topic4_sef_hfo/xy_block_joint_observation_pilot_20260916')
TRAJ = RUN / 'xy_block/units/xy_left_20/2511_847401/workers/trajectory'
PATIENT = ROOT / 'results/paper-ready-figure/fig2/candidates/exemplar_rank_trajectory_20261009/source_snapshot'
TARGET = ROOT / 'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/shaft_aware_patient_training_target.npz'
PATIENT_BLOCK = Path('/mnt/epilepsia_data/interilca_inter_results/all_data_lns/1146/all_recs/114600102_0026')
CONTACTS = ['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tfr(x, fs, band):
    """Fig1 full-window S^3 rule on isolated real 250-ms tiles (not stitched).

    Magnitude STFT, 50-ms Hamming window, 80% overlap, same-size FFT,
    constant detrend; Gaussian sigma=1.5 bins BEFORE frequency selection.
    Input-specific frequency supports are explicit, not assumed equivalent.
    """
    n = int(.05 * fs)
    spectra = []
    for row in np.asarray(x, float):
        f, t, s = spectrogram(row, fs=fs, window='hamming', nperseg=n,
                             noverlap=int(.8*n), nfft=n, detrend='constant',
                             mode='magnitude', scaling='density')
        spectra.append(gaussian_filter(s, sigma=1.5))
    keep = (f >= band[0]) & (f <= band[1])
    s = np.asarray(spectra)[:, keep]
    weights = s**3
    denom = weights.sum(axis=(1, 2))
    assert np.all(denom > 0)
    tc = (weights*t[None, None, :]).sum(axis=(1, 2))/denom*1000
    fc = (weights*f[keep][None, :, None]).sum(axis=(1, 2))/denom
    return dict(tfr_magnitude=s, tfr_relative=s/s.max(axis=(1, 2))[:, None, None],
                tfr_time_ms=t*1000-125, tfr_frequency_hz=f[keep],
                tfr_centroid_ms=tc-125, tfr_centroid_hz=fc,
                tfr_local_rank=rankdata(tc, method='average'))


def main():
    run = read(TRAJ.with_suffix('.json'))
    assert sha(TRAJ.with_suffix('.npz')) == run['arrays_sha256']
    mechanism = read(FIG4 / 'source/mechanism_source.json')
    assert mechanism['mode_showcases']['events'][0]['event_index'] == 10
    event = run['events'][10]
    assert event['mode'] == 1 and event['primary_eligible']
    assert event['window_ms'] == [4238., 4488.]
    with np.load(TRAJ.with_suffix('.npz')) as z:
        names = z['contact_names'].astype(str).tolist()
        indices = [names.index(n) for n in CONTACTS]
        envelope = z['contact_envelope'].astype(float)
        dt = float(z['contact_envelope_dt_ms'])
        model_mu = z['centroid_ms'][10]
        model_ranks = rankdata(model_mu, method='average')
    cfg = read(NATIVE / 'config/topic4_continuous_core_state_r1.json')
    obs_path = Path(cfg['sources']['observation_contract']['path'])
    contract = read(obs_path)
    assert sha(obs_path) == cfg['sources']['observation_contract']['sha256']
    lo, hi = [int(t/dt) for t in event['window_ms']]
    full = envelope[:, lo:hi]
    baseline = np.maximum(contract['baseline'], np.quantile(full, .1, axis=1))
    weight = np.maximum(full-baseline[:, None], 0.)
    times = (np.arange(lo, hi)+.5)*dt
    reconstructed = (weight*times).sum(1)/weight.sum(1)
    # Frozen arrays use float32 envelopes; original observer used float64.
    err = float(np.max(abs(reconstructed-model_mu)))
    assert err < 1e-3, err
    with np.load(PATIENT / 'event_cache.npz') as z:
        patient_names = z['contact_order'].astype(str).tolist()
        pi = [patient_names.index(n) for n in CONTACTS]
        assert int(z['TA_event_pos']) == 6344
        assert np.all(z['TA_participant'][pi])
        patient_signal = z['TA_band_tile'][pi]
        previous_peak_component_mu = z['TA_fig1a_centroid_ms'][pi]
    with np.load(TARGET) as z:
        assert z['contact_names'].astype(str).tolist() == names
        row = np.flatnonzero(z['patient_train_event_indices'] == 6344)
        assert len(row) == 1
        row = int(row[0])
        patient_native_mu = z['patient_train_onsets'][row].astype(float)*1000
        patient_native_ranks = z['patient_train_ranks'][row].astype(float)+1
        np.testing.assert_array_equal(patient_native_ranks, rankdata(patient_native_mu))
    patient_meta = read(PATIENT / 'fig2_panelc_metadata.json')['exemplar']['TA']
    fs = patient_meta['fs_hz']
    assert fs == 1024 and patient_signal.shape == (len(CONTACTS), 256)
    # Recover the actual legacy stitched-window origin from packed sample counts.
    # Never infer a timestamp from rank or snap it to the newly calculated TFR.
    packs_path = Path(str(PATIENT_BLOCK)+'_packedTimes_withFreqCent.npy')
    lag_path = Path(str(PATIENT_BLOCK)+'_lagPat_withFreqCent.npz')
    packs = np.load(packs_path)
    packed_row = int(np.argmin(abs(packs[:,0]-patient_meta['t_in_block_sec'])))
    assert packed_row == 270 and abs(packs[packed_row,0]-patient_meta['t_in_block_sec']) < 1e-3
    segment_start = np.floor(packs[packed_row,0]/200)*200
    segment_t = segment_start+np.arange(int(200*fs))/fs
    segment_ids = np.flatnonzero((packs[:,0]>=segment_t[0]) & (packs[:,1]<=segment_t[-1]))
    sample_counts = np.array([np.sum((segment_t>=a)&(segment_t<=b)) for a,b in packs[segment_ids]])
    slot = int(np.flatnonzero(segment_ids == packed_row)[0])
    stitched_start_ms = float(sample_counts[:slot].sum()/fs*1000)
    assert stitched_start_ms == 1250. and sample_counts[slot] == 256
    with np.load(lag_path) as z:
        raw_ix = [z['chnNames'].tolist().index(n) for n in names]
        raw_times = z['lagPatRaw'][raw_ix,packed_row]*1000
        assert np.all(z['eventsBool'][raw_ix,packed_row])
    np.testing.assert_allclose(raw_times, patient_native_mu, atol=1e-4, rtol=0)
    patient_native_plot_ms = patient_native_mu[indices]-stitched_start_ms-125
    model_tfr = tfr(full[indices], 1000/dt, (20., 200.))
    # Lower display frequencies are derived from the same STFT, without moving centroids or ranks.
    broad=tfr(full[indices],1000/dt,(0.,200.))
    for key in ['tfr_magnitude','tfr_relative','tfr_time_ms','tfr_frequency_hz']:
        model_tfr[key]=broad[key]
    patient_tfr = tfr(patient_signal, fs, (50., 300.))
    arrays = dict(contact_names=np.array(CONTACTS), native_all_contact_names=np.array(names),
                  model_signal=full[indices], model_signal_time_ms=times-np.mean(event['window_ms']),
                  model_fs_hz=np.array(1000/dt), model_window_absolute_ms=event['window_ms'],
                  model_native_centroid_absolute_ms=model_mu,
                  model_native_centroid_ms=model_mu[indices]-np.mean(event['window_ms']),
                  model_native_rank_all=model_ranks, model_native_rank=model_ranks[indices],
                  patient_signal=patient_signal, patient_fs_hz=np.array(fs),
                  patient_signal_time_ms=np.arange(256)/fs*1000-125,
                  patient_native_stitched_centroid_ms=patient_native_mu,
                  patient_native_centroid_ms=patient_native_plot_ms,
                  patient_stitched_window_start_ms=np.array(stitched_start_ms),
                  patient_segment_packed_sample_counts=sample_counts,
                  patient_native_rank_all=patient_native_ranks,
                  patient_native_rank=patient_native_ranks[indices],
                  patient_previous_component_centroid_ms=previous_peak_component_mu)
    arrays.update({f'model_{k}':v for k, v in model_tfr.items()})
    arrays.update({f'patient_{k}':v for k, v in patient_tfr.items()})
    with np.load(FIG4 / 'source/burst_readout_arrays.npz') as z:
        assert z['names'].tolist() == ['ICL3','ICL4','ICL5']
        arrays.update(model_existing_30_80_burst=z['waveforms'][0],
                      model_existing_peak_ms=z['peak_times_ms'][0],
                      model_existing_burst_time_ms=z['time_ms'])
    score = read(RUN / 'scores/xy_left_20.json')
    plan = read(RUN / 'plan.json')
    objective_path = Path(plan['objective']['artifact'])
    assert sha(objective_path) == plan['objective']['sha256'] == score['objective_sha256']
    display = next(r for r in csv.DictReader(open(FIG4 / 'source/B_visible_workpoints.csv'))
                   if r['candidate'] == 'xy_left_20')
    errors = [float(display[k]) for k in ['rank_error', 'within_rod_order_error', 'participation_error']]
    j = np.mean([g['scaled'] for g in score['groups'].values()])
    assert abs(j-score['J']) < 1e-14
    for g in score['groups'].values():
        assert abs(g['A']-g['B']-g['D_off']) < 1e-14
    arrays.update(display_errors=errors, optimizer_J=np.array(j),
                  diagnostic_J_joint=np.array(float(display['J_joint'])))
    out = HERE / 'c_panel_data.npz'
    np.savez_compressed(out, **arrays)
    paths = [TRAJ.with_suffix('.npz'), TRAJ.with_suffix('.json'), TARGET, packs_path, lag_path,
             PATIENT/'event_cache.npz', PATIENT/'fig2_panelc_metadata.json',
             FIG4/'source/burst_readout_arrays.npz', FIG4/'source/B_visible_workpoints.csv',
             RUN/'scores/xy_left_20.json', RUN/'plan.json', objective_path, obs_path,
             NATIVE/'src/topic4_observation_repaired.py',
             NATIVE/'src/topic4_streaming_spike_readout.py',
             NATIVE/'src/topic4_three_observable_label_free.py',
             NATIVE/'src/topic4_three_observable_objective.py',
             NATIVE/'scripts/report_topic4_label_free_dense.py',
             NATIVE/'scripts/audit_topic4_joint_observation.py',
             ROOT/'scripts/plot_topic5_interictal_event_envelope_field.py',
             ROOT/'ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py',
             ROOT/'results/paper-ready-figure/fig1/source/code_snapshot/scripts/paper_figures/restore_fig1_legacy_spectrum.py']
    meta = dict(status='PENDING_AUTHOR_VISUAL_REVIEW', panel='new C',
        interface='render_c(fig, rect), normalized figure coordinates [left,bottom,width,height]',
        nominal_size_mm=[151,73], contacts_top_to_bottom=CONTACTS,
        purpose='Show genuine model and patient contact timing; explain event-distribution fitting, not a paired-event waveform loss.',
        model=dict(subject='epilepsiae_1146', public_label='E10', workpoint='xy_left_20',
            event_index=10, frozen_mode=1, mode='MTA', window_absolute_ms=event['window_ms'],
            seed_topology=2511, seed_dynamics=847401, trajectory_duration_ms=60000,
            signal='Gaussian contact average of E-spike counts in 2-ms bins, Gaussian temporal sigma=5 ms; no voltage forward model.',
            TFR_input='Unfiltered frozen firing-density trace, NOT its 30-80 Hz plotting derivative.',
            native_centroid='sum_t t*max(envelope_t-max(reference_baseline,q10_event),0)/sum_t max(envelope_t-max(reference_baseline,q10_event),0); 250-ms window, bin-center times.',
            native_centroid_reconstruction_max_abs_error_ms=err,
            existing_A_markers='Peaks of the 30-80 Hz derivative at 76/68/62 ms in the 4310-4430 ms crop; neither observer centroid nor TFR centroid.',
            spectrum_limit='Temporal smoothing and pulse shape dominate this low-frequency rate spectrum. It does not establish a model HFO oscillation.'),
        patient=dict(subject='epilepsiae_1146', mode='TA', event_pos=6344,
            block=patient_meta['block'], packed_start_in_block_sec=patient_meta['t_in_block_sec'],
            sample_rate_hz=fs, source_signal='Actual SEEG: CAR, 50/100/150/200/250-Hz notch, 80-250-Hz bandpass; frozen 250-ms band_tile.',
            membership='Exact event ID in the 30049-event development target. Do not infer membership of the later 19770-event FIT split from this alone.',
            selection='Reuse previously selected Fig2 TA exemplar; not selected here for closest model similarity.',
            previous_cache_centroid='Peak enhancement connected-component centroid, threshold 0.70. Not reused as the new S^3 centroid.',
            original_fitting_centroid='Legacy lagPatRaw S^3 time-frequency centroid on a stitched event timeline; stored seconds converted to milliseconds. Its rank is preserved.',
            fitting_marker_time_recovery=dict(block_packed_row=packed_row,
                preprocessing_segment_seconds=[float(segment_start),float(segment_start+200)],
                event_slot_in_segment=slot, stitched_start_ms=stitched_start_ms,
                sample_counts=sample_counts.tolist(),
                rule='Original producer selects inclusive sample endpoints, stitches packed tiles per 200-s segment. Sum preceding tile sample counts / 1024, subtract this from stored fitting centroid; then subtract 125 ms for centered display.',
                legacy_code_boundary='The legacy source has a 500-ms default, but these actual packed artifacts are 250 ms. Window size is taken from artifacts, not the legacy default.',
                plotted_ms=patient_native_plot_ms.tolist()),
            native_target_row=row),
        TFR=dict(algorithm='magnitude STFT, Hamming 50 ms, 80% overlap, nfft=nperseg, constant detrend; Gaussian sigma=1.5 in frequency/time bins BEFORE crop; W=S^3.',
            model_nperseg=25, model_noverlap=20, patient_nperseg=51, patient_noverlap=40,
            model_frequency_support_hz=[20,200], model_display_frequency_limits_hz=[0,80], model_available_spectrum_hz=[0,200], patient_frequency_support_hz=[50,300],
            time_centroid_formula='t_c=sum_tf t*S(t,f)^3 / sum_tf S(t,f)^3',
            frequency_centroid_formula='f_c=sum_tf f*S(t,f)^3 / sum_tf S(t,f)^3',
            normalization='Each channel/event magnitude divided by its own maximum; not power or cross-contact amplitude.',
            time_origin='Each 250-ms event window center; no channel-specific shifts, no event-specific time stretch.',
            relation_to_current_Fig1='Same full-window S^3 rule, but isolated 250-ms tiles rather than full stitched preprocessing segments; newly derived display, not exact legacy lagPatRaw replay.',
            new_centroid_used_by_optimizer=False,
            marker_legend='Legend omitted on canvas per author. Amber circle: newly calculated full-tile S^3 TFR centroid. Black top tick: stored timing used for rank. Thin amber lines connect adjacent contacts only within the same shaft.',
            model_centroid_ms=model_tfr['tfr_centroid_ms'].tolist(),
            patient_centroid_ms=patient_tfr['tfr_centroid_ms'].tolist(),
            model_centroid_hz=model_tfr['tfr_centroid_hz'].tolist(),
            patient_centroid_hz=patient_tfr['tfr_centroid_hz'].tolist()),
        rank=dict(display='Original fitting-readout rank among all 15 participating contacts, 1-based; all 15 contacts shown in SCL9–6 then ICL11–1 anatomical shaft order. Not reranked in this subset.',
            model=model_ranks[indices].tolist(), patient=patient_native_ranks[indices].tolist(),
            TFR_centroid_rank_is_stored_separately=True),
        loss=dict(plotted_formula='J = (1/3) sum_g D_g_off/s_g',
            groups=['rank_pattern','local_and_interrod_timing','participation'],
            D_off='||mean(phi_model)-mean(phi_patient_FIT)||^2 - [mean(||phi_model||^2)-||mean(phi_model)||^2]/(N-1)',
            scales='Frozen positive medians of squared embedding distances in patient CAL draws of 16 events; never re-estimated from this example.',
            feature_map='Frozen multiscale random Fourier features; rank/participation additionally include linear features; timing includes within-ICL, within-SCL and intershaft delays.',
            N_model=score['N'], J=float(j), J_joint=float(display['J_joint']),
            J_joint_role='Offline joint-feature diagnostic for xy_left_20, NOT its optimization loss. Later stages 13-16 jointly optimize J and J_joint.',
            current_panel_mapping='new F/G/H/I = accepted old C/E/B/D',
            display_errors=dict(zip(['mean_rank','within_rod_order','participation'], errors)),
            display_average=float(np.mean(errors)),
            display_average_formula='L_display=(epsilon_rank+epsilon_order+epsilon_participation)/3; descriptive summary, not optimizer J.',
            display_error_definition='Each error first averages absolute model-patient differences within SCL and ICL, then averages the two shafts equally. Rank: conditional participating normalized mean ranks; order: within-shaft co-participating pair precedence probabilities; participation: contact participation probabilities.',
            statistical_unit='A full 60-s native trajectory (151 eligible events after 1500-ms burnin), not the one illustrated event; comparator is the frozen patient FIT event distribution.',
            stages='1-5 label-conditioned development; 6-12 label-free J; 13-16 J and J_joint multiobjective. No single sum of the three displayed errors is the universal optimization objective.'),
        provenance={str(p):sha(p) for p in paths}, data_sha256=sha(out),
        validation=dict(real_patient_and_native_model=True, shared_contact_order=True,
            no_time_warp_or_waveform_synthesis=True, no_legacy_imports=True,
            observer_centroid_is_not_peak=True, observer_centroid_is_not_TFR_centroid=True,
            frozen_rank_arrays_preserved=True, frozen_optimizer_score_arithmetic_verified=True,
            frozen_objective_artifact_identity_verified=True,
            current_artifacts_edited=False))
    (HERE/'c_panel_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n')
    evidence = HERE / 'c_provenance'
    evidence.mkdir(exist_ok=True)
    for p in paths:
        if p.suffix == '.py' or p == obs_path or p == RUN/'scores/xy_left_20.json':
            shutil.copyfile(p, evidence/p.name)
    print(json.dumps(dict(data=str(out), model_native_rank=meta['rank']['model'],
                         patient_native_rank=meta['rank']['patient'],
                         model_TFR_ms=meta['TFR']['model_centroid_ms'],
                         patient_TFR_ms=meta['TFR']['patient_centroid_ms'],
                         centroid_reconstruction_error_ms=err, J=float(j)), indent=2))


if __name__ == '__main__':
    main()
