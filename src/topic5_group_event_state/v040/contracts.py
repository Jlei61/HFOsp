"""The four machine-readable contract tables and the G0 runner.

G0 fixes definitions and repairs interfaces; it is not a global qualification
gate.  Anything that is independently valid continues while a repair is pending.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import conditions as CD
from . import checks as K
from .objective import (HORIZON_WEIGHTS, VIEW_WEIGHTS, FAMILIES, VIEWS, SECONDARY, SELECTION_HORIZON,
                        SELECTION_VIEWS)
from .model import COND_DIM
from ..v0312.numerics import LATENT
from .train import RunConfig, ARMS, FINAL_SEEDS, atomic_json, source_digest, load_run, file_hash
from .engine import DONOR_RULE


def support_table(payload, cfg):
    decision = D.short_history_decision(payload, cfg['subject'], cfg['split_seed'],
                                        tuple(cfg['short_history_candidates']))
    H = decision['selected_short_history_minutes']
    binding = {}
    pk = payload['packets']
    for stage, role in (('inner0', 'inner'), ('inner1', 'inner'), ('outer', 'outer')):
        split = D.build_split(payload, cfg['subject'], cfg['split_seed'], stage, 'S-E')
        table = D.target_table(payload, split, role, 30)
        qs = np.unique(table[:, 2]) if len(table) else np.empty(0, int)
        allowed = D.input_mask(split, role)
        binds = 0
        lost = []
        for q in qs:
            t = pk['end'][q]
            s0 = int(split['episode_start'][q])
            s1 = max(s0, int(np.searchsorted(pk['start'], t - H * 60 - 1e-6)))
            if s1 > s0:
                binds += 1
            f = np.arange(s0, q + 1)
            f = f[allowed[f] & (pk['release'][f] <= t)]
            h = np.arange(s1, q + 1)
            h = h[allowed[h] & (pk['release'][h] <= t)]
            lost.append(int((pk['event_hi'][f] - pk['event_lo'][f]).sum() - (pk['event_hi'][h] - pk['event_lo'][h]).sum()))
        lost = np.asarray(lost)
        binding[stage] = dict(
            n_queries=int(len(qs)), short_window_truncates_prefix=int(binds),
            truncation_fraction=float(binds / max(len(qs), 1)),
            queries_losing_events=int((lost > 0).sum()),
            median_events_removed_when_binding=float(np.median(lost[lost > 0])) if (lost > 0).any() else 0.)
    strata = {}
    for stage in ('inner0', 'inner1', 'outer'):
        rows = decision['stages'][stage]['windows'][H]['rows']
        mem = [r for r in rows if r['memory_support']]
        strata[stage] = dict(
            n_queries=len(rows), memory_support_queries=len(mem),
            recent_event_quartiles=(np.percentile([r['readable_recent_events'] for r in rows],
                                                  [0, 25, 50, 75, 100]).tolist() if rows else None),
            earlier_event_quartiles=(np.percentile([r['readable_older_events'] for r in mem],
                                                   [0, 25, 50, 75, 100]).tolist() if mem else None),
            memory_support_without_earlier_events=int(sum(r['readable_older_events'] == 0 for r in mem)))
    return dict(
        subject=cfg['subject'], split_seed=cfg['split_seed'],
        short_history=decision,
        short_history_binding=binding,
        memory_support=dict(
            definition=('short-window coverage>=0.5, support-end age<=H/2 and at least 30 further minutes of '
                        'legal published exposure before the short window'),
            strata=strata,
            caveat=('a sufficiently observed quiet history is valid information, but it does not license the '
                    'claim that several earlier events were integrated')),
        support_classes=dict(
            window_not_observable='the amplifier record does not cover the window',
            observed_but_unpublished='the window was recorded but no block covering it had been released',
            partially_published='some of the observable seconds were released by query time',
            published_and_quiet='fully published exposure carrying zero interictal events',
            published_with_events='fully published exposure carrying at least one event'),
        release=CD.availability_note(),
        training_cutoff=D.training_cutoff_audit(payload, cfg['subject'], cfg['split_seed']),
        limitations=[
            'INNER0 reaches the two-hour support rule on 1/12 main-horizon queries; two-hour eligibility is '
            'not broadly satisfied and the earlier-history contrast is weakly supported on that origin',
            'neither 30 nor 120 minutes met the 80% quorum on both INNER origins, so the longest registered '
            'candidate replaced the shorter one under the pre-registered rule',
            'count and marks share one closed one-hour block release; a distinct earlier count publication '
            'time is not inferred'])


def training_objectives_table(prep):
    pk = prep.payload['packets']
    n_ratio = int(prep.band_ratio.shape[1])
    n_xlag = int(prep.xlag.shape[1])
    return dict(
        optimization=dict(horizon_weights=HORIZON_WEIGHTS, view_weights=VIEW_WEIGHTS,
                          current_packet_reconstruction_weight=0.0,
                          entry='src/topic5_group_event_state/v040/objective.py:training_loss',
                          denominator=('expected number of scored windows of that view and horizon in a batch; '
                                       'fixed before every microbatch so accumulation is boundary independent'),
                          denominator_entry='src/topic5_group_event_state/v040/train.py:training_normalizers'),
        selection=dict(score=f'0.5*spatial@{SELECTION_HORIZON} + 0.5*morphology@{SELECTION_HORIZON}',
                       views=list(SELECTION_VIEWS), weighting='window-equal-weight',
                       entry='src/topic5_group_event_state/v040/objective.py:selection_score',
                       monitored_only=['count', 'load', 'legacy_packet_joint_component_score'],
                       note='count, load and the preserved v0312 packet-joint score never select a checkpoint'),
        targets=[
            dict(name='count', distribution='negative binomial on the exposure-integrated rate',
                 grid='13-point 5-second trapezoid on the observed support', coordinate='log rate',
                 support='every valid window, including zero-event windows',
                 weight='w_h/3', role='trained and monitored; not in the main selection',
                 entry='v040/objective.py:window_scores -> v0312/model.py:count_log_prob'),
            dict(name='shaft_composition', distribution='multinomial given the total participating contacts',
                 coordinate='softmax logits', support='windows with at least one participating contact',
                 weight='w_h/3', role=f'trained; enters the main selection at h={SELECTION_HORIZON}',
                 conditioning='conditioned on the total participation count, not on the number of events',
                 entry='v040/objective.py:window_scores -> v0312/model.py:composition_log_prob'),
            dict(name='band_ratio', distribution='per-component Normal on FIT-standardised coordinates',
                 n_components=n_ratio, support='event-time paths with a finite original component',
                 weight=f'w_h/3 shared by the three morphology families',
                 role=f'trained; enters the main selection at h={SELECTION_HORIZON}',
                 entry='v040/objective.py:morphology_terms'),
            dict(name='signed_xlag', distribution='per-component Normal on FIT-standardised coordinates',
                 n_components=n_xlag, support='event-time paths with a finite original component',
                 weight='shares w_h/3', role=f'trained; enters the main selection at h={SELECTION_HORIZON}',
                 entry='v040/objective.py:morphology_terms'),
            dict(name='delay_iqr', distribution='point mass at exact zero plus a Normal on the standardised log',
                 n_components=1, support='events with a finite delay IQR', weight='shares w_h/3',
                 role=f'trained; enters the main selection at h={SELECTION_HORIZON}',
                 entry='v040/objective.py:morphology_terms'),
            dict(name='total_load', distribution='Normal on the FIT-standardised log total load',
                 support='windows with at least one event', weight='0.25*w_h',
                 role='trained and monitored; not in the main selection',
                 entry='v040/objective.py:window_scores -> v0312/model.py:load_log_prob'),
            dict(name='current_packet_reconstruction', distribution=None, support=None, weight=0.0,
                 role='no independent target exists; the h=1 target is the next minute, not a reconstruction'),
            dict(name='conditional_identity / continuation / seizure', distribution='consumer-defined',
                 support='frozen-consumer FIT and validation', weight=0.0,
                 role='never in the producer loss and never in checkpoint selection')],
        morphology_estimator=dict(
            per_component='-log mean_s p(y_bik | z_s) on the event-time prediction path',
            order=['mixture per valid scalar component', 'mean over the valid components of a family',
                   'equal weight over the families present for that event',
                   'equal weight over the events with morphology support in the window',
                   'equal weight over the windows with at least one scorable event'],
            companion='event-equal-weight Delta_event is reported alongside the window-equal-weight main effect',
            zero_event_windows='score the count target; conditional morphology is not applicable and is not zero-filled',
            entry='v040/objective.py:morphology_terms, v040/objective.py:aggregate'),
        preserved_legacy=dict(
            name='legacy_packet_joint_component_score',
            definition=('the v0312 rule: sum every event and component log-probability inside the packet, mix '
                        'over paths once, normalise by the total valid component count'),
            status='monitored only; never divided by an event count and renamed as the new estimator',
            entry='v040/objective.py:legacy_packet_joint_component_score'),
        transforms=dict(band_ratio='FIT median/IQR per component', signed_xlag='FIT median/IQR per component',
                        delay_iqr='FIT median/IQR of log positive values, with an explicit zero mass',
                        total_load='FIT median/IQR of the log total load',
                        packet_inputs='FIT median/IQR per column, clipped to +-8',
                        conditions='FIT median/IQR of the continuous entries only; flags are never rescaled'),
        conditions=dict(names=list(CD.NAMES), dim=int(CD.DIM), readout_conditioning_dim=int(COND_DIM),
                        layout='intraday sin/cos at the evaluation time, then the query-time condition vector',
                        entry='v040/conditions.py'),
        fit_support=dict(fit_packets=int(prep.split['train_packet'].sum()),
                         late_release_packets_removed=int(prep.split['late_release_excluded_from_fit'].sum()),
                         fit_events=int(prep.split['train_packet'][D.event_packets(prep.payload)].sum())))


def state_export_table():
    return dict(
        exported=dict(name='QueryState', m=dict(dim=LATENT, meaning='posterior mean of the generative state'),
                      P=dict(shape=[LATENT, LATENT], meaning='posterior covariance'),
                      cond=dict(dim=int(CD.DIM), names=list(CD.NAMES)),
                      metadata=['query_packet', 'query_time', 'source_time', 'release_time',
                                'information_age_minutes', 'available_exposure_seconds', 'readable_events',
                                'prefix_start', 'input_digest', 'producer_hash'],
                      entry='v040/engine.py:QueryState.export'),
        withheld=dict(inference_gru_c64='the inference GRU hidden state is never handed to a consumer',
                      within_minute_event_memory='the per-event GRU state is internal to the encoder',
                      reason='the single export contract keeps every consumer reading the same object'),
        consumer_routes=dict(
            H1=dict(reads='posterior samples of the same Q, advanced by f and read out by g',
                    note='P is part of the predictive uncertainty, not a separate score'),
            H2a_A=dict(reads='m only, 24 dimensions, standardised inside the consumer FIT',
                       forbidden=['inference GRU c', 'covariance diagonal']),
            H2a_B=dict(reads='m only, alongside the genuinely observed event prefix'),
            S_A=dict(reads='the fixed functional coordinates of the frozen g applied to the same Q'),
            S_B=dict(reads='the same m and C as H2a-A, through the frozen interictal contact head'),
            common_C=dict(reads='the legal clock, measurement support and deterministic recording timers',
                          availability='offered identically to every arm and every consumer')),
        representation_classes=dict(
            static_trait='FIT contact dictionary, geometry and templates; never refitted on the scored period',
            unlearned_input_driven=('a checkpoint selected at step 0 still varies with history through the '
                                    'fixed recurrent map; this is not a static trait'),
            learned='parameters that moved away from initialisation, evidenced by the parameter inventory',
            decided_by=['selected_updates', 'parameter_inventory relative_update', 'consumer weight change']),
        identity=dict(same_state_means='one producer, one export function and its fixed projections',
                      does_not_mean='that consumer input tensors are identical or that Q is a sufficient statistic'))


def consumer_routes_table(prep):
    community = np.asarray(prep.payload['shaft_index'])
    sizes = {int(c): int((community == c).sum()) for c in np.unique(community)}
    return dict(
        H2a_A=dict(
            query='the last registered minute query strictly before the event, reading only release<=q_i',
            task='fine identity given the event size or coarse composition; not an unconditional set forecast',
            conditions=['C', 'C+H', 'C+S', 'C+H+S'], weighting='event-equal-weight',
            budget=dict(max_steps=400, validate_every=20, patience=6, optimizer='AdamW', lr=.01,
                        weight_decay=1e-3, report_step_zero=True),
            normalisation=('per community c given K_c: exp(sum_{j in A_c} l_j) / e_{K_c}(exp l_c); '
                           'community log-probabilities are summed'),
            units='K_c=0 and K_c=|c| are deterministic; an event whose communities are all deterministic is not a unit',
            communities=sizes,
            entry='v040/consumers.py:h2a_a, reusing v0312/frozen.py:conditional_set_lp and identity_units'),
        wrong_time=dict(head='the fixed C+H+S conditional identity head', swapped='the state slot only',
                        unchanged=['recipient C', 'recipient H', 'target', 'K_c'], donor_rule=DONOR_RULE,
                        selection='nearest legal FIT donor by time distance; no target, seizure or score is read',
                        no_donor='locally not estimable; the criteria are never relaxed',
                        entry='v040/engine.py:donor_match, v040/engine.py:wrong_time_control'),
        H2a_B=dict(prefix='the genuinely observed prefix of the event, in occurrence order',
                   forbidden=['final K', 'normalised rank', 'whole-event waveform', 'suffix statistics',
                              'event end time'],
                   endpoints=['next contact', 'next synchronous group', 'STOP'],
                   protocols=['teacher forced one step', 'free continuation'],
                   entry='v040/consumers.py:h2a_b'),
        S_A=dict(windows=dict(primary=[-7200, -1800], secondary=[-1800, -300]),
                 coordinates=['shaft softmax posterior mean', 'band_ratio predicted mean per component',
                              'signed_xlag predicted mean per component', 'delay IQR zero probability',
                              '(1-p0)-weighted normalised positive standardised log-IQR conditional mean'],
                 forbidden=['future real K', 'future participation', 'future morphology'],
                 reference='the FIT reference state distribution at the identical query, C and MC noise',
                 primary_output='per case window, per coordinate signed mean difference against matched controls',
                 overall='sqrt(0.5*spatial + 0.5*mean over the three morphology families of mean squared standardised differences)',
                 caveat='non-negative by construction; it is not a positivity verdict and no latent axis is picked',
                 entry='v040/consumers.py:s_a'),
        S_B=dict(query='the last legal minute query at least 5 minutes before EEG onset',
                 predictor='the frozen interictal H2a-A contact head on the same m and C',
                 target='clinical onset [0,10] seconds, CAR, 1-150 Hz baseline robust-z broadband activation',
                 primary='within-community Spearman on identical valid contacts, equal weight over qualified communities',
                 minimum_contacts_per_community=3, undefined='constant prediction or constant target',
                 auxiliary='all-contact Spearman, retained as the legacy convention with a cross-community bias caveat',
                 forbidden='substituting continuous broadband energy into the IED set likelihood',
                 entry='v040/consumers.py:s_b'),
        S_C=dict(status='NOT_RUN', requires=['complete observable person-hours', 'event annotation',
                                             'legal time validation'],
                 note='a missing denominator limits S-C only; it does not block S-A or S-B'),
        no_new_seizure_head='the first package adds no seizure-supervised head')


def run_g0(cfg, device='cuda:0'):
    """Emit the four contract tables and run the interface checks."""
    root = Path(cfg['results_root'])
    out = root / 'contracts'
    out.mkdir(parents=True, exist_ok=True)
    packet_path = Path(cfg['packets_root']) / f"{cfg['subject']}.pt"
    payload = torch.load(packet_path, weights_only=False, map_location='cpu')
    support = support_table(payload, cfg)
    H = support['short_history']['selected_short_history_minutes']
    atomic_json(support, out / 'support.json')
    run = RunConfig(subject=cfg['subject'], arm_name='S_marks', inputs='P_marks', arm='state',
                    stage='inner1', short_history_minutes=H, packets_root=cfg['packets_root'],
                    out_dir=str(root / 'runs'), device=device, eval_stride=60, eval_paths=8, eval_chunk=12)
    model, prep = load_run(run)
    atomic_json(training_objectives_table(prep), out / 'training_objectives.json')
    atomic_json(state_export_table(), out / 'state_export.json')
    atomic_json(consumer_routes_table(prep), out / 'consumer_routes.json')
    results = dict(
        morphology_estimator=K.check_morphology_estimator(model, prep, run, 'inner'),
        weight_formula=K.check_weight_formula(model, prep, run),
        set_normalization=K.check_set_normalization(),
        release_cutoff=K.check_release_cutoff(payload, cfg['subject'], cfg['split_seed']),
        replay_and_streams=K.check_replay_and_streams(model, prep, run, 'inner'),
        extreme_scores=K.check_extreme_scores(model, prep, run, 'inner'))
    status = {k: v['status'] for k, v in results.items()}
    report = dict(status='COMPLETE' if all(v != 'FAILED' for v in status.values()) else 'FAILED',
                  scope='interface verification at initialisation; not a scientific result and not a training stage',
                  checks_status=status, results=results, source_digest=source_digest()[0],
                  packets_sha256=file_hash(packet_path), short_history_minutes=H,
                  note='G0 fixes definitions and repairs interfaces; independently valid work continues in parallel')
    atomic_json(report, out / 'g0_checks.json')
    return dict(status=report['status'], short_history_minutes=H, checks_status=status,
                contracts=[str(out / f) for f in ('support.json', 'training_objectives.json',
                                                  'state_export.json', 'consumer_routes.json')])
