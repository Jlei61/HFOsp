# Topic 5 subject-specific `T_spectral_best` refinement

**Status:** v1.2 implementation contract. Numerical morphology-scoring weights were fixed before the E1146/E1084/E442/E583/E916 stress test. The stress test exposed a distinct failure mode—an isolated later sustained state could be mistaken for onset—and the temporal-recurrence acceptance gate below was locked before full-cohort execution.

## Scientific target

The analysis has two separate decisions:

1. **Phenotype existence:** does this seizure express a sustained, spatially distributed broadband-energy episode?
2. **Onset localization:** if the phenotype exists, which candidate change point best matches the onset morphology repeated across this patient's other seizures?

A seizure without the broadband phenotype receives no `T_spectral_best`. The algorithm must not create a time merely to obtain complete coverage.

## Fixed phenotype gate

The v1.1 label-blind episode detector remains unchanged: five common 1–80 Hz bands, fixed timing contacts, subject-level leave-one-seizure-out distal-background calibration, multiband and spatial support, future-5-s occupancy persistence, and positive step diagnostics.

An event is automatically eligible for `T_spectral_best` only when at least one sustained broadband episode is connected to the target seizure interval. A prior episode that returns to background before both annotations remains `prior_candidate_manual_only`; it is visualized but is not automatically promoted to a seizure onset. An event with no sustained broadband episode remains `phenotype_absent`.

## Candidate change points

The earliest connected sustained broadband episode is first locked as the onset episode. Later connected episodes are treated as subsequent ictal state changes and cannot compete for seizure onset. Within the locked onset episode, candidate points are local maxima of the consensus upward-step curve from 10 s before to 3 s after the confirmed state begins. The v1.1 change point and episode start are always included. Candidates closer than 0.5 s are collapsed, retaining the stronger step, with at most 12 candidates.

Each candidate receives a fixed-dimensional signature containing:

- the five-band positive step profile;
- the five-band fraction of contacts supporting a positive step and elevated post-level;
- the five-band temporal trajectory sampled from 2 s before to 5 s after the candidate;
- generic change quality from consensus strength, spectral breadth, spatial support, and proximity to the confirmed state onset.

The detector does not read interictal A/B labels, scaffold correlations, or outcome variables.

## Patient-specific leave-one-seizure-out prototype

Training seeds are phenotype-positive seizures whose v1.1 candidate interval is stable (90% width no greater than 5 s). For each target seizure, its own seed is removed before forming the prototype. At least three other stable seizures are required.

The patient prototype is the normalized coordinate-wise median of the remaining candidate signatures. Prototype coherence is the median cosine similarity between training seeds and the prototype. The prototype is used only when coherence is at least 0.65; otherwise localization falls back to event-local generic quality and is explicitly flagged as lacking a coherent patient template.

## Selection and uncertainty

With a coherent prototype, candidate score is:

`0.65 × patient-prototype similarity + 0.35 × generic change quality`.

Without a coherent prototype, the score is generic change quality alone. If candidates are within 0.03 score units, the earliest is chosen because the target is episode onset rather than the largest later ictal peak.

Uncertainty is estimated with 100 contact-bootstrap resamples and bootstrap resampling of the other-seizure training seeds. Outputs include the 5th–95th percentile interval, interval width, selection consistency within 1 s, best-versus-second score margin, prototype similarity, prototype coherence, and whether the patient prototype was actually used.

## Patient-level temporal-recurrence acceptance gate

Phenotype existence and a candidate change point are not sufficient to call that point an accepted onset. Pilot visual review showed that a seizure can have an obvious ictal change near annotation time but only enter a sustained broadband state 15–20 s later. That later state remains a valid spectral candidate, but it is not automatically promoted to onset.

For each target, stable v1.1 seed times from all other seizures define patient-level timing modes. The support radius is adaptive: for every training seed, compute its distance to the second-nearest training seed; take the 90th percentile of these distances, with a 2 s resolution floor. A target candidate is accepted as `accepted_subject_recurrent` only when at least two other seizures fall within this radius, so the mode has at least three seizures including the target. This rule allows multiple recurrent timing modes rather than forcing one patient median.

When fewer than three other stable seizures exist, the point is retained as `candidate_no_subject_timing_template`. When a template exists but fewer than two other seizures support the point, it is retained as `candidate_temporally_unanchored`. Both classes remain visible in per-seizure figures with uncertainty intervals but do not populate the accepted `T_spectral_best` field without manual adjudication.

## Required outputs

- one row per seizure, including phenotype status, candidate status, and blank accepted `T_spectral_best` when phenotype-ineligible or temporally unanchored;
- one raw-waveform/multiband diagnostic figure per seizure with EEG onset, clinical onset, v1.1 candidate, and v1.2 `T_spectral_best` shown separately;
- patient-level coverage, prototype coherence, timing median/IQR, and bootstrap stability;
- cohort summaries folded by subject, never by pooled seizure count alone.

## Ictal-cache integration

Accepted times are materialized as a parallel, versioned ictal cache rather than overwriting the clinical-onset-referenced source cache. Only `accepted_subject_recurrent` seizures enter aligned arrays; phenotype-absent, prior-only, temporally unanchored, and no-template events remain explicit exclusions in the alignment inventory. The five common 1–80 Hz traces retain their source values and fixed timing-contact ordering, while every time grid is re-zeroed to `T_spectral_best`. Clinical and EEG onset offsets in the new frame are retained per event so downstream analyses can distinguish the spectral reference from the annotations.

Cache coverage and analysis eligibility remain different contracts. Additional narrow-axis subjects may be cached as a sensitivity tier when they retain a valid baseline, complete interval, at least 300 s separation from the previous seizure, and at least six resolved narrow-axis contacts. Waiving the primary 80% montage-coverage threshold permits cache construction only; those additions do not enter primary cohort inference unless they later satisfy the primary admission contract.

## Dataset-specific annotation reference

Epilepsiae retains the original clinical-onset-referenced source cache. EEG and clinical annotations are both available for assigning sustained episodes to seizures, and accepted arrays are re-zeroed from clinical onset to `T_spectral_best`.

Yuquan has EEG onset annotations only. Its source-cache zero is therefore the recorded EEG onset, episode assignment uses that single anchor, and accepted arrays are re-zeroed from EEG onset to `T_spectral_best`. Yuquan outputs must set `annotation_mode=eeg_only`, `clinical_onset_available=false`, and `cache_zero_reference=eeg_onset`; clinical-onset fields remain missing rather than duplicating the EEG time under a clinical label. This annotation difference is retained in every manifest and cache inventory and prevents an EEG-only sensitivity extension from being presented as independent clinical-onset validation.

## Claim boundary

`T_spectral_best` is the best algorithmic localization of a specifically defined, patient-recurrent broadband-onset pattern. It is not a universal seizure-onset label. Events without the phenotype, ambiguous prior-only episodes, isolated late broadband states, and events lacking a patient timing template remain without an accepted time; their candidate times are retained separately for review.
