# Topic 5 `T_spectral` episode adjudication design

**Status:** algorithmic v1.1 locked for full-cohort execution after the
E1146/E442/E583/E916 stress test. v1.1 corrects the persistence implementation to match the
predefined future-5-s occupancy rule; no numerical threshold was retuned. Manual adjudication
remains pending, so automatic output is not a final clinical seizure-onset label and thresholds
must not be retuned on the cohort result.

## Scientific target

`T_spectral` is the onset of a sustained, spatially distributed broadband-energy episode
that is connected to the target annotated seizure. It is not the largest spectral change in
a peri-ictal window and it is not a replacement for the clinical EEG-onset annotation.

Only seizures expressing this phenotype receive a precise `T_spectral`. Non-expressors are
kept as phenotype-negative or uncertain observations rather than force-aligned.

## Fixed input contract

- Primary frequency support: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, and gamma
  30–80 Hz.
- Contacts: the fixed masked rank-displacement / lagPat-valid timing-contact set. A/B labels,
  template ranks, and scaffold correlations are not read by the detector.
- Signal: per-contact log band power, robust-normalized by the existing cache contract and
  re-centred to the event's distal background.
- Calibration: subject-level leave-one-seizure-out distal-background samples. The same
  false-positive threshold is used across seizures; the target seizure does not calibrate
  its own gate unless no other seizure exists.

## Automatic episode gates

An automatic broadband episode must pass all of the following provisional gates:

1. **Background-extreme level:** contact-by-band energy exceeds its subject-level
   leave-one-seizure-out distal-background Q95.
2. **Spectral breadth:** at least 3/5 bands are active, including at least one low-frequency
   band (delta/theta/alpha) and one high-frequency band (beta/gamma).
3. **Spatial support:** at least 25% of timing contacts and at least two contacts show
   co-activation in at least three bands.
4. **Persistence:** the broadband state occupies at least 60% of the following 5 s and brief
   gaps no longer than 1 s are merged. This is a future-window occupancy rule, not a requirement
   for five uninterrupted seconds of broadband state.
5. **Upward change:** the episode onset neighbourhood contains a positive consensus step
   above a robust subject-background z threshold of 3.0. The reported change point is the
   peak of the earliest complete multiband/spatial step connected within the preceding 10 s
   to the confirmed sustained state. This separates state confirmation from onset localization
   and avoids reporting the later stable plateau as onset.
6. **Temporal precision:** band/contact bootstrap gives a 90% onset interval no wider than
   3 s. Episodes failing this gate remain broadband episodes without a precise onset.

Every connected broadband episode receives a candidate change time and bootstrap interval.
Primary eligibility still requires all gates and CI width no greater than 3 s; a candidate
with CI width no greater than 5 s but an incomplete step gate is sensitivity-only.

The numerical gates above are intentionally visible in every diagnostic output. They are
pilot values, not post-review cohort claims.

## Episode-to-seizure assignment

All broadband episodes are detected before reading EEG or clinical onset markers. Markers
are used only in a second stage to assign an episode to the target seizure.

- An episode is automatically connected when its sustained state overlaps the interval from
  the earlier of EEG/clinical onset through 20 s after the later marker.
- A strong earlier episode that returns to background before that interval is labelled
  `separate_prior_episode`, not selected as `T_spectral`.
- Manual review may override assignment when the annotation itself is clearly displaced, but
  the override and reason must be recorded.

## Allowed event-level outcomes

- `confirmed_precise_T`
- `broadband_but_imprecise_T`
- `narrowband_transition`
- `separate_prior_episode`
- `no_detectable_broadband_transition`
- `artifact_or_uncertain`

Only `confirmed_precise_T` enters the primary `T_spectral`-aligned analysis.
`broadband_but_imprecise_T` is sensitivity-only.

## Review workflow

1. **Blind pass:** anonymized time axis; no EEG/clinical marker and no subject/seizure label.
   The reviewer judges whether a sustained broadband transition exists and records its time.
2. **Revealed pass:** EEG/clinical markers and automatic episodes are shown. The reviewer
   decides whether the selected episode belongs to the target seizure and checks raw-signal
   artifact.
3. Thresholds are frozen only after the four-subject pilot covers a high-alignment positive
   subject (E1146), a mixed subject (E442), and low-alignment/negative subjects (E583/E916).
4. After freezing, the complete cohort is rerun without subject-specific retuning.

## Required per-seizure diagnostics

- raw traces on exactly the timing contacts;
- five-band spatial energy heatmap;
- five-band and consensus energy traces;
- contact-by-time number of active bands;
- consensus step z, number of active bands, and number of broadband-active contacts;
- automatic episode spans and onset uncertainty;
- EEG/clinical markers only in the revealed version;
- a manifest row containing every automatic gate and blank manual-adjudication fields.
