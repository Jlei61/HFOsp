"""Map a new simulation to the unchanged paper waveform painter."""
import numpy as np
from modeling.panel_analysis import classify_model
from plotting.paper.fig4_waveform import waveform_panel,DISPLAY


def waveforms(run,patient_templates,names):
    # The historical painter encodes TA as 1 and TB as 0.
    labels=1-classify_model(run['observed']['centroid_ms'],patient_templates)
    ids=np.array([i for i in run['observed']['primary_event_indices']
                  if run['observed']['events'][i]['window_ms'][0]>=1500],dtype=int)
    if len(np.unique(labels[ids]))<2:
        raise ValueError('Both model template classes are required for the paper Fig4F pair')
    source=dict(events=run['observed']['events'],actual_duration_ms=float(run['time_ms'][-1]))
    arrays=dict(event_mode=labels,recruitment_ms=np.asarray(run['observed']['recruitment_ms'],float),
                contact_envelope_dt_ms=run['envelope_dt_ms'],contact_envelope=run['envelope'])
    return waveform_panel(source,arrays,ids,np.array([names.index(n) for n in DISPLAY]))
