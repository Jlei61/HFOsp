"""Wait for one frozen phase, then produce the complete read-only review package.

No dispatch, nomination, new score or change of physical/observation contract.
"""
import argparse
import subprocess
import sys
import time
from scripts import analyze_topic4_propagation_recovery_night as stats
from scripts import plot_topic4_recovery_parameter_response as paired
from scripts import plot_topic4_recovery_temporal_support as temporal
from scripts import audit_topic4_paired_input_streams as inputs
from scripts import audit_topic4_local_burst_observer as burst
from scripts import audit_topic4_recovery_patient_support as support
from scripts.paper_figures import plot_topic4_recovery_review as fields
from scripts import plot_topic4_recovery_distributions as distributions
from scripts import plot_topic4_recovery_core_timing as core


def main(phase,wait=False):
    spec=stats.rt.read(stats.night.OUT/f'{phase}_units.json')
    pending=list(spec['units'])
    while pending:
        pending=[u for u in pending if not stats.an.run.complete(stats.an.run.result_path(spec['stage'],*u))]
        if not pending:break
        if not wait:raise RuntimeError(f'{len(pending)} incomplete units; no partial result labelled complete')
        if time.time()>stats.rt.read(stats.night.OUT/'plan.json')['hard_stop_unix']+600:
            raise RuntimeError('physical deadline passed; inspect incomplete units')
        time.sleep(30)
    stats.main(phase,True)
    paired.main(phase)
    temporal.main(phase)
    inputs.main(phase)
    burst.main(phase)
    support.main(phase)
    fields.main(phase)
    fields.main(phase,'primary')
    distributions.main(phase)
    for cid in dict.fromkeys(c for c,_,_ in spec['units']):core.main(phase,cid)
    stats.rt.write(stats.night.OUT/f'{phase}_analysis_complete.json',dict(
        status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',phase=phase,units=len(spec['units']),
        completed_unix=time.time(),producer=__file__,producer_sha256=stats.rt.sha(__file__),
        interpretation='Artifact generation completed; no patient propagation acceptance or model freeze.'))
    print(phase.upper()+'_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();main(a.phase,a.wait)
