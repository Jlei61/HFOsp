"""Reuse original A4 and independent count/field readouts for the unit repair."""
from common import OUT,read,write,log
from physical_delay_count_rate import DEST
import audit_fine_forcing_pair as original
import argparse


def register():
    original.DEST=DEST;original.register()


def audit():
    original.DEST=DEST;original.audit(partial=False)
    raw=read(DEST/'scientific_comparison.json');write(DEST/'scientific_comparison_raw.json',raw)
    previous=read(OUT/'conditioned_refractory_fine_forcing/scientific_comparison.json')
    native=next(r for r in raw['rows'] if r['label']=='native')
    old=dict(next(r for r in previous['rows'] if r['label']=='recorded_drive_binomial_seed1_fine_forcing'))
    new=dict(next(r for r in raw['rows'] if r['label']=='recorded_drive_binomial_seed1_fine_forcing'))
    old['label']='legacy_delay_split';new['label']='physical_delay_split'
    raw['rows']=[native,old,new]
    raw['matched_comparison']='One correctedtrajectory versus completedoriginalfineforcing counttrajectory; exactsameinitialization, seed, graph,response,Z/Mandinput. Only delayunitsinprivatevariance differ.'
    raw['core_event_rows']=[r for r in raw['core_event_rows'] if r['label'] in ['native','recorded_drive_binomial_seed1_fine_forcing']]
    for r in raw['core_event_rows']:
        if r['label']!='native':r['label']='physical_delay_split'
    raw['core_event_rows'] += [{**r,'label':'legacy_delay_split'} for r in previous['core_event_rows'] if r['label']=='recorded_drive_binomial_seed1_fine_forcing']
    raw['variance_unit_audit']=str(OUT/'physical_delay_variance_split/result.json')
    write(DEST/'scientific_comparison.json',raw)
    log('PHYSICAL DELAY SCIENTIFIC COMPARISON',[(r['label'],r['original_six_passed'],r['original_six_checks']) for r in raw['rows']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','audit']);a=p.parse_args();register() if a.command=='register' else audit()
