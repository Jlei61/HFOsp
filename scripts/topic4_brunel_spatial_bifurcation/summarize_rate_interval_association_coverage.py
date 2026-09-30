"""Match current stability brackets to existing full-waveform root evidence.

New brackets remain explicit; old association files never certify a newly
expanded interval simply because they reference the same source filename.
"""
from complete_rate_positive_stability import DEST, read, write
from pathlib import Path
import time


def identity(row):
    return (row['family'], tuple(Path(e['analyzed_orbit']).resolve() for e in row['ends']))


def main():
    current=read(DEST/'current_interval_evidence.json')
    old=read(DEST/'current_interval_root_associations.json')
    fine=read(DEST/'interval_root_temporal_verification.json')
    prior={identity(r):r for r in old['rows']}
    rows=[]
    for bracket in current['opposite_or_dimension_change_brackets']:
        previous=prior.get(identity(bracket))
        checked=[]
        if previous:
            for root in previous['nearby_roots']:
                if not root['within_sampled_polyline_neighborhood']:
                    continue
                records=[r for r in fine['rows'] if r['family']==bracket['family']
                         and r['site_indices']==[e['site_index'] for e in bracket['ends']]
                         and r['internal_label']==root['internal_label']]
                for r in records:
                    unchanged=True
                    for stamp in r['profile_fingerprints']:
                        path=Path(stamp['path']);stat=path.stat()
                        unchanged &= (stat.st_size==stamp['size'] and stat.st_mtime_ns==stamp['mtime_ns'])
                    if unchanged and r['association_retained']:
                        checked.append(dict(label=r['label'],internal_label=r['internal_label'],
                            evidence=str(DEST/'interval_root_temporal_verification.json'),
                            root_validation_status=r['root_validation_status']))
        rows.append(dict(**bracket,known_root_associations=checked,
            association_status='EXISTING_ROOT_NEIGHBORHOOD_CHECKED' if checked else 'ROOT_ASSOCIATION_PENDING',
            interval_certified=False))
    result=dict(status='CURRENT_ASSOCIATION_COVERAGE_CHECKED',timestamp=time.time(),
        coverage_source=str(DEST/'current_interval_evidence.json'),rows=rows,
        current_brackets=len(rows),brackets_with_checked_known_root=sum(bool(r['known_root_associations']) for r in rows),
        pending_brackets=sum(not r['known_root_associations'] for r in rows),
        global_branch_completeness=False,
        scope='Match exact ordered endpoint identities and unchanged full-waveform evidence. '
              'Pending brackets may contain already known folds or other crossings; they do not establish a new bifurcation. '
              'A known root association does not certify the number of roots or absence of compensating crossings.')
    write(DEST/'current_interval_association_coverage.json',result)
    print('ASSOCIATION COVERAGE',result['current_brackets'],result['brackets_with_checked_known_root'],result['pending_brackets'])


if __name__=='__main__':main()
