"""Persist the actual coverage and agreement of full stationary root counts."""
from rate_stationary_root_count import *
from datetime import datetime,timezone


def main():
    plan=read(DEST/'batch_plan.json');rows=[]
    tasks=[(f'J{J:.8f}',J,None) for J in plan['baseline_J']]
    tasks +=[(f'branch{q["branch_index"]:04d}',q['J_EE_core'],q['branch_index']) for q in plan['branch_sites']]
    for tag,J,index in tasks:
        records=[]
        for off in plan['offsets']:
            path=DEST/f'{tag}_offset{off:g}.json'
            if path.exists():
                q=read(path);assert q['equilibrium_residual']<1e-8
                records.append(dict(source=str(path),offset=off,status=q['status'],count=q['positive_root_count'],
                    maximum_phase_increment=q['refinements'][-1]['maximum_phase_increment'],
                    tail_bounds=[q['right_half_plane_tail_row_bound'],q['high_frequency_tail_row_bound']]))
        resolved=len(records)==len(plan['offsets']) and all(q['status']=='NUMERICALLY_RESOLVED' for q in records)
        agreement=resolved and len({q['count'] for q in records})==1
        rows.append(dict(site=tag,J_EE_core=J,branch_index=index,records=records,
            status='PAIRED_OFFSET_AGREEMENT' if agreement else 'OFFSET_DISAGREEMENT' if resolved else 'PENDING',
            unstable_root_count=records[0]['count'] if agreement else None))
    live=False
    try:os.kill(plan['pid'],0);live=True
    except OSError:pass
    out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),planned_states=len(tasks),
        completed_contours=sum(len(q['records']) for q in rows),
        resolved_paired_states=sum(q['status']=='PAIRED_OFFSET_AGREEMENT' for q in rows),
        worker_pid=plan['pid'],worker_alive=live,rows=rows,
        inventory_complete=False,
        scope='Argument-principle counts of the full rate characteristic determinant at sampled equilibria. All tail frequencies bounded; interior contour resolution numerical. Equal counts at two offsets do not prove the narrow strip is empty, or exclude paired crossings between parameter sites.')
    write(DEST/'summary.json',out)
    print('STATIONARY COUNT COVERAGE',out['completed_contours'],'/',2*len(tasks),
        'paired states',out['resolved_paired_states'],'worker alive',live,flush=True)


if __name__=='__main__':main()
