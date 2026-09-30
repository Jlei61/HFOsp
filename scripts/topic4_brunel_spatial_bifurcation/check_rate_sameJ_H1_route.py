"""Oversampled equation/positivity checks on every displayed H1 route point."""
from rate_periodic_accuracy import *
from run_rate_sameJ_basin_bridge import DEST


def main(device):
    original=read(DEST/'displayed_H1_path_filter_check.json')
    repaired=read(DEST/'displayed_H1_path_refinement.json')
    assert repaired['status']=='COMPLETE'
    mapping={q['source']:q for q in repaired['rows']};s=RateField();rows=[]
    target=DEST/'displayed_H1_path_continuous_check.json'
    prior=read(target)['rows'] if target.exists() else [];done={q['source']:q for q in prior}
    for index,q in enumerate(original['rows']):
        source=q['orbit']
        if source in done and done[source]['status']=='PASS':
            rows.append(done[source]);continue
        if source in mapping:
            match=mapping[source];path=match['orbit'];check=match['resolution']
        else:
            path=source;check=defect(s,path,device,harmonic_chunk_size=32,stream_harmonics=False)
            check['filter_state_check']=q['filter_state']
        passed=(check['maximum_group_defect_Hz']<.1 and max(check['regional_defect_Hz'])<.001
                and check['minimum_rate_Hz']>=-1e-9 and check['filter_state_check']['positive'])
        rows.append(dict(source=source,orbit=path,status='PASS' if passed else 'UNRESOLVED',check=check))
        if index%20==0 or not passed:
            write(target,dict(status='RUNNING',expected=len(original['rows']),completed=len(rows),rows=rows))
            print('ROUTE CHECK',index,rows[-1]['status'],flush=True)
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        if not passed:break
    passed=len(rows)==len(original['rows']) and all(q['status']=='PASS' for q in rows)
    write(target,dict(status='PASS' if passed else 'INCOMPLETE',expected=len(original['rows']),completed=len(rows),rows=rows,
        continuation_breaks=original['continuation_breaks'],
        scope='All displayed H1-route samples have positive constituent filters and pass the declared four-times oversampled nonlinear-defect thresholds. This is numerical branch geometry, not a complete interval stability classification or a proof of global H1-to-burst connection.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--wait-for-refinement',action='store_true');a=p.parse_args()
    if a.wait_for_refinement:
        started=time.time()
        while True:
            progress=read(DEST/'displayed_H1_path_refinement.json')
            if progress['status']=='COMPLETE':break
            assert progress['status']=='RUNNING',progress['status']
            assert time.time()-started<3600,'Timed out waiting for refinement'
            os.kill(progress['pid'],0)
            time.sleep(5)
    main(a.device)
