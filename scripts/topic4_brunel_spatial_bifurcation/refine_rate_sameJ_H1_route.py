"""Repair underresolved physical rate filters on the displayed H1 return path."""
from rate_periodic_accuracy import *
from run_rate_sameJ_basin_bridge import DEST


def main(device):
    audit=read(DEST/'displayed_H1_path_filter_check.json')
    destination=DEST/'displayed_H1_path_refinement.json'
    prior=read(destination)['rows'] if destination.exists() else []
    done={q['source']:q for q in prior};rows=[]
    bad=[q for q in audit['rows'] if not q['filter_state']['positive']]
    for index,q in enumerate(bad):
        source=q['orbit']
        if source in done and done[source]['status']=='RESOLUTION_CHECKED':
            rows.append(done[source]);continue
        print('ROUTE REFINE',index,len(bad),source,flush=True)
        path,check=prepare(source,device,max_N=2048,host_krylov=False,check_filter_states=True,
            harmonic_chunk_size=32,stream_harmonics=False)
        original=np.load(source);fine=np.load(path)
        r=resample(original['r'],len(fine['r']),axis=0)
        dr=float(np.linalg.norm(fine['r']-r)/max(np.linalg.norm(r-r.mean(0)),1e-12))
        dp=abs(float(fine['T']/original['T'])-1)
        same=abs(float(fine['J']-original['J']))<1e-12 and dr<.01 and dp<.001
        row=dict(source=source,orbit=str(path),status=check['status'] if same else 'BRANCH_MATCH_REVIEW',
            resolution=check,relative_waveform_change=dr,relative_period_change=dp,
            branch_match_pass=same)
        rows.append(row)
        write(destination,dict(status='RUNNING',expected=len(bad),completed=len(rows),rows=rows,pid=os.getpid()))
        print('ROUTE RESULT',index,row['status'],dr,dp,flush=True)
        if row['status']!='RESOLUTION_CHECKED':break
    complete=len(rows)==len(bad) and all(q['status']=='RESOLUTION_CHECKED' for q in rows)
    write(destination,dict(status='COMPLETE' if complete else 'INCOMPLETE',expected=len(bad),completed=len(rows),rows=rows,
        scope='Same-J waveform corrections for underresolved points; retained old data, checked constituent-filter positivity, oversampled nonlinear defect and closeness to the original branch. No global H1-to-burst connection is inferred.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)
