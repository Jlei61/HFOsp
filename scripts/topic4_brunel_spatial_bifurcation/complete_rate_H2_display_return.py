"""Check the complete displayed H2 return through LPC13 and its nearby flip.

Preserve continuation order, physical filters, exact J, and the full spatial
waveform when replacing old coarse samples with corrected temporal meshes.
"""
from complete_rate_positive_stability import *
from plot_rate_periodic_completion import families
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    a=p.parse_args();folder=DEST/'H2_display_return';folder.mkdir(exist_ok=True)
    worker=folder/'worker.json';s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    source=families()['B'];last=next(i for i,q in enumerate(source) if Path(q['path']).stem=='arcB_0103_N64')
    source=source[:last+1];rows=[]
    def status(stage,**kw):write(worker,dict(status=stage,pid=os.getpid(),completed=len(rows),total=len(source),**kw))
    try:
        for i,item in enumerate(source):
            out=folder/f'{i:03d}.json'
            if out.exists() and read(out).get('status')=='PASS':rows.append(read(out));continue
            release(a.device)
            while float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                  '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<3*1024:
                status('WAITING_GPU_RESOURCE',index=i);time.sleep(30)
            status('CHECKING_PROFILE',index=i,source=item['path'])
            path,check=prepare(Path(item['path']),a.device,max_N=512,check_filter_states=True,harmonic_chunk_size=64)
            while (check['maximum_group_defect_Hz']>1e-3 or max(check['regional_defect_Hz'])>1e-5) and check['N']<512:
                path,check=prepare(path,a.device,max_N=512,min_N=2*check['N'],
                    check_filter_states=True,harmonic_chunk_size=64)
            assert check['status']=='RESOLUTION_CHECKED' and check['filter_state_check']['positive']
            assert check['maximum_group_defect_Hz']<=1e-3 and max(check['regional_defect_Hz'])<=1e-5
            before,after=np.load(item['path']),np.load(path);N=max(len(before['r']),len(after['r']))
            x,y=[resample(z['r']*1000,N,axis=0) for z in [before,after]]
            d,phase=distances(x[:,None,:],y,weights)
            rms=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
            change=float(d[0]/max(rms,1e-12));period=abs(float(after['T'])/float(before['T'])-1)
            assert abs(float(after['J'])-float(before['J']))<1e-12
            assert change<.02 and period<.01,(i,change,period)
            q=dict(index=i,source=item['path'],orbit=str(path),status='PASS',check=check,
                phase_aligned_relative_waveform_change=change,relative_period_change=period,
                branch_match_pass=True)
            write(out,q);rows.append(q)
        write(DEST/'H2_display_return.json',dict(status='PASS',rows=rows,
            continuation_order_source='families B prefix ending at arcB_0103_N64',
            scope='Full-space periodic geometry and positive physical profiles; no interval stability classification.'))
        status('BATCH_FINISHED')
    except Exception as exc:status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
