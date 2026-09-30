"""Project sources only, with every target and physical delay retained."""
from shared_source import *
from scipy import sparse
import time

def project(matrices,source_groups,source_counts,target_inverse):
    N=len(target_inverse);P=len(source_counts);D=len(matrices)
    rows=[];cols=[];values=[];checks=[];original_nnz=0
    for d,mat in enumerate(matrices):
        if not mat.nnz:continue
        assert d>0,'Kernel assumes strictly positive native delays'
        c=mat.tocoo();b=source_groups[c.col];target=target_inverse[c.row]
        part=sparse.coo_matrix((c.data/source_counts[b],(b,target)),shape=(P,N)).tocsr()
        part.sum_duplicates();part.sort_indices();p=part.tocoo()
        recovered=np.bincount(p.col,weights=p.data*source_counts[p.row],minlength=N)
        expected=np.empty(N);expected[target_inverse]=np.asarray(mat.sum(1)).ravel()
        error=float(np.max(abs(recovered-expected)));assert np.allclose(recovered,expected,rtol=2e-12,atol=1e-12)
        rows.append(p.row.astype(np.int32));cols.append((d*N+p.col).astype(np.int32));values.append(p.data)
        checks.append(dict(delay_steps=d,target_total_max_abs_error=error));original_nnz+=mat.nnz
    result=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(P,D*N)).tocsr()
    result.sort_indices()
    return result,dict(original_nnz=original_nnz,projected_nnz=result.nnz,per_delay=checks)

def main(partition='adaptive1'):
    t=time.time();OUT.mkdir(exist_ok=True)
    if partition=='half':make_half_model()
    folder=operator_directory(partition)
    if (folder/'prepared.json').exists():return
    s,*rest=runtime.setup(J,1.,848101);applied=rest[-2]
    assert applied['identity']==read(PRIOR/'model_config.json')['identity']
    z=model(partition);N=len(z['order']);P=len(z['count']);D=s.net['max_delay_steps']+1
    inverse=np.empty(N,np.int64);inverse[z['order']]=np.arange(N);target_group=z['group'][z['order']]
    allchecks=[]
    for kind,offset,number in [('ampa',0,s.n_e),('gaba',s.n_e,s.n_i)]:
        mat,checks=project(s.net[kind+'_by_delay'],z['group'][offset:offset+number],z['count'],inverse)
        # Averaging these target-specific profiles recovers the old block mean exactly.
        c=mat.tocoo()
        if partition=='adaptive1':
            a=target_group[c.col%N]
            coarse=sparse.coo_matrix((c.data/z['count'][a],(c.row,(c.col//N)*P+a)),shape=(P,D*P)).tocsr()
            old=sparse.load_npz(BASE/f'{kind}_mean_adaptive1.npz')
        else:
            parent=model();oldinverse=np.empty(N,int);oldinverse[parent['order']]=np.arange(N)
            source=z['parent_group'][c.row];dst=oldinverse[z['order'][c.col%N]]
            coarse=sparse.coo_matrix((c.data*z['count'][c.row]/parent['count'][source],(source,(c.col//N)*N+dst)),shape=(len(parent['count']),D*N)).tocsr()
            op=np.load(OUT/f'{kind}_source_operator.npz')
            old=sparse.csr_matrix((op['weight'],op['delay'].astype(np.int64)*N+op['target'],op['ptr']),shape=coarse.shape)
        delta=coarse-old
        error=float(abs(delta.data).max()) if delta.nnz else 0.;assert error<1e-11
        checks['parent_operator_max_abs_error']=error;checks['kind']=kind;allchecks.append(checks)
        np.savez_compressed(folder/f'{kind}_source_operator.npz',ptr=mat.indptr.astype(np.int64),
            target=(mat.indices%N).astype(np.int32),delay=(mat.indices//N).astype(np.int16),weight=mat.data)
        print(kind,'nnz',mat.nnz,'seconds',time.time()-t,flush=True)
    write(folder/'prepared.json',dict(status='COMPLETE',groups=P,cells=N,delay_slots=D,J=J,seconds=time.time()-t,checks=allchecks,partition=partition,
        projection='sum of actual incoming jumps for each individual target, each source population and each physical delay, divided by source population size',
        identity=applied['identity'],parent_model=str(BASE/'model_adaptive1.npz'),fitted_parameters=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['adaptive1','half'],default='adaptive1');main(p.parse_args().partition)
