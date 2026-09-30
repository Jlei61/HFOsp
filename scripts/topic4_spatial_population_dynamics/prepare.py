"""Build delayed block-average graph, preserving the original jump units."""
from shared import *
from scipy import sparse
import time,argparse

def main(partition='adaptive'):
    suffix='' if partition=='adaptive' else '_'+partition
    t=time.time();s,groups,loading,det,applied,cores=runtime.setup(J,1.,848101)
    assert applied['identity']==read(PRIOR/'model_config.json')['identity']
    group,region,pos,tiles=make_partition(partition=='adaptive1');P=group.max()+1;count=np.bincount(group);D=s.net['max_delay_steps']+1
    order=np.argsort(group,kind='stable');ptr=np.r_[0,np.cumsum(count)]
    contacts=[]
    for xy in s.montage.contacts:
        w=np.exp(-np.sum((s.positions_e-xy)**2,axis=1)/(2*.25**2));w/=w.sum();contacts.append(np.r_[w,np.zeros(s.n_i)])
    weights=np.array(contacts).T;population_weights=np.stack([np.bincount(group,weights=weights[:,j],minlength=P) for j in range(15)])
    core_ids=np.flatnonzero(np.isin(region,[0,1]));ext_index=np.full(len(group),-1,np.int32);ext_index[core_ids]=np.arange(len(core_ids))
    sample=np.concatenate([np.flatnonzero(region==k)[np.linspace(0,np.sum(region==k)-1,min(120,np.sum(region==k)),dtype=int)] for k in range(6)])
    inv=np.empty(len(group),int);inv[order]=np.arange(len(group));raster_index=np.full(len(group),-1,np.int32);raster_index[inv[sample]]=np.arange(len(sample))
    xy=np.minimum(pos.astype(int),19);field=xy[:,1]*20+xy[:,0]
    np.savez_compressed(OUT/f'model{suffix}.npz',group=group,region=region,positions=pos,tiles=tiles,count=count,order=order,ptr=ptr,
        vtheta=s.vtheta[order],region_sorted=region[order],weights_sorted=weights[order],field_sorted=field[order],
        ext_index=ext_index[order],core_ids=core_ids,core_regions=region[core_ids],raster_index=raster_index,
        raster_neuron_ids=sample,raster_regions=region[sample],contact_weights=population_weights,
        contact_xy=s.contact_xy,contact_names=s.contact_names,ne=s.n_e)
    checks=[]
    for kind,off in [('ampa',0),('gaba',s.n_e)]:
        rows=[];cols=[];vals=[];squares=[]
        for d,m in enumerate(s.net[kind+'_by_delay']):
            if not m.nnz:continue
            co=m.tocoo();a=group[co.row];b=group[co.col+off]
            rows.append(b);cols.append(d*P+a);vals.append(co.data/(count[a]*count[b]));squares.append(co.data**2/(count[a]*count[b]))
        row=np.concatenate(rows);col=np.concatenate(cols)
        for moment,data in [('mean',vals),('second',squares)]:
            mat=sparse.coo_matrix((np.concatenate(data),(row,col)),shape=(P,D*P)).tocsr();mat.sum_duplicates();mat.sort_indices()
            sparse.save_npz(OUT/f'{kind}_{moment}{suffix}.npz',mat)
            c=mat.tocoo();recovered=float(np.sum(c.data*count[c.row]*count[c.col%P]))
            expected=float(sum(np.sum(m.data if moment=='mean' else m.data**2) for m in s.net[kind+'_by_delay']))
            assert np.isclose(recovered,expected,rtol=2e-12)
            checks.append(dict(kind=kind,moment=moment,nnz=mat.nnz,relative_error=abs(recovered-expected)/expected))
        print(kind,'prepared',time.time()-t,flush=True)
    write(OUT/f'prepared{suffix}.json',dict(status='COMPLETE',J=J,groups=P,delay_slots=D,checks=checks,seconds=time.time()-t,
        state='all original neuron thresholds and dynamical particles retained; connectivity replaced by spatial population blocks',
        projection='M_ab(d)=sum(A_ij)/(N_a*N_b); inputs use actual source population counts per native 0.1 ms step'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['adaptive','adaptive1'],default='adaptive');a=p.parse_args();main(a.partition)
