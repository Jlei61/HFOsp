"""Project the same Fig.5 graph to rate populations and retain source lineage."""
from common import *
from scipy import sparse
import time


def main():
    folder=OUT/'operators';folder.mkdir(exist_ok=True)
    geo=dict(np.load(GRID/'geometry.npz'));prep=read(GRID/'prepared.json')
    cell=[];theta=[];counts=[];regions=[];membership=np.full(32000,-1,dtype=np.int32)
    for c in range(400):
        ids=np.flatnonzero(geo['cell_e']==c);ids=ids[np.argsort(geo['vtheta_e'][ids],kind='stable')]
        chunks=[x for x in np.array_split(ids,8) if len(x)]
        merged=[]
        for ids in chunks:
            mean=float(geo['vtheta_e'][ids].mean())
            if merged and abs(mean-float(geo['vtheta_e'][merged[-1]].mean()))<1e-12:
                merged[-1]=np.r_[merged[-1],ids]
            else:merged.append(ids)
        for ids in merged:
            membership[ids]=len(cell);cell.append(c);theta.append(float(geo['vtheta_e'][ids].mean()))
            counts.append(len(ids));regions.append(np.bincount(geo['g175'][ids],minlength=3))
    nE=len(cell)
    for c in range(400):
        ids=np.flatnonzero(geo['cell_i']==c)
        if len(ids):
            cell.append(c+400);theta.append(18.);counts.append(len(ids));regions.append(np.zeros(3))
    cell=np.array(cell,dtype='int32');theta=np.array(theta);counts=np.array(counts);P=len(cell)
    total_counts=np.r_[geo['count_e'],geo['count_i']]
    weight=counts/total_counts[cell]
    assert np.all(membership>=0) and counts.sum()==40000
    assert np.allclose(np.bincount(cell,weights=weight,minlength=800),1.)
    operators=[]
    for paths in [('ee','ie'),('ei','ii')]:
        rows=[];cols=[];vals=[]
        for target_population,key in enumerate(paths):
            op=sparse.load_npz(GRID/f'delay_{key}.npz').tocoo()
            rows.append(op.row+400*target_population)
            cols.append((op.col//400+1)*800+op.col%400+(400 if key.endswith('i') else 0))
            vals.append(op.data)
        joined=sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(800,(prep['max_delay_steps']+1)*800)).tocsr()
        operators.append(joined)
    np.savez_compressed(folder/'groups.npz',cell=cell,theta=theta,count=counts,weight=weight,
        population=np.r_[np.zeros(nE,dtype=int),np.ones(P-nE,dtype=int)],nE=nE,
        region_counts=np.array(regions),membership_e=membership,centers_mm=geo['centers_mm'],
        cell_count_e=geo['count_e'],cell_count_i=geo['count_i'])
    for name,op in zip(['ampa','gaba'],operators):sparse.save_npz(folder/f'{name}_delay.npz',op)
    write(folder/'definition.json',dict(source=str(GRID),graph_identity=prep['graph_identity'],
        spatial_grid=20,bin_mm=1.,E_groups=nE,I_groups=P-nE,rate_states=2*P,
        synaptic_states=4*800,slow_states=2*nE,
        total_continuous_states=2*P+4*800+2*nE,
        delays_ms=[.1,prep['max_delay_steps']*.1],delay_history_scope='Numerical DDE history, not individual-neuron or voltage-density coordinates',
        threshold_closure='Eight equal-count empirical threshold chunks per spatial E cell; identical threshold chunks merged',
        only_population_rate_sources=True,mean_operator_source='Same realized topology6101 projected to original1mm grid'))
    write(folder/'params.json',prep)
    print('groups',nE,P-nE,'continuous states',2*P+4*800+2*nE,flush=True)


if __name__=='__main__':main()
