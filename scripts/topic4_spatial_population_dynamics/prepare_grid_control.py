"""Same 1-mm spatial projection as the failed rate: state-representation control."""
from shared import *
from scipy import sparse

def main():
    base=np.load(OUT/'model.npz');old=np.load(PRIOR/'grid20/model.npz');group=old['group'];P=len(old['count']);count=old['count']
    region=base['region'];order=np.argsort(group,kind='stable');ptr=np.r_[0,np.cumsum(count)]
    undo=np.argsort(base['order']);th=base['vtheta'][undo];weights=base['weights_sorted'][undo]
    ext=base['ext_index'][undo];field=base['field_sorted'][undo]
    raster_old=base['raster_neuron_ids'];inverse=np.empty(len(order),int);inverse[order]=np.arange(len(order))
    raster_index=np.full(len(order),-1,np.int32);raster_index[inverse[raster_old]]=np.arange(len(raster_old))
    data={k:base[k] for k in base.files}
    data.update(group=group,count=count,order=order,ptr=ptr,vtheta=th[order],region_sorted=region[order],
        weights_sorted=weights[order],ext_index=ext[order],field_sorted=field[order],raster_index=raster_index,
        contact_weights=old['contact_weights'],tiles=np.array([[x,y,1.] for y in range(20) for x in range(20)]))
    np.savez_compressed(OUT/'model_grid20.npz',**data)
    D=read(PRIOR/'grid20/prepared.json')['delay_bins']+1;checks=[]
    for kind in ('ampa','gaba'):
        m=sparse.load_npz(PRIOR/f'grid20/{kind}_delay.npz').tocoo()
        source=m.col%P;target=m.row;delay=m.col//P+1
        new=sparse.coo_matrix((m.data/count[source],(source,delay*P+target)),shape=(P,D*P)).tocsr()
        sparse.save_npz(OUT/f'{kind}_mean_grid20.npz',new)
        checks.append(dict(kind=kind,nnz=new.nnz))
    write(OUT/'prepared_grid20.json',dict(status='COMPLETE',groups=P,delay_slots=D,checks=checks,
        control='Exactly the failed grid20 rate mean-connectivity operator, original empirical thresholds and actual cell states'))

if __name__=='__main__':main()
