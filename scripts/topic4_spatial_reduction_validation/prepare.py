"""Actual graph -> space x core identity x E/I; no template/trajectory fitting."""
from common import *
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
import argparse,time

def quadrature(v,n=8):
    v=np.asarray(v);unique=np.unique(v)
    if len(unique)==1:return np.full(n,v[0]),np.r_[1.,np.zeros(n-1)]
    cen=v.mean();width=v.std();x=(v-cen)/width;q=np.ones(len(v))/np.sqrt(len(v))
    prev=np.zeros(len(v));beta=0.;basis=[];aa=[];bb=[]
    for k in range(min(n,len(unique))):
        basis.append(q.copy());z=x*q-beta*prev;alpha=q@z;z-=alpha*q
        for _ in range(2):
            for b in basis:z-=b*(b@z)
        bn=np.linalg.norm(z);aa.append(alpha)
        if k==n-1 or bn<1e-13:break
        bb.append(bn);prev=q;q=z/bn;beta=bn
    nodes,vec=eigh_tridiagonal(np.array(aa),np.array(bb[:len(aa)-1]))
    return np.r_[cen+width*nodes,np.full(n-len(nodes),v[-1])],np.r_[vec[0]**2,np.zeros(n-len(nodes))]

def main():
    start=time.time();s,groups,loading,det,applied,cores=runtime.setup(J,1.,848101)
    assert applied['identity']==read(V10/'native/a/applied_physics.json')['identity']
    region=np.r_[np.where(cores[0]>=0,cores[0],2),np.where(cores[1]>=0,cores[1]+3,5)]
    positions=np.r_[s.positions_e,s.positions_i];p=s.params
    keys=['dt','tau_m_E','tau_m_I','tau_ref_E','tau_ref_I','tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA','V_reset','J_ext_E','J_ext_I','sigma_n','tau_n','nu_ext_ratio']
    cfg=dict(params={k:float(getattr(p,k)) for k in keys},signal_per_ms=float(p.nu_ext_ratio*runtime.compute_nu_theta(p)[0]),
        identity=applied['identity'],core_centers=applied['candidate']['centers_mm'],core_radii=applied['candidate']['radii_mm'],
        J=J,topology=2511,rate_time_constants_ms=[5.,2.5],rate_response_calibrated_on_current_network=False)
    write(OUT/'model_config.json',cfg)
    for grid in (10,20):
        folder=OUT/f'grid{grid}';folder.mkdir(exist_ok=True)
        group,reg,cell=partition(positions,region,grid);P=len(reg);count=np.bincount(group)
        assert np.array_equal(np.bincount(reg,weights=count,minlength=6),np.bincount(region,minlength=6))
        tm=np.where(reg<3,p.tau_m_E,p.tau_m_I);D=s.net['max_delay_steps']
        threshold,tw=zip(*(quadrature(s.vtheta[group==i]) for i in range(P)))
        contact=[]
        for xy in s.montage.contacts:
            d=np.linalg.norm(s.positions_e-xy,axis=1);w=np.exp(-d*d/(2*.25**2));w/=w.sum()
            contact.append(np.bincount(group[:s.n_e],weights=w,minlength=P))
        ee=(reg[:,None]<2)&(reg[None,:]<2)&(reg[:,None]==reg[None,:])
        projcheck={}
        for kind,off,rise in [('ampa',0,p.tau_r_AMPA),('gaba',s.n_e,p.tau_r_GABA)]:
            rows=[];cols=[];ww=[];qq=[]
            for d,mat in enumerate(s.net[kind+'_by_delay']):
                if not mat.nnz:continue
                coo=mat.tocoo();a=group[coo.row];b=group[coo.col+off]
                rows.append(a);cols.append((d-1)*P+b);ww.append(coo.data/count[a])
                qq.append((coo.data*rise/tm[a])**2/count[a])
            rows=np.concatenate(rows);cols=np.concatenate(cols);w=np.concatenate(ww);q=np.concatenate(qq)
            W=sparse.coo_matrix((w,(rows,cols)),shape=(P,D*P)).tocsr();W.sum_duplicates();W.sort_indices()
            Q=sparse.coo_matrix((q,(rows,cols%P)),shape=(P,P)).tocsr();Q.sum_duplicates();Q.sort_indices()
            sparse.save_npz(folder/f'{kind}_delay.npz',W);sparse.save_npz(folder/f'{kind}_variance.npz',Q)
            assert np.all(W.data>0) and np.all(Q.data>0)
            reconstructed=float(np.dot(count,np.asarray(W.sum(1)).ravel()))
            original=float(sum(m.data.sum() for m in s.net[kind+'_by_delay']))
            assert np.isclose(reconstructed,original,rtol=1e-12)
            projcheck[kind]=dict(delay_nnz=W.nnz,variance_nnz=Q.nnz,weighted_edge_sum_relative_error=abs(reconstructed-original)/original)
        np.savez_compressed(folder/'model.npz',group=group,region=reg,cell=cell,count=count,
            threshold=np.array(threshold),threshold_weight=np.array(tw),contact_weights=np.array(contact),
            positions=positions,contact_xy=s.contact_xy,contact_names=s.contact_names,ne=s.n_e)
        write(folder/'prepared.json',dict(status='COMPLETE',grid=grid,group_count=P,delay_bins=D,
            threshold_nodes=8,split='space x core identity x E/I',checks=projcheck,
            formulas='mean: exact native gating/current expectation; recurrent variance: instantaneous independent-spike moments',seconds=time.time()-start))
        print(grid,P,projcheck,flush=True)

if __name__=='__main__':main()
