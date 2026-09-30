"""Finer spatial-delay operators from the unchanged realized Fig.5 graph."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['TOPIC4_MANUAL_ARM']='manual_hard'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/topic4_fig5_z_state'))
from common import old
OUT=ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916'
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'


def main(grid=40):
    start=time.time();dest=OUT/f'coarse_{grid}';dest.mkdir(exist_ok=False)
    assert (ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1/substrate.json').exists()
    s,tr,frozen,identity=old.setup(9108401)
    ref=json.loads((SOURCE/'approx/coarse_20/prepared.json').read_text())
    assert identity==ref['graph_identity']
    n=grid**2;ne=s.n_e
    def cell(pos):
        ij=np.clip(np.floor(pos/20*grid).astype(int),0,grid-1)
        return ij[:,1]*grid+ij[:,0]
    ce,ci=cell(s.positions_e),cell(s.positions_i);allcells=np.r_[ce,ci]
    counts=[np.bincount(ce,minlength=n),np.bincount(ci,minlength=n)]
    geo=dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
    geo.update(cell_e=ce,cell_i=ci,count_e=counts[0],count_i=counts[1])
    np.savez_compressed(dest/'geometry.npz',**geo)
    md=dict(np.load(SOURCE/'approx/coarse_20/model.npz'))
    md={k:v for k,v in md.items() if np.ndim(v)==0};md.update(n_grid=grid,count_e=counts[0],count_i=counts[1])
    np.savez_compressed(dest/'model.npz',**md)
    for kind,src in [('ampa',ce),('gaba',ci)]:
        blocks=[[],[]];squares=[[],[]]
        for d,mat in enumerate(s.net[kind+'_by_delay'][1:],1):
            co=mat.tocoo()
            for pop in (0,1):
                take=(co.row<ne) if pop==0 else (co.row>=ne)
                rows=allcells[co.row[take]];cols=src[co.col[take]]
                tau=s.params.tau_m_E if pop==0 else s.params.tau_m_I
                rise=s.params.tau_r_AMPA if kind=='ampa' else s.params.tau_r_GABA
                physical=co.data[take]*rise/tau
                denom=counts[pop][rows]
                assert np.all(denom>0)
                blocks[pop].append(sparse.coo_matrix((physical/denom,(rows,cols)),shape=(n,n)).tocsr())
                squares[pop].append(sparse.coo_matrix((physical**2/denom,(rows,cols)),shape=(n,n)).tocsr())
        for pop,key in enumerate(('ee','ie') if kind=='ampa' else ('ei','ii')):
            w=sparse.hstack(blocks[pop],format='csr');q=sparse.hstack(squares[pop],format='csr')
            sparse.save_npz(dest/f'delay_{key}.npz',w);sparse.save_npz(dest/f'vdelay_{key}.npz',q)
            # Neuron-weighted incoming totals must agree with the coarser projection.
            ow=sparse.load_npz(SOURCE/f'approx/coarse_20/delay_{key}.npz')
            oq=sparse.load_npz(SOURCE/f'approx/coarse_20/vdelay_{key}.npz')
            oc=np.load(SOURCE/'approx/coarse_20/model.npz')['count_e' if pop==0 else 'count_i']
            totals=[float(counts[pop]@np.asarray(x.sum(1)).ravel()) for x in (w,q)]
            oldtot=[float(oc@np.asarray(x.sum(1)).ravel()) for x in (ow,oq)]
            assert np.allclose(totals,oldtot,rtol=1e-10)
            print('built',grid,key,w.nnz,'seconds',round(time.time()-start,1),flush=True)
    ref.update(grid=grid,threshold_groups='empirical particles',empty_E_cells=int((counts[0]==0).sum()),empty_I_cells=int((counts[1]==0).sum()),
               seconds=time.time()-start,scope='Unchanged native graph, 0.5mm spatial bins; empty population bins carry zero outgoing activity.')
    (dest/'prepared.json').write_text(json.dumps(ref,indent=2)+'\n')


if __name__=='__main__':main()
