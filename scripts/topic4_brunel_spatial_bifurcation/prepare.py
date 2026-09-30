"""Mean and shot-noise variance operators from the same realized SNN graph.

The second moment is formed BEFORE spatial aggregation. Squaring already
aggregated weights would change the diffusion approximation.
"""
from common import *
import argparse

def main(grids):
    sys.path.insert(0,str(ROOT/'scripts/topic4_fig5_z_state'))
    # The legacy common module name collides with this directory's common.
    import importlib.util
    spec=importlib.util.spec_from_file_location('brunel_native_source',ROOT/'scripts/topic4_fig5_z_state/common.py')
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    sim,_,_,identity=source.old.setup(9108401)
    for grid in grids:
        dest=OUT/f'operators/g{grid}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'prepared.json').exists(): continue
        geo=dict(np.load(PROJECTED/f'g{grid}/geometry.npz'));prep=read(PROJECTED/f'g{grid}/prepared.json')
        assert identity==prep['graph_identity']
        group=geo['cell_group'];size=geo['group_size'];P=len(size);records=[]
        for name in ('ampa','gaba'):
            src=group[:32000] if name=='ampa' else group[32000:]
            rise=sim.params.tau_r_AMPA if name=='ampa' else sim.params.tau_r_GABA
            rr=[];cc=[];vv=[];qq=[];maximum_error=0.
            for delay,mat in enumerate(sim.net[name+'_by_delay']):
                if not mat.nnz:continue
                c=mat.tocoo();physical=c.data/(np.where(c.row<32000,20.,10.)/rise)
                rr.append(group[c.row]);cc.append(src[c.col]+(delay-1)*P)
                vv.append(physical/size[group[c.row]]);qq.append(physical**2/size[group[c.row]])
            rows=np.concatenate(rr);cols=np.concatenate(cc)
            for label,values in [('mean',vv),('variance',qq)]:
                a=sparse.coo_matrix((np.concatenate(values),(rows,cols)),shape=(P,P*prep['max_delay_steps'])).tocsr()
                if label=='mean':
                    old=sparse.load_npz(PROJECTED/f'g{grid}/delay_{name}.npz');difference=a-old
                    maximum_error=float(abs(difference.data).max()) if difference.nnz else 0.
                    assert maximum_error<1e-10
                sparse.save_npz(dest/f'{label}_{name}.npz',a)
            records.append(dict(pathway=name,mean_operator_error=maximum_error))
            print('projected',grid,name,flush=True)
        np.savez_compressed(dest/'geometry.npz',**geo)
        write(dest/'prepared.json',dict(params=prep['params'],graph_identity=identity,grid=grid,
            groups=P,max_delay_steps=prep['max_delay_steps'],nu_ext_per_ms=prep['nu_ext_per_ms'],checks=records,
            source=str(PROJECTED/f'g{grid}'),variance_definition='mean over target cells of sum of INDIVIDUAL squared physical weights'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--grids',type=int,nargs='+',default=[20,40]);main(ap.parse_args().grids)
