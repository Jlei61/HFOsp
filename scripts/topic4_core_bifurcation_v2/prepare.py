"""Project the current realized core SNN, preserving masks, moments and delay bins."""
from pathlib import Path
import os,sys,json
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(ROOT/'scripts/topic4_burst_regime'))
import runtime
sys.path.insert(0,str(ROOT/'scripts/topic4_core_burst_onset_v1'))
from run import Drive
OUT=ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s,groups,loading,det,applied,cores=runtime.setup(1.,1.,848101)
    p=s.params;ne=s.n_e;N=ne+s.n_i
    region=np.r_[np.where(cores[0]>=0,cores[0],2),np.where(cores[1]>=0,cores[1]+3,5)]
    names=['Core A E','Core B E','Surround E','Core A I','Core B I','Surround I']
    count=np.bincount(region,minlength=6);D=s.net['max_delay_steps']
    W=np.zeros((D,6,6));Q=np.zeros_like(W)
    tau=np.where(np.arange(6)<3,p.tau_m_E,p.tau_m_I)
    for kind,offset,rise in [('ampa',0,p.tau_r_AMPA),('gaba',ne,p.tau_r_GABA)]:
        for d,mat in enumerate(s.net[kind+'_by_delay']):
            if not mat.nnz:continue
            assert d>=1
            coo=mat.tocoo();i=region[coo.row];j=region[coo.col+offset]
            weight=coo.data*rise/tau[i]
            W[d-1]+=np.bincount(i*6+j,weights=weight,minlength=36).reshape(6,6)/count[:,None]
            Q[d-1]+=np.bincount(i*6+j,weights=weight**2,minlength=36).reshape(6,6)/count[:,None]
    drive=Drive(s,cores[0],848101,0.)
    nu=np.full(6,drive.signal);nu[:2]=drive.matched
    np.savez_compressed(OUT/'projected_graph.npz',W=W,Q=Q,count=count,region=region,
        names=names,vtheta=s.vtheta,positions=np.r_[s.positions_e,s.positions_i],nu=nu,
        stochastic_fraction=np.array([1,1,0,0,0,0]))
    cfg=dict(params={k:float(getattr(p,k)) for k in ('dt','tau_m_E','tau_m_I','tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA','V_reset','V_th','tau_ref_E','tau_ref_I','J_ext_E','J_ext_I')},
        graph_identity=applied['identity'],names=names,counts=count.tolist(),delay_bins=D,
        source='Frozen burst_regime_map_20260914 topology 2511, threshold depth 1; source EE multiplier 1.',
        projection='Mean and squared physical edge weights per receiving neuron, each original 0.1-ms delay bin retained. E core masks are separate from surround.',
        external_input='Core E private Poisson retained in diffusion moments; shared OU removed. Other E and all I retain deterministic external drive.')
    (OUT/'projection.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(status='PROJECTED',counts=count.tolist(),delay_bins=D,W=W.sum(0).tolist())))

if __name__=='__main__':main()
