"""Independent finite-difference check of the full variational GPU update."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,numpy as np,torch
from topic4_fig5_D_physical_model import Orbit,OUT
from topic4_fig5_z_branch_dynamics import initialize
from floquet_topic4_fig5_D_cycles import ReturnMap
from topic4_fig5_z_frequency_parallel import install

def main():
    install(4);a=np.load(OUT/'q1.25_Ha.npz');D=float(a['D']);o=Orbit(D,8,1.25);m=o.m
    initialize(o.eq,a['r_hz'],D);r=np.r_[m.r_u,m.r_i];ret=ReturnMap(o,np.tile(r,(8,1)),200.,0,3)
    re=m.cell_rate_e();ri=m.r_i
    for key in ['ee','ei','ie','ii']:
        ex=key[-1]=='e';aa,bb=(m.arA,m.adA) if ex else (m.arG,m.adG);source=re if ex else ri
        val=getattr(m,'v_'+key)@source
        if key=='ee':val=val+m.je*m.je*m.nu_sig
        if key=='ie':val=val+m.ji*m.ji*m.nu_sig
        for j,co in enumerate([aa*aa,bb*bb,aa*bb]):m.y[key][j]=co/(1-co)*val
    state=m.state_dict();rng=np.random.default_rng(908);delta=rng.normal(size=ret.dimension)
    delta[ret.cuts[0]:ret.cuts[2]]*=.001
    delta[ret.cuts[6]:]*=.001
    parts=[delta[aa:bb].reshape((1,)+s) for aa,bb,s in zip(ret.cuts[:-1],ret.cuts[1:],ret.shapes)]
    rr,ii,mm,g,c,y,he,hi=[ret.tensor(x) for x in parts]
    products=torch.sparse.mm(ret.matrix,torch.cat([he.flatten(),hi.flatten()])[:,None]).reshape(4,2,m.n,1)
    nr,ni,nm,ng,nc,ny=ret._linear_update(rr,ii,mm,g,c,y,products,ret.ge[0],ret.gi[0])
    hre=torch.cat([((rr*ret.w).reshape(1,m.n,m.K).sum(2))[:,None],he[:,:-1]],dim=1);hri=torch.cat([ii[:,None],hi[:,:-1]],dim=1)
    expected=torch.cat([x.flatten() for x in [nr,ni,nm,ng,nc,ny,hre,hri]]).cpu().numpy()
    def flat():return np.concatenate([x.ravel() for x in [m.r_u,m.r_i,m.m_u,np.array([m.gAE,m.gGE,m.gAI,m.gGI]),np.array([m.cAE,m.cGE,m.cAI,m.cGI]),np.array([m.y[k] for k in ['ee','ei','ie','ii']]),m.hE,m.hI]])
    base=flat()
    def setflat(x):
        parts=[x[aa:bb].reshape(s).copy() for aa,bb,s in zip(ret.cuts[:-1],ret.cuts[1:],ret.shapes)]
        m.r_u,m.r_i,m.m_u,g,c,y,m.hE,m.hI=parts
        m.gAE,m.gGE,m.gAI,m.gGI=g;m.cAE,m.cGE,m.cAI,m.cGI=c
        for k,yy in zip(['ee','ei','ie','ii'],y):m.y[k]=yy
    rows=[]
    for eps in [1e-4,5e-5,1e-5]:
        ans=[]
        for sign in [-1,1]:
            setflat(base+sign*eps*delta);m.step(np.full(m.n,m.nu_sig),m.nu_sig);ans.append(flat())
        fd=(ans[1]-ans[0])/(2*eps);rows.append(dict(epsilon=eps,max_abs_error=float(max(abs(fd-expected))),relative_norm_error=float(np.linalg.norm(fd-expected)/np.linalg.norm(expected))))
    result=dict(checks=rows,scope='CPU smooth nonlinear map vs GPU analytic full-state tangent; M/filter/delay histories included',pass_check=all(r['relative_norm_error']<1e-7 for r in rows))
    (OUT/'variational_step_qa.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':main()
