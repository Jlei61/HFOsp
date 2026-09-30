"""Local density poles on the fixed-noise-marginal tangent space.

Linearize the actual limited finite-volume transport at a stationary PDF.
The limiter active set is held fixed; finite-difference checks and mesh checks
remain necessary at switching surfaces. This tests selected local groups,
not a substitute for the closed-network spectrum.
"""
from stationary_response import *
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs


def voltage_jacobian(moved,edges,nodes,ratio,decay,drive,nr):
    K,width=moved.shape;nv=len(edges)-1
    c=(edges[:-1]+edges[1:])/2;w=np.diff(edges);rows=[];cols=[];vals=[]
    def add(s,target,source,value):
        for j,v in zip(source,value):
            if v!=0:rows.append(s*width+target);cols.append(s*width+j);vals.append(v)
    for s in range(K):
        p=moved[s];current=nodes[s]*ratio+drive
        for j in range(nv+1):
            v=c[j] if j<nv else 11.;dest=decay*v+(1-decay)*current
            if j==nv:
                if dest>=edges[-1]:add(s,nv+nr-1,[j],[1.])
                elif dest<=c[0]:add(s,0,[j],[1.])
                elif dest>=c[-1]:add(s,nv-1,[j],[1.])
                else:
                    lo=np.searchsorted(c,dest,side='right')-1;f=(dest-c[lo])/(c[lo+1]-c[lo])
                    add(s,lo,[j],[1-f]);add(s,lo+1,[j],[f])
                continue
            slope={}
            if 0<j<nv-1:
                dl=(p[j]/w[j]-p[j-1]/w[j-1])/(c[j]-c[j-1])
                dr=(p[j+1]/w[j+1]-p[j]/w[j])/(c[j+1]-c[j])
                if dl*dr>0:
                    if abs(dl)<=abs(dr):slope={j:1/(w[j]*(c[j]-c[j-1])),j-1:-1/(w[j-1]*(c[j]-c[j-1]))}
                    else:slope={j+1:1/(w[j+1]*(c[j+1]-c[j])),j:-1/(w[j]*(c[j+1]-c[j]))}
            left=dest-.5*decay*w[j];right=dest+.5*decay*w[j]
            if left>=edges[-1]:add(s,nv+nr-1,[j],[1.]);continue
            at=max(0,min(nv-1,np.searchsorted(edges,left,side='right')-1))
            prev=-.5*w[j];used={}
            while at<nv and edges[at]<right:
                b=max(prev,min(.5*w[j],(edges[at+1]-dest)/decay))
                coef={i:.5*(b*b-prev*prev)*val for i,val in slope.items()}
                coef[j]=coef.get(j,0.)+(b-prev)/w[j]
                add(s,at,list(coef),list(coef.values()))
                for i,val in coef.items():used[i]=used.get(i,0.)+val
                prev=b
                if b>=.5*w[j]:break
                at+=1
            remainder={i:-val for i,val in used.items()};remainder[j]=remainder.get(j,0.)+1.
            target=nv+nr-1 if right>=edges[-1] else min(at,nv-1)
            add(s,target,list(remainder),list(remainder.values()))
        for j in range(1,nr):add(s,nv+j-1,[nv+j],[1.])
    return sparse.coo_matrix((vals,(rows,cols)),shape=(K*width,K*width)).tocsr()


def run(args):
    source=Path(args.equilibrium);folder=source/'local_poles'
    folder.mkdir(parents=True,exist_ok=False)
    config=read(source/'config.json');geo=dict(np.load(OPERATORS/'geometry.npz'))
    with np.load(source/'stationary_local_state.npz') as z:state={k:z[k] for k in z.files}
    response=source/'response_eps0.001_200ms/susceptibility.npz'
    with np.load(response) as z:response_size=np.sum(abs(z['kernel_hz_per_mv']),axis=0)
    groups=[]
    for pop in (0,1):
        for region in (0,1,2):
            ids=np.flatnonzero((geo['population']==pop)&(geo['group_region']==region))
            if not len(ids):continue
            groups.extend(ids[np.argsort(response_size[ids])[-2:]])
            groups.extend([ids[np.argmin(state['current_mv'][ids])],ids[np.argmax(state['current_mv'][ids])]])
    groups=np.unique(groups).astype(int)
    m=LocalStationaryDensity(state['theta'][groups],state['population'][groups],state['current_mv'][groups],
        config['degree'],config['voltage_dv'],args.device,basis_mode=config.get('basis_mode','legacy'))
    m.F[:]=cp.asarray(state['F'][groups]);m.map();baseline=cp.asnumpy(m.Q)
    A,nodes,ratio,decay=[cp.asnumpy(x) for x in (m.A,m.nodes,m.ratio,m.decay)]
    F=state['F'][groups];rng=np.random.default_rng(9201);directions=[];jacobians=[];rows=[]
    for at,g in enumerate(groups):
        nr=int(cp.asnumpy(m.refs)[at]);width=m.nv+nr
        f=F[at,:,:width];moved=A@f
        V=voltage_jacobian(moved,m.edges[at],nodes,ratio[at],decay[at],state['current_mv'][g],nr)
        mass_error=float(np.max(abs(np.asarray(V.sum(0)).ravel()-1)))
        def project(x):
            x=x.reshape(m.K,width);return x-x.mean(1)[:,None]
        def mv(x):
            v=project(x);y=(V@(A@v).ravel()).reshape(m.K,width)
            return project(y).ravel()
        op=LinearOperator((m.K*width,m.K*width),matvec=mv,dtype=float)
        values,vectors=eigs(op,k=8,which='LM',tol=2e-8,maxiter=8000,ncv=40,
            v0=project(rng.normal(size=(m.K,width))).ravel())
        residual=max(np.linalg.norm(mv(vectors[:,j])-values[j]*vectors[:,j])/np.linalg.norm(vectors[:,j]) for j in range(len(values)))
        direction=f*rng.normal(size=f.shape)
        shape=abs(f)/np.maximum(abs(f).sum(1)[:,None],1e-300)
        direction-=direction.sum(1)[:,None]*shape
        full=np.zeros_like(F[at]);full[:,:width]=direction;directions.append(full)
        jac=np.zeros_like(F[at]);jac[:,:width]=(V@(A@direction).ravel()).reshape(m.K,width);jacobians.append(jac)
        row=dict(group=int(g),population=int(state['population'][g]),threshold=float(state['theta'][g]),
            current_mv=float(state['current_mv'][g]),maximum_local_multiplier_modulus=float(max(abs(values))),
            eigenvalues=np.c_[values.real,values.imag],eigenpair_residual=float(residual),
            mass_conservation_error=mass_error)
        rows.append(row);print('local poles',g,row['maximum_local_multiplier_modulus'],flush=True)
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),latest=row))
    direction=cp.asarray(np.asarray(directions));jacobian=np.asarray(jacobians);checks=[]
    for eps in (1e-4,1e-5,1e-6):
        m.F[:]=cp.asarray(F)+eps*direction;m.map();plus=cp.asnumpy(m.Q)
        m.F[:]=cp.asarray(F)-eps*direction;m.map();minus=cp.asnumpy(m.Q)
        fd=(plus-minus)/(2*eps)
        relative=np.sum(abs(fd-jacobian),axis=(1,2))/np.maximum(np.sum(abs(jacobian),axis=(1,2)),1e-30)
        checks.append(dict(epsilon=eps,max_relative_L1_error=float(relative.max()),per_group=relative))
    report=dict(status='SELECTED_LOCAL_POLE_AUDIT_COMPLETE',D=config['D'],groups=rows,
        selected_groups=len(groups),all_groups=len(geo['population']),finite_difference=checks,
        dynamic_M_small_gain_bound=float(.0005*response_size[geo['population']==0].max()),
        scope='Selected worst-susceptibility/extreme-drive populations; fixed external marginal constraints remove mass modes',
        limitation='Limiter active-set Jacobian; full network and numerical mesh qualification remain separate')
    write(folder/'audit.json',report);print(folder,checks,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--equilibrium',type=Path,required=True)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
