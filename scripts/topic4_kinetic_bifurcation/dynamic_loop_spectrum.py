"""Frequency-domain loop spectrum of the density, synapse, M and delay model.

Rates are coarse spatial-cell averages; threshold subgroups are eliminated
exactly at the linear-response level, including their own dynamic M feedback.
This first scan is not a certified stability count: frequency/impulse/epsilon
refinement and local-pole checks remain explicit requirements.
"""
from autonomous_density import *
from scipy import sparse
from scipy.sparse.linalg import eigs


def aggregate_complex(index, values, n):
    return np.bincount(index,weights=values.real,minlength=n)+1j*np.bincount(index,weights=values.imag,minlength=n)


class DynamicLoop:
    def __init__(self, response_folder):
        self.response_folder=Path(response_folder);self.source=self.response_folder.parent
        status=read(self.response_folder/'status.json');qa=status.get('qa',{})
        assert qa.get('max_relative_dc_error',float('inf'))<.01, 'Unqualified local susceptibility DC response'
        assert qa.get('max_tail_fraction_of_absolute_response',float('inf'))<.01, 'Unresolved local susceptibility tail'
        assert qa.get('usable_for_dynamic_spectrum',True), 'Local response audit failed'
        self.config=read(self.source/'config.json')
        with np.load(self.response_folder/'susceptibility.npz') as z:
            self.h=z['kernel_hz_per_mv'];self.t=z['time_s']
        self.geo=dict(np.load(OPERATORS/'geometry.npz'));self.prep=read(OPERATORS/'prepared.json')
        self.p=self.prep['params'];self.n=1600;self.P=3200
        self.group_cell=self.geo['group_cell']+self.geo['population']*self.n
        self.e=self.geo['population']==0
        size=np.bincount(self.group_cell,weights=self.geo['group_size'],minlength=self.P)
        self.w=self.geo['group_size']/size[self.group_cell]
        if (self.source/'checkpoint.npz').exists():
            with np.load(self.source/'checkpoint.npz') as z:self.Z=z['Z']
        else:
            # Fine equilibria can store independent PDFs as a disk array.
            # Reconstruct exactly the same prescribed spatial Z path.
            from equilibrium_predictor import EquilibriumProblem
            problem=EquilibriumProblem(Path(self.config['table']),'pchip')
            self.Z,_=problem.resource(self.config['D'])
        source=ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916/coarse_40'
        rr=[];cc=[];dd=[];vv=[]
        for key,target,src in [('ee',0,0),('ie',1,0),('ei',0,1),('ii',1,1)]:
            W=sparse.load_npz(source/f'delay_{key}.npz').tocoo()
            rr.append(W.row+target*self.n);cc.append(W.col%self.n+src*self.n)
            dd.append(W.col//self.n+1);vv.append(W.data)
        row=np.concatenate(rr);col=np.concatenate(cc);delay=np.concatenate(dd);val=np.concatenate(vv)
        keys,inverse=np.unique(row.astype(np.int64)*self.P+col,return_inverse=True)
        self.row=(keys//self.P).astype(np.int32);self.col=(keys%self.P).astype(np.int32)
        self.depth=int(delay.max())
        self.delay_by_edge=sparse.coo_matrix((val,(inverse,delay)),shape=(len(keys),self.depth+1)).tocsr()
        self.ptr=np.r_[0,np.cumsum(np.bincount(self.row,minlength=self.P))].astype(np.int32)
        self.ampa=self.col<self.n
        self.row_sum_A=np.bincount(row[col<self.n],weights=val[col<self.n],minlength=self.P)
        self.row_sum_G=np.bincount(row[col>=self.n],weights=val[col>=self.n],minlength=self.P)
        self.tm=np.r_[np.full(self.n,self.p['tau_m_E']),np.full(self.n,self.p['tau_m_I'])]
        self.abs_h=np.sum(abs(self.h),axis=0)

    def synapse(self,z,rise,decay):
        ar=np.exp(-DT/rise);ad=np.exp(-DT/decay)
        return self.tm/rise*(DT/1000.)*(1-ad)/((1-ar/z)*(1-ad/z))

    def matrix(self,s):
        z=np.exp(s*DT/1000.)
        chi=np.exp(-s*self.t)@self.h
        m=(DT/1000.)/(z-(1-DT/1000.))
        effective=chi/(1+.0005*self.e*chi*m)
        S0=aggregate_complex(self.group_cell,self.w*effective,self.P)
        SZ=aggregate_complex(self.group_cell,self.w*self.Z*effective,self.P)
        a=self.synapse(z,self.p['tau_r_AMPA'],self.p['tau_d_AMPA'])*S0
        g=-self.synapse(z,self.p['tau_r_GABA'],self.p['tau_d_GABA'])*SZ
        phase=np.exp(-s*DT/1000.*np.arange(self.depth+1))
        weights=self.delay_by_edge@phase
        data=weights*np.where(self.ampa,a[self.row],g[self.row])
        return sparse.csr_matrix((data,self.col,self.ptr),shape=(self.P,self.P))

    def high_frequency_bound(self,hz):
        # For frequencies from hz to Nyquist, discrete synaptic/M magnitudes
        # decrease monotonically. Delays have modulus one; |chi| <= sum |h|.
        z=np.exp(2j*np.pi*hz*DT/1000.)
        m=abs((DT/1000.)/(z-(1-DT/1000.)))
        den=1-.0005*self.e*self.abs_h*m
        if np.min(den)<=0:return float('inf')
        bound=self.abs_h/den
        s0=np.bincount(self.group_cell,weights=self.w*bound,minlength=self.P)
        sz=np.bincount(self.group_cell,weights=self.w*self.Z*bound,minlength=self.P)
        a=abs(self.synapse(z,self.p['tau_r_AMPA'],self.p['tau_d_AMPA']))
        g=abs(self.synapse(z,self.p['tau_r_GABA'],self.p['tau_d_GABA']))
        return float(np.max(s0*a*self.row_sum_A+sz*g*self.row_sum_G))


def contracted_phase(eigenvalues,inner=.6,outer=.9):
    # Radially contract eigenvalues strictly inside the unit disk. This cannot
    # cross +1. It removes harmless phase accumulation only when the complete
    # spectrum outside 'inner' is captured; refinement still has to establish
    # a winding number of the resulting continuous frequency curve.
    scale=np.clip((abs(eigenvalues)-inner)/(outer-inner),0.,1.)
    return float(np.angle(np.prod(1-scale*eigenvalues)))


def self_check():
    tests=[]
    for rho,count in [(.7,0),(1.2,2)]:
        t=np.linspace(0,np.pi,2001)
        ev=np.array([rho*np.exp(.7j),rho*np.exp(-.7j)])
        phase=np.unwrap([contracted_phase(ev*np.exp(-1j*w)) for w in t])
        observed=-int(round((phase[-1]-phase[0])/np.pi))
        assert observed==count
        tests.append(dict(radius=rho,expected_unstable=count,observed=observed))
    return tests


def run(args):
    folder=Path(args.response);model=DynamicLoop(folder)
    if args.tag:
        folder=folder/args.tag;folder.mkdir(parents=True,exist_ok=False)
    limits=[(f,model.high_frequency_bound(f)) for f in (100.,200.,400.,800.,1600.,3200.,5000.)]
    safe=[f for f,b in limits if b<.4]
    upper=safe[0] if safe else 5000.
    base=np.array([0,.01,.025,.05,.1,.25,.5,1,2,3,5,7,10,15,20,30,50,80,120,200,350,600,1000,1600,3200,5000.])
    freqs=np.unique(np.r_[base[base<upper],upper])
    if args.subdivisions>1:
        freqs=np.unique(np.concatenate([np.linspace(a,b,args.subdivisions+1)
            for a,b in zip(freqs[:-1],freqs[1:])]))
    rows=[]
    rng=np.random.default_rng(6131);v0=rng.normal(size=model.P)+1j*rng.normal(size=model.P)
    vectors=[]
    for f in freqs:
        K=model.matrix(2j*np.pi*f)
        values,modes=eigs(K,k=args.modes,which='LM',v0=v0,tol=1e-8,maxiter=3000,ncv=max(2*args.modes+1,50))
        order=np.argsort(-abs(values));values=values[order];modes=modes[:,order]
        near=np.argmin(abs(values-1));captured=bool(abs(values[-1])<.6)
        row=dict(frequency_hz=float(f),outer_spectrum_captured=captured,
            maximum_loop_eigenvalue_modulus=float(abs(values).max()),smallest_computed_modulus=float(abs(values[-1])),
            distance_to_plus_one=float(abs(values[near]-1)),
            nearest_eigenvalue=[float(values[near].real),float(values[near].imag)],
            contracted_determinant_phase=contracted_phase(values),
            eigenvalues=np.c_[values.real,values.imag])
        rows.append(row);vectors.append(modes[:,near])
        print('loop D',model.config['D'],'f',f,'max',row['maximum_loop_eigenvalue_modulus'],
              'distance',row['distance_to_plus_one'],'outer',captured,flush=True)
        write(folder/'loop_scan_progress.json',dict(status='RUNNING',pid=os.getpid(),latest=row))
    phase=np.unwrap([r['contracted_determinant_phase'] for r in rows])
    coarse_count=-int(round((phase[-1]-phase[0])/np.pi))
    np.savez_compressed(folder/'loop_modes.npz',frequency_hz=freqs,mode_EI=np.asarray(vectors))
    write(folder/'loop_spectrum.json',dict(status='COARSE_DYNAMIC_LOOP_SCAN_COMPLETE',D=model.config['D'],
        frequency_subdivisions=args.subdivisions,response_source=str(Path(args.response).resolve()),
        self_checks=self_check(),high_frequency_bounds=limits,points=rows,
        coarse_winding_candidate=coarse_count if all(r['outer_spectrum_captured'] for r in rows) else None,
        classification='NOT_CERTIFIED',
        required=['adaptive frequency/winding convergence','impulse duration and finite-difference convergence',
                  'local density pole/control checks','direct full-map eigenmode growth validation']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--response',type=Path,required=True)
    ap.add_argument('--modes',type=int,default=24)
    ap.add_argument('--subdivisions',type=int,default=1);ap.add_argument('--tag');run(ap.parse_args())
