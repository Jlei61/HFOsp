"""Training-only fit of one preregistered response-bank candidate."""
from response_bank import DEST
from common import *
from scipy import sparse
from scipy.sparse.linalg import spsolve


def neighbors(grids):
    shape=tuple(map(len,grids));ids=np.arange(np.prod(shape)).reshape(shape)
    rows=[];cols=[];values=[];edge=0
    for axis,grid in enumerate(grids):
        scale=np.median(np.diff(grid))
        for ix in np.ndindex(shape):
            if ix[axis]+1==shape[axis]:continue
            j=list(ix);j[axis]+=1;wt=scale/(grid[j[axis]]-grid[ix[axis]])
            rows.extend([edge,edge]);cols.extend([ids[ix],ids[tuple(j)]]);values.extend([wt,-wt]);edge+=1
    return sparse.coo_matrix((values,(rows,cols)),shape=(edge,ids.size)).tocsr()


def main():
    c=read(OUT/'response_bank_candidate_contract.json');DEST.mkdir(exist_ok=True)
    raw=read(BASE/'dynamic_assay/rows.json')['rows'];s=model();all_values={};summaries=[]
    frequencies=np.array(c['all_training_frequencies_hz']);lam=2j*np.pi*frequencies/1000
    taus=np.array(c['filter_times_ms']);basis=lam[:,None]*taus/(1+lam[:,None]*taus)
    for pop in 'EI':
        rows=[r for r in raw if r['pop']==pop]
        grids=[np.array(sorted({r[k] for r in rows})) for k in ['x','sigma_E','sigma_I']]
        shape=tuple(map(len,grids));n=int(np.prod(shape));index={}
        points={}
        for ix in np.ndindex(shape):
            key=tuple(grids[j][ix[j]] for j in range(3));index[key]=np.ravel_multi_index(ix,shape);points[key]={}
        for r in rows:points[(r['x'],r['sigma_E'],r['sigma_I'])][(r['channel'],r['frequency_hz'])]=r
        xyz=np.array(list(index));mu=11+7*xyz[:,0];ve=(7*xyz[:,1])**2;vi=(7*xyz[:,2])**2
        static=s.spline[pop].evaluate(mu,ve,vi,np.full(n,18.))
        gains=np.array([static['d_mu']*7000,static['d_ve']*49000,static['d_vi']*49000])
        L=neighbors([np.arcsinh(grids[0]),grids[1],grids[2]])
        regularizer=sparse.kron(L.T@L,sparse.eye(5),format='csc')
        ridge=c['ridge']*sparse.eye(n*5,format='csc')
        values=[]
        for channel in range(3):
            R=np.zeros((n,len(frequencies)),complex);sem=np.ones(R.shape);eligible=np.zeros(n,bool)
            scale=7. if channel==0 else 49.
            for key,p in points.items():
                j=index[key];dc=p[(0,0.)]
                if (channel,0.) not in p:continue
                mean_snr=abs(complex(*dc['response']))/max(dc['sem'],1e-12)
                rr=np.array([complex(*p[(channel,float(f))]['response']) for f in frequencies])*scale
                ss=np.array([p[(channel,float(f))]['sem'] for f in frequencies])*scale
                ac_snr=max(abs(rr))/max(np.median(ss),1e-12)
                eligible[j]=mean_snr>=10 and (channel==0 or ac_snr>=10)
                if channel:
                    syn=1+lam*s.tau[channel-1]/2;rr*=syn;ss*=abs(syn)
                R[j]=rr;sem[j]=np.maximum(ss,c['SEM_floor_fraction']*max(abs(gains[channel,j]),.01*abs(gains[0,j]),1e-5))
            A=gains[0,:,None,None]*basis[None,:,:]/sem[:,:,None]
            target=(R-gains[channel,:,None])/sem
            A[~eligible]=0;target[~eligible]=0
            def solve(mask,smoothing):
                a=A[:,mask];y=target[:,mask]
                gram=np.einsum('nfj,nfk->njk',a.conj(),a).real
                rhs=np.einsum('nfj,nf->nj',a.conj(),y).real
                matrix=sparse.block_diag(gram,format='csc')+smoothing*regularizer+ridge
                x=spsolve(matrix,rhs.ravel()).reshape(n,5)
                defect=np.linalg.norm(matrix@x.ravel()-rhs.ravel())/max(np.linalg.norm(rhs),1)
                assert defect<1e-7,defect
                return x,float(defect)
            fit_mask=np.isin(frequencies,c['selection_fit_frequencies_hz']);held_mask=~fit_mask
            tried=[]
            for smoothing in c['smoothing_candidates']:
                x,defect=solve(fit_mask,smoothing)
                error=np.einsum('nfj,nj->nf',A[:,held_mask],x)-target[:,held_mask]
                score=float(np.mean(abs(error[eligible])**2))
                tried.append(dict(smoothing=smoothing,training_frequency_holdout_weighted_MSE=score,linear_residual=defect))
            chosen=min(tried,key=lambda r:r['training_frequency_holdout_weighted_MSE'])['smoothing']
            x,defect=solve(np.ones(len(frequencies),bool),chosen);values.append(x.T.reshape((5,)+shape))
            predicted=gains[channel,:,None]+gains[0,:,None]*np.einsum('fj,nj->nf',basis,x)
            training_error=np.sqrt(np.mean(abs(predicted-R)**2,axis=1))/np.maximum(np.max(abs(R),axis=1),1e-8)
            summary=dict(pop=pop,channel=channel,eligible_workpoints=int(eligible.sum()),total_grid_points=n,
                         chosen_smoothing=chosen,selection=tried,final_linear_residual=defect,
                         median_training_relative_RMS=float(np.median(training_error[eligible])),
                         p90_training_relative_RMS=float(np.percentile(training_error[eligible],90)),
                         max_coefficient=float(np.max(abs(x))))
            summaries.append(summary);log('BANK FIT',summary)
        all_values.update({'x_'+pop:grids[0],'sE_'+pop:grids[1],'sI_'+pop:grids[2],
                           'values_'+pop:np.concatenate(values,axis=0)})
    np.savez_compressed(DEST/'coefficients.npz',**all_values)
    write(DEST/'fit_result.json',dict(status='TRAINING_FIT_COMPLETE',rows=summaries,
          source=str(BASE/'dynamic_assay/rows.json'),contract=str(OUT/'response_bank_candidate_contract.json'),
          validation='NOT_YET_RUN',scope='One response candidate trained only on single-cell assays; no network or bifurcation acceptance.'))


if __name__=='__main__':main()
