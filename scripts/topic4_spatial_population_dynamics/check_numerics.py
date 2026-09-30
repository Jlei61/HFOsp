"""Check the factored population kernel against neuron-resolved native algebra."""
from shared import *
from particles import evolve,parameters,scatter,deterministic_trace
from scipy import sparse

def main(use_gains=False):
    p=parameters();dt,ae,be,ai,bi,de,di,re,ri,incE,incI,signal,reset=p
    rng=np.random.default_rng(404);P=4;N=20;M=7;steps=400
    ptr=np.arange(0,N+1,5);popreg=np.array([0,2,3,5]);reg=np.repeat(popreg,5)
    th=np.linspace(12.2,18.,N);exidx=np.r_[np.arange(5),np.full(15,-1)]
    ext=rng.poisson(.16,size=(steps,5)).astype(np.uint8)
    mats=[]
    for source in [[0,1],[2,3]]:
        a=np.zeros((P,M*P))
        for s in source:
            for target in range(P):a[s,(1+(s+target)%6)*P+target]=float(rng.uniform(4,9))
        mat=sparse.csr_matrix(a);mats.append((mat.indptr.astype(np.int64),mat.indices.astype(np.int64),mat.data))
    state=(np.full(N,reset),np.zeros(N,np.int32),np.zeros(N),np.zeros(N),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros((M,P)),np.zeros((M,P)))
    weight=np.full((N,15),1./10);weight[reg>=3]=0
    gains=rng.uniform(.7,1.3,(N,2)) if use_gains else np.ones((N,2))
    gains=gains.reshape(P,5,2);gains/=gains.mean(1)[:,None,:];gains=gains.reshape(N,2)
    det=deterministic_trace(steps)
    result=evolve(ext,0,ptr,popreg,th,exidx,np.arange(N),weight,np.arange(N),mats[0],mats[1],p,state,
        gains if use_gains else None,det if use_gains else None)
    v=np.full(N,reset);ref=np.zeros(N,int);sE=np.zeros(N);sI=sE.copy();IE=sE.copy();II=sE.copy()
    rings=[np.zeros((M,P)),np.zeros((M,P))];spikes=[]
    for t in range(steps):
        raw=np.full(N,signal*dt);raw[:5]=ext[t]
        sE*=ae;sI*=ai
        sE+=np.repeat(rings[0][t%M],5)*gains[:,0];sI+=np.repeat(rings[1][t%M],5)*gains[:,1]
        rings[0][t%M]=0;rings[1][t%M]=0
        sE+=raw*np.where(reg<3,incE,incI)
        IE=sE+(IE-sE)*be;II=sI+(II-sI)*bi
        ref=np.maximum(ref-1,0);free=ref==0;net=IE-II
        v=np.where(free,net+(v-net)*np.where(reg<3,de,di),reset)
        fire=free&(v>=th);v[fire]=reset;ref[fire]=np.where(reg<3,re,ri)[fire]
        spikes.append(fire);c=fire.reshape(P,5).sum(1)
        for ring,(ind,col,val) in zip(rings,mats):
            for b in range(P):
                for k in range(ind[b],ind[b+1]):ring[(t+col[k]//P)%M,col[k]%P]+=val[k]*c[b]
    spikes=np.array(spikes)
    assert np.array_equal(result[3],spikes)
    assert np.allclose(v,state[0],rtol=0,atol=2e-10)
    assert np.array_equal(ref,state[1])
    base=np.where(reg<2,0.,np.where(reg<3,det[-1,0],det[-1,1]))
    reconstructed=state[3]+base+(np.repeat(state[5],5)-base)*gains[:,0]
    assert np.allclose(IE,reconstructed,rtol=0,atol=2e-10)
    assert np.allclose(II,np.repeat(state[7],5)*gains[:,1],rtol=0,atol=2e-10)
    assert np.array_equal(result[0],spikes.reshape(-1,20,P,5).sum((1,3)))
    # Chunk boundaries must not reset synapses, refractory state or delay history.
    split=(np.full(N,reset),np.zeros(N,np.int32),np.zeros(N),np.zeros(N),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros(P),np.zeros((M,P)),np.zeros((M,P)))
    first=evolve(ext[:200],0,ptr,popreg,th,exidx,np.arange(N),weight,np.arange(N),mats[0],mats[1],p,split,
        gains if use_gains else None,det[:200] if use_gains else None)
    second=evolve(ext[200:],200,ptr,popreg,th,exidx,np.arange(N),weight,np.arange(N),mats[0],mats[1],p,split,
        gains if use_gains else None,det[200:] if use_gains else None)
    for i in range(5):assert np.array_equal(np.concatenate([first[i],second[i]]),result[i])
    checks=read(OUT/'prepared.json')['checks']
    assert all(row['relative_error']<2e-12 for row in checks)
    write(OUT/f'numerical_validation{"_gain" if use_gains else ""}.json',dict(status='PASS_IMPLEMENTATION_ONLY',native_algebra_spike_parity=True,
        voltage_max_error=float(np.max(abs(v-state[0]))),synapse_max_error=float(np.max(abs(IE-reconstructed))),
        chunk_parity=True,projected_graph_moments=checks,scientific_correspondence='NOT_IMPLIED'))
    print('Numerical checks passed; scientific correspondence not implied.',flush=True)

if __name__=='__main__':main();main(True)
