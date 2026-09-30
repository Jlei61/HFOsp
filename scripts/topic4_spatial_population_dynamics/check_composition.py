"""Independent neuron-synapse calculation for the six-component kernel."""
from shared import *
from composition_particles import evolve,initial
from particles import parameters,deterministic_trace
from scipy import sparse

def main():
    p=parameters();dt,ae,be,ai,bi,de,di,re,ri,incE,incI,signal,reset=p
    rng=np.random.default_rng(405);P=6;N=30;M=7;steps=400;ptr=np.arange(0,N+1,5)
    popreg=np.arange(6);reg=np.repeat(popreg,5);th=np.linspace(12.2,18.,N)
    exidx=np.r_[np.arange(10),np.full(20,-1)];ext=rng.poisson(.16,(steps,10)).astype(np.uint8)
    mats=[]
    for sources in [range(3),range(3,6)]:
        a=np.zeros((P,M*P))
        for src in sources:
            for dst in range(P):a[src,(1+(src+dst)%6)*P+dst]=rng.uniform(4,9)
        m=sparse.csr_matrix(a);mats.append((m.indptr.astype(np.int64),m.indices.astype(np.int64),m.data))
    gains=rng.uniform(.3,1.7,(P,5,6));gains/=gains.mean(1)[:,None,:];gains=gains.reshape(N,6)
    det=deterministic_trace(steps);weight=np.full((N,15),1./15);weight[reg>=3]=0
    state=initial(N,P,M,p)
    args=(ptr,popreg,th,exidx,np.arange(N),weight,np.arange(N),mats[0],mats[1],p)
    result=evolve(ext,0,*args,state,gains,det)
    v=np.full(N,reset);ref=np.zeros(N,int);sE=np.zeros(N);sI=np.zeros(N);IE=np.zeros(N);II=np.zeros(N)
    # Reference keeps individual synaptic currents and individual target delay queues.
    rings=[np.zeros((M,N)),np.zeros((M,N))];spikes=[]
    for t in range(steps):
        raw=np.full(N,signal*dt);raw[:10]=ext[t]
        sE=sE*ae+rings[0][t%M]+raw*np.where(reg<3,incE,incI)
        sI=sI*ai+rings[1][t%M];rings[0][t%M]=0;rings[1][t%M]=0
        IE=sE+(IE-sE)*be;II=sI+(II-sI)*bi
        ref=np.maximum(ref-1,0);free=ref==0;net=IE-II
        v=np.where(free,net+(v-net)*np.where(reg<3,de,di),reset)
        fire=free&(v>=th);v[fire]=reset;ref[fire]=np.where(reg<3,re,ri)[fire];spikes.append(fire)
        count=fire.reshape(P,5).sum(1)
        for ring,(ind,col,val) in zip(rings,mats):
            for b in range(P):
                for k in range(ind[b],ind[b+1]):
                    d=col[k]//P;a=col[k]%P
                    ring[(t+d)%M,5*a:5*a+5]+=gains[5*a:5*a+5,b]*val[k]*count[b]
    spikes=np.array(spikes);assert np.array_equal(result[3],spikes)
    assert np.allclose(v,state[0],rtol=0,atol=2e-10);assert np.array_equal(ref,state[1])
    rec=state[5].repeat(5,axis=0)*gains
    base=np.where(reg<2,0.,np.where(reg<3,det[-1,0],det[-1,1]))
    E=rec[:,:3].sum(1)+base+state[3];I=rec[:,3:].sum(1)
    assert np.allclose(IE,E,rtol=0,atol=2e-10);assert np.allclose(II,I,rtol=0,atol=2e-10)
    assert np.array_equal(result[0],spikes.reshape(-1,20,P,5).sum((1,3)))
    split=initial(N,P,M,p)
    first=evolve(ext[:200],0,*args,split,gains,det[:200]);second=evolve(ext[200:],200,*args,split,gains,det[200:])
    for i in range(5):assert np.array_equal(np.concatenate([first[i],second[i]]),result[i])
    for a,b in zip(state,split):assert np.array_equal(a,b)
    write(OUT/'numerical_validation_composition.json',dict(status='PASS_IMPLEMENTATION_ONLY',native_algebra_spike_parity=True,
        voltage_max_error=float(abs(v-state[0]).max()),current_max_error=float(max(abs(IE-E).max(),abs(II-I).max())),
        chunk_parity=True,scientific_correspondence='NOT_IMPLIED'))
    print('Composition checks passed.',flush=True)

if __name__=='__main__':main()
