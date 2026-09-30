"""Independent original-edge reference and singleton-source limit."""
from shared_source import *
from source_operator import project
from source_simulation import evolve,initial,parameters
from scipy import sparse

def main(gpu=False):
    rng=np.random.default_rng(406);N=30;ne=15;M=7;steps=400
    reg=np.repeat(np.arange(6),5);th=np.linspace(12.2,18.,N);p=parameters();p[11]=4.2
    dt,ae,be,ai,bi,de,di,re,ri,incE,incI,signal,reset=p
    exidx=np.r_[np.arange(10),np.full(20,-1)];ext=rng.poisson(.42,(steps,10)).astype(np.uint8)
    edges=[]
    for offset in (0,ne):
        kinds=[]
        for d in range(M):
            mat=rng.uniform(1.5,3.,(N,ne))*(rng.random((N,ne))<.13) if d else np.zeros((N,ne))
            kinds.append(sparse.csr_matrix(mat))
        edges.append(kinds)
    weight=rng.uniform(0,1,(N,15));weight[reg>=3]=0;weight/=weight.sum(0)
    checks=[]
    for singleton in (False,True):
        group=np.arange(N) if singleton else np.repeat(np.arange(6),5);count=np.bincount(group);P=len(count);ptr=np.r_[0,np.cumsum(count)]
        pr=reg[ptr[:-1]];ops=[]
        for mats,offset in zip(edges,(0,ne)):
            mat,_=project(mats,group[offset:offset+ne],count,np.arange(N))
            ops.append((mat.indptr.astype(np.int64),(mat.indices%N).astype(np.int32),(mat.indices//N).astype(np.int16),mat.data))
        args=(ptr,pr,th,exidx,np.arange(N),weight,np.arange(N),ops[0],ops[1],p)
        state=initial(N,M,p);result=evolve(ext,0,*args,state)
        v=np.full(N,reset);ref=np.zeros(N,int);sE=np.zeros(N);sI=np.zeros(N);IE=np.zeros(N);II=np.zeros(N)
        rings=[np.zeros((M,N)),np.zeros((M,N))];spikes=[]
        for t in range(steps):
            raw=np.full(N,signal*dt);raw[:10]=ext[t]
            sE=sE*ae+rings[0][t%M];sI=sI*ai+rings[1][t%M]
            sE+=raw*np.where(reg<3,incE,incI);rings[0][t%M]=0;rings[1][t%M]=0
            IE=sE+(IE-sE)*be;II=sI+(II-sI)*bi
            ref=np.maximum(ref-1,0);free=ref==0;net=IE-II
            v=np.where(free,net+(v-net)*np.where(reg<3,de,di),reset)
            fire=free&(v>=th);v[fire]=reset;ref[fire]=np.where(reg<3,re,ri)[fire];spikes.append(fire)
            source=np.bincount(group,weights=fire,minlength=P)[group]/count[group]
            if singleton:assert np.array_equal(source,fire)
            for ring,mats,offset in zip(rings,edges,(0,ne)):
                for d,mat in enumerate(mats):
                    if d:ring[(t+d)%M]+=mat@source[offset:offset+ne]
        spikes=np.array(spikes);assert spikes[:,:ne].any() and spikes[:,ne:].any();assert np.array_equal(result[3],spikes)
        assert np.array_equal(ref,state[1]);assert np.allclose(v,state[0],rtol=0,atol=2e-10)
        assert np.allclose(IE,state[3],rtol=0,atol=2e-10);assert np.allclose(II,state[5],rtol=0,atol=2e-10)
        binned=spikes.reshape(-1,20,N).sum(1)
        assert np.array_equal(result[0],np.stack([binned[:,group==a].sum(1) for a in range(P)],axis=1))
        assert np.array_equal(result[2][:,:ne],binned[:,:ne]);assert not result[2][:,ne:].any()
        assert np.allclose(result[1],binned@weight,rtol=0,atol=2e-10)
        split=initial(N,M,p);first=evolve(ext[:200],0,*args,split);second=evolve(ext[200:],200,*args,split)
        for a,b in zip(state,split):assert np.array_equal(a,b)
        for i in range(5):assert np.array_equal(np.concatenate([first[i],second[i]]),result[i])
        checks.append(dict(singleton_sources=singleton,spike_parity=True,source_reference='original edge matrices applied to current source-group mean spikes',
            voltage_max_abs_error=float(abs(v-state[0]).max()),current_max_abs_error=float(max(abs(IE-state[3]).max(),abs(II-state[5]).max())),
            chunk_parity=True,contact_and_field_parity=True,total_spikes=int(spikes.sum())))
        if gpu:
            from source_gpu import GPU
            engine=GPU(*args,initial(N,M,p));actual=engine.evolve(ext,0);gst=engine.host_state()
            for j in (0,2,3):assert np.array_equal(actual[j],result[j]),('GPU integer output',j)
            for j in (1,4):assert np.allclose(actual[j],result[j],rtol=0,atol=2e-10),('GPU floating output',j)
            for j,(a,b) in enumerate(zip(gst,state)):assert np.array_equal(a,b),('GPU state',j,float(np.max(abs(a-b))))
            checks[-1]['gpu_state_bitwise_equal']=True
    write(OUT/('numerical_validation_gpu.json' if gpu else 'numerical_validation.json'),dict(status='PASS_IMPLEMENTATION_ONLY',checks=checks,scientific_correspondence='NOT_IMPLIED'))
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');main(parser.parse_args().gpu)
