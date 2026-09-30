"""Existing native field records: spatial subdivision versus equal-size shuffles."""
from pathlib import Path
import numpy as np,json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/native_input_bridge'
OP=ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators'
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/fields'


def main():
    g={n:dict(np.load(OP/f'g{n}/geometry.npz')) for n in [20,40]}
    assert np.array_equal(g[20]['original_positions'],g[40]['original_positions'])
    coarse=g[20]['cell_group'];fine=g[40]['cell_group'];P=len(g[40]['group_size'])
    parent=np.empty(P,dtype=int)
    for j in range(P):
        ids=np.unique(coarse[fine==j]);assert len(ids)==1;parent[j]=ids[0]
    rng=np.random.default_rng(920115);randoms=[]
    for repeat in range(10):
        shuffled=fine.copy()
        for j in range(len(g[20]['group_size'])):
            cell=np.flatnonzero(coarse==j);shuffled[cell]=rng.permutation(fine[cell])
        assert np.array_equal(np.bincount(shuffled),g[40]['group_size'])
        assert np.array_equal(parent[shuffled],coarse)
        randoms.append(shuffled[:32000])
    maps=[coarse[:32000],fine[:32000]]+randoms;geos=[g[20]]+[g[40]]*11
    configs=[]
    for ids,geo in zip(maps,geos):
        size=np.bincount(ids,minlength=len(geo['group_size']));den=np.maximum(size,1)
        masks=[(geo['population']==0)&(geo['group_region']==j) for j in range(3)]
        configs.append((ids,size,den,masks))
    rows=[]
    for f in sorted(SOURCE.glob('*.npz')):
        if int(f.stem.split('_')[1])<=90000 or int(f.stem.split('_')[0])>=103700:continue
        z=np.load(f)
        for k,tm in enumerate(z['zm_step']*.1):
            if not 9000<=tm<10370:continue
            cur=z['ie'][k].astype(float)-z['z'][k].astype(float)*z['ii'][k].astype(float)-.0005*z['m'][k].astype(float)
            for method,(ids,size,den,masks) in enumerate(configs):
                mu=np.bincount(ids,weights=cur,minlength=len(size))/den
                var=np.maximum(np.bincount(ids,weights=cur*cur,minlength=len(size))/den-mu*mu,0)
                rows.extend([tm,method,j,float(np.average(var[mask],weights=size[mask]))] for j,mask in enumerate(masks))
    a=np.array(rows);result=[]
    for lo,hi in [(9000,9420),(9420,9868.5),(9868.5,10370)]:
        for reg in range(3):
            v=[float(a[(a[:,0]>=lo)&(a[:,0]<hi)&(a[:,1]==m)&(a[:,2]==reg),3].mean()) for m in range(12)]
            result.append(dict(window_ms=[lo,hi],region=reg,g20_native_within_variance_mv2=v[0],g40_native_within_variance_mv2=v[1],
                retained_fraction=v[1]/v[0],matched_size_shuffle_mean_mv2=float(np.mean(v[2:])),
                matched_size_shuffle_range_mv2=[min(v[2:]),max(v[2:])],shuffle_retained_fraction=float(np.mean(v[2:])/v[0])))
    np.savez_compressed(OUT/'spatial_resolution_spread.npz',rows=a,columns=['time_ms','method','region','within_variance_mv2'])
    (OUT/'spatial_resolution_spread.json').write_text(json.dumps(dict(status='READ_ONLY_DESCRIPTIVE_COMPLETE',rows=result,
        source=str(SOURCE),sampling_ms=5,shuffle_seed=920115,shuffles=10,
        control='Finegroup labels shuffled within each parentcoarsegroup; exactgroupN and originalcoarsegroup membership preserved. Removes spatialsubdivision while retaining smallerNs.',
        scope='One existing native history; shuffles are descriptive partitioncontrols, not independentnetworkexperiments. ActualeffectiveinhibitionincludesZ/M. No finenetwork simulation, privatevarianceclosure validation or bifurcation.'),indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
