"""Delivery integrity and scientific gate are distinct outputs."""
from common import *
from PIL import Image

def main():
    numerical=read(OUT/'numerical_validation.json');assert numerical['status']=='PASS_IMPLEMENTATION_ONLY'
    sizes=np.bincount(np.load(V10/'native/a/trajectory.npz')['region'],minlength=6)
    checks=[]
    for seed in SEEDS:
        z=np.load(OUT/'native'/str(seed)/'trajectory.npz');assert z['six_counts'].shape==(6000,6)
        if seed==848101:
            old=np.load(V10/'native/a/trajectory.npz')
            assert np.array_equal(z['six_counts'],old['six_group_counts_2ms'])
            assert np.array_equal(z['contact_envelope'].astype(np.float32),old['contact_envelope'])
            assert np.array_equal(z['field_counts'],old['sheet_activity_counts'])
        assert z['nu_core'].shape==(120000,2)
        assert np.array_equal(z['field_counts'].sum((1,2)),z['six_counts'][:,:3].sum(1))
        for grid in (10,20,40):
            reg=z[f'region_{grid}'];counts=z[f'counts_{grid}'];group=z[f'group_{grid}']
            assert np.array_equal(np.bincount(reg[group],minlength=6),sizes)
            for j in range(6):assert np.array_equal(counts[:,reg==j].sum(1),z['six_counts'][:,j])
            assert np.isclose(z[f'contact_weights_{grid}'].sum(1),1).all()
        checks.append(dict(kind='native',seed=seed,group_and_field_conservation=True))
    for grid in (10,20):
        z=np.load(OUT/'rate'/f'grid{grid}_seed848101.npz');native=np.load(OUT/'native/848101/trajectory.npz')
        assert np.array_equal(z['nu_core'],native['nu_core'])
        assert np.allclose(z['field_counts'].sum((1,2)),z['six_counts'][:,:3].sum(1),rtol=2e-6,atol=1e-5)
        assert np.isfinite(z['six_counts']).all() and z['six_counts'].min()>=0
        checks.append(dict(kind='rate',grid=grid,exact_afferent_input=True,field_conservation=True))
    images=[]
    for p in (OUT/'figures').glob('*.png'):
        with Image.open(p) as im:im.load();assert min(im.size)>600;images.append(dict(name=p.name,size=im.size))
    assert len(images)==6
    report=read(OUT/'comparison.json');assert report['bifurcation_allowed'] is False
    write(OUT/'validation.json',dict(status='PASS_OUTPUT_AND_NUMERICAL_CHECKS',checks=checks,images=images,
        scientific_status={g:q['first_screen'] for g,q in report['rate'].items()},
        bifurcation_allowed=False,human_visual_acceptance='PENDING'))
    print('Validated output integrity; no bifurcation gate inferred.',flush=True)

if __name__=='__main__':main()
