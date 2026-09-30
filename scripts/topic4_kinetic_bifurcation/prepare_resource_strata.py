"""Prepare the proposed Z-strata geometry and exactly lifted g40 communication.

This is a candidate representation, not a dynamical acceptance or parameter
fit. Thresholds retain their old parent values; adaptation sharing is recorded
explicitly and must be implemented by the consuming density model.
"""
from autonomous_density import *
from z_quadrature import resource_strata
from scipy import sparse
from interictal_common import weights as contact_weights


def prepare(levels):
    folder=OUT/'operators'/f'selected_g40_theta0.25_Zstrata{levels}'
    folder.mkdir(parents=True,exist_ok=False)
    old=dict(np.load(OPERATORS/'geometry.npz'))
    strata=resource_strata(old,field_at(.225)[0],levels)
    parent=strata['parent_group'];assignment=strata['cell_stratum'];size=strata['stratum_size']
    P=len(parent);n=1600;prep=read(OPERATORS/'prepared.json')
    g={k:old[k][parent] for k in ('population','group_cell','group_region','threshold_mv')}
    position=old['original_positions']
    g.update(cell_group=assignment,group_size=size,parent_group=parent,
        parent_group_size=old['group_size'],original_positions=position,
        centers_mm=old['centers_mm'],threshold_error_mv=old['threshold_error_mv'],
        positions=np.column_stack([np.bincount(assignment,weights=position[:,i])/size for i in (0,1)]))
    for key,w in zip(('contact_rate_weights','contact_current_weights'),contact_weights()):
        value=np.column_stack([np.bincount(assignment[:32000],weights=w[:,i],minlength=P) for i in range(w.shape[1])])
        assert np.max(abs(value.sum(0)-old[key].sum(0)))<1e-12
        g[key]=value
    assert np.array_equal(g['threshold_mv'][assignment],old['threshold_mv'][old['cell_group']])
    np.savez_compressed(folder/'geometry.npz',**g)
    source=Path(prep['communication_source']);pop=g['population'];cell=g['group_cell']
    ids=np.arange(P);coarse=cell+pop*n
    counts=np.bincount(coarse,weights=size,minlength=2*n)
    lift=sparse.coo_matrix((np.ones(P),(ids,coarse)),shape=(P,2*n)).tocsr()
    depth=prep['max_delay_steps'];history=np.random.default_rng(6119).uniform(0,.01,(depth,P))
    checks={};records=[]
    for kind,(name,paths) in enumerate((('ampa',('ee','ie')),('gaba',('ei','ii')))):
        take=pop==kind
        restrict=sparse.coo_matrix((size[take]/counts[coarse[take]],(cell[take],ids[take])),shape=(n,P)).tocsr()
        original=sparse.vstack([sparse.load_npz(source/f'delay_{k}.npz') for k in paths]).tocsr()
        rows=[];cols=[];data=[]
        for d in range(depth):
            part=(lift@original[:,d*n:(d+1)*n]@restrict).tocoo()
            rows.append(part.row);cols.append(part.col+d*P);data.append(part.data)
        W=sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(P,depth*P)).tocsr()
        expected=lift@(original@(restrict@history.T).T.reshape(-1));actual=W@history.reshape(-1)
        error=float(np.max(abs(actual-expected)));assert error<1e-10
        reference=np.zeros(2*n);reference[coarse]=actual
        target_error=float(np.max(abs(actual-reference[coarse])));assert target_error<1e-11
        sparse.save_npz(folder/f'delay_{name}.npz',W)
        checks[name]=dict(arbitrary_delayed_activity_error=error,same_target_cell_input_error=target_error,terms=W.nnz)
        records.append(dict(kind=name,terms=W.nnz));print(name,checks[name],flush=True)
    prep.pop('projection_checks',None)
    prep.update(groups=P,operators=records,communication_qa=checks,
        scope='Candidate fixed empirical Z strata; selected g40 mean communication exactly lifted',
        parent_operators=str(OPERATORS),resource_strata_per_E_group=levels,
        resource_membership='Fixed empirical rank quantiles along the reference Z path; no output-based membership',
        thresholds='Retained parent-group values, identical to the Z-only particle pilot',
        M_sharing='Dynamic original parent-group mean shared by its resource strata; consumers must enforce this constraint',
        candidate_approximations=['parent-group threshold quadrature','finite resource-strata representation',
            'dynamic parent-group M','deterministic population limit and density discretization'],
        scientific_acceptance='NOT_VALIDATED; needs the paired finite-particle and density correspondence checks')
    write(folder/'prepared.json',prep)
    write(folder/'preparation_qa.json',dict(status='ALGEBRAIC_PREPARATION_PASS',groups=P,
        mean_resource_preserved=True,original_threshold_assignment_preserved=True,communication_checks=checks,
        scope='Geometry, physical resource representation and exact communication lifting only; no model/dynamics acceptance'))
    print(folder,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--levels',type=int,default=2)
    prepare(ap.parse_args().levels)
