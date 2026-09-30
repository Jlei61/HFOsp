"""Condense only the linear equations of full multiple shooting.

All nonlinear shooting nodes remain independent. Exact block elimination
reduces the Newton linear solve to node zero and period; back-substitution
recovers every node update. A bounded step scales the entire recovered update.
This differs from eliminating nonlinear nodes by forward time integration.
"""
from common import np, write, log
from core_a_periodic_hookstep import krylov


def condensation(derivatives, slopes, fractions, normal, residual, wT):
    K=len(derivatives); N=normal.size
    f=residual[:-1].reshape(K,N)
    affine=np.zeros(N); time_column=np.zeros(N)
    for J,s,a,r in zip(derivatives,slopes,fractions,f):
        affine=(J(affine) if np.any(affine) else affine)+r
        time_column=(J(time_column) if np.any(time_column) else time_column)+s*a
    rhs=np.r_[affine,residual[-1]]
    def operator(v):
        q=v[:-1].copy()
        for J in derivatives:q=J(q)
        return np.r_[v[:-1]-q-time_column*(v[-1]/wT),normal@v[:-1]]
    def reconstruct(v):
        nodes=[v[:-1].copy()]; dt=v[-1]/wT; products=[]
        for j,J in enumerate(derivatives):
            product=J(nodes[j]); products.append(product)
            if j<K-1:nodes.append(product+slopes[j]*fractions[j]*dt+f[j])
        delta=np.r_[np.concatenate(nodes),v[-1]]
        actual=np.r_[np.concatenate([nodes[(j+1)%K]-products[j]-slopes[j]*fractions[j]*dt for j in range(K)]),normal@nodes[0]]
        return delta,actual
    return rhs,operator,reconstruct


def condensed_proposal(derivatives, slopes, fractions, normal, residual,
                       wT, P, m_factor, maximum, folder, linear_tolerance,
                       original_operator):
    rhs,operator,reconstruct=condensation(derivatives,slopes,fractions,normal,residual,wT)
    def right(v):
        u=v.copy(); u[:-1].reshape(-1,P)[4]*=m_factor; return u
    # The condensed RHS can be amplified. Tighten its relative solve target
    # to preserve the requested tolerance in the original full equations.
    tolerance=min(linear_tolerance,linear_tolerance*np.linalg.norm(residual)/np.linalg.norm(rhs))
    proposal,progress=krylov(lambda v:v-operator(v),rhs,maximum,folder,P,m_factor,
        right_override=right,linear_tolerance=tolerance,
        operator_description='Exactly condensed full multiple-shooting Newton system. Nonlinear nodes remain independent; not a Floquet eigensolver.')
    compact,meta=proposal(1e30)
    delta,actual=reconstruct(compact); norm=float(np.linalg.norm(delta))
    independent=original_operator(delta)
    parity=float(np.linalg.norm(actual-independent)/max(np.linalg.norm(independent),np.linalg.norm(residual),1e-15))
    assert parity<1e-10,('Condensed reconstruction disagrees with original full linear operator',parity)
    actual=independent
    check=dict(condensed_dimension=normal.size+1,full_dimension=residual.size,
        reconstructed_vs_original_full_operator_relative_error=parity,
        condensed_relative_residual=float(np.linalg.norm(operator(compact)-rhs)/np.linalg.norm(rhs)),
        original_full_linear_relative_residual=float(np.linalg.norm(actual-residual)/np.linalg.norm(residual)),
        requested_original_full_tolerance=linear_tolerance,
        requested_condensed_tolerance=tolerance,full_newton_step_norm=norm)
    write(folder/'condensed_linear_check.json',check); log('CONDENSED FULL NEWTON CHECK',check)
    def bounded(radius):
        alpha=min(1.,radius/max(norm,1e-30))
        return alpha*delta,dict(method='Whole-direction damped condensed Newton',radius=float(radius),
            step_norm=alpha*norm,damping=alpha,
            predicted_relative_residual=float(np.linalg.norm(residual-alpha*actual)/np.linalg.norm(residual)))
    return bounded,progress


def dense_check():
    """Independent dense cyclic solve including period and phase border."""
    rng=np.random.default_rng(925639); rows=[]
    for K in [2,4,6,12]:
        for N in [3,7]:
            matrices=[rng.normal(size=(N,N))*.3 for _ in range(K)]
            slopes=rng.normal(size=(K,N)); normal=rng.normal(size=N); normal/=np.linalg.norm(normal)
            for fixed_last in [False,True]:
                fractions=np.zeros(K) if fixed_last else np.full(K,1/K)
                if fixed_last:fractions[-1]=1
                residual=rng.normal(size=K*N+1); wT=.01
                derivatives=[lambda v,M=M:M@v for M in matrices]
                rhs,L,reconstruct=condensation(derivatives,slopes,fractions,normal,residual,wT)
                small=np.column_stack([L(v) for v in np.eye(N+1)])
                compact=np.linalg.solve(small,rhs); delta,product=reconstruct(compact)
                full=np.zeros((K*N+1,K*N+1))
                for j in range(K):
                    full[j*N:(j+1)*N,j*N:(j+1)*N]-=matrices[j]
                    jj=(j+1)%K; full[j*N:(j+1)*N,jj*N:(jj+1)*N]+=np.eye(N)
                    full[j*N:(j+1)*N,-1]=-slopes[j]*fractions[j]/wT
                full[-1,:N]=normal
                expected=np.linalg.solve(full,residual)
                error=float(np.linalg.norm(delta-expected)/np.linalg.norm(expected))
                product_error=float(np.linalg.norm(product-full@delta)/max(np.linalg.norm(product),1e-15))
                assert error<1e-10 and product_error<1e-10,(K,N,error,product_error)
                # Damping scales the entire recovered update, including
                # the affine back-substitution contribution at every node.
                alpha=.17; damp_error=float(np.linalg.norm(full@(alpha*delta)-alpha*product)/max(np.linalg.norm(product),1e-15))
                assert damp_error<1e-10
                rows.append(dict(K=K,N=N,fixed_last=fixed_last,solution_error=error,product_error=product_error,damping_error=damp_error))
    return dict(status='PASS',cases=rows,scope='Exact linear block elimination, full reconstruction and whole-update damping; original nonlinear-flow checks remain required per shooting run.')
