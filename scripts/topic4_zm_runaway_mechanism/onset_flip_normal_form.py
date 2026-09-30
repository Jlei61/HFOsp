"""Map normal-form algebra, including noncritical-state quadratic response.

For a full Poincare map P, J q = mu q and p J = mu p, p q = 1.
The parameterization H(w)=x+w q+h2*w^2/2 removes its quadratic term:
 (mu^2 I-J) h2 = B(q,q).
At mu=-1 the cubic coefficient is
 c = p C(q,q,q)/6 + p B(q,h2)/2.
The second term is essential; a raw cubic fit along q is not this coefficient.
This finite-dimensional MAP algebra applies to the implemented full delayed
history map; it does not by itself qualify its continuous-DDE approximation.
"""
from common import np,OUT,write


def cubic_from_curved_pair(plus,minus,h,linear,dual):
    return float(dual@((plus-minus)/(2*h)-linear)/(h*h))


def dense_check():
    # Independent two-variable polynomial maps have an analytic center
    # reduction: F1=-x+a*x^2+b*x^3+c*x*y, F2=lambda*y+d*x^2.
    # Its flip coefficient is b+a^2+c*d/(1-lambda), including the
    # noncritical y response and removal of the x quadratic term.
    rng=np.random.default_rng(925940);rows=[]
    for k in range(12):
        a,b,c,d=rng.uniform(-1,1,4);lam=rng.uniform(-.6,.6)
        L=np.eye(2)+rng.normal(size=(2,2))*.15;inv=np.linalg.inv(L)
        J=L@np.diag([-1.,lam])@inv
        norm=np.linalg.norm(L[:,0]);q=L[:,0]/norm;p=inv[0]*norm
        B=L@np.array([2*a,2*d])/(norm*norm)
        h2=np.linalg.solve(np.eye(2)-J,B)
        expected=(b+a*a+c*d/(1-lam))/(norm*norm)
        def F(v):
            x,y=inv@v;return L@np.array([-x+a*x*x+b*x**3+c*x*y,lam*y+d*x*x])
        estimates=[]
        for h in [2e-3,1e-3]:
            plus=F(h*q+.5*h*h*h2);minus=F(-h*q+.5*h*h*h2)
            estimates.append(cubic_from_curved_pair(plus,minus,h,J@q,p))
        rich=(4*estimates[1]-estimates[0])/3
        err=abs(rich-expected)/max(1.,abs(expected))
        assert err<1e-8,(k,rich,expected,err)
        rows.append(dict(case=k,expected=expected,raw_estimates=estimates,extrapolated=rich,relative_error=err))
    out=OUT/'core_a_bifurcation_type_20260924/numerical_checks/flip_map_normal_form'
    out.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',cases=rows,scope='Map normal-form algebra and curved-pair coefficient estimator only. Not a spatial-model coefficient or bifurcation certificate.')
    write(out/'result.json',result);return result


if __name__=='__main__':print(dense_check())
