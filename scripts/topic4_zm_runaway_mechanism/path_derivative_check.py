"""Verify exact spatial-path derivatives and nonsmooth-knot semantics."""
from native_path import *


def main():
    s=model();rows=[]
    for name,attach in [('rate',attach_rate_entry_path),('native',attach_native_path),('fine',attach_fine_rate_entry_path)]:
        attach(s);knots=s.path_D_knots
        for j in range(len(knots)-1):
            D=float((knots[j]+knots[j+1])/2);h=(knots[j+1]-knots[j])*1e-3
            s.set_D(D);expected=path_Z_derivative(s,D)
            s.set_D(D+h);plus=s.Z.copy();s.set_D(D-h);minus=s.Z.copy()
            fd=(plus-minus)/(2*h)
            rel=float(np.linalg.norm(fd-expected)/np.linalg.norm(expected))
            assert rel<1e-8,(name,j,rel)
            rows.append(dict(family=name,segment=j,relative_FD_error=rel))
        for j in range(1,len(knots)-1):
            right=(s.path_Z_fields[j+1]-s.path_Z_fields[j])/(knots[j+1]-knots[j])
            assert np.array_equal(path_Z_derivative(s,float(knots[j])),right)
    write(OUT/'path_derivative_check.json',dict(status='PASS',rows=rows,
        change='Analytic segment slopes replace finite differences; the spatial parameter paths and nonlinear equations are unchanged.',
        knot_rule='Right derivative at interior knots; left derivative at final endpoint. A join is not by itself a smooth bifurcation.'))
    log('SPATIAL PATH DERIVATIVES PASS',rows)


if __name__=='__main__':main()
