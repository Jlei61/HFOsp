"""Check intersections in the full group state, not just the plotted mean."""
from continue_D import *
from scipy.spatial import cKDTree


def load_points(base,names):
    states=[];records=[];tangents=[]
    for name in names:
        for row in read(base/name/'result.json')['rows']:
            if not row.get('converged',True):continue
            file=base/name/f'point{row["index"]:04d}.npz'
            with np.load(file) as z:
                states.append(np.r_[z['r']/RS,float(z['D'])/DS]);tangents.append(z['tangent'].copy())
            records.append(str(file))
    return np.array(states),np.array(tangents),records


def main(a):
    s=ZMRate();base=DEST/'g20'
    xx,tt,files=load_points(base,[a.new_branch]);yy,uu,oldfiles=load_points(base,a.old_branch)
    dist,nearest=cKDTree(yy).query(xx);order=np.argsort(dist);results=[]
    for i in order[:20]:
        j=nearest[i];alignment=float(tt[i]@uu[j])
        if abs(alignment)<.9:continue
        D=xx[i,-1]*DS;pred=yy[j].copy();pred[-1]=xx[i,-1]
        border=np.zeros(s.P+1);border[-1]=1
        x,ok,it=correct(s,pred,border)
        err=float(max(abs(x[:-1]-xx[i,:-1]))*RS*1000)
        residual=float(max(abs(s.residual(x[:-1]*RS,D)))*1000)
        match=bool(ok and err<1e-5 and residual<2e-8)
        row=dict(new_state=files[i],old_state=oldfiles[j],D=D,scaled_initial_distance=float(dist[i]),
            tangent_alignment=alignment,fixed_D_correction_converged=ok,
            max_group_rate_difference_hz=err,corrected_residual_hz=residual,connection_verified=match)
        results.append(row);print(row,flush=True)
        if match:break
    write(DEST/a.output,dict(status='COMPLETE',connection_verified=any(r['connection_verified'] for r in results),rows=results,
        scope='Identity of complete rate-group equilibrium states at the same D, not proximity of global rates'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--new-branch',required=True)
    p.add_argument('--old-branch',action='append',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
