"""Identify phase-like directions in a candidate native-map spectrum.

The original model is advanced by two native steps. First- and second-order
forward chords are diagnostics of the observed orbit direction, not a claim
that the discrete map has an exact continuous-time neutral phase symmetry.
No Floquet or cycle classification follows from this audit alone.
"""
from cycle_monodromy import *


def run(args):
    cfg=read(args.source/'config.json');scfg=read(args.spectrum/'config.json')
    H=np.load(args.spectrum/'arnoldi.npz')['H'];n=H.shape[1]
    folder=args.spectrum/f'phase_audit_k{n:02d}';folder.mkdir(exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(args.source);coords=StateCoordinates(m)
    x=coords.pack(m);m.advance_step();y=coords.pack(m);m.advance_step();z=coords.pack(m)
    first=(y-x)/DT;second=(-3*x+4*y-z)/(2*DT)
    f=float(cp.linalg.norm(first).get());s=float(cp.linalg.norm(second).get())
    angle=float((cp.dot(first,second)/(f*s)).get())
    first/=f;second/=s
    coeff=np.empty(n);coeff_first=np.empty(n)
    for i in range(n):
        q=cp.asarray(np.load(Path(scfg['krylov_storage'])/f'q{i:03d}.npy'))
        coeff[i]=float(cp.dot(second,q).get());coeff_first[i]=float(cp.dot(first,q).get())
        del q
    vals,vec=np.linalg.eig(H[:n,:n]);order=np.argsort(-abs(vals));vals=vals[order];vec=vec[:,order]
    overlap=abs(coeff@vec)/np.linalg.norm(vec,axis=0)
    overlap_first=abs(coeff_first@vec)/np.linalg.norm(vec,axis=0)
    rows=[dict(eigenvalue=[v.real,v.imag],modulus=abs(v),ritz_residual=abs(H[n,n-1]*vec[-1,j]),
        overlap_with_second_order_phase_chord=overlap[j],overlap_with_first_order_phase_chord=overlap_first[j]) for j,v in enumerate(vals)]
    result=dict(status='COMPLETE_DIAGNOSTIC',dimension=n,D=cfg['D'],phase_chord_cosine_first_second=angle,
        relative_unrepresented_phase_chord_norm=np.sqrt(max(0.,1-coeff@coeff)),rows=rows,
        scope='Native-map orbit direction from time chords; an eigenvalue near one is not assumed or removed',
        candidate_monodromy_source=str(args.spectrum),source=str(args.source))
    write(folder/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--spectrum',type=Path,required=True);ap.add_argument('--device',type=int,default=0)
    run(ap.parse_args())
