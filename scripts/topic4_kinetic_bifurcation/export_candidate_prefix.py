"""Freeze a completed Arnoldi prefix solely as a shooting preconditioner."""
from cycle_monodromy import *


def run(args):
    cfg=read(args.source/'config.json');n=args.dimension
    data=read(args.source/'spectrum_progress.json');assert len(data['iterations'])>=n
    H=np.load(args.source/'arnoldi.npz')['H'][:n+1,:n].copy()
    assert H.shape==(n+1,n)
    folder=OUT/'candidate_monodromy'/args.label;folder.mkdir(parents=True,exist_ok=False)
    values,vectors=np.linalg.eig(H[:n,:n]);order=np.argsort(-abs(values));values=values[order];vectors=vectors[:,order]
    if not args.skip_mode_files:
        source=Path(cfg['source']);c=read(source/'config.json')
        m=AutonomousDensity(c['D'],c['degree'],c['voltage_dv'],args.device,basis_mode=c.get('basis_mode','legacy'))
        m.restore(source);t=NetworkTangent(m);coords=StateCoordinates(m)
        basis=[cp.asarray(np.load(Path(cfg['krylov_storage'])/f'q{i:03d}.npy')) for i in range(n)]
    for j in range(0 if args.skip_mode_files else min(3,n)):
        for part,cfs in [('real',vectors[:,j].real),('imag',vectors[:,j].imag)]:
            if np.linalg.norm(cfs)<1e-12:continue
            v=cp.zeros(coords.size)
            for q,a in zip(basis,cfs):v+=a*q
            coords.unpack(v,t,project=True)
            np.savez_compressed(folder/f'mode_{j:02d}_{part}.npz',**{k:cp.asnumpy(t.canonical(k)) for k in STATE_NAMES})
    row=data['iterations'][n-1]
    assert np.allclose(values.real,row['ritz_real']) and np.allclose(values.imag,row['ritz_imag'])
    write(folder/'config.json',dict(cfg,krylov_dimension=n,prefix_of=str(args.source.resolve()),
        usage='Approximate real-mode preconditioner only; all shooting steps must reduce the original full-map residual'))
    write(folder/'result.json',dict(status='PREFIX_EXPORTED_FOR_PRECONDITIONING',final=row,
        acceptance='NOT_A_CERTIFIED_SPECTRUM',mode_source=str(args.source.resolve())))
    np.savez_compressed(folder/'arnoldi.npz',H=H)
    print(folder,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--dimension',type=int,required=True);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--skip-mode-files',action='store_true',help='Save the fixed complete Arnoldi projection without optional eigenvector exports')
    run(ap.parse_args())
