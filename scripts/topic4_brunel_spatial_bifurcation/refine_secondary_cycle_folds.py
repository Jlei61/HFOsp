"""Check every secondary turning candidate in the already traced Hopf families."""
from rate_periodic import *
import subprocess

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--N',type=int,default=128);p.add_argument('--from-mesh',type=int);a=p.parse_args()
 jobs=[('arcA',74,'A','LPC_A2'),('arcA',76,'A','LPC_A3'),('arcA',87,'B','LPC_A4'),('arcA',106,'A','LPC_A5')]
 jobs += [('arcB',i,'B' if i==102 else 'A',f'LPC_B{j+2}') for j,i in enumerate([102,110,116,126,130,142,150])]
 rows=[]
 for family,i,core,label in jobs:
  dest=PERIODIC_OUT/f'{label}_N{a.N}.json'
  args=[sys.executable,str(Path(__file__).with_name('rate_cycle_fold_mean.py')),
    str(PERIODIC_OUT/f'orbits/{family}_{i-1:04d}_N64.npz'),str(PERIODIC_OUT/f'orbits/{family}_{i+1:04d}_N64.npz'),
    '--core',core,'--label',label,'--N',str(a.N),'--device',str(a.device)]
  if a.from_mesh:
   prior=PERIODIC_OUT/f'{label}_N{a.from_mesh}.json'
   if not prior.exists():
    print('SKIP missing previous refinement',label,flush=True);continue
   q=read(prior);args[2:4]=[q['orbit'],q['orbit']];args+=['--radius','.001']
  log=PERIODIC_OUT/f'{label}_N{a.N}.log'
  with log.open('w') as f:done=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  q=dict(label=label,family=family,turn_index=i,returncode=done.returncode,result=str(dest) if dest.exists() else None,log=str(log))
  rows.append(q);write(PERIODIC_OUT/f'secondary_fold_batch_N{a.N}.json',dict(rows=rows,status='RUNNING' if len(rows)<len(jobs) else 'FINISHED'))
  print(q,flush=True)

if __name__=='__main__':main()
