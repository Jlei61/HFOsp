"""Necessity control: preserve the carried M field, suppress only its update."""
from runner import *


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args()
    s=model();initial=BASE/'runs/A4_det_meandrive/checkpoints/t7000ms.npz';rows=[]
    for t,Z in fields(s,'native'):
        if t not in [9000,9420,9870]:continue
        rows.append(run_condition(s,Z,f'native_Z{t}_carried_fixedM',12000,a.device,initial,dynamic_m=False))
        write(OUT/'fixed_m_controls.json',dict(status='RUNNING',rows=rows))
    write(OUT/'fixed_m_controls.json',dict(status='COMPLETE',rows=rows))
