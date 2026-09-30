"""Render exact-pair readouts only after the active physical solve succeeds."""
from common import *
import argparse,os,subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--after-pid',type=int,required=True)
    parser.add_argument('--pair-source',type=Path)
    parser.add_argument('--output-prefix',default='sameJ_burst')
    args=parser.parse_args()
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/Bleading_extension')
    result=args.pair_source or folder/'sameJ_pair.json'
    if args.pair_source is not None:assert args.output_prefix!='sameJ_burst'
    worker=folder/((result.stem+'_delivery_worker.json') if args.pair_source else 'sameJ_delivery_worker.json')
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    proc=Path(f'/proc/{args.after_pid}/cmdline')
    identity=proc.read_bytes() if proc.exists() else b''
    if identity:
        assert b'solve_rate_sameJ_burst_pair.py' in identity
        status('WAITING_SAME_J_SOLVE',dependency=args.after_pid)
        while proc.exists():
            try:active=proc.read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:break
            time.sleep(20)
    if not result.exists():
        status('SAME_J_RESULT_NOT_AVAILABLE',dependency=args.after_pid);return
    q=read(result)
    if q['status']!='TWO_PHYSICAL_SAME_J_PERIODIC_SOLUTIONS':
        status('SAME_J_PHYSICAL_CHECKS_PENDING',result_status=q['status']);return
    status('COMPUTING_CONTACT_STATISTICS_AND_RENDERING')
    code=subprocess.call([sys.executable,'-u',str(Path(__file__).with_name('plot_rate_sameJ_burst_pair.py')),
                          '--pair-source',str(result),'--output-prefix',args.output_prefix])
    status('READOUT_FIGURES_GENERATED' if code==0 else 'READOUT_PRODUCTION_FAILED',
        returncode=code,human_visual_acceptance='PENDING')
    if code:raise SystemExit(code)


if __name__=='__main__':main()
