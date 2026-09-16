#!/usr/bin/env python3
"""Route accepted diagnostic producers to V4 without editing historical sources."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    kind=sys.argv.pop(1)
    from scripts import run_topic4_joint_xy_component_search as current
    if kind=='figure':
        from scripts.paper_figures import plot_topic4_joint_replicated_expanded as producer
        producer.run.OUT=current.OUT;producer.run.CONFIG=current.CONFIG
    elif kind=='raw':
        from scripts import audit_topic4_xy_raw_propagation_video as producer
        old=json.loads((ROOT/'config/topic4_joint_xy_kernel_v3.json').read_text())
        new=json.loads(current.CONFIG.read_text())
        if old['observation']!=new['observation']:
            raise RuntimeError('raw producer observation is no longer shared')
        producer.SEARCH=current.OUT
    else:
        raise ValueError(kind)
    producer.main()


if __name__=='__main__':main()
