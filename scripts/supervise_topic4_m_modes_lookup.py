#!/usr/bin/env python3
"""Adopt existing M workers and accelerate only future dispatches, same40 jobs."""
from pathlib import Path
import sys
import supervise_topic4_m_modes_overnight as base


def main():
    qa=base.read(base.ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/scatter_lookup_qa/full_network_qa.json')
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical']
    wrapper=base.ROOT/'scripts/run_topic4_m_modes_scatter_lookup.py'
    original_popen=base.subprocess.Popen
    def dispatch(args,*positional,**kwargs):
        command=list(args)
        if str(base.RUNNER) in command and 'worker' in command:
            assert command[:3]==[sys.executable,'-u',str(base.RUNNER)]
            command.insert(2,str(wrapper))
        return original_popen(command,*positional,**kwargs)
    # base.discover still sees the original runner path in wrapped command lines.
    base.subprocess.Popen=dispatch
    base.main()


if __name__=='__main__':main()
