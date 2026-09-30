"""Check the non-integer-millisecond graph block against ordinary steps."""
from periodic_flow_closure import *


def main():
    s=model();attach_native_path(s);s.set_D(.2193352638617445)
    rates=np.full(s.P,.001);initial=s.equilibrium_state(rates);rows=[]
    for dt in [.05,.025,.0125]:
        depth=round(s.delays[-1]/dt)+1
        history=np.broadcast_to(rates,(depth,s.P)).copy()
        history*=1+.01*np.sin(np.arange(depth)[:,None]*.013)
        e=EndpointIntegrator(s,dt=dt,initial=initial,history=history,dynamic_z=False,device=1)
        ordinary=EndpointIntegrator(s,dt=dt,initial=initial,history=history,dynamic_z=False,device=1)
        block=StepBlock(e,128);block.run()
        for _ in range(128):ordinary.step()
        expected=np.roll(ordinary.history.get(),-ordinary.tick%ordinary.depth,axis=0)
        state_equal=np.array_equal(e.y.get(),ordinary.y.get())
        history_equal=np.array_equal(e.history.get(),expected)
        assert state_equal and history_equal
        rows.append(dict(dt_ms=dt,state_bitwise_equal=state_equal,canonical_history_bitwise_equal=history_equal))
        del e,ordinary,block;gc.collect()
    write(OUT/'periodic_flow_block_check.json',dict(status='PASS',rows=rows))
    print(rows)


if __name__=='__main__':main()
