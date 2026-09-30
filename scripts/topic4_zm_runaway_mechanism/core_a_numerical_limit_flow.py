"""Use the existing passive flow recorder with a declared integrator/history.

The initialization uses identical bin-integrated physical history at all
meshes. No response, graph, forcing or resource parameter is fitted.
"""
from common import OUT, read, write
from onset_exponential_midpoint import ExponentialMidpointEngine
from check_spatial_midpoint_convergence import conservative_history
from onset_state_continuation import build as original_build
import core_a_candidate_flow_profile as recorder
from pathlib import Path
import argparse


def main(a):
    numerical=OUT/'core_a_bifurcation_type_20260924/numerical_checks/exponential_midpoint'
    assert read(numerical/'spatial_identity.json')['status']=='PASS'
    assert read(numerical/'short_spatial_convergence/analysis.json')['status']=='SHORT_WINDOW_ORDER_AND_COMMON_LIMIT_PASS'
    out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'numerical_contract.json').exists()
    write(out/'numerical_contract.json',dict(method=a.method,dt_ms=a.dt,source_dt_ms=a.source_dt,
          source=str(Path(a.source).resolve()),duration_ms=a.duration,
          common_initial_history='Original source values are flux-bin averages. Subdivide each parent bin conservatively as a common piecewise-constant history, preserving every original integral and initial refractory occupancy. All continuous fast/M states unchanged.',
          physical_model='Locked3479group spatial renewalrate, originalgraph/thresholds/physicaldelays/privateQ/conditioned39+locked transientcorrection. AllZheld, allE Mdynamic, constantoriginalinput.',
          numerical_checks=str(numerical),
          interpretation='Longer finite-time numerical control. A120ms convergence pass does not certify a long-time attractor, a critical parameter or a bifurcation type. Old tangent/Floquet operators cannot be applied to the new integrator.',model_promoted=False))
    def build(device,dt):
        if a.method=='old_endpoint':return original_build(device,dt)
        e=ExponentialMidpointEngine(dt=dt,device=device);e.graph();return e
    def history(state,e,source_dt):
        base,qa=conservative_history(state,e,source_dt)
        write(out/'initial_history_qa.json',qa);return base
    recorder.build=build;recorder.regrid_state=history
    recorder.main(a)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--method',choices=['old_endpoint','exponential_midpoint'],required=True)
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--source-dt',type=float,default=.05)
    p.add_argument('--duration',type=int,default=5000);p.add_argument('--device',type=int,default=0)
    p.add_argument('--target-native-time',type=float);main(p.parse_args())
