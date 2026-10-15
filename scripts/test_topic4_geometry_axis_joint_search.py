"""Checks that prevent budget, objective and paired-design mistakes; no simulation."""
from pathlib import Path
import sys, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_geometry_axis_joint_search as run


class Contracts(unittest.TestCase):
    def test_display_reindexes_patient_channels_by_name(self):
        from scripts.report_topic4_geometry_axis_joint_search import contact_reindex
        source=['SCL6','ICL1','SCL9'];target=['ICL1','SCL9','SCL6']
        np.testing.assert_array_equal(np.array([6,1,9])[contact_reindex(source,target)],[1,9,6])

    def test_pareto_preserves_negative_values_and_tradeoffs(self):
        a=[[-2,3],[-1,1],[0,0],[1,2],[-1,1]]
        self.assertEqual(run.pareto_indices(a),[0,1,2,4])

    def test_nomination_ignores_diagnostic_labels(self):
        rows=[dict(candidate=c,J=a,J_joint=b,TA_count=0,TB_error=900) for c,a,b in [('a',0,4),('b',4,0),('c',1,1),('d',5,5)]]
        norm=dict(center=[0,0],scale=[1,1])
        before=run.nominees(rows,norm)
        for r in rows:r.update(TA_count=1000,TB_error=-1000)
        self.assertEqual(before,run.nominees(rows,norm))
        self.assertEqual({r['candidate'] for r in before},{'a','b','c'})

    def test_unscorable_does_not_become_zero(self):
        rows=[dict(candidate='bad',J=None,J_joint=None),dict(candidate='ok',J=2,J_joint=2)]
        result=run.nominees(rows,dict(center=[0,0],scale=[1,1]))
        self.assertEqual([r['candidate'] for r in result],['ok'])

    def test_factorial_varies_only_three_prespecified_coordinates(self):
        p=run.rt.read(run.OUT/'plan.json');cs=run.candidates();anchor=run.old.vector(cs['g1_axis3_plus'])
        cells=p['interaction_cells'];self.assertEqual(len(cells),24)
        self.assertEqual(len({c['candidate'] for c in cells}),24)
        combinations=set()
        for cell in cells:
            c=cs[cell['candidate']];x=run.old.vector(c);run.validate(c,p)
            np.testing.assert_allclose(x[:3],anchor[:3],rtol=0,atol=1e-10)
            np.testing.assert_allclose(x[3:],[cell['right_y_mm'],cell['EE_core_to_out_scale'],cell['angle_offset_deg']],rtol=0,atol=1e-10)
            combinations.add(tuple(x[3:]))
        self.assertEqual(len(combinations),24)
        self.assertEqual(p['budget']['max_new_runs'],len(p['interaction_new_ids'])+3*8)
        self.assertEqual(p['budget']['additional_seeds'],0)

    def test_no_core_boundary_clipping_or_overlap_in_search_box(self):
        p=run.rt.read(run.OUT/'plan.json');lo=np.array(p['bounds']['lower']);hi=np.array(p['bounds']['upper'])
        from itertools import product
        radius=np.array(run.candidates()[p['reference_id']]['radii_mm'])
        for bits in product([0,1],repeat=6):
            self.assertTrue(run.feasible_vector(np.where(bits,hi,lo),lo,hi,radius))

    def test_all_historical_feedback_has_matching_objective_and_physics(self):
        p=run.rt.read(run.OUT/'plan.json');cs=run.candidates();reference=cs[p['reference_id']]
        rows=[r for r in run.scores() if r['candidate'] in p['historical_ids']]
        self.assertEqual(len(rows),184)
        for r in rows:
            self.assertEqual(r['objective_sha256'],p['objective']['sha256'])
            self.assertEqual(run.old.background(cs[r['candidate']]),run.old.background(reference))
            self.assertEqual(r['N'],r['joint']['N'])
            self.assertAlmostEqual(r['joint']['D_off'],r['joint']['A']-r['joint']['B'],places=10)
            np.testing.assert_allclose(r['vector'],run.old.vector(cs[r['candidate']]),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
