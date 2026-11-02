"""Compare the expanded mean/variance expression with frozen training scores."""
from pathlib import Path
import csv
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_feature_map', HERE/'i_j_analysis.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frozen = json.loads((HERE/'i_j_frozen_spec.json').read_text())
with np.load(HERE/'i_j_frozen_parameters.npz') as z:
    parameters = dict(z)
rows = list(csv.DictReader((HERE/'i_original_angle_response.csv').open()))
records = {r['candidate']: r for r in json.loads((HERE/'i_provenance.json').read_text())['records']}
checks = []
with np.load(HERE/'i_events.npz') as events:
    for row in rows:
        cid = row['candidate']
        features = module.mapped_features(events[cid+'_centroid_ms'], parameters, frozen)
        components = []
        for g in module.GROUPS:
            phi = features[g]
            n = len(phi)
            mu_m = phi.mean(axis=0)
            mu_d = parameters[g+'_target']
            # This is the central moment with divisor N, not sample SD squared.
            v = np.sum((phi-mu_m)**2)/n
            expanded = (np.sum((mu_m-mu_d)**2)-v/(n-1))/frozen['scales'][g]
            # Independent sum/self-pair scorer frozen before this display edit.
            original = module.d_off(phi, mu_d)/frozen['scales'][g]
            np.testing.assert_allclose(expanded, original, rtol=0, atol=1e-12)
            components.append(expanded)
        j = np.sum(components)/3
        np.testing.assert_allclose(components, records[cid]['J_components'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(j, records[cid]['J'], rtol=0, atol=1e-12)
        checks.append({'candidate':cid, 'expanded_J':float(j), 'frozen_J':records[cid]['J']})

score = json.loads((HERE/'c_provenance/xy_left_20.json').read_text())
c_value = sum((v['A']-v['B'])/frozen['scales'][g] for g,v in score['groups'].items())/3
np.testing.assert_allclose(c_value, score['J'], rtol=0, atol=1e-12)
result = {'all_pass':True, 'independent_scans':checks, 'components_checked':len(checks)*3,
          'C_example_expanded_J':c_value, 'C_example_frozen_J':score['J'],
          'variance_definition':'v_g = mean_i ||phi_g(i)-mu_g^M||^2, divisor N',
          'patient_target':'Frozen FIT mean, no patient variance subtraction',
          'signed_distance':'No clipping or square root'}
(HERE/'expanded_formula_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'all_pass':True, 'components_checked':len(checks)*3,
                  'C_example_expanded_J':c_value}, indent=2))
