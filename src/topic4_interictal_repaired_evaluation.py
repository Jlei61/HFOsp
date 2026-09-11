"""Patient-only calibrated development diagnostics, without a model acceptance gate.

FIT/CAL/PROBE are disjoint blocks of previously reused development data. Neither
these labels nor novel metric formulas make this an independent clinical test.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from src.lagpat_rank_audit import build_masked_kmeans_features
from src.topic4_observation_repaired import order_error
from src.topic4_joint_xy_kernel import event_kernel_features, fit_kernel_maps, kernel_map
from src.topic4_xy_direction import onset_directions, direction_histogram, direction_distance


def validate(t):
    t = np.asarray(t, float)
    if t.ndim != 2 or np.isinf(t).any():
        raise ValueError('finite-or-missing events x contacts required')
    return t


def rank_features(t):
    t = validate(t)
    mask = np.isfinite(t)
    ranks = np.zeros_like(t)
    for i, row in enumerate(t):
        ranks[i, mask[i]] = rankdata(row[mask[i]], method='average') - 1
    # Canonical routine re-ranks only participating contacts; never phantom ranks.
    return build_masked_kmeans_features(ranks.T, mask.T, impute='event_median')


def span(t):
    t = validate(t); good = np.isfinite(t).sum(1) >= 2
    return np.nanmax(t[good], axis=1) - np.nanmin(t[good], axis=1)


class RepairedEvaluator:
    def __init__(self, patient, blocks, xy, groups, *, seed=2026090619):
        self.patient = validate(patient); self.blocks = np.asarray(blocks)
        self.xy, self.groups = np.asarray(xy), groups
        rng = np.random.default_rng(seed)
        unique = np.unique(blocks); rng.shuffle(unique)
        self.partition = {'FIT': unique[:len(unique)//2],
                          'CAL': unique[len(unique)//2:3*len(unique)//4],
                          'PROBE': unique[3*len(unique)//4:]}
        self.index = {k: np.flatnonzero(np.isin(blocks, v)) for k, v in self.partition.items()}
        fit, cal = (self.patient[self.index[k]] for k in ('FIT', 'CAL'))
        first = np.nanmin(fit, axis=1)[:, None]
        lag = fit - first
        self.scale = float(np.median(lag[lag > 0]))
        xf = rank_features(fit); xp = rank_features(self.patient[self.index['PROBE']])
        # Discovery K is not fixed to two. Reproducibility assessed on the SAME
        # probe events under independently fitted FIT-block halves.
        halves = np.array_split(self.partition['FIT'], 2)
        xx = [rank_features(self.patient[np.isin(blocks, b)]) for b in halves]
        self.k_scan = []
        models = {}
        for k in range(2, 6):
            km = KMeans(k, n_init=10, random_state=seed).fit(xf)
            labels = km.labels_; fractions = np.bincount(labels, minlength=k)/len(labels)
            probes = [KMeans(k, n_init=10, random_state=seed+j+1).fit(x).predict(xp) for j, x in enumerate(xx)]
            ami = float(adjusted_mutual_info_score(*probes))
            sil = float(silhouette_score(xf, labels, sample_size=min(2000, len(xf)), random_state=seed))
            self.k_scan.append({'k': k, 'silhouette': sil, 'cross_block_probe_AMI': ami,
                                'minimum_fit_fraction': float(fractions.min()),
                                'eligible': ami >= .8 and fractions.min() >= .05})
            models[k] = km
        eligible = [r for r in self.k_scan if r['eligible']]
        # A failure to identify stable modes does not silently invent two modes.
        self.modes_stable = bool(eligible)
        self.k = max(eligible, key=lambda r: (r['silhouette'], -r['k']))['k'] if eligible else 1
        self.km = models[self.k] if eligible else KMeans(1, n_init=1, random_state=seed).fit(xf)
        self.fit = fit; self.cal = cal
        features = event_kernel_features(fit, self.xy, groups, self.scale)
        self.maps = fit_kernel_maps(features, seed=seed, n_fourier=512)
        self.means = {k: kernel_map(v, self.maps[k]).mean(0, dtype=float) for k, v in features.items()}
        self.fit_joint = features['joint']
        self.fit_labels = self.km.predict(xf)
        self.trees = [cKDTree(self.fit_joint[self.fit_labels == m]) for m in range(self.k)]
        self.cal_labels = self.km.predict(rank_features(cal))
        caljoint = event_kernel_features(cal, self.xy, groups, self.scale)['joint']
        self.radii = []
        for m in range(self.k):
            distance = self.trees[m].query(caljoint[self.cal_labels == m], k=5)[0].mean(1)
            if len(distance) < 50:
                raise RuntimeError('insufficient CAL support for a mode')
            self.radii.append(np.quantile(distance, [.90, .99]))
        self.radii = np.asarray(self.radii)
        self.reference_direction = direction_histogram(onset_directions(fit, self.xy))
        self.cache_probe = None

    def classify(self, t):
        t = validate(t)
        if t.shape[1] != self.patient.shape[1]:
            raise ValueError('contact count mismatch')
        readable = np.isfinite(t).sum(1) >= 2
        labels = np.full(len(t), -1, int); state = np.zeros(len(t), int)
        distances = np.full(len(t), np.nan)
        if readable.any():
            ix = np.flatnonzero(readable)
            labels[ix] = self.km.predict(rank_features(t[ix]))
            x = event_kernel_features(t[ix], self.xy, self.groups, self.scale)['joint']
            for m in range(self.k):
                local = np.flatnonzero(labels[ix] == m); selected = ix[local]
                d = self.trees[m].query(x[local], k=5)[0].mean(1)
                distances[selected] = d
                state[selected[d <= self.radii[m, 0]]] = 1
                state[selected[d > self.radii[m, 1]]] = -1
        return labels, state, distances

    def metrics(self, t, seeds=None, *, detail=True):
        t = validate(t); n = len(t)
        seeds = np.zeros(n, int) if seeds is None else np.asarray(seeds)
        if seeds.shape != (n,):
            raise ValueError('one network ID per event required')
        labels, state, distance = self.classify(t)
        result = {'n_events': n, 'status': 'ESTIMABLE_DEVELOPMENT' if n else 'NOT_ESTIMABLE_NO_EVENTS',
                  'supported_fraction': float(np.mean(state == 1)) if n else None,
                  'unsupported_fraction': float(np.mean(state == -1)) if n else None,
                  'indeterminate_fraction': float(np.mean(state == 0)) if n else None,
                  'n_unreadable': int(np.sum(labels < 0)), 'final_acceptance': None}
        if not n:
            return {**result, 'modes': [], 'mode_presence_fraction': None, 'joint_distance': None}
        features = event_kernel_features(t, self.xy, self.groups, self.scale)
        result['kernel_distances'] = {k: float(np.sum((kernel_map(v, self.maps[k]).mean(0, dtype=float)-self.means[k])**2)) for k, v in features.items()}
        result['joint_distance'] = result['kernel_distances']['joint']
        result['direction_distance'] = direction_distance(direction_histogram(onset_directions(t, self.xy)), self.reference_direction)
        probe = self.patient[self.index['PROBE']]
        if self.cache_probe is None:
            pl, ps, _ = self.classify(probe)
            pf = event_kernel_features(probe, self.xy, self.groups, self.scale)['joint']
            self.cache_probe = (pl, ps, pf)
        pl, ps, pf = self.cache_probe
        modes = []
        for m in range(self.k):
            ids = (labels == m) & (state == 1)
            all_ids = labels == m
            unit_counts = {str(s): int(np.sum(ids & (seeds == s))) for s in np.unique(seeds)}
            # Operational detection is displayed, not promoted to calibrated equivalence.
            observed = int(ids.sum()) >= 3 and sum(v > 0 for v in unit_counts.values()) >= 2
            ref = self.fit[self.fit_labels == m]; current = t[all_ids]
            a, b = span(current), span(ref)
            row = {'mode': m, 'n_assigned': int(all_ids.sum()), 'n_supported': int(ids.sum()),
                   'supported_by_seed': unit_counts, 'operationally_observed': observed if len(np.unique(seeds))>=2 else None,
                   'mixture_fraction_all_events': float(all_ids.sum()/n),
                   'patient_fit_fraction': float(np.mean(self.fit_labels == m)),
                   'span_median_ms': float(np.median(a)) if len(a) else None,
                   'span_iqr_ms': float(np.subtract(*np.quantile(a, [.75,.25]))) if len(a) else None,
                   'patient_span_iqr_ms': float(np.subtract(*np.quantile(b, [.75,.25]))),
                   'span_wasserstein_ms': float(wasserstein_distance(a,b)) if len(a) else None,
                   'participation_mae': float(np.abs(np.isfinite(current).mean(0)-np.isfinite(ref).mean(0)).mean()) if len(current) else None}
            q = (pl == m) & (ps == 1)
            if ids.any() and q.any():
                xq = pf[q]
                # One-way query coverage depends on model N; matched-N self baselines
                # are calculated after real event counts are known, not larger-N bins.
                d = cKDTree(features['joint'][ids]).query(xq)[0]
                row['patient_probe_neighborhood_coverage'] = float(np.mean(d <= self.radii[m, 0]))
            else:
                row['patient_probe_neighborhood_coverage'] = 0. if q.any() else None
            if detail and len(current):
                pairs = []
                for i in range(t.shape[1]):
                    for j in range(i+1, t.shape[1]):
                        d1 = current[:,j]-current[:,i]; d1=d1[np.isfinite(d1)]
                        d2 = ref[:,j]-ref[:,i]; d2=d2[np.isfinite(d2)]
                        if len(d1) >= 5 and len(d2) >= 5:
                            pairs.append({'i':i,'j':j,'n_model':len(d1),
                                'signed_median_model_ms':float(np.median(d1)),
                                'signed_median_patient_ms':float(np.median(d2)),
                                'order_TV_three_states_at_2ms':order_error(d1,d2,2.),
                                'signed_lag_wasserstein_ms':float(wasserstein_distance(d1,d2))})
                row['pair_diagnostics'] = pairs
                row['order_TV_at_2ms'] = float(np.mean([v['order_TV_three_states_at_2ms'] for v in pairs])) if pairs else None
                row['signed_lag_wasserstein_ms'] = float(np.mean([v['signed_lag_wasserstein_ms'] for v in pairs])) if pairs else None
                row['rank_variation'] = float(np.var(rank_features(current), axis=0).mean())
                row['direction_distance'] = direction_distance(direction_histogram(onset_directions(current,self.xy)),direction_histogram(onset_directions(ref,self.xy)))
            modes.append(row)
        result['modes'] = modes
        result['mode_presence_fraction'] = float(np.mean([r['operationally_observed'] for r in modes])) if self.modes_stable and len(np.unique(seeds))>=2 else None
        result['mode_presence_status'] = 'ESTIMABLE' if len(np.unique(seeds))>=2 else 'NOT_ESTIMABLE_SINGLE_NETWORK'
        return result

    def manifest(self):
        return {'role': 'REPAIRED_DEVELOPMENT_EVALUATION_NO_FINAL_MODEL_GATE',
                'partition_blocks': {k:v.tolist() for k,v in self.partition.items()},
                'partition_event_counts': {k:len(v) for k,v in self.index.items()},
                'K': self.k, 'stable_mode_discovery_pass': self.modes_stable,
                'k_scan':self.k_scan,'fit_time_scale_ms':self.scale,
                'support_radii_q90_q99':self.radii.tolist(),
                'support_rule':'mean distance to 5 FIT neighbors within assigned rank mode; <= CAL q90 supported; >q99 unsupported; otherwise indeterminate',
                'mode_detection_rule':'at least 3 supported events on at least 2 networks; operational diagnostic, not an equivalence test',
                'tie_rule':'2 ms resolution sensitivity, not established physiological timing uncertainty',
                'independence':'All blocks previously used for development; contact XY and historical patient axis are inherited upstream. No untouched-patient validation claim.',
                'final_model_acceptance_rule':None}
