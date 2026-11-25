"""
FACE (Feasible Actionable Counterfactual Explanations) explainer.

Retrieves counterfactuals via shortest-path search on a density-weighted
k-NN graph of the training set. CFs are real training patients, so no
plausibility snapping is needed beyond immutability enforcement.

Retrieval is constrained on three tiers: immutable features are enforced
after retrieval, labs are near-fixed via an aggregate distance budget
expressed in tolerance widths, and treatment variables are free but at
least one discrete treatment decision must differ. The search target is
the query patient's own prediction plus a margin rather than the cohort
median, so a retrieved case is always better than this patient rather
than merely better than average.

Lab distance is aggregated across labs rather than applied as an
independent constraint per lab. A per-lab conjunction empties the
candidate pool on a cohort this size. Direction priors are redundant
once labs are held near-fixed and are skipped in that configuration.

Candidates are ranked rather than argmin-selected, and the improvement
is re-verified after immutability enforcement. Enforcement replaces the
neighbour's immutable values with the query's, so the row that is
finally predicted is not the row that was scored during graph search,
and a candidate that beat the threshold as a whole patient can fail to
beat it as a hybrid.

When no candidate falls inside the widest budget, the search falls back
to the lab-closest candidates, but refuses entirely beyond
max_fallback_distance. A patient with no comparable bloodwork in the
training set yields no counterfactual rather than one drawn from a
clinically unrelated case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from xai_decision_making.explainability.base import (
    AggregatedSanityResult, CounterfactualExplainer, ExplainerModality,
    ExplainerRegistry, Explanation, ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks, compute_cf_confidence, compute_content_deltas,
    direction_compatible_mask, enforce_immutables, extract_scalar_prediction,
    resolve_direction_priors, risk_predict_fn, snap_categoricals,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config,
)
from xai_decision_making.explainability.counterfactual.feature_policy import (
    CF_FREE_FLAGS, LAB_TOLERANCE, is_sentinel,
)

registry = ExplainerRegistry()

LAB_BUDGETS = (1.0, 1.5, 2.0, 3.0)
FALLBACK_N = 5


@registry.register
class FACEExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_neighbours=10, density_bandwidth=1.0,
                 distance_metric="euclidean", density_threshold=0.0, immutable=None,
                 max_graph_size=2000, time_horizon=None, direction_priors=None,
                 lock_labs=True, require_treatment_change=True, rel_margin=0.5,
                 sparsity_weight=0.5, hide_sentinel_deltas=True, verify_top_n=10,
                 max_fallback_distance=5.0, random_state=1, rel_floor=0.25):
        super().__init__(model_adapter)
        self._k = n_neighbours
        self._bandwidth = density_bandwidth
        self._metric = distance_metric
        self._density_threshold = density_threshold
        self._max_graph_size = max_graph_size
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)

        self._lock_labs = lock_labs
        self._require_treatment_change = require_treatment_change
        self._rel_margin = rel_margin
        self._sparsity_weight = sparsity_weight
        self._hide_sentinel_deltas = hide_sentinel_deltas
        self._verify_top_n = verify_top_n
        self._max_fallback_distance = max_fallback_distance
        self._rel_margin = rel_margin
        self._rel_floor = rel_floor
        self._random_state = random_state

        self._graph = self._graph_data = self._graph_data_raw = None
        self._graph_clinical = None
        self._graph_risks = self._risk_median = self._risk_std = self._scaler = None

        self._last_relaxation = None
        self._last_n_candidates = None
        self._last_threshold = None
        self._last_null_reason = None
        self._last_fallback = False
        self._last_verified = False
        self._last_rank_tried = 0

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _build_graph(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        data_raw = x_train[feat_names].fillna(0).values.astype(np.float64)
        if len(data_raw) > self._max_graph_size:
            rng = np.random.default_rng(self._random_state)
            sel = rng.choice(len(data_raw), self._max_graph_size, replace=False)
            data_raw = data_raw[sel]
        n = len(data_raw)

        self._scaler = MinMaxScaler()
        lo = np.nanpercentile(data_raw, 1, axis=0)
        hi = np.nanpercentile(data_raw, 99, axis=0)
        self._scaler.fit(np.clip(data_raw, lo, hi))
        data_norm = self._scaler.transform(data_raw)

        dist_matrix = cdist(data_norm, data_norm, metric=self._metric)
        kde = KernelDensity(bandwidth=self._bandwidth, kernel="gaussian")
        kde.fit(data_norm)
        densities = np.exp(kde.score_samples(data_norm))

        rows, cols, weights = [], [], []
        for i in range(n):
            neighbours = np.argsort(dist_matrix[i])[1:self._k + 1]
            for j in neighbours:
                min_density = max(min(densities[i], densities[j]), 1e-12)
                if min_density >= self._density_threshold:
                    w = dist_matrix[i, j] / min_density
                    rows.extend([i, j])
                    cols.extend([j, i])
                    weights.extend([w, w])

        self._graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        self._graph_data = data_norm
        self._graph_data_raw = data_raw
        self._graph_clinical = self._to_clinical(data_raw, feat_names)
        self._graph_risks = self._predict_fn(data_raw)
        self._risk_median = float(np.median(self._graph_risks))
        self._risk_std = float(np.std(self._graph_risks))

    def _to_clinical(self, arr, feat_names):
        df = pd.DataFrame(np.atleast_2d(arr), columns=feat_names)
        try:
            out = self._adapter.inverse_transform_numeric(df)
        except Exception:
            out = df
        return out.values.astype(np.float64)

    def _lab_distance(self, x_query_clin, feat_names):
        dist = np.zeros(len(self._graph_clinical))
        n_used = 0
        for j, f in enumerate(feat_names):
            spec = LAB_TOLERANCE.get(f)
            if spec is None:
                continue
            q = float(x_query_clin[j])
            if is_sentinel(f, q):
                continue
            kind, tol = spec
            col = self._graph_clinical[:, j]
            if kind == "abs":
                dist += np.abs(col - q) / tol
            else:
                dist += np.abs(
                    np.log10(np.maximum(col, 1e-6)) - np.log10(max(q, 1e-6))
                ) / tol
            n_used += 1
        return dist / max(n_used, 1)

    def _treatment_differs_mask(self, x_query_raw, feat_names):
        if not self._require_treatment_change:
            return np.ones(len(self._graph_data_raw), dtype=bool)
        idx = [j for j, f in enumerate(feat_names) if f in CF_FREE_FLAGS]
        if not idx:
            return np.ones(len(self._graph_data_raw), dtype=bool)
        return ~np.all(
            np.isclose(self._graph_data_raw[:, idx], x_query_raw[idx]), axis=1
        )

    def _rank_targets(self, reachable, sp_dists, x_query_raw, feat_names):
        dists = sp_dists[reachable]
        if self._sparsity_weight <= 0 or len(reachable) == 1:
            return reachable[np.argsort(dists)]

        mutable_idx = [j for j, f in enumerate(feat_names) if f not in self._immutable]
        if not mutable_idx:
            return reachable[np.argsort(dists)]

        changed = np.array([
            int(np.sum(~np.isclose(
                self._graph_data_raw[t][mutable_idx], x_query_raw[mutable_idx]
            )))
            for t in reachable
        ], dtype=np.float64)

        d_max = dists.max() if dists.max() > 0 else 1.0
        score = dists / d_max + self._sparsity_weight * changed / len(mutable_idx)
        return reachable[np.argsort(score)]

    def _build_cf_row(self, index, query_row, feat_names):
        cf_df = pd.DataFrame([self._graph_data_raw[index]], columns=feat_names)
        cf_df = snap_categoricals(cf_df, self._categorical_values)
        return enforce_immutables(query_row, cf_df, self._immutable)

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._graph is None:
            self._build_graph()

        self._last_relaxation = None
        self._last_n_candidates = 0
        self._last_null_reason = None
        self._last_fallback = False
        self._last_verified = False
        self._last_rank_tried = 0

        feat_names = self._adapter.feature_names()
        x_query_raw = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        x_query_norm = self._scaler.transform(x_query_raw.reshape(1, -1))[0]
        x_query_clin = self._to_clinical(x_query_raw, feat_names)[0]

        orig = float(self._predict_fn(x_query_raw.reshape(1, -1))[0])
        margin = self._rel_margin * (self._risk_std or 0.0)
        if target_outcome is not None:
            threshold = float(target_outcome)
        elif self._higher_is_better:
            threshold = min(orig + margin, orig * (1.0 + self._rel_floor))
        else:
            threshold = max(orig - margin, orig * (1.0 - self._rel_floor))
        self._last_threshold = threshold

        pool = (self._graph_risks > threshold) if self._higher_is_better \
            else (self._graph_risks < threshold)
        if not pool.any():
            self._last_null_reason = "no_candidate_beats_patient"
            return None

        if self._direction_priors and not self._lock_labs:
            pool = pool & direction_compatible_mask(
                self._graph_data_raw, x_query_raw, feat_names, self._direction_priors
            )
            if not pool.any():
                self._last_null_reason = "direction_priors_exhausted"
                return None

        pool = pool & self._treatment_differs_mask(x_query_raw, feat_names)
        if not pool.any():
            self._last_null_reason = "no_candidate_with_different_treatment"
            return None

        target_mask = None
        if self._lock_labs:
            lab_dist = self._lab_distance(x_query_clin, feat_names)
            for budget in LAB_BUDGETS:
                cand = pool & (lab_dist <= budget)
                if cand.any():
                    target_mask = cand
                    self._last_relaxation = budget
                    break
            if target_mask is None:
                idx = np.where(pool)[0]
                keep = idx[np.argsort(lab_dist[idx])[:FALLBACK_N]]
                if float(lab_dist[keep].min()) > self._max_fallback_distance:
                    self._last_relaxation = float(lab_dist[keep].min())
                    self._last_null_reason = "no_comparable_labs"
                    return None
                target_mask = np.zeros_like(pool)
                target_mask[keep] = True
                self._last_relaxation = float(lab_dist[keep].max())
                self._last_fallback = True
        else:
            target_mask = pool

        entry_node = int(np.argmin(
            cdist(x_query_norm.reshape(1, -1), self._graph_data, metric=self._metric)[0]
        ))
        sp_dists = shortest_path(self._graph, directed=False, indices=entry_node)
        reachable = np.where(target_mask & np.isfinite(sp_dists))[0]
        if len(reachable) == 0:
            self._last_null_reason = "no_reachable_candidate"
            return None
        self._last_n_candidates = int(len(reachable))

        ranked = self._rank_targets(reachable, sp_dists, x_query_raw, feat_names)
        query_row = patient_features[feat_names].head(1)
        orig_check = extract_scalar_prediction(
            self._adapter.predict(query_row.fillna(0), time_horizon=self._time_horizon)
        )

        for rank, t in enumerate(ranked[:self._verify_top_n], start=1):
            cf_df = self._build_cf_row(int(t), query_row, feat_names)
            cf_check = extract_scalar_prediction(
                self._adapter.predict(cf_df.head(1), time_horizon=self._time_horizon)
            )
            self._last_rank_tried = rank
            if orig_check is None or cf_check is None:
                return cf_df
            improved = (cf_check > orig_check) if self._higher_is_better \
                else (cf_check < orig_check)
            if improved:
                self._last_verified = True
                return cf_df

        self._last_null_reason = "no_candidate_improves_after_immutables"
        return None

    def _strip_sentinel_deltas(self, content, raw_query, raw_cf=None):
        if not self._hide_sentinel_deltas or not content or not raw_query:
            return content, []
        raw_cf = raw_cf or {}
        hidden = [k for k in content
                  if is_sentinel(k, raw_query.get(k, 0))
                  or is_sentinel(k, raw_cf.get(k, 0))]
        if not hidden:
            return content, []
        return {k: v for k, v in content.items() if k not in hidden}, sorted(hidden)

    def explain(self, patient_features, patient_id=None):
        feat_names = self._adapter.feature_names()
        query = patient_features[feat_names].fillna(0).head(1)
        cf = self.generate_counterfactual(patient_features)

        if cf is not None:
            content, raw_query, raw_cf = compute_content_deltas(
                self._adapter, query, cf, feat_names
            )
        else:
            content, raw_query, raw_cf = {}, None, None

        content, hidden_sentinels = self._strip_sentinel_deltas(content, raw_query, raw_cf)

        orig_pred = extract_scalar_prediction(
            self._adapter.predict(query, time_horizon=self._time_horizon)
        )
        cf_pred = extract_scalar_prediction(
            self._adapter.predict(cf.head(1), time_horizon=self._time_horizon)
        ) if cf is not None else None

        confidence_metrics = compute_cf_confidence(
            raw_query, raw_cf, orig_pred, cf_pred,
            higher_is_better=self._higher_is_better,
        )
        if not isinstance(confidence_metrics, dict):
            confidence_metrics = {}
        if orig_pred is not None and cf_pred is not None:
            confidence_metrics["improves_on_patient"] = bool(
                (cf_pred > orig_pred) if self._higher_is_better else (cf_pred < orig_pred)
            )
        confidence_metrics["verified_improvement"] = self._last_verified
        confidence_metrics["rank_selected"] = self._last_rank_tried
        confidence_metrics["tolerance_scale"] = self._last_relaxation
        confidence_metrics["fallback_used"] = self._last_fallback
        confidence_metrics["n_candidates"] = self._last_n_candidates
        confidence_metrics["target_threshold"] = self._last_threshold
        confidence_metrics["null_reason"] = self._last_null_reason
        confidence_metrics["labs_locked"] = self._lock_labs
        confidence_metrics["treatment_change_required"] = self._require_treatment_change
        confidence_metrics["hidden_sentinel_features"] = hidden_sentinels
        if content:
            confidence_metrics["treatment_features_changed"] = sorted(
                set(content) & CF_FREE_FLAGS
            )

        return self._make_explanation(
            patient_id, content=content, method="FACE",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf,
            higher_is_better=self._higher_is_better,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)