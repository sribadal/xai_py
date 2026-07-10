"""
Shared utilities for attribution-based explainers.

Provides the risk prediction wrapper, SHAP-to-dict conversion,
sanity check runner, and TreeExplainer probe used across all
attribution methods.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    ExplainerModality,
)
from xai_decision_making.explainability.sanity_checks import (
    ATTRIBUTION_CHECKS,
    DeletionFaithfulnessCheck,
)


def _risk_predict_fn(adapter, time_horizon=None):
    col_names = adapter.feature_names()

    if time_horizon is not None:
        def _predict_at_t(X):
            df = pd.DataFrame(X, columns=col_names)
            return adapter.predict_at_horizon(df, time_horizon)
        return _predict_at_t

    def _predict(X):
        scores = []
        for _, row in pd.DataFrame(X, columns=col_names).iterrows():
            patient = row.to_frame().T.reset_index(drop=True)
            try:
                scores.append(adapter.predict(patient)["risk_score"])
            except Exception:
                scores.append(np.nan)
        return np.array(scores)

    return _predict

def _attrs_from_shap_values(shap_values, feature_names):
    return dict(zip(feature_names, np.asarray(shap_values).ravel()))

def _run_sanity_checks(explainer, held_out_df, checks):
    n = len(held_out_df)
    checks = [dataclasses.replace(c, max_patients=n) for c in checks]
    results = [check.run(explainer, held_out_df) for check in checks]
    agg = AggregatedSanityResult(
        modality=ExplainerModality.ATTRIBUTION,
        target=explainer._adapter.target,
        explainer_name=type(explainer).__name__,
        check_results=results,
    )
    agg.print_summary()
    return agg

def _try_tree_explainer(model):
    try:
        import shap
        return shap.TreeExplainer(model)
    except Exception as exc:
        return None