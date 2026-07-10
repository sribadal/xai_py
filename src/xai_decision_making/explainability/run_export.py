"""
Export per-patient explanations across all modalities to JSON. Loads an
RSF or ConDOR adapter, optionally filters to the cross-model overlap
cohort, stratifies ConDOR patients by baseline QoL, then runs attribution,
counterfactual, case-based, and rule-based explainers at baseline and
each requested time horizon. Supports resume, shard-parallel execution,
and selective recomputation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xai_decision_making.explainability.adapters import (
    HORIZON_LABELS,
    QOL_HORIZON_LABELS,
    load_rsf_adapter,
    load_condor_adapter,
)

from xai_decision_making.explainability.attribution import (
    SurvLIMEExplainer,
    PermutationImportanceExplainer,
    MinimalDepthExplainer,
    IntegratedGradientsExplainer,
    DeepLiftExplainer,
    DeepLiftSHAPExplainer,
    TreeSHAPExplainer,
)

from xai_decision_making.explainability.counterfactual import (
    DiCEExplainer,
    FOCUSExplainer,
    FACEExplainer,
    LOREExplainer,
    WachterExplainer,
    CCHVAEExplainer,
    REVISEExplainer,
    CounterNetExplainer,
    TreatmentScenarioCFExplainer,
)

from xai_decision_making.explainability.case_based import (
    KNNRetriever,
    EmbeddingRetriever,
    PrototypeExplainer,
)

from xai_decision_making.explainability.rule_based import (
    LORERuleExplainer,
    RuleFitExplainer,
    AnchorsExplainer,
    SkopeRulesExplainer,
)

warnings.filterwarnings("ignore")
SKIPPED_BY_FILTER = object()

STRATIFY_SEED = 42
STRATA_BOUNDS = {
    "low": (-float("inf"), 60.0),
    "mid": (60.0, 85.0),
    "high": (85.0, float("inf")),
}

OVERLAP_RSF_FILE = "test_set_processed.csv"
OVERLAP_CONDOR_FILE = "test_set_pacap.csv"
OVERLAP_ID_COLUMN = "key_nkr"
_LINKAGE_COLUMNS = ("key_nkr", "key_eid")


def _load_overlap_ids(rsf_path, condor_path, id_column):
    if not os.path.exists(rsf_path):
        raise FileNotFoundError(
            f"--overlap-only: cannot find RSF test file at {rsf_path}"
        )
    if not os.path.exists(condor_path):
        raise FileNotFoundError(
            f"--overlap-only: cannot find ConDOR test file at {condor_path}"
        )

    def _read_keys(path):
        header = pd.read_csv(path, low_memory=False, nrows=0).columns.tolist()
        cols = [c for c in _LINKAGE_COLUMNS if c in header]
        if id_column not in cols:
            raise ValueError(
                f"overlap id_column '{id_column}' not found in {path} "
                f"(available: {header[:10]}...)"
            )
        return pd.read_csv(path, usecols=cols, low_memory=False)

    rsf_df = _read_keys(rsf_path)
    condor_df = _read_keys(condor_path)

    rsf_ids = set(rsf_df[id_column].dropna().astype("int64").tolist())
    condor_ids = set(condor_df[id_column].dropna().astype("int64").tolist())
    overlap = rsf_ids & condor_ids

    linkage_frames = []
    for df in (rsf_df, condor_df):
        if set(_LINKAGE_COLUMNS).issubset(df.columns):
            sub = df[list(_LINKAGE_COLUMNS)].dropna()
            linkage_frames.append(sub.astype("int64"))
    if linkage_frames:
        linkage = pd.concat(linkage_frames).drop_duplicates().reset_index(drop=True)
    else:
        linkage = pd.DataFrame(columns=list(_LINKAGE_COLUMNS))

    return overlap, linkage


def _translate_ids(ids, source_col, target_col, linkage):
    if source_col == target_col:
        return {int(i) for i in ids}
    if linkage.empty:
        return set()
    if source_col not in linkage.columns or target_col not in linkage.columns:
        return set()
    mapping = dict(zip(
        linkage[source_col].astype("int64"),
        linkage[target_col].astype("int64"),
    ))
    return {mapping[int(i)] for i in ids if int(i) in mapping}


def _resolve_overlap_for_adapter(overlap_ids, linkage, adapter_id_column):
    if adapter_id_column == OVERLAP_ID_COLUMN:
        return {int(p) for p in overlap_ids}
    if linkage is None or linkage.empty:
        return set()
    return _translate_ids(
        overlap_ids,
        source_col=OVERLAP_ID_COLUMN,
        target_col=adapter_id_column,
        linkage=linkage,
    )


def _split_by_overlap(held_out, patient_ids, meta, overlap_ids, linkage=None):
    adapter_id_col = meta.get("id_column", OVERLAP_ID_COLUMN)
    overlap_for_compare = _resolve_overlap_for_adapter(
        overlap_ids, linkage, adapter_id_col,
    )

    keep_mask = np.array(
        [int(pid) in overlap_for_compare for pid in patient_ids],
        dtype=bool,
    )
    overlap_idx = np.where(keep_mask)[0].tolist()
    non_overlap_idx = np.where(~keep_mask)[0].tolist()

    raw_slice = meta.get("raw_test_slice")

    def _view(idx):
        if not idx:
            return (held_out.iloc[0:0].reset_index(drop=True), [], None)
        ho = held_out.iloc[idx].reset_index(drop=True)
        ids = [patient_ids[i] for i in idx]
        rs = raw_slice.iloc[idx].reset_index(drop=True) if raw_slice is not None else None
        return (ho, ids, rs)

    return _view(overlap_idx), _view(non_overlap_idx)


def _filter_to_overlap(held_out, patient_ids, meta, overlap_ids, linkage=None):
    (held_out_sub, patient_ids_sub, raw_sub), _ = _split_by_overlap(
        held_out, patient_ids, meta, overlap_ids, linkage=linkage,
    )

    if not len(patient_ids_sub):
        raise RuntimeError(
            "--overlap-only: no loaded patients are in the cross-model "
            "intersection."
        )

    meta_sub = dict(meta)
    if raw_sub is not None:
        meta_sub["raw_test_slice"] = raw_sub
    meta_sub["overlap_filter"] = {
        "mode": "overlap-only",
        "applied": True,
        "rsf_file": OVERLAP_RSF_FILE,
        "condor_file": OVERLAP_CONDOR_FILE,
        "id_column": OVERLAP_ID_COLUMN,
        "adapter_id_column": meta.get("id_column"),
        "intersection_size": int(len(overlap_ids)),
        "loaded_before": int(len(held_out)),
        "loaded_after": int(len(held_out_sub)),
    }

    return held_out_sub, patient_ids_sub, meta_sub


def _reorder_overlap_priority(held_out, patient_ids, meta, overlap_ids, linkage=None):
    (overlap_view, non_overlap_view) = _split_by_overlap(
        held_out, patient_ids, meta, overlap_ids, linkage=linkage,
    )
    ov_ho, ov_ids, ov_raw = overlap_view
    non_ho, non_ids, non_raw = non_overlap_view

    held_out_re = pd.concat([ov_ho, non_ho], ignore_index=True)
    patient_ids_re = list(ov_ids) + list(non_ids)

    meta_re = dict(meta)
    if meta.get("raw_test_slice") is not None:
        if ov_raw is None:
            meta_re["raw_test_slice"] = non_raw
        elif non_raw is None:
            meta_re["raw_test_slice"] = ov_raw
        else:
            meta_re["raw_test_slice"] = pd.concat(
                [ov_raw, non_raw], ignore_index=True,
            )
    meta_re["overlap_filter"] = {
        "mode": "overlap-priority",
        "applied": True,
        "rsf_file": OVERLAP_RSF_FILE,
        "condor_file": OVERLAP_CONDOR_FILE,
        "id_column": OVERLAP_ID_COLUMN,
        "adapter_id_column": meta.get("id_column"),
        "intersection_size": int(len(overlap_ids)),
        "loaded_total": int(len(held_out)),
        "loaded_overlap": int(len(ov_ids)),
        "loaded_non_overlap": int(len(non_ids)),
    }

    return held_out_re, patient_ids_re, meta_re


def _stratify_condor_patients(
    adapter, held_out, patient_ids, meta, n_target,
    mandatory_ids=None, linkage=None,
):
    baseline_qols = np.empty(len(held_out))
    for i in range(len(held_out)):
        row = held_out.iloc[[i]]
        pred = adapter.predict(row)
        baseline_qols[i] = pred["risk_score"]

    strata = {}
    for name, (lo, hi) in STRATA_BOUNDS.items():
        mask = (baseline_qols >= lo) & (baseline_qols < hi)
        strata[name] = np.where(mask)[0]

    mandatory_idx = []
    if mandatory_ids:
        adapter_id_col = meta.get("id_column", OVERLAP_ID_COLUMN)
        mandatory_set = _resolve_overlap_for_adapter(
            mandatory_ids, linkage, adapter_id_col,
        )
        mandatory_idx = [
            i for i, pid in enumerate(patient_ids)
            if int(pid) in mandatory_set
        ]

    non_empty = [k for k, v in strata.items() if len(v) > 0]
    if not non_empty:
        return held_out.head(n_target), patient_ids[:n_target], meta

    remaining_budget = max(0, n_target - len(mandatory_idx))
    mandatory_set_idx = set(mandatory_idx)

    rng = np.random.default_rng(STRATIFY_SEED)
    selected_from_strata = []

    if remaining_budget > 0:
        available_strata = {}
        for name, idxs in strata.items():
            avail = [i for i in idxs.tolist() if i not in mandatory_set_idx]
            available_strata[name] = np.array(avail, dtype=int)

        non_empty_avail = [k for k, v in available_strata.items() if len(v) > 0]
        if non_empty_avail:
            base_per = remaining_budget // len(non_empty_avail)
            rem = remaining_budget - base_per * len(non_empty_avail)
            quotas = {
                k: (base_per + (1 if i < rem else 0))
                for i, k in enumerate(non_empty_avail)
            }
            for name in non_empty_avail:
                idxs = available_strata[name]
                take = min(quotas[name], len(idxs))
                if take == 0:
                    continue
                chosen = rng.choice(idxs, size=take, replace=False)
                selected_from_strata.extend(chosen.tolist())

    selected_idx = sorted(set(mandatory_idx) | set(selected_from_strata))

    held_out_sub = held_out.iloc[selected_idx].reset_index(drop=True)
    patient_ids_sub = [patient_ids[i] for i in selected_idx]

    meta_sub = dict(meta)
    if meta_sub.get("raw_test_slice") is not None:
        meta_sub["raw_test_slice"] = (
            meta_sub["raw_test_slice"].iloc[selected_idx].reset_index(drop=True)
        )
    meta_sub["stratification"] = {
        "seed": STRATIFY_SEED,
        "bounds": {k: list(v) for k, v in STRATA_BOUNDS.items()},
        "strata_sizes": {k: int(len(v)) for k, v in strata.items()},
        "mandatory_count": int(len(mandatory_idx)),
        "stratified_count": int(len(selected_from_strata)),
        "selected_indices": [int(i) for i in selected_idx],
        "selected_qols": [float(baseline_qols[i]) for i in selected_idx],
    }

    return held_out_sub, patient_ids_sub, meta_sub


ATTRIBUTION_RSF = {
    "lime": SurvLIMEExplainer,
    "permutation": PermutationImportanceExplainer,
    "minimal_depth": MinimalDepthExplainer,
    "tree_shap": TreeSHAPExplainer,
}
ATTRIBUTION_CONDOR = {
    "lime": SurvLIMEExplainer,
    "permutation": PermutationImportanceExplainer,
    "deep_lift": DeepLiftExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
    "deep_lift_shap": DeepLiftSHAPExplainer,
}

COUNTERFACTUAL = {
    "dice": DiCEExplainer,
    "cchvae": CCHVAEExplainer,
    "face": FACEExplainer,
    "lore_counterfactual": LOREExplainer,
    "wachter": WachterExplainer,
    "revise": REVISEExplainer,
    "counternet": CounterNetExplainer,
    "treatment_scenario_cf": TreatmentScenarioCFExplainer,
}

CASE_BASED = {
    "knn": KNNRetriever,
    "embedding": EmbeddingRetriever,
    "prototype": PrototypeExplainer,
}

RULE_BASED = {
    "lore": LORERuleExplainer,
    "rulefit": RuleFitExplainer,
    "anchors": AnchorsExplainer,
    "skope_rules": SkopeRulesExplainer,
}


def build_registry(model_type):
    return {
        "attribution": ATTRIBUTION_RSF if model_type == "rsf" else ATTRIBUTION_CONDOR,
        "counterfactual": COUNTERFACTUAL,
        "case_based": CASE_BASED,
        "rule_based": RULE_BASED,
    }


def horizon_skip_modalities():
    return {"case_based"}


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(pid):
    cleaned = _SAFE_RE.sub("_", str(pid))
    return cleaned or "unknown"


def _to_jsonable(obj, _depth=0):
    if _depth > 10:
        return f"<max-depth: {type(obj).__name__}>"

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, np.generic):
        try:
            return _to_jsonable(obj.item(), _depth + 1)
        except Exception:
            return str(obj)

    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x, _depth + 1) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [
            {str(k): _to_jsonable(v, _depth + 1) for k, v in row.items()}
            for row in obj.to_dict(orient="records")
        ]

    if isinstance(obj, pd.Series):
        return {str(k): _to_jsonable(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, pd.Index):
        return [_to_jsonable(x, _depth + 1) for x in obj.tolist()]

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x, _depth + 1) for x in obj]

    if hasattr(obj, "x") and hasattr(obj, "y") and type(obj).__name__ == "StepFunction":
        return {
            "type": "step_function",
            "times": [_to_jsonable(t, _depth + 1) for t in np.asarray(obj.x).tolist()],
            "values": [_to_jsonable(v, _depth + 1) for v in np.asarray(obj.y).tolist()],
        }

    if hasattr(obj, "__fspath__"):
        return str(obj)

    try:
        from enum import Enum
        if isinstance(obj, Enum):
            return obj.name
    except Exception:
        pass

    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return f"<unserializable: {type(obj).__name__}>"


def _serialize_explanation(exp):
    return {
        "content": _to_jsonable(getattr(exp, "content", None)),
        "metadata": _to_jsonable(getattr(exp, "metadata", {}) or {}),
    }


def parse_only_filter(only_args):
    if not only_args:
        return None
    out = set()
    for token in only_args:
        if ":" in token:
            modality, method = token.split(":", 1)
            out.add((modality.strip(), method.strip()))
        else:
            out.add((None, token.strip()))
    return out


def should_recompute(modality, method, existing, only_filter, refresh):
    if only_filter is not None:
        return (None, method) in only_filter or (modality, method) in only_filter
    if refresh:
        return True
    if existing is None:
        return True
    prev = existing.get(modality, {}).get(method)
    if prev is None:
        return True
    if "error" in prev:
        return True
    return False


def load_existing_patient(fpath):
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def should_instantiate(modality, method, only_filter):
    if only_filter is None:
        return True
    return (None, method) in only_filter or (modality, method) in only_filter


def instantiate_explainers(
    adapter, registry, horizon_labels, time_horizons=None, only_filter=None,
):
    instances = {}
    for modality, methods in registry.items():
        instances[modality] = {}
        for name, cls in methods.items():
            if not should_instantiate(modality, name, only_filter):
                instances[modality][name] = SKIPPED_BY_FILTER
                continue
            try:
                if modality == "case_based" and time_horizons:
                    instances[modality][name] = cls(
                        adapter,
                        time_horizons=time_horizons,
                        horizon_labels=horizon_labels,
                    )
                else:
                    instances[modality][name] = cls(adapter)
            except Exception:
                instances[modality][name] = None
                traceback.print_exc(limit=2)

    horizon_instances = {}
    skip_modalities = horizon_skip_modalities()
    if time_horizons:
        for t_val in time_horizons:
            label = horizon_labels.get(t_val, f"{t_val}")
            horizon_instances[label] = {}
            for modality, methods in registry.items():
                if modality in skip_modalities:
                    continue
                horizon_instances[label][modality] = {}
                for name, cls in methods.items():
                    if not should_instantiate(modality, name, only_filter):
                        horizon_instances[label][modality][name] = SKIPPED_BY_FILTER
                        continue
                    try:
                        inst = cls(adapter, time_horizon=t_val)
                        horizon_instances[label][modality][name] = inst
                    except TypeError:
                        horizon_instances[label][modality][name] = None
                    except Exception:
                        horizon_instances[label][modality][name] = None

    return instances, horizon_instances


def explain_patient(
    instances, horizon_instances, patient_df, patient_id,
    existing_explanations, only_filter, refresh,
):
    out = {}
    actions = {}

    for modality, methods in instances.items():
        out[modality] = {}
        actions[modality] = {}

        for name, explainer in methods.items():
            prev = None
            if existing_explanations is not None:
                prev = existing_explanations.get(modality, {}).get(name)

            recompute = should_recompute(
                modality, name, existing_explanations, only_filter, refresh
            )

            if not recompute and prev is not None:
                out[modality][name] = prev
                actions[modality][name] = "reused"
                continue

            if explainer is SKIPPED_BY_FILTER:
                if prev is not None:
                    out[modality][name] = prev
                    actions[modality][name] = "reused"
                continue

            if explainer is None:
                out[modality][name] = {
                    "error": "instantiation failed",
                    "error_type": "InstantiationError",
                }
                actions[modality][name] = "computed"
                continue

            try:
                exp = explainer.explain(patient_df, patient_id=patient_id)
                out[modality][name] = _serialize_explanation(exp)
            except Exception as exc:
                out[modality][name] = {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            actions[modality][name] = "computed"

    if horizon_instances:
        out["time_horizons"] = {}

        existing_hz_all = None
        if existing_explanations:
            existing_hz_all = existing_explanations.get("time_horizons")

        for label, modalities in horizon_instances.items():
            out["time_horizons"][label] = {}
            existing_hz = existing_hz_all.get(label) if existing_hz_all else None

            for modality, methods in modalities.items():
                out["time_horizons"][label][modality] = {}

                for name, explainer in methods.items():
                    prev = None
                    if existing_hz is not None:
                        prev = existing_hz.get(modality, {}).get(name)

                    recompute = should_recompute(
                        modality, name, existing_hz, only_filter, refresh
                    )

                    if not recompute and prev is not None:
                        out["time_horizons"][label][modality][name] = prev
                        continue

                    if explainer is SKIPPED_BY_FILTER:
                        if prev is not None:
                            out["time_horizons"][label][modality][name] = prev
                        continue

                    if explainer is None:
                        out["time_horizons"][label][modality][name] = {
                            "error": "not applicable for this horizon",
                            "error_type": "NotApplicable",
                        }
                        continue

                    try:
                        exp = explainer.explain(patient_df, patient_id=patient_id)
                        out["time_horizons"][label][modality][name] = (
                            _serialize_explanation(exp)
                        )
                    except Exception as exc:
                        out["time_horizons"][label][modality][name] = {
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }

    return out, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["rsf", "condor"], default="rsf")
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--n-patients", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--time-horizons", nargs="*", type=int, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--overlap-only", action="store_true")
    parser.add_argument("--overlap-priority", action="store_true")
    parser.add_argument("--overlap-rsf-file", type=str, default=None)
    parser.add_argument("--overlap-condor-file", type=str, default=None)
    parser.add_argument("--shard", type=str, default=None)
    parser.add_argument("--skip-manifest", action="store_true")

    args = parser.parse_args()

    if args.overlap_only and args.overlap_priority:
        parser.error("--overlap-only and --overlap-priority are mutually exclusive")

    if args.train_data is None:
        args.train_data = os.path.join(
            ROOT, "data",
            "dev_set_processed.csv" if args.model_type == "rsf" else "dev_set_pacap.csv",
        )
    if args.test_data is None:
        args.test_data = os.path.join(
            ROOT, "data",
            "test_set_processed.csv" if args.model_type == "rsf" else "test_set_pacap.csv",
        )

    out_dir = args.output_dir or os.path.join(ROOT, "exports", args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    only_filter = parse_only_filter(args.only)

    if args.model_type == "rsf":
        horizon_labels = HORIZON_LABELS
    else:
        horizon_labels = QOL_HORIZON_LABELS

    time_horizons = args.time_horizons

    overlap_ids = None
    overlap_linkage = None
    overlap_mode = None
    if args.overlap_only:
        overlap_mode = "overlap-only"
    elif args.overlap_priority:
        overlap_mode = "overlap-priority"

    if overlap_mode is not None:
        rsf_path = args.overlap_rsf_file or os.path.join(
            ROOT, "data", OVERLAP_RSF_FILE,
        )
        condor_path = args.overlap_condor_file or os.path.join(
            ROOT, "data", OVERLAP_CONDOR_FILE,
        )
        overlap_ids, overlap_linkage = _load_overlap_ids(
            rsf_path, condor_path, OVERLAP_ID_COLUMN,
        )

    overlap_active = overlap_mode is not None

    if args.model_type == "rsf":
        if overlap_active:
            load_n = min(max(args.n_patients * 4, 5000), 10000)
        else:
            load_n = args.n_patients
        adapter, held_out, patient_ids, meta = load_rsf_adapter(
            args.train_data, args.test_data, load_n, args.artifact,
        )
    else:
        if args.no_stratify and not overlap_active:
            load_n = args.n_patients
        else:
            mult = 1
            if not args.no_stratify:
                mult *= 30
            if overlap_active:
                mult *= 4
            load_n = min(max(args.n_patients * mult, 200), 2000)
        adapter, held_out, patient_ids, meta = load_condor_adapter(
            args.train_data, args.test_data, load_n, args.artifact,
        )

    feature_names = adapter.feature_names()

    mandatory_ids_for_stratify = None
    if args.overlap_only:
        held_out, patient_ids, meta = _filter_to_overlap(
            held_out, patient_ids, meta, overlap_ids,
            linkage=overlap_linkage,
        )
        if args.model_type == "rsf" and len(held_out) > args.n_patients:
            held_out = held_out.head(args.n_patients).reset_index(drop=True)
            patient_ids = patient_ids[:args.n_patients]
            if meta.get("raw_test_slice") is not None:
                meta["raw_test_slice"] = (
                    meta["raw_test_slice"]
                    .head(args.n_patients)
                    .reset_index(drop=True)
                )
    elif args.overlap_priority:
        held_out, patient_ids, meta = _reorder_overlap_priority(
            held_out, patient_ids, meta, overlap_ids,
            linkage=overlap_linkage,
        )
        if args.model_type == "rsf" and len(held_out) > args.n_patients:
            adapter_id_col = meta.get("id_column", OVERLAP_ID_COLUMN)
            overlap_in_adapter_space = _resolve_overlap_for_adapter(
                overlap_ids, overlap_linkage, adapter_id_col,
            )
            held_out = held_out.head(args.n_patients).reset_index(drop=True)
            patient_ids = patient_ids[:args.n_patients]
            if meta.get("raw_test_slice") is not None:
                meta["raw_test_slice"] = (
                    meta["raw_test_slice"]
                    .head(args.n_patients)
                    .reset_index(drop=True)
                )
        if args.model_type == "condor":
            mandatory_ids_for_stratify = overlap_ids

    if args.model_type == "condor" and not args.no_stratify:
        held_out, patient_ids, meta = _stratify_condor_patients(
            adapter, held_out, patient_ids, meta, args.n_patients,
            mandatory_ids=mandatory_ids_for_stratify,
            linkage=overlap_linkage,
        )

    assert len(patient_ids) == len(held_out), (
        f"patient_ids ({len(patient_ids)}) and held_out ({len(held_out)}) "
        f"length mismatch"
    )

    if args.shard:
        shard_k, shard_n = map(int, args.shard.split("/"))
        total = len(held_out)
        chunk = total // shard_n
        start = shard_k * chunk
        end = start + chunk if shard_k < shard_n - 1 else total
        held_out = held_out.iloc[start:end].reset_index(drop=True)
        patient_ids = patient_ids[start:end]
        if meta.get("raw_test_slice") is not None:
            meta["raw_test_slice"] = (
                meta["raw_test_slice"]
                .iloc[start:end]
                .reset_index(drop=True)
            )

    registry = build_registry(args.model_type)
    instances, horizon_instances = instantiate_explainers(
        adapter,
        registry,
        horizon_labels=horizon_labels,
        time_horizons=time_horizons,
        only_filter=only_filter,
    )

    manifest_patients = []
    t_start = time.time()
    totals = {"computed": 0, "reused": 0}
    seen_filenames = set()

    for i, (_, row) in enumerate(held_out.iterrows()):
        patient_df = row.to_frame().T.reset_index(drop=True)
        real_id = patient_ids[i]
        safe_id = sanitize_filename(real_id)
        fname = f"patient_{safe_id}.json"

        raw_slice = meta.get("raw_test_slice")
        raw_row = raw_slice.iloc[i] if raw_slice is not None else None

        if fname in seen_filenames:
            continue
        seen_filenames.add(fname)

        fpath = os.path.join(out_dir, fname)
        existing = load_existing_patient(fpath)
        existing_explanations = existing.get("explanations") if existing else None

        try:
            try:
                pred_json = _to_jsonable(adapter.predict(patient_df, full=True))
            except TypeError:
                pred_json = _to_jsonable(adapter.predict(patient_df))
        except Exception as exc:
            pred_json = {"error": str(exc), "error_type": type(exc).__name__}

        explanations, actions = explain_patient(
            instances, horizon_instances,
            patient_df, real_id,
            existing_explanations, only_filter, args.refresh,
        )

        n_computed = sum(1 for mods in actions.values() for a in mods.values() if a == "computed")
        n_reused = sum(1 for mods in actions.values() for a in mods.values() if a == "reused")
        totals["computed"] += n_computed
        totals["reused"] += n_reused

        record = {
            "patient_id": real_id,
            "model_type": args.model_type,
            "features": {str(k): _to_jsonable(v) for k, v in row.items()},
            "features_raw": (
                {str(k): _to_jsonable(v) for k, v in raw_row.items()}
                if raw_row is not None else None
            ),
            "prediction": pred_json,
            "explanations": explanations,
        }

        if args.dry_run:
            return

        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w") as f:
            json.dump(record, f, indent=2)

        method_status = {
            modality: {
                name: ("ok" if "error" not in result
                       else result.get("error_type", "error"))
                for name, result in methods.items()
            }
            for modality, methods in explanations.items()
            if modality != "time_horizons"
        }
        manifest_patients.append({
            "patient_id": real_id,
            "file": fname,
            "methods": method_status,
        })

    if not args.skip_manifest:
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_type": args.model_type,
            "model_meta": _to_jsonable(meta),
            "n_patients": len(manifest_patients),
            "feature_names": feature_names,
            "horizon_unit": (
                "days" if args.model_type == "rsf" else "months"
            ) if time_horizons else None,
            "time_horizons": (
                {str(t): horizon_labels.get(t, f"{t}") for t in time_horizons}
                if time_horizons else None
            ),
            "modalities": {
                modality: list(methods.keys())
                for modality, methods in registry.items()
            },
            "patients": manifest_patients,
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()