"""
Adapter layer between trained clinical models and the explainability framework.

Wraps Random Survival Forest (RSF) and ConDOR (Conditional Diffusion for
Ordinal Regression) models behind a unified SurvivalModelAdapter interface,
exposing prediction, preprocessing, inverse transformation, and training
data access required by counterfactual, case-based, and feature-attribution
explainers. Includes checkpoint loading, encoder recovery, and treatment
feature construction for both model families.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# old models had other structure, after restructuring we needed to do this to make it work 
import xai_decision_making.models
import xai_decision_making.models.survival
import xai_decision_making.models.survival.survival_utils
sys.modules["models"] = xai_decision_making.models
sys.modules["models.survival"] = xai_decision_making.models.survival
sys.modules["models.survival.survival_utils"] = xai_decision_making.models.survival.survival_utils

from xai_decision_making.experiments.qol_data_ablation_study import (
    ID_COL,
    TIME_COL,
    STAGE_COL,
    StaticEncoderBundle,
    stage_to_ordinal,
    transform_static_row,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xai_decision_making.explainability.base import (
    ModelTarget,
    SurvivalModelAdapter as _BaseSurvivalAdapter,
)

RSF_ID_COL = "key_eid"

TREATMENT_FEATURE_NAMES = [
    "ts_surg",
    "had_surg",
    "impulse_surg",
    "ts_syst",
    "had_syst",
    "impulse_syst",
]

DEFAULT_TIME_HORIZONS = [90, 180, 365]
HORIZON_LABELS = {90: "3_months", 180: "6_months", 365: "12_months"}
DEFAULT_QOL_HORIZONS_MONTHS = [1, 3, 6, 9, 12]
QOL_HORIZON_LABELS = {
    1: "1_month",
    3: "3_months",
    6: "6_months",
    9: "9_months",
    12: "12_months",
}


def _months_to_t_norm(months):
    return float(months) / 60.0


def _preprocess_batch(wrapper, df, feature_names):
    aligned = df.reindex(columns=feature_names)
    arr = wrapper._prepare_input(aligned)
    names = feature_names
    if wrapper.pipeline is not None:
        cleaner = wrapper.pipeline.named_steps.get("cleaner")
        if cleaner is not None:
            out = cleaner.get_feature_names_out()
            if out is not None:
                names = list(out)
    return arr, names


def _find_condor_checkpoint(pattern):
    search_dir = Path(ROOT) / "artifacts" / "condor_models"
    if not search_dir.is_dir():
        raise FileNotFoundError(
            f"Paper checkpoint directory not found: {search_dir}"
        )
    candidates = list(search_dir.glob("**/*.pt"))
    if pattern:
        candidates = [c for c in candidates if pattern in str(c)]
    if not candidates:
        raise FileNotFoundError(
            f"No ConDOR checkpoints found (filter: {pattern!r})\n"
            f"Searched: {search_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _recover_encoder_columns(encoders_dict):
    e = encoders_dict
    num_imputer = e.get("static_num_imputer") or e.get("num_imputer")
    num_scaler = e.get("static_num_scaler") or e.get("num_scaler")
    cat_imputer = e.get("static_cat_imputer") or e.get("cat_imputer")
    cat_encoder = e.get("static_cat_encoder") or e.get("cat_encoder")

    if num_imputer is None or num_scaler is None or cat_encoder is None:
        raise RuntimeError(
            "Checkpoint encoder dict is missing required sklearn objects. "
            f"Got keys: {list(e.keys())}"
        )

    saved_num_cols = e.get("static_num_cols") or e.get("num_cols")
    saved_cat_cols = e.get("static_cat_cols") or e.get("cat_cols")

    if not saved_num_cols:
        fni = getattr(num_imputer, "feature_names_in_", None)
        if fni is None:
            raise RuntimeError(
                "Cannot recover numeric column names: checkpoint has no "
                "static_num_cols and num_imputer.feature_names_in_ is None."
            )
        num_cols = list(fni)
    else:
        num_cols = list(saved_num_cols)

    if not saved_cat_cols:
        fni = getattr(cat_imputer, "feature_names_in_", None) if cat_imputer is not None else None
        if fni is None:
            n_cat = getattr(cat_encoder, "n_features_in_", 0) or 0
            if n_cat > 0:
                raise RuntimeError(
                    "Cannot recover categorical column names: checkpoint "
                    "has no static_cat_cols and cat_imputer.feature_names_in_ "
                    f"is None, but cat_encoder.n_features_in_ = {n_cat}."
                )
            cat_cols = []
        else:
            cat_cols = list(fni)
    else:
        cat_cols = list(saved_cat_cols)

    expected_num = getattr(num_scaler, "n_features_in_", None)
    if expected_num is not None and len(num_cols) != expected_num:
        raise RuntimeError(
            f"Numeric column count mismatch: recovered {len(num_cols)} names "
            f"({num_cols}) but num_scaler.n_features_in_ = {expected_num}."
        )

    expected_cat = getattr(cat_encoder, "n_features_in_", None)
    if expected_cat is not None and len(cat_cols) != expected_cat:
        raise RuntimeError(
            f"Categorical column count mismatch: recovered {len(cat_cols)} "
            f"names ({cat_cols}) but cat_encoder.n_features_in_ = {expected_cat}."
        )

    return num_imputer, num_scaler, cat_imputer, cat_encoder, num_cols, cat_cols


class WrappedSurvivalAdapter(_BaseSurvivalAdapter):

    def __init__(self, wrapper, x_train_raw, feature_names, raw_train_full=None):
        self._wrapper = wrapper
        self._raw_feature_names = feature_names
        arr, names = _preprocess_batch(wrapper, x_train_raw, feature_names)
        self._x_train = pd.DataFrame(arr, columns=names)
        self._raw_train_full = raw_train_full
        if self._raw_train_full is not None:
            self._raw_train_full = self._raw_train_full.reset_index(drop=True)
            if len(self._raw_train_full) != len(self._x_train):
                self._raw_train_full = None
        train_risks = self._wrapper.model.predict(arr)
        self._train_risks = np.sort(train_risks)
        self._risk_min = float(self._train_risks.min())
        self._risk_max = float(self._train_risks.max())

    def _scale_risk(self, raw_risk):
        if self._risk_max <= self._risk_min:
            return 50.0
        scaled = (raw_risk - self._risk_min) / (self._risk_max - self._risk_min) * 100.0
        return float(np.clip(scaled, 0.0, 100.0))

    @property
    def target(self):
        return ModelTarget.SURVIVAL

    @property
    def higher_is_better(self):
        return False

    @property
    def prediction_saturation_bounds(self):
        return (0.01, 0.99)

    @property
    def risk_scale_factor(self):
        denom = self._risk_max - self._risk_min
        if denom <= 0:
            return None
        return denom / 100.0

    @property
    def raw_model(self):
        return self._wrapper.model

    def feature_names(self):
        return list(self._x_train.columns)

    def training_data(self):
        return self._x_train

    def training_data_raw(self):
        return self._raw_train_full

    def preprocess(self, df):
        arr, names = _preprocess_batch(self._wrapper, df, self._raw_feature_names)
        return pd.DataFrame(arr, columns=names)

    def predict(self, patient_features, time_horizon=None):
        x = patient_features.values
        risk = self._wrapper.model.predict(x)
        try:
            surv_fns = self._wrapper.model.predict_survival_function(x)
        except Exception:
            surv_fns = None

        result = {
            "risk_score": self._scale_risk(risk[0]),
            "risk_score_raw": float(risk[0]),
            "survival_curve": surv_fns,
        }

        if surv_fns is not None:
            surv_fn = surv_fns[0]
            for t_days, label in HORIZON_LABELS.items():
                try:
                    s_t = float(surv_fn(t_days))
                    result[f"survival_prob_{label}"] = round(s_t, 4)
                    result[f"mortality_risk_{label}"] = round(1.0 - s_t, 4)
                except Exception:
                    result[f"survival_prob_{label}"] = None
                    result[f"mortality_risk_{label}"] = None

        if time_horizon is not None:
            label = HORIZON_LABELS.get(time_horizon, f"{time_horizon}d")
            mort_key = f"mortality_risk_{label}"
            if mort_key in result and result[mort_key] is not None:
                result["risk_score"] = result[mort_key]
                result["risk_score_raw"] = result[mort_key]

        return result

    def predict_at_horizon(self, patient_features, time_days):
        x = patient_features.values
        surv_fns = self._wrapper.model.predict_survival_function(x)
        return np.array([1.0 - float(sf(time_days)) for sf in surv_fns])

    def inverse_transform_numeric(self, df):
        return df.copy()


class DifferentiableRDMWrapper(nn.Module):

    def __init__(
        self,
        rdm,
        tdm,
        config_obj,
        *,
        use_static_rdm,
        use_static_tdm,
        use_treatment,
        use_mortality_risk,
        static_dim,
        treat_dim,
        default_stage=4,
        default_t0=0.0,
    ):
        super().__init__()
        self.rdm = rdm
        self.tdm = tdm
        self._config = config_obj
        self._use_static_rdm = use_static_rdm
        self._use_static_tdm = use_static_tdm
        self._use_treatment = use_treatment
        self._use_mortality_risk = use_mortality_risk
        self._static_dim = static_dim
        self._treat_dim = treat_dim
        self._default_stage = default_stage
        self._default_t0 = default_t0

    def _split_input(self, x_input):
        static = x_input[:, :self._static_dim]
        end = self._static_dim + self._treat_dim
        if self._treat_dim > 0 and x_input.shape[1] >= end:
            treat = x_input[:, self._static_dim:end]
        else:
            treat = torch.zeros(
                x_input.shape[0], self._treat_dim,
                device=x_input.device, dtype=x_input.dtype,
            )
        return static, treat

    def _resolve_stage(self, bs, device, stage):
        if stage is None:
            return torch.full((bs,), self._default_stage,
                              dtype=torch.long, device=device)
        if isinstance(stage, (int, np.integer)):
            return torch.full((bs,), int(stage),
                              dtype=torch.long, device=device)
        if isinstance(stage, torch.Tensor):
            y = stage.to(device=device, dtype=torch.long)
        else:
            y = torch.as_tensor(np.asarray(stage), device=device, dtype=torch.long)
        if y.dim() == 0:
            y = y.expand(bs)
        elif y.shape[0] == 1 and bs > 1:
            y = y.expand(bs)
        elif y.shape[0] != bs:
            raise ValueError(
                f"stage tensor shape {tuple(y.shape)} does not match batch size {bs}"
            )
        return y

    def _generate_baseline(self, x_input, stage=None):
        static_features, _ = self._split_input(x_input)
        device = static_features.device
        bs = static_features.shape[0]
        y0 = self._resolve_stage(bs, device, stage)
        t0 = torch.full((bs,), self._default_t0, dtype=torch.float32, device=device)
        static_arg = static_features if self._use_static_rdm else None
        risk_arg = (
            torch.zeros(bs, 1, device=device) if self._use_mortality_risk else None
        )
        gen = torch.Generator(device=device).manual_seed(1)
        x_t = torch.randn(bs, 1, device=device, dtype=static_features.dtype, generator=gen)

        step_scale = 1.0 / float(self._config.rdm_steps)
        for i in reversed(range(self._config.rdm_steps)):
            t_step = torch.full((bs,), i, device=device, dtype=torch.float32)
            pred_noise = self.rdm(
                x_t, t_step, y0, t0,
                static=static_arg,
                mortality_risk=risk_arg,
            )
            x_t = x_t - pred_noise * step_scale
        return torch.clamp(x_t, 0.0, 1.0)

    def _roll_forward(self, x_baseline, x_input, target_t_norm, stage=None):
        static_features, treat_features = self._split_input(x_input)
        device = x_baseline.device
        bs = x_baseline.shape[0]
        target = torch.full((bs,), float(target_t_norm),
                            dtype=torch.float32, device=device)
        label = self._resolve_stage(bs, device, stage)
        static_arg = static_features if self._use_static_tdm else None
        treat_arg = treat_features if self._use_treatment else None

        x_traj = x_baseline.clone()
        for d in range(self._config.diffusion_steps):
            ds = torch.full((bs,), d, device=device, dtype=torch.long)
            delta = self.tdm(
                x_traj, ds, target, label,
                static=static_arg, treat=treat_arg,
            )
            x_traj = torch.clamp(x_traj + delta, 0.0, 1.0)
        return x_traj

    def forward(self, x_input, target_t_norm=None, stage=None):
        x_baseline = self._generate_baseline(x_input, stage=stage)
        if target_t_norm is None or target_t_norm <= 0.0:
            return torch.clamp(x_baseline * 100.0, 0, 100)
        x_at_t = self._roll_forward(x_baseline, x_input, target_t_norm, stage=stage)
        return torch.clamp(x_at_t * 100.0, 0, 100)


class WrappedConDORAdapter(_BaseSurvivalAdapter):

    def __init__(
        self,
        models,
        config_obj,
        x_train_static,
        *,
        use_static_rdm,
        use_static_tdm=False,
        use_treatment=False,
        use_mortality_risk=False,
        default_stage=4,
        default_t0=0.0,
        encoder_bundle=None,
        stages_by_id=None,
        raw_train_full=None,
    ):
        self._ord, self._rdm, self._tdm = models
        self._config = config_obj
        self._x_train = x_train_static
        self._use_static_rdm = use_static_rdm
        self._use_static_tdm = use_static_tdm
        self._use_treatment = use_treatment
        self._use_mortality_risk = use_mortality_risk
        self._encoder = encoder_bundle
        self._stages_by_id = stages_by_id or {}
        self._raw_train_full = raw_train_full
        if self._raw_train_full is not None:
            self._raw_train_full = self._raw_train_full.reset_index(drop=True)
            if len(self._raw_train_full) != len(self._x_train):
                self._raw_train_full = None

        self._diff_model = DifferentiableRDMWrapper(
            self._rdm, self._tdm, config_obj,
            use_static_rdm=use_static_rdm,
            use_static_tdm=use_static_tdm,
            use_treatment=use_treatment,
            use_mortality_risk=use_mortality_risk,
            static_dim=config_obj.static_dim,
            treat_dim=getattr(config_obj, "treat_dim", len(TREATMENT_FEATURE_NAMES)),
            default_stage=default_stage,
            default_t0=default_t0,
        )

    @property
    def target(self):
        return ModelTarget.SURVIVAL

    @property
    def higher_is_better(self):
        return True

    @property
    def prediction_saturation_bounds(self):
        return (1.0, 99.0)

    @property
    def raw_model(self):
        return self._diff_model

    @property
    def feeds_zscored(self):
        return True

    def feature_names(self):
        return list(self._x_train.columns)

    def training_data(self):
        return self._x_train

    def training_data_raw(self):
        return self._raw_train_full

    def preprocess(self, df):
        return df.reindex(columns=self.feature_names()).copy()

    def _to_tensor(self, patient_features):
        x = torch.FloatTensor(patient_features.values).to(self._config.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _resolve_stage_for_call(self, stage, patient_id):
        if stage is not None:
            return stage
        if patient_id is not None:
            key = str(patient_id)
            if key in self._stages_by_id:
                return int(self._stages_by_id[key])
        return None

    def predict(self, patient_features, time_horizon=None, full=False,
                stage=None, patient_id=None):
        x = self._to_tensor(patient_features)
        resolved_stage = self._resolve_stage_for_call(stage, patient_id)
        result = {}

        with torch.no_grad():
            if full:
                x_baseline = self._diff_model._generate_baseline(x, stage=resolved_stage)
                qol_baseline = torch.clamp(x_baseline * 100.0, 0, 100)
                result["qol_baseline"] = float(qol_baseline.item())
                for months in DEFAULT_QOL_HORIZONS_MONTHS:
                    t_norm = _months_to_t_norm(months)
                    x_at_t = self._diff_model._roll_forward(
                        x_baseline, x, t_norm, stage=resolved_stage,
                    )
                    qol_t = torch.clamp(x_at_t * 100.0, 0, 100)
                    result[f"qol_{QOL_HORIZON_LABELS[months]}"] = round(
                        float(qol_t.item()), 4
                    )
                if time_horizon is None:
                    result["risk_score"] = result["qol_baseline"]
                else:
                    months = int(time_horizon)
                    label = QOL_HORIZON_LABELS.get(months)
                    if label is not None:
                        result["risk_score"] = result[f"qol_{label}"]
                    else:
                        x_at_t = self._diff_model._roll_forward(
                            x_baseline, x, _months_to_t_norm(months),
                            stage=resolved_stage,
                        )
                        qol_t = torch.clamp(x_at_t * 100.0, 0, 100)
                        result["risk_score"] = round(float(qol_t.item()), 4)
            else:
                t_norm = None if time_horizon is None else _months_to_t_norm(int(time_horizon))
                qol = self._diff_model(x, target_t_norm=t_norm, stage=resolved_stage)
                result["risk_score"] = float(qol.item())

        return result

    def predict_at_horizon(self, patient_features, months,
                           stage=None, patient_id=None):
        x = self._to_tensor(patient_features)
        resolved_stage = self._resolve_stage_for_call(stage, patient_id)
        with torch.no_grad():
            qol_t = self._diff_model(
                x, target_t_norm=_months_to_t_norm(months), stage=resolved_stage,
            )
        return qol_t.detach().cpu().numpy().reshape(-1)

    def inverse_transform_numeric(self, df):
        if self._encoder is None:
            return df.copy()
        scaler = getattr(self._encoder, "num_scaler", None)
        num_cols = list(getattr(self._encoder, "num_cols", []))
        if scaler is None or not num_cols:
            return df.copy()

        result = df.copy()
        present = [c for c in num_cols if c in result.columns]
        if not present:
            return result

        if len(present) == len(num_cols):
            result[num_cols] = scaler.inverse_transform(result[num_cols].values)
        else:
            means = getattr(scaler, "mean_", None)
            scales = getattr(scaler, "scale_", None)
            if means is not None and scales is not None:
                for c in present:
                    i = num_cols.index(c)
                    result[c] = result[c] * scales[i] + means[i]
        return result


def load_rsf_adapter(train_data, test_data, n_patients, artifact=None):
    from xai_decision_making.models.survival.survival_utils import load_pickle
    from xai_decision_making.models.survival.analysis.evaluate_saved_artifacts import (
        coerce_pandas_string_to_object,
        ensure_columns,
        infer_raw_input_columns,
        parse_artifact_name,
    )

    artifacts_dir = os.path.join(ROOT, "artifacts", "survival_models")
    filenames = sorted(f for f in os.listdir(artifacts_dir) if f.endswith(".pkl"))
    results = []
    for fn in filenames:
        if artifact and artifact not in fn:
            continue
        parsed = parse_artifact_name(fn)
        if parsed is not None:
            results.append((fn, parsed))

    if not results:
        raise FileNotFoundError(
            f"No RSF artifacts in {artifacts_dir} (filter: {artifact!r})"
        )
    if len(results) > 1:
        listing = "\n  ".join(fn for fn, _ in results)
        raise ValueError(
            f"--artifact filter {artifact!r} matched {len(results)} files; "
            f"narrow it down to exactly one:\n  {listing}"
        )

    filename, parsed = results[0]
    wrapper = load_pickle(os.path.join(artifacts_dir, filename))
    wrapper.model.n_jobs = 1

    raw_train_csv = pd.read_csv(train_data, low_memory=False)
    raw_cols = infer_raw_input_columns(wrapper, raw_train_csv)
    raw_train = coerce_pandas_string_to_object(
        ensure_columns(raw_train_csv, raw_cols)
    )

    raw_test_csv = pd.read_csv(test_data, low_memory=False)
    raw_test = coerce_pandas_string_to_object(
        ensure_columns(raw_test_csv, raw_cols)
    )

    feature_names = infer_raw_input_columns(wrapper, raw_train)

    adapter = WrappedSurvivalAdapter(
        wrapper, raw_train, feature_names,
        raw_train_full=raw_train_csv,
    )

    if RSF_ID_COL not in raw_test_csv.columns:
        raise KeyError(
            f"Expected patient ID column {RSF_ID_COL!r} in {test_data}. "
            f"Columns available: {list(raw_test_csv.columns)[:10]}..."
        )

    test_slice = raw_test.head(n_patients)
    patient_ids = [str(v) for v in raw_test_csv[RSF_ID_COL].head(n_patients).tolist()]
    held_out = adapter.preprocess(test_slice)

    meta = vars(parsed)
    meta["filename"] = filename
    meta["id_column"] = RSF_ID_COL
    meta["raw_test_slice"] = test_slice.reset_index(drop=True)
    meta["risk_scale_factor"] = adapter.risk_scale_factor
    meta["risk_min"] = adapter._risk_min
    meta["risk_max"] = adapter._risk_max

    return adapter, held_out, patient_ids, meta


def _build_treatment_block(clinical_df, n_rows, time_col, reference_days=180):
    from xai_decision_making.experiments.qol_data_ablation_study import build_treatment_matrix

    if time_col in clinical_df.columns:
        try:
            mat = build_treatment_matrix(clinical_df, time_col=time_col)
            if mat.shape == (n_rows, len(TREATMENT_FEATURE_NAMES)):
                return mat
        except Exception:
            pass

    out = np.zeros((n_rows, len(TREATMENT_FEATURE_NAMES)), dtype=float)

    surg_col = None
    for cand in ("op_datum", "chir_int1"):
        if cand in clinical_df.columns:
            surg_col = cand
            break

    syst_cols = [c for c in clinical_df.columns
                 if str(c).startswith("syst_start_int")]

    if surg_col is None and not syst_cols:
        return out

    for i, (_, row) in enumerate(clinical_df.head(n_rows).iterrows()):
        if surg_col is not None:
            surg_day = pd.to_numeric(row.get(surg_col), errors="coerce")
            if pd.notna(surg_day) and float(surg_day) <= reference_days:
                ts = (reference_days - float(surg_day)) / 30.44 / 60.0
                out[i, 0] = ts
                out[i, 1] = 1.0

        for sc in syst_cols:
            syst_day = pd.to_numeric(row.get(sc), errors="coerce")
            if pd.notna(syst_day) and float(syst_day) <= reference_days:
                ts = (reference_days - float(syst_day)) / 30.44 / 60.0
                if ts > out[i, 3]:
                    out[i, 3] = ts
                out[i, 4] = 1.0
                break

    return out


def load_condor_adapter(train_data, test_data, n_patients, artifact=None):
    from xai_decision_making.experiments.qol_data_ablation_study import (
        ID_COL,
        TIME_COL,
        StaticEncoderBundle,
        transform_static_row,
    )
    from xai_decision_making.models.synthesis.condor_core import (
        ConDORConfig,
        OrdinalRegression,
        RDM,
        TDM,
    )

    ckpt_path = _find_condor_checkpoint(artifact)
    tmp_conf = ConDORConfig(static_dim=0)
    try:
        ckpt = torch.load(ckpt_path, map_location=tmp_conf.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=tmp_conf.device)

    config_args = ckpt.get("config_args", {})
    conf = ConDORConfig(**config_args)

    ord_net = OrdinalRegression(conf).to(conf.device)
    rdm_net = RDM(conf).to(conf.device)
    tdm_net = TDM(conf).to(conf.device)

    if "ordinal" in ckpt and ckpt["ordinal"] is not None:
        ord_net.load_state_dict(ckpt["ordinal"])
    rdm_net.load_state_dict(ckpt["rdm"])
    tdm_net.load_state_dict(ckpt["tdm"])

    models = (ord_net, rdm_net, tdm_net)

    use_static_rdm = bool(config_args.get("use_static_rdm", False))
    use_static_tdm = bool(config_args.get("use_static_tdm", False))
    use_treatment = bool(config_args.get("use_treatment", False))
    use_mortality_risk = bool(config_args.get("use_mortality_risk", False))

    if not (use_static_rdm or use_static_tdm):
        raise RuntimeError(
            "Checkpoint was trained without static features. "
            "Use one with clinical features."
        )

    if "encoders" not in ckpt:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has no 'encoders' key. "
            "Use a fold checkpoint that bundles the sklearn pipeline."
        )

    (
        num_imputer,
        num_scaler,
        cat_imputer,
        cat_encoder,
        num_cols,
        cat_cols,
    ) = _recover_encoder_columns(ckpt["encoders"])

    enc = StaticEncoderBundle(
        num_imputer=num_imputer,
        num_scaler=num_scaler,
        cat_imputer=cat_imputer,
        cat_encoder=cat_encoder,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    clin_train = pd.read_csv(train_data, low_memory=False)
    clin_test = pd.read_csv(test_data, low_memory=False)

    train_unique = clin_train.drop_duplicates(subset=[ID_COL], keep="first").copy()
    test_unique = clin_test.drop_duplicates(subset=[ID_COL], keep="first").copy()

    missing_num = [c for c in num_cols if c not in train_unique.columns]
    missing_cat = [c for c in cat_cols if c not in train_unique.columns]
    if missing_num or missing_cat:
        raise RuntimeError(
            f"Training CSV is missing columns the encoder expects.\n"
            f"  Missing numeric: {missing_num}\n"
            f"  Missing categorical: {missing_cat}"
        )

    if num_cols:
        for c in num_cols:
            if c in train_unique.columns:
                train_unique[c] = pd.to_numeric(train_unique[c], errors="coerce")
            if c in test_unique.columns:
                test_unique[c] = pd.to_numeric(test_unique[c], errors="coerce")

    sample_vec = transform_static_row(train_unique.iloc[0], enc)
    static_dim = len(sample_vec)
    assert static_dim == conf.static_dim, (
        f"Encoder dim={static_dim} != checkpoint static_dim={conf.static_dim}."
    )

    static_feat_names = list(num_cols)
    n_cat = static_dim - len(num_cols)
    if n_cat > 0:
        try:
            cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
            static_feat_names += (
                cat_names if len(cat_names) == n_cat
                else [f"cat_{i}" for i in range(n_cat)]
            )
        except Exception:
            static_feat_names += [f"cat_{i}" for i in range(n_cat)]

    x_train_static_arr = np.vstack(
        [transform_static_row(row, enc) for _, row in train_unique.iterrows()]
    )
    test_slice = test_unique.head(n_patients)
    patient_ids = [str(v) for v in test_slice[ID_COL].tolist()]
    held_out_static_arr = np.vstack(
        [transform_static_row(row, enc) for _, row in test_slice.iterrows()]
    )

    treat_dim = getattr(conf, "treat_dim", len(TREATMENT_FEATURE_NAMES))

    if use_treatment:
        train_treat = _build_treatment_block(
            train_unique, len(train_unique), TIME_COL,
        )
        test_treat = _build_treatment_block(
            test_slice, len(test_slice), TIME_COL,
        )

        if train_treat.shape[1] != treat_dim:
            if train_treat.shape[1] < treat_dim:
                pad = treat_dim - train_treat.shape[1]
                train_treat = np.hstack(
                    [train_treat, np.zeros((train_treat.shape[0], pad))]
                )
                test_treat = np.hstack(
                    [test_treat, np.zeros((test_treat.shape[0], pad))]
                )
            else:
                train_treat = train_treat[:, :treat_dim]
                test_treat = test_treat[:, :treat_dim]

        treat_feat_names = list(TREATMENT_FEATURE_NAMES[:treat_dim])
        if len(treat_feat_names) < treat_dim:
            treat_feat_names += [
                f"treat_{i}" for i in range(len(treat_feat_names), treat_dim)
            ]
        feat_names_full = static_feat_names + treat_feat_names
        x_train_full = np.hstack([x_train_static_arr, train_treat])
        held_out_full = np.hstack([held_out_static_arr, test_treat])
    else:
        feat_names_full = static_feat_names
        x_train_full = x_train_static_arr
        held_out_full = held_out_static_arr

    x_train_static = pd.DataFrame(x_train_full, columns=feat_names_full)
    held_out = pd.DataFrame(held_out_full, columns=feat_names_full)

    stages_by_id = {}
    for src_df in (train_unique, test_slice):
        if STAGE_COL in src_df.columns:
            for _, srow in src_df.iterrows():
                stages_by_id[str(srow[ID_COL])] = int(
                    stage_to_ordinal(srow.get(STAGE_COL))
                )

    adapter = WrappedConDORAdapter(
        models,
        conf,
        x_train_static,
        use_static_rdm=use_static_rdm,
        use_static_tdm=use_static_tdm,
        use_treatment=use_treatment,
        use_mortality_risk=use_mortality_risk,
        encoder_bundle=enc,
        stages_by_id=stages_by_id,
        raw_train_full=train_unique,
    )

    meta = {
        "model_type": "condor",
        "checkpoint": str(ckpt_path),
        "use_static_rdm": use_static_rdm,
        "use_static_tdm": use_static_tdm,
        "use_treatment": use_treatment,
        "use_mortality_risk": use_mortality_risk,
        "static_dim": conf.static_dim,
        "treat_dim": treat_dim if use_treatment else 0,
        "treatment_feature_names": (
            list(TREATMENT_FEATURE_NAMES[:treat_dim]) if use_treatment else []
        ),
        "id_column": ID_COL,
        "raw_test_slice": test_slice.reset_index(drop=True),
        "higher_is_better": True,
    }

    return adapter, held_out, patient_ids, meta