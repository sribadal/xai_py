"""
ConDOR ablation study with patient-level K-fold cross-validation.

Full pipeline for training and evaluating conditional ordinal diffusion
models for longitudinal QoL prediction: data loading/merging, static
feature encoding, treatment matrix construction, sequence building,
RDM/TDM training with early stopping, and per-horizon RMSE evaluation.
Runs seven study configurations comparing baseline, RDM, TDM, and
combined feature conditioning.
"""
import argparse
import copy
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from xai_decision_making import config as main_config
except Exception:
    main_config = None

from xai_decision_making.models.synthesis.condor_core import ConDORConfig, OrdinalRegression, RDM, TDM
from xai_decision_making.preprocessing.qol.qlq_c30_scoring import calculate_qlq_c30_scores

try:
    from xai_decision_making.models.survival import config_survival as survival_config
except Exception:
    survival_config = None

NUM_VARS = [
    "cci", "ca199", "bilirubine", "albumine", "ldh", "crp",
]
CAT_VARS = [
    "leeft_cat", "gesl", "seizoen",
]

CATEGORY_LEVELS = {
    "leeft_cat": ["6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"],
    "gesl": ["1", "2"],
    "seizoen": ["1", "2", "3", "4"],
}

DEFAULT_ID_COL = "key_nkr"
DEFAULT_TIME_COL = "proms_dat_int"
DEFAULT_STAGE_COL = "stadium"

def _get_col_from_config(attr, default):
    if main_config is None:
        return default
    return getattr(main_config, attr, default)

ID_COL = _get_col_from_config("KEY_NK", DEFAULT_ID_COL)
TIME_COL = _get_col_from_config("PROMS_DATE_COL", DEFAULT_TIME_COL)
STAGE_COL = _get_col_from_config("STAGE_COL", DEFAULT_STAGE_COL)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _safe_study_name(name):
    safe = name.lower().strip()
    safe = safe.replace(" ", "_")
    safe = safe.replace("+", "")
    safe = safe.replace("(", "")
    safe = safe.replace(")", "")
    safe = re.sub(r"[^a-z0-9_]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe

def _normalize_col_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def _as_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def _resolve_time_col(df, preferred):
    if preferred in df.columns:
        return preferred

    norm_to_col = {_normalize_col_name(c): c for c in df.columns}

    explicit_candidates = [
        preferred,
        f"{preferred}_int",
        "proms_dat_int",
        "proms_datum_int",
        "promsdatum",
        "promsdate",
        "proms_date",
        "proms_dat",
        "proms_time",
        "proms_days",
        "days_since_diagnosis",
        "time_since_diagnosis",
        "followup_days",
        "follow_up_days",
        "meetmoment",
        "tijd",
        "tijdstip",
    ]

    for cand in explicit_candidates:
        if cand in df.columns:
            return cand
        cand_norm = _normalize_col_name(cand)
        if cand_norm in norm_to_col:
            return norm_to_col[cand_norm]

    keyword_hits = []
    for c in df.columns:
        lc = str(c).lower()
        has_context = any(
            k in lc
            for k in ["prom", "follow", "meet", "time", "tijd", "month", "day"]
        )
        has_timeish = any(
            k in lc
            for k in ["datum", "date", "int", "days", "day", "months", "month", "tijd"]
        )
        if has_context and has_timeish:
            keyword_hits.append(c)

    if len(keyword_hits) == 1:
        return keyword_hits[0]

    raise KeyError(
        f"TIME_COL '{preferred}' not found in PACAP. "
        f"Potential time-like candidates: {keyword_hits[:15]}. "
        f"Available: {list(df.columns)[:25]}"
    )

def stage_to_ordinal(val):
    if pd.isna(val):
        return 4

    s = str(val).strip().upper()

    mapping = {
        "I": 0, "IA": 0, "IB": 0, "1": 0, "1A": 0, "1B": 0,
        "II": 1, "IIA": 1, "IIB": 1, "2": 1, "2A": 1, "2B": 1,
        "III": 2, "3": 2,
        "IV": 3, "4": 3, "M": 3,
        "X": 4, "UNKNOWN": 4, "NAN": 4, "NONE": 4, "": 4,
    }

    if s in mapping:
        return mapping[s]

    try:
        n = int(float(s))
        if n == 1:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 3
        return 4
    except Exception:
        return 4


def rmse(a, b):
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a_arr - b_arr) ** 2)))

class ConDORSequenceDataset(Dataset):
    def __init__(self, sequences, max_len = 10):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        t_vals = np.asarray(seq["t"], dtype=float)
        x_vals = np.asarray(seq["x"], dtype=float)
        y_vals = np.asarray(seq["y"], dtype=int)

        treat_vals = np.asarray(
            seq.get("treat", np.zeros((len(t_vals), 0), dtype=float)),
            dtype=float,
        )

        actual_len = min(len(t_vals), self.max_len)
        pad_len = self.max_len - actual_len

        x = np.pad(x_vals[:actual_len], (0, pad_len), constant_values=0.0)
        t = np.pad(t_vals[:actual_len], (0, pad_len), constant_values=-1.0)
        y = np.pad(y_vals[:actual_len], (0, pad_len), constant_values=0)
        mask = np.pad(
            np.ones(actual_len, dtype=int),
            (0, pad_len),
            constant_values=0,
        ).astype(bool)

        static = np.asarray(seq.get("static", np.zeros(0, dtype=float)), dtype=float)

        if treat_vals.ndim == 1:
            treat_vals = treat_vals.reshape(-1, 1)

        treat_dim = treat_vals.shape[1] if treat_vals.ndim == 2 else 0
        if treat_dim > 0:
            treat_pad = np.zeros((pad_len, treat_dim), dtype=float)
            treat = np.concatenate([treat_vals[:actual_len], treat_pad], axis=0)
        else:
            treat = np.zeros((self.max_len, 0), dtype=float)

        patient_id = seq.get("patient_id", None)
        mortality_risk = float(seq.get("mortality_risk", 0.0))

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(t),
            torch.LongTensor(y),
            torch.BoolTensor(mask),
            torch.FloatTensor(static),
            torch.FloatTensor(treat),
            patient_id,
            torch.FloatTensor([mortality_risk]),
        )

def _make_onehot_encoder(categories=None):
    kwargs = dict(handle_unknown="ignore")
    if categories is not None:
        kwargs["categories"] = categories
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

@dataclass
class StaticEncoderBundle:
    num_imputer: SimpleImputer
    num_scaler: StandardScaler
    cat_imputer: SimpleImputer
    cat_encoder: OneHotEncoder
    num_cols: list
    cat_cols: list


def fit_static_encoders(
    clinical_df,
    *,
    id_col,
    stage_col,
    num_cols=None,
    cat_cols=None,
    exclude_cols=None,
):

    if num_cols is None and cat_cols is None:
        import warnings
        warnings.warn(
            "fit_static_encoders called without explicit num_cols/cat_cols; "
            "using every-numeric-column inference, which produces the "
            "281-dim leaky feature vector.",
            RuntimeWarning,
        )
        exclude = set(exclude_cols or [])
        exclude.add(id_col)
        if stage_col in clinical_df.columns:
            exclude.add(stage_col)
        df = clinical_df.copy()
        usable_cols = [c for c in df.columns if c not in exclude]
        df = df[usable_cols].copy()
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]
    else:
        num_cols = list(num_cols or [])
        cat_cols = list(cat_cols or [])
        missing_num = [c for c in num_cols if c not in clinical_df.columns]
        missing_cat = [c for c in cat_cols if c not in clinical_df.columns]
        if missing_num or missing_cat:
            raise KeyError(
                "fit_static_encoders: columns missing from clinical_df.\n"
                f"  Missing numeric: {missing_num}\n"
                f"  Missing categorical: {missing_cat}"
            )
        df = clinical_df.copy()

    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_categories = [CATEGORY_LEVELS[c] for c in cat_cols] if cat_cols else None
    cat_encoder = _make_onehot_encoder(categories=cat_categories)

    if num_cols:
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        from xai_decision_making.preprocessing.variables.clinical_variables import TYPICAL_RANGES
        for c in num_cols:
            rng = TYPICAL_RANGES.get(c)
            if rng is not None:
                lo, hi = rng
                df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

        num_imputer.fit(df[num_cols])
        num_scaler.fit(num_imputer.transform(df[num_cols]))

    if cat_cols:
        cat_imputer.fit(df[cat_cols].astype(str))
        cat_encoder.fit(cat_imputer.transform(df[cat_cols].astype(str)))

    return StaticEncoderBundle(
        num_imputer=num_imputer,
        num_scaler=num_scaler,
        cat_imputer=cat_imputer,
        cat_encoder=cat_encoder,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

def transform_static_row(row, enc):
    from xai_decision_making.preprocessing.variables.clinical_variables import TYPICAL_RANGES
    num_part = np.empty((1, 0), dtype=float)
    cat_part = np.empty((1, 0), dtype=float)

    if enc.num_cols:
        ndf = pd.DataFrame([{c: row.get(c, np.nan) for c in enc.num_cols}])
        for c in enc.num_cols:
            ndf[c] = pd.to_numeric(ndf[c], errors="coerce")
            rng = TYPICAL_RANGES.get(c)
            if rng is not None:
                lo, hi = rng
                ndf.loc[(ndf[c] < lo) | (ndf[c] > hi), c] = np.nan
        num_part = enc.num_scaler.transform(
            enc.num_imputer.transform(ndf[enc.num_cols])
        )

    if enc.cat_cols:
        cdf = pd.DataFrame([{c: row.get(c, "") for c in enc.cat_cols}])
        cat_part = enc.cat_encoder.transform(
            enc.cat_imputer.transform(cdf[enc.cat_cols].astype(str))
        )

    return np.hstack([num_part, cat_part]).astype(float).reshape(-1)

def build_treatment_matrix(merged_df, *, time_col):
    n_rows = len(merged_df)
    out = np.zeros((n_rows, 6), dtype=float)

    if time_col not in merged_df.columns or ID_COL not in merged_df.columns:
        return out

    surg_col = None
    for cand in ("op_datum", "chir_int1"):
        if cand in merged_df.columns:
            surg_col = cand
            break

    syst_cols = [c for c in merged_df.columns if str(c).startswith("syst_start_int")]

    if surg_col is None and not syst_cols:
        return out

    work = merged_df[[ID_COL, time_col]].copy()
    work["_row_idx"] = np.arange(n_rows)
    work["_t_days"] = pd.to_numeric(work[time_col], errors="coerce")

    surg_first = (
        merged_df.groupby(ID_COL)[surg_col].first() if surg_col is not None else None
    )
    syst_first = (
        merged_df.groupby(ID_COL)[syst_cols].first() if syst_cols else None
    )

    def _surg_days(pid):
        if surg_first is None:
            return []
        v = pd.to_numeric(pd.Series([surg_first.loc[pid]]), errors="coerce").iloc[0]
        return [float(v)] if pd.notna(v) else []

    def _syst_days(pid):
        if syst_first is None:
            return []
        vals = pd.to_numeric(syst_first.loc[pid], errors="coerce").values
        return [float(v) for v in vals if pd.notna(v)]

    for pid, group in work.sort_values([ID_COL, time_col]).groupby(ID_COL):
        surg_list = _surg_days(pid)
        syst_list = _syst_days(pid)

        prev_had_surg, prev_surg_day = False, None
        prev_had_syst, prev_syst_day = False, None

        for _, r in group.iterrows():
            t_curr = r["_t_days"]
            row_idx = int(r["_row_idx"])
            if pd.isna(t_curr):
                continue

            past_surg = [d for d in surg_list if d <= t_curr]
            if past_surg:
                latest = max(past_surg)
                ts_surg = (t_curr - latest) / 30.44 / 60.0
                had_surg = 1.0
                impulse_surg = (
                    1.0
                    if (not prev_had_surg
                        or (prev_surg_day is not None and latest != prev_surg_day))
                    else 0.0
                )
                prev_surg_day, prev_had_surg = latest, True
            else:
                ts_surg, had_surg, impulse_surg = 0.0, 0.0, 0.0

            past_syst = [d for d in syst_list if d <= t_curr]
            if past_syst:
                latest = max(past_syst)
                ts_syst = (t_curr - latest) / 30.44 / 60.0
                had_syst = 1.0
                impulse_syst = (
                    1.0
                    if (not prev_had_syst
                        or (prev_syst_day is not None and latest != prev_syst_day))
                    else 0.0
                )
                prev_syst_day, prev_had_syst = latest, True
            else:
                ts_syst, had_syst, impulse_syst = 0.0, 0.0, 0.0

            out[row_idx, 0] = ts_surg
            out[row_idx, 1] = had_surg
            out[row_idx, 2] = impulse_surg
            out[row_idx, 3] = ts_syst
            out[row_idx, 4] = had_syst
            out[row_idx, 5] = impulse_syst

    return out

MORTALITY_CACHE_PATH = Path("outputs") / "mortality_risk_cache.pkl"

def load_survival_model(path = None):
    if survival_config is None:
        print("Survival config not importable; mortality risk disabled.")
        return None

    if path is not None:
        p = Path(path)
        if not p.exists():
            print(f"Survival model not found: {p}")
            return None
        with p.open("rb") as f:
            model = pickle.load(f)
        print(f"Loaded survival model: {p}")
        return model

    candidate_dirs = []
    for attr in ["SURVIVAL_MODELS_DIR", "MODELS_DIR"]:
        d = getattr(survival_config, attr, None)
        if d:
            candidate_dirs.append(Path(d))

    for models_dir in candidate_dirs:
        if not models_dir.exists():
            continue
        candidates = sorted(models_dir.glob("*.pkl"))
        if candidates:
            chosen = candidates[-1]
            with chosen.open("rb") as f:
                model = pickle.load(f)
            print(f"Auto-selected survival model: {chosen}")
            return model

    print("No survival model .pkl files found; mortality risk disabled.")
    return None

def precompute_mortality_risk_lookup(
    survival_model,
    *,
    cache_path = MORTALITY_CACHE_PATH,
):
    if survival_model is None or survival_config is None:
        return {}

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "risk_lookup" in obj:
                print(
                    f"Loaded mortality risk cache: {cache_path} "
                    f"({len(obj['risk_lookup'])} patients)"
                )
                return obj["risk_lookup"]
        except Exception:
            pass

    train_path = Path(getattr(survival_config, "TRAIN_DATA_PATH", ""))
    if not train_path.exists():
        print(f"Survival TRAIN_DATA_PATH not found: {train_path}")
        return {}

    df = pd.read_csv(train_path, low_memory=False)

    if ID_COL not in df.columns:
        print(f"ID col '{ID_COL}' not found in survival data; mortality risk disabled.")
        return {}

    if hasattr(survival_model, "feature_names_"):
        feats = list(survival_model.feature_names_)
    else:
        feats = list(getattr(survival_config, "PRETREATMENT_QOL_FEATURES", []))

    feats = [c for c in feats if c in df.columns]
    if not feats:
        print("No matching survival features found in TRAIN_DATA_PATH.")
        return {}

    risk_lookup = {}

    for pid in df[ID_COL].dropna().unique():
        row = df[df[ID_COL] == pid].iloc[0]
        x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}], columns=feats)
        try:
            score = float(survival_model.predict(x)[0])
        except Exception:
            score = 0.5
        risk_lookup[pid] = score

    vals = list(risk_lookup.values())
    if vals:
        vmin = float(min(vals))
        vmax = float(max(vals))
        if vmax > vmin:
            risk_lookup = {k: (v - vmin) / (vmax - vmin) for k, v in risk_lookup.items()}
        else:
            risk_lookup = {k: 0.0 for k in risk_lookup.keys()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump({"risk_lookup": risk_lookup}, f)

    print(f"Saved mortality risk cache: {cache_path}")
    return risk_lookup

def lookup_mortality_risk(
    risk_lookup,
    patient_id,
):
    if not risk_lookup:
        return 0.0

    for key in (patient_id, str(patient_id)):
        if key in risk_lookup:
            return float(risk_lookup[key])

    try:
        ik = int(patient_id)
        if ik in risk_lookup:
            return float(risk_lookup[ik])
    except Exception:
        pass

    return 0.5

def load_merged(pacap_path, clinical_path):
    df_p = pd.read_csv(pacap_path, low_memory=False)
    df_c = pd.read_csv(clinical_path, low_memory=False)

    if ID_COL not in df_p.columns or ID_COL not in df_c.columns:
        raise KeyError(
            f"ID_COL '{ID_COL}' must exist in both PACAP and clinical data."
        )

    resolved_time_col = _resolve_time_col(df_p, TIME_COL)
    if resolved_time_col != TIME_COL:
        print(
            f"TIME_COL '{TIME_COL}' not found in PACAP; "
            f"using '{resolved_time_col}' instead."
        )
        df_p = df_p.rename(columns={resolved_time_col: TIME_COL})

    scores = calculate_qlq_c30_scores(df_p)
    summary_col = "Summary Score: QLQ-C30 Summary Score"
    if summary_col not in scores.columns:
        raise KeyError("QLQ-C30 scoring did not produce summary score column.")

    df_p = df_p.copy()
    df_p["qol_summary"] = scores[summary_col] / 100.0

    df_m = pd.merge(df_p, df_c, on=ID_COL, how="left")

    df_m[TIME_COL] = _as_numeric(df_m[TIME_COL])
    if df_m[TIME_COL].notna().sum() == 0:
        raise ValueError(
            f"Resolved TIME_COL '{TIME_COL}' exists but could not be parsed as "
            "numeric. The model expects a numeric longitudinal time variable."
        )

    df_m = df_m.dropna(subset=[TIME_COL, "qol_summary"]).copy()
    df_m["time_months"] = df_m[TIME_COL] / 30.44
    df_m["time_normalized"] = df_m["time_months"] / 60.0
    df_m = df_m.dropna(subset=["time_normalized"]).copy()
    df_m = df_m[df_m["time_normalized"] >= 0].copy()

    if STAGE_COL in df_m.columns:
        df_m["label_ordinal"] = df_m[STAGE_COL].apply(stage_to_ordinal).astype(int)
    else:
        df_m["label_ordinal"] = 4

    return df_m

def build_sequences_from_merged(
    merged_df,
    clinical_df,
    *,
    use_static,
    use_treatment,
    encoders,
    fit_encoders,
    risk_lookup,
):
    clinical_unique = clinical_df.drop_duplicates(subset=[ID_COL], keep="first").copy()

    static_map = {}
    static_dim = 0

    if use_static:
        if fit_encoders or encoders is None:
            encoders = fit_static_encoders(
            clinical_unique,
            id_col=ID_COL,
            stage_col=STAGE_COL,
            num_cols=NUM_VARS,
            cat_cols=CAT_VARS,
            )

        for _, row in clinical_unique.iterrows():
            pid = row[ID_COL]
            static_map[pid] = transform_static_row(row, encoders)

        if static_map:
            static_dim = next(iter(static_map.values())).shape[0]
        else:
            dummy_row = pd.Series(
                {c: np.nan for c in encoders.num_cols} |
                {c: "" for c in encoders.cat_cols}
            )
            static_dim = transform_static_row(dummy_row, encoders).shape[0]

    treat_mat = None
    if use_treatment:
        treat_mat = build_treatment_matrix(merged_df, time_col=TIME_COL)
        merged_df = merged_df.copy()
        merged_df["_treat_idx"] = np.arange(len(merged_df))

    sequences = []
    grouped = merged_df.sort_values([ID_COL, "time_normalized"]).groupby(ID_COL)

    for patient_id, group in grouped:
        if len(group) < 2:
            continue

        t_vals = group["time_normalized"].values.astype(float)
        x_vals = group["qol_summary"].values.astype(float)

        stage_val = int(group["label_ordinal"].iloc[0])
        y_vals = np.full(len(group), stage_val, dtype=int)

        if use_static:
            static_vec = static_map.get(
                patient_id,
                np.zeros(static_dim, dtype=float),
            )
        else:
            static_vec = np.zeros(0, dtype=float)

        if use_treatment and treat_mat is not None:
            idxs = group["_treat_idx"].values.astype(int)
            treat_vals = treat_mat[idxs]
        else:
            treat_vals = np.zeros((len(group), 0), dtype=float)

        sequences.append(
            {
                "patient_id": patient_id,
                "t": t_vals,
                "x": x_vals,
                "y": y_vals,
                "static": static_vec,
                "treat": treat_vals,
                "mortality_risk": lookup_mortality_risk(risk_lookup, patient_id),
            }
        )

    return sequences, encoders, static_dim

def _ordinal_loss(
    ord_net,
    x,
    t,
    y,
    mask,
):
    mask_flat = mask.view(-1).bool()
    if not mask_flat.any():
        return None

    probs = ord_net(x.view(-1, 1)[mask_flat], t.view(-1, 1)[mask_flat])
    y_true = y.view(-1)[mask_flat].long()

    valid = (y_true >= 0) & (y_true < probs.shape[1])
    if not valid.any():
        return None

    probs = probs[valid]
    y_true = y_true[valid]
    idx = torch.arange(len(y_true), device=probs.device)

    return -torch.log(probs[idx, y_true] + 1e-6).mean()

def _ordinal_preds(
    ord_net,
    x,
    t,
    y,
    mask,
):
    mask_flat = mask.view(-1).bool()
    if not mask_flat.any():
        return [], []

    probs = ord_net(x.view(-1, 1)[mask_flat], t.view(-1, 1)[mask_flat])
    y_true = y.view(-1)[mask_flat].long()

    valid = (y_true >= 0) & (y_true < probs.shape[1])
    if not valid.any():
        return [], []

    probs = probs[valid]
    y_true = y_true[valid]
    pred = torch.argmax(probs, dim=1)

    return pred.cpu().numpy().tolist(), y_true.cpu().numpy().tolist()

def train_one_fold(
    *,
    fold_id,
    train_seqs,
    val_seqs,
    static_dim,
    use_static_rdm,
    use_static_tdm,
    use_treatment,
    use_mortality_risk,
    use_ordinal,
    epochs,
    patience,
):
    config = ConDORConfig(
        static_dim=static_dim,
        use_static_rdm=use_static_rdm,
        use_static_tdm=use_static_tdm,
        use_treatment=use_treatment,
        use_mortality_risk=use_mortality_risk,
    )

    ord_net = OrdinalRegression(config).to(config.device)
    rdm_net = RDM(config).to(config.device)
    tdm_net = TDM(config).to(config.device)

    opt_ord = optim.Adam(ord_net.parameters(), lr=1e-3)
    opt_rdm = optim.Adam(rdm_net.parameters(), lr=1e-3)
    opt_tdm = optim.Adam(tdm_net.parameters(), lr=5e-4 if use_treatment else 1e-3)

    train_loader = DataLoader(
        ConDORSequenceDataset(train_seqs, max_len=10),
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ConDORSequenceDataset(val_seqs, max_len=10),
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        ord_net.train()
        rdm_net.train()
        tdm_net.train()

        train_loss_total = 0.0
        train_batches = 0

        for batch in train_loader:
            x, t, y, mask, static, treat, patient_ids, mortality_risk = batch
            x = x.to(config.device)
            t = t.to(config.device)
            y = y.to(config.device)
            mask = mask.to(config.device)
            static = static.to(config.device)
            treat = treat.to(config.device)
            mortality_risk = mortality_risk.to(config.device).float()

            loss_batch = 0.0

            if use_ordinal:
                loss_ord = _ordinal_loss(ord_net, x, t, y, mask)
                if loss_ord is not None:
                    opt_ord.zero_grad()
                    loss_ord.backward()
                    opt_ord.step()
                    loss_batch += float(loss_ord.item())

            x_0 = x[:, 0].unsqueeze(1)
            noise = torch.randn_like(x_0)
            ts = torch.randint(0, config.rdm_steps, (x_0.shape[0],), device=config.device)
            x_noisy = x_0 + noise * (ts.unsqueeze(1).float() / float(config.rdm_steps))

            static_arg_rdm = static if use_static_rdm else None
            risk_arg = mortality_risk if use_mortality_risk else None

            pred_noise = rdm_net(
                x_noisy, ts.float(), y[:, 0], t[:, 0],
                static=static_arg_rdm, mortality_risk=risk_arg,
            )
            loss_rdm = nn.MSELoss()(pred_noise, noise)
            opt_rdm.zero_grad()
            loss_rdm.backward()
            opt_rdm.step()
            loss_batch += float(loss_rdm.item())

            if x.shape[1] > 1:
                seq_mask = mask[:, 1:].bool()
                curr_x = x[:, :-1][seq_mask].unsqueeze(1)
                next_x = x[:, 1:][seq_mask].unsqueeze(1)
                t_t = t[:, 1:][seq_mask]
                t_y = y[:, 1:][seq_mask]

                static_seq = None
                if use_static_tdm:
                    static_seq = static.unsqueeze(1).expand(-1, x.shape[1] - 1, -1)
                    static_seq = static_seq[seq_mask]

                treat_seq = None
                if use_treatment:
                    treat_seq = treat[:, 1:][seq_mask]

                if len(curr_x) > 0:
                    ds = torch.randint(
                        0, config.diffusion_steps, (curr_x.shape[0],), device=config.device,
                    )
                    alpha = ds.float().unsqueeze(1) / float(config.diffusion_steps)
                    target = (1 - alpha) * curr_x + alpha * next_x
                    true_delta = (next_x - curr_x) / float(config.diffusion_steps)

                    pred_delta = tdm_net(target, ds, t_t, t_y, static=static_seq, treat=treat_seq)
                    loss_tdm = nn.MSELoss()(pred_delta, true_delta)
                    opt_tdm.zero_grad()
                    loss_tdm.backward()
                    opt_tdm.step()
                    loss_batch += float(loss_tdm.item())

            train_loss_total += loss_batch
            train_batches += 1

        ord_net.eval()
        rdm_net.eval()
        tdm_net.eval()

        val_loss_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, t, y, mask, static, treat, patient_ids, mortality_risk = batch
                x = x.to(config.device)
                t = t.to(config.device)
                y = y.to(config.device)
                mask = mask.to(config.device)
                static = static.to(config.device)
                treat = treat.to(config.device)
                mortality_risk = mortality_risk.to(config.device).float()

                loss_val_batch = 0.0

                if use_ordinal:
                    loss_ord = _ordinal_loss(ord_net, x, t, y, mask)
                    if loss_ord is not None:
                        loss_val_batch += float(loss_ord.item())

                x_0 = x[:, 0].unsqueeze(1)
                noise = torch.randn_like(x_0)
                ts = torch.randint(0, config.rdm_steps, (x_0.shape[0],), device=config.device)
                x_noisy = x_0 + noise * (ts.unsqueeze(1).float() / float(config.rdm_steps))

                static_arg_rdm = static if use_static_rdm else None
                risk_arg = mortality_risk if use_mortality_risk else None

                pred_noise = rdm_net(
                    x_noisy, ts.float(), y[:, 0], t[:, 0],
                    static=static_arg_rdm, mortality_risk=risk_arg,
                )
                loss_rdm = nn.MSELoss()(pred_noise, noise)
                loss_val_batch += float(loss_rdm.item())

                if x.shape[1] > 1:
                    seq_mask = mask[:, 1:].bool()
                    curr_x = x[:, :-1][seq_mask].unsqueeze(1)
                    next_x = x[:, 1:][seq_mask].unsqueeze(1)
                    t_t = t[:, 1:][seq_mask]
                    t_y = y[:, 1:][seq_mask]

                    static_seq = None
                    if use_static_tdm:
                        static_seq = static.unsqueeze(1).expand(-1, x.shape[1] - 1, -1)
                        static_seq = static_seq[seq_mask]

                    treat_seq = None
                    if use_treatment:
                        treat_seq = treat[:, 1:][seq_mask]

                    if len(curr_x) > 0:
                        ds = torch.randint(
                            0, config.diffusion_steps, (curr_x.shape[0],), device=config.device,
                        )
                        alpha = ds.float().unsqueeze(1) / float(config.diffusion_steps)
                        target = (1 - alpha) * curr_x + alpha * next_x
                        true_delta = (next_x - curr_x) / float(config.diffusion_steps)

                        pred_delta = tdm_net(target, ds, t_t, t_y, static=static_seq, treat=treat_seq)
                        loss_tdm = nn.MSELoss()(pred_delta, true_delta)
                        loss_val_batch += float(loss_tdm.item())

                val_loss_total += loss_val_batch
                val_batches += 1

        avg_train = train_loss_total / max(train_batches, 1)
        avg_val = val_loss_total / max(val_batches, 1)

        if (epoch + 1) % 10 == 0:
            print(
                f"Fold {fold_id + 1} | Epoch {epoch + 1}/{epochs} | "
                f"Train {avg_train:.4f} | Val {avg_val:.4f}"
            )

        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            best_state = {
                "ordinal": copy.deepcopy(ord_net.state_dict()),
                "rdm": copy.deepcopy(rdm_net.state_dict()),
                "tdm": copy.deepcopy(tdm_net.state_dict()),
            }
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Fold {fold_id + 1}: early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        ord_net.load_state_dict(best_state["ordinal"])
        rdm_net.load_state_dict(best_state["rdm"])
        tdm_net.load_state_dict(best_state["tdm"])

    return (ord_net, rdm_net, tdm_net), config

def evaluate_sequences(
    *,
    models,
    config,
    sequences,
    use_static_rdm,
    use_static_tdm,
    use_treatment,
    use_mortality_risk,
    use_ordinal,
):
    ord_net, rdm_net, tdm_net = models
    ord_net.eval()
    rdm_net.eval()
    tdm_net.eval()

    loader = DataLoader(
        ConDORSequenceDataset(sequences, max_len=10),
        batch_size=1,
        shuffle=False,
    )

    rdm_real = []
    rdm_gen = []
    full_real = []
    full_gen = []

    time_specific_real = {1: [], 3: [], 6: [], 9: [], 12: []}
    time_specific_gen = {1: [], 3: [], 6: [], 9: [], 12: []}

    ordinal_preds = []
    ordinal_true = []

    time_targets = {1: 1 / 60.0, 3: 0.05, 6: 0.10, 9: 0.15, 12: 12 / 60.0}
    time_tolerance = 0.02

    with torch.no_grad():
        for batch in loader:
            x, t, y, mask, static, treat, patient_ids, mortality_risk = batch
            x = x.to(config.device)
            t = t.to(config.device)
            y = y.to(config.device)
            mask = mask.to(config.device)
            static = static.to(config.device)
            treat = treat.to(config.device)
            mortality_risk = mortality_risk.to(config.device).float()

            if use_ordinal:
                pred_list, true_list = _ordinal_preds(ord_net, x, t, y, mask)
                ordinal_preds.extend(pred_list)
                ordinal_true.extend(true_list)

            static_arg = static if use_static_rdm else None
            risk_arg = mortality_risk if use_mortality_risk else None

            x_t = torch.randn_like(x[:, 0].unsqueeze(1))
            for i in reversed(range(config.rdm_steps)):
                t_step = torch.full(
                    (x.shape[0],), i, device=config.device, dtype=torch.float32,
                )
                pred_noise = rdm_net(
                    x_t, t_step, y[:, 0], t[:, 0],
                    static=static_arg, mortality_risk=risk_arg,
                )
                x_t = x_t - pred_noise * (1.0 / float(config.rdm_steps))

            x0_gen = torch.clamp(x_t * 100.0, 0, 100)
            rdm_real.append(float(x[:, 0].item() * 100.0))
            rdm_gen.append(float(x0_gen.item()))

            if x.shape[1] > 1:
                seq_mask = mask[:, 1:].bool()
                curr_x = x[:, :-1][seq_mask].unsqueeze(1)
                next_x = x[:, 1:][seq_mask].unsqueeze(1)
                t_t = t[:, 1:][seq_mask]
                t_y = y[:, 1:][seq_mask]

                static_seq = None
                if use_static_tdm:
                    static_seq = static.unsqueeze(1).expand(-1, x.shape[1] - 1, -1)
                    static_seq = static_seq[seq_mask]

                treat_seq = None
                if use_treatment:
                    treat_seq = treat[:, 1:][seq_mask]

                if len(curr_x) > 0:
                    x_traj = curr_x.clone()
                    for d in range(config.diffusion_steps):
                        ds = torch.full(
                            (x_traj.shape[0],), d, device=config.device, dtype=torch.long,
                        )
                        delta = tdm_net(x_traj, ds, t_t, t_y, static=static_seq, treat=treat_seq)
                        x_traj = torch.clamp(x_traj + delta, 0.0, 1.0)

                    x_final = torch.clamp(x_traj * 100.0, 0, 100)
                    next_x_scaled = next_x * 100.0

                    full_real.extend(next_x_scaled.squeeze(1).cpu().numpy().flatten().tolist())
                    full_gen.extend(x_final.squeeze(1).cpu().numpy().flatten().tolist())

                    t_np = t_t.cpu().numpy().flatten()
                    next_np = next_x_scaled.squeeze(1).cpu().numpy().flatten()
                    gen_np = x_final.squeeze(1).cpu().numpy().flatten()

                    months = (t_np * 60.0).round().astype(int)
                    for i, m in enumerate(months):
                        if m <= 0:
                            continue
                        time_specific_real.setdefault(int(m), []).append(float(next_np[i]))
                        time_specific_gen.setdefault(int(m), []).append(float(gen_np[i]))

                    for m, target in time_targets.items():
                        diffs = np.abs(t_np - target)
                        close = np.where(diffs < time_tolerance)[0]
                        if len(close) > 0:
                            j = close[np.argmin(diffs[close])]
                            time_specific_real[m].append(float(next_np[j]))
                            time_specific_gen[m].append(float(gen_np[j]))

    results = {
        "RDM_RMSE": rmse(rdm_real, rdm_gen),
        "Full_RMSE": rmse(full_real, full_gen),
        "RMSE_1M": rmse(time_specific_real.get(1, []), time_specific_gen.get(1, [])),
        "RMSE_3M": rmse(time_specific_real.get(3, []), time_specific_gen.get(3, [])),
        "RMSE_6M": rmse(time_specific_real.get(6, []), time_specific_gen.get(6, [])),
        "RMSE_9M": rmse(time_specific_real.get(9, []), time_specific_gen.get(9, [])),
        "RMSE_12M": rmse(time_specific_real.get(12, []), time_specific_gen.get(12, [])),
    }

    if use_ordinal and len(ordinal_preds) > 0:
        preds = np.asarray(ordinal_preds)
        true = np.asarray(ordinal_true)
        results["Ordinal_Accuracy"] = float(np.mean(preds == true))
    else:
        results["Ordinal_Accuracy"] = float("nan")

    min_n = 100
    for m in sorted(time_specific_real.keys()):
        if m in (1, 3, 6, 9, 12):
            continue
        if m <= 0:
            continue
        if len(time_specific_real[m]) >= min_n:
            results[f"RMSE_{m}M"] = rmse(time_specific_real[m], time_specific_gen[m])

    return results

def run_study(
    study_name,
    *,
    use_static_rdm,
    use_static_tdm,
    use_treatment,
    use_mortality_risk,
    use_ordinal,
    n_folds,
    epochs,
    patience,
    survival_model_path = None,
    use_black_envelope = False,
):
    safe = _safe_study_name(study_name)
    out_dir = Path("outputs") / f"study_{safe}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== STUDY: {study_name} ({n_folds}-fold) ===")
    print(f"Saving to: {out_dir}")
    if use_black_envelope:
        print("Black envelope evaluation: ENABLED")
    else:
        print("Black envelope evaluation: disabled (use --black-envelope to enable)")

    if use_mortality_risk:
        model = load_survival_model(survival_model_path)
        risk_lookup = precompute_mortality_risk_lookup(model)
    else:
        risk_lookup = None

    print("Loading dev and test sets...")
    df_dev = load_merged("data/dev_set_pacap.csv", "data/dev_set.csv")
    df_test = load_merged("data/test_set_pacap.csv", "data/test_set.csv")
    df_combined = pd.concat([df_dev, df_test], ignore_index=True)

    df_clin_dev = pd.read_csv("data/dev_set.csv", low_memory=False)
    df_clin_test = pd.read_csv("data/test_set.csv", low_memory=False)
    df_clin_combined = pd.concat([df_clin_dev, df_clin_test], ignore_index=True)

    unique_patients = df_combined[ID_COL].dropna().unique()
    print(f"Unique patients: {len(unique_patients)}")

    use_static_any = use_static_rdm or use_static_tdm

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_patients)):
        train_ids = set(unique_patients[train_idx])
        val_ids = set(unique_patients[val_idx])

        train_df = df_combined[df_combined[ID_COL].isin(train_ids)].copy()
        val_df = df_combined[df_combined[ID_COL].isin(val_ids)].copy()

        train_clin = df_clin_combined[df_clin_combined[ID_COL].isin(train_ids)].copy()
        val_clin = df_clin_combined[df_clin_combined[ID_COL].isin(val_ids)].copy()

        train_seqs, enc, static_dim = build_sequences_from_merged(
            train_df,
            train_clin,
            use_static=use_static_any,
            use_treatment=use_treatment,
            encoders=None,
            fit_encoders=True,
            risk_lookup=risk_lookup,
        )

        val_seqs, _, _ = build_sequences_from_merged(
            val_df,
            val_clin,
            use_static=use_static_any,
            use_treatment=use_treatment,
            encoders=enc,
            fit_encoders=False,
            risk_lookup=risk_lookup,
        )

        print(
            f"\nFold {fold_idx + 1}/{n_folds}: "
            f"train_seqs={len(train_seqs)}, "
            f"val_seqs={len(val_seqs)}, "
            f"static_dim={static_dim}"
        )

        if len(train_seqs) == 0 or len(val_seqs) == 0:
            print(f"Skipping fold {fold_idx + 1}: empty sequences.")
            continue

        models, conf = train_one_fold(
            fold_id=fold_idx,
            train_seqs=train_seqs,
            val_seqs=val_seqs,
            static_dim=static_dim,
            use_static_rdm=use_static_rdm,
            use_static_tdm=use_static_tdm,
            use_treatment=use_treatment,
            use_mortality_risk=use_mortality_risk,
            use_ordinal=use_ordinal,
            epochs=epochs,
            patience=patience,
        )

        cv_metrics = evaluate_sequences(
            models=models,
            config=conf,
            sequences=val_seqs,
            use_static_rdm=use_static_rdm,
            use_static_tdm=use_static_tdm,
            use_treatment=use_treatment,
            use_mortality_risk=use_mortality_risk,
            use_ordinal=use_ordinal,
        )
        cv_metrics["fold"] = fold_idx + 1
        cv_metrics["eval_set"] = "CV_Validation"
        all_rows.append(cv_metrics)
        print(f"Fold {fold_idx + 1} CV metrics: {cv_metrics}")

        be_metrics = None
        if use_black_envelope:
            be_pacap = "data/black_envelop_test_set_pacap.csv"
            be_clin_path = "data/black_envelop_test_set.csv"

            if os.path.exists(be_pacap) and os.path.exists(be_clin_path):
                df_be = load_merged(be_pacap, be_clin_path)
                df_be_clin = pd.read_csv(be_clin_path, low_memory=False)

                be_seqs, _, _ = build_sequences_from_merged(
                    df_be,
                    df_be_clin,
                    use_static=use_static_any,
                    use_treatment=use_treatment,
                    encoders=enc,
                    fit_encoders=False,
                    risk_lookup=risk_lookup,
                )

                if len(be_seqs) > 0:
                    be_metrics = evaluate_sequences(
                        models=models,
                        config=conf,
                        sequences=be_seqs,
                        use_static_rdm=use_static_rdm,
                        use_static_tdm=use_static_tdm,
                        use_treatment=use_treatment,
                        use_mortality_risk=use_mortality_risk,
                        use_ordinal=use_ordinal,
                    )
                    be_metrics["fold"] = fold_idx + 1
                    be_metrics["eval_set"] = "Black_Envelope"
                    all_rows.append(be_metrics)
                    print(f"Fold {fold_idx + 1} Black Envelope metrics: {be_metrics}")
                else:
                    print(f"Fold {fold_idx + 1}: no valid Black Envelope sequences.")
            else:
                print(f"Black envelope files not found at {be_pacap} / {be_clin_path}.")

        ord_net, rdm_net, tdm_net = models
        fold_rmse = (
            float(be_metrics["Full_RMSE"])
            if be_metrics is not None and not pd.isna(be_metrics.get("Full_RMSE"))
            else float(cv_metrics.get("Full_RMSE", float("nan")))
        )

        checkpoint = {
            "ordinal": ord_net.state_dict(),
            "rdm": rdm_net.state_dict(),
            "tdm": tdm_net.state_dict(),
            "config_args": {
                "static_dim": conf.static_dim,
                "use_static_rdm": use_static_rdm,
                "use_static_tdm": use_static_tdm,
                "use_treatment": use_treatment,
                "use_mortality_risk": use_mortality_risk,
            },
            "encoders": {
                "static_num_imputer": enc.num_imputer if enc is not None else None,
                "static_num_scaler": enc.num_scaler if enc is not None else None,
                "static_cat_imputer": enc.cat_imputer if enc is not None else None,
                "static_cat_encoder": enc.cat_encoder if enc is not None else None,
                "static_num_cols": enc.num_cols if enc is not None else [],
                "static_cat_cols": enc.cat_cols if enc is not None else [],
            },
            "cv_metrics": cv_metrics,
            "be_metrics": be_metrics,
        }

        ckpt_path = out_dir / f"fold_{fold_idx + 1}_rmse_{fold_rmse:.2f}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    df_results = pd.DataFrame(all_rows)
    results_path = out_dir / f"results_{safe}.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")

    if len(df_results) == 0:
        print("No results to summarize.")
        return

    if "eval_set" in df_results.columns:
        for eval_set in sorted(df_results["eval_set"].dropna().unique()):
            sub = df_results[df_results["eval_set"] == eval_set]
            if len(sub) == 0:
                continue

            print(f"\n=== Summary: {study_name} [{eval_set}] ===")

            if "Full_RMSE" in sub.columns:
                print(
                    f"Full_RMSE: "
                    f"{sub['Full_RMSE'].mean():.4f} ± {sub['Full_RMSE'].std():.4f}"
                )
            if "RDM_RMSE" in sub.columns:
                print(
                    f"RDM_RMSE:  "
                    f"{sub['RDM_RMSE'].mean():.4f} ± {sub['RDM_RMSE'].std():.4f}"
                )
            if "Ordinal_Accuracy" in sub.columns:
                acc = sub["Ordinal_Accuracy"].dropna()
                if len(acc) > 0:
                    print(
                        f"Ordinal_Accuracy: "
                        f"{acc.mean():.4f} ± {acc.std():.4f}"
                    )

            # Per-month RMSE summary
            month_cols = [c for c in sub.columns if re.match(r"RMSE_\d+M$", c)]
            for mc in sorted(month_cols, key=lambda c: int(re.search(r"\d+", c).group())):
                vals = sub[mc].dropna()
                if len(vals) > 0:
                    print(f"{mc}: {vals.mean():.4f} ± {vals.std():.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Run QoL ConDOR ablation study with patient-level CV."
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-ordinal", action="store_true",
        help="Enable ordinal regression auxiliary task",
    )
    parser.add_argument(
        "--use-mortality-risk", action="store_true",
        help="Use pre-computed mortality risk as RDM conditioning",
    )
    parser.add_argument(
        "--survival-model-path", type=str, default=None,
        help="Path to pickled survival model for mortality risk",
    )
    parser.add_argument(
        "--black-envelope", action="store_true",
        help="Evaluate on black envelope test set each fold (opt-in)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    common = dict(
        use_mortality_risk=args.use_mortality_risk,
        use_ordinal=args.use_ordinal,
        n_folds=args.folds,
        epochs=args.epochs,
        patience=args.patience,
        survival_model_path=args.survival_model_path,
        use_black_envelope=args.black_envelope,
    )

    run_study(
        "Baseline",
        use_static_rdm=False,
        use_static_tdm=False,
        use_treatment=False,
        **common,
    )

    run_study(
        "RDM with Clinical Features",
        use_static_rdm=True,
        use_static_tdm=False,
        use_treatment=False,
        **common,
    )

    run_study(
        "TDM with Treatment",
        use_static_rdm=False,
        use_static_tdm=False,
        use_treatment=True,
        **common,
    )

    run_study(
        "RDM Clinical Features + TDM Treatment",
        use_static_rdm=True,
        use_static_tdm=False,
        use_treatment=True,
        **common,
    )


if __name__ == "__main__":
    main()