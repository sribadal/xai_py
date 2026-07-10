"""
EORTC QLQ-C30 scoring implementation.

Computes all 15 scale scores (5 functional, 9 symptom, 1 global health)
and the QLQ-C30 Summary Score according to the EORTC scoring manual.
Handles out-of-range values, missing data (half-rule), and version
differences (v1, v2, v3, v303).
"""
import numpy as np
import pandas as pd
from xai_decision_making.preprocessing.variables.qol_scales import get_qlq_c30_scales


def calculate_qlq_c30_scores(data, version=3, column_prefix="qlqc30"):
    df = data.copy()
    scores = pd.DataFrame(index=df.index)
    col_map = {f"q{i}": f"{column_prefix}{i}" for i in range(1, 31)}
    scales = get_qlq_c30_scales(version)

    for scale_name, scale_info in scales.items():
        mapped_items = [col_map[item] for item in scale_info["items"]]
        item_range = scale_info["range"]
        scale_type = scale_info["type"]

        for item in scale_info["items"]:
            col = col_map[item]
            mask = (df[col] < 1.0) | (df[col] > scale_info["range"] + 1)
            df.loc[mask, col] = np.nan

        non_missing = df[mapped_items].notna().sum(axis=1)
        raw_score = df[mapped_items].mean(axis=1)
        valid_scores = non_missing >= len(mapped_items) / 2

        if scale_type == "F":
            scores[scale_name] = np.where(
                valid_scores, (1 - (raw_score - 1) / item_range) * 100, np.nan
            )
        else:
            scores[scale_name] = np.where(
                valid_scores, ((raw_score - 1) / item_range) * 100, np.nan
            )

    scale_descriptions = {
        "QL": "Global health status/QoL",
        "QL2": "Global health status/QoL (revised)",
        "PF": "Physical functioning",
        "PF2": "Physical functioning (revised)",
        "RF": "Role functioning",
        "RF2": "Role functioning (revised)",
        "EF": "Emotional functioning",
        "CF": "Cognitive functioning",
        "SF": "Social functioning",
        "FA": "Fatigue",
        "NV": "Nausea and vomiting",
        "PA": "Pain",
        "DY": "Dyspnoea",
        "SL": "Insomnia",
        "AP": "Appetite loss",
        "CO": "Constipation",
        "DI": "Diarrhoea",
        "FI": "Financial difficulties",
    }

    scores.columns.name = "Scale"
    for col in scores.columns:
        if col in scale_descriptions:
            scores.rename(
                columns={col: f"{col}: {scale_descriptions[col]}"}, inplace=True
            )

    functional_scales = [
        col
        for col in scores.columns
        if any(fs in col for fs in ["PF", "RF", "EF", "CF", "SF"])
    ]
    symptom_scales = [
        col
        for col in scores.columns
        if any(ss in col for ss in ["FA", "NV", "PA", "DY", "SL", "AP", "CO", "DI"])
    ]

    required_scales = functional_scales + symptom_scales

    if len(required_scales) == 13:
        summary_calc = scores.copy()
        for symptom_scale in symptom_scales:
            summary_calc[symptom_scale] = 100 - summary_calc[symptom_scale]
        summary_score = summary_calc[required_scales].mean(axis=1, skipna=False)
        scores["Summary Score: QLQ-C30 Summary Score"] = summary_score

    return scores