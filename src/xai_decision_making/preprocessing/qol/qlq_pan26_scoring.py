"""
EORTC QLQ-PAN26 scoring implementation.

Computes the 7 symptom scale scores for the pancreatic cancer module
(pain, digestive, jaundice, bowel, body image, emotional, other) per
the EORTC scoring manual. All PAN26 scales are symptom scales where
higher scores indicate worse symptom burden.
"""
import numpy as np
import pandas as pd
from xai_decision_making.preprocessing.variables.qol_scales import get_qlq_c30_scales


def calculate_qlq_pan26_scores(data, column_prefix="qlqpan"):
    df = data.copy()
    scores = pd.DataFrame(index=df.index)
    col_map = {f"pan{i}": f"{column_prefix}{i}" for i in range(1, 27)}

    scales = {
        "PANPAIN": {
            "items": ["pan1", "pan2", "pan3", "pan4"],
            "range": 3,
            "type": "S",
            "desc": "Pancreatic pain",
        },
        "PANDIGEST": {
            "items": ["pan5", "pan6", "pan7", "pan8"],
            "range": 3,
            "type": "S",
            "desc": "Digestive symptoms",
        },
        "PANJAUND": {
            "items": ["pan9", "pan10", "pan11"],
            "range": 3,
            "type": "S",
            "desc": "Jaundice symptoms",
        },
        "PANBOWEL": {
            "items": ["pan12", "pan13", "pan14"],
            "range": 3,
            "type": "S",
            "desc": "Altered bowel habit",
        },
        "PANIMAGE": {
            "items": ["pan15", "pan16"],
            "range": 3,
            "type": "S",
            "desc": "Body image/self-perception",
        },
        "PANEMOT": {
            "items": ["pan17", "pan18", "pan19", "pan20", "pan21"],
            "range": 3,
            "type": "S",
            "desc": "Emotional functioning",
        },
        "PANOTHER": {
            "items": ["pan22", "pan23", "pan24", "pan25", "pan26"],
            "range": 3,
            "type": "S",
            "desc": "Other symptoms",
        },
    }

    for scale_name, scale_info in scales.items():
        mapped_items = [col_map[item] for item in scale_info["items"]]
        item_range = scale_info["range"]

        non_missing = df[mapped_items].notna().sum(axis=1)
        raw_score = df[mapped_items].mean(axis=1)
        valid_scores = non_missing >= len(mapped_items) / 2

        scores[scale_name] = np.where(
            valid_scores, ((raw_score - 1) / item_range) * 100, np.nan
        )

    for scale_name, scale_info in scales.items():
        desc = scale_info["desc"]
        scores.rename(columns={scale_name: f"{scale_name}: {desc}"}, inplace=True)

    return scores