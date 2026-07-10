"""
EORTC QLQ-C30 scale definitions by questionnaire version.

Maps each scale (functional, symptom, global health) to its constituent
items, response range, and scale type. Supports versions 1, 2, 3, and
303 (QLQ-C30+3) as defined in the EORTC scoring manual.
"""
def get_qlq_c30_scales(version=3):
    if version == 3:
        scales = {
            "QL2": {"items": ["q29", "q30"], "range": 6, "type": "QL"},
            "PF2": {"items": ["q1", "q2", "q3", "q4", "q5"], "range": 3, "type": "F"},
            "RF2": {"items": ["q6", "q7"], "range": 3, "type": "F"},
            "EF": {"items": ["q21", "q22", "q23", "q24"], "range": 3, "type": "F"},
            "CF": {"items": ["q20", "q25"], "range": 3, "type": "F"},
            "SF": {"items": ["q26", "q27"], "range": 3, "type": "F"},
            "FA": {"items": ["q10", "q12", "q18"], "range": 3, "type": "S"},
            "NV": {"items": ["q14", "q15"], "range": 3, "type": "S"},
            "PA": {"items": ["q9", "q19"], "range": 3, "type": "S"},
            "DY": {"items": ["q8"], "range": 3, "type": "S"},
            "SL": {"items": ["q11"], "range": 3, "type": "S"},
            "AP": {"items": ["q13"], "range": 3, "type": "S"},
            "CO": {"items": ["q16"], "range": 3, "type": "S"},
            "DI": {"items": ["q17"], "range": 3, "type": "S"},
            "FI": {"items": ["q28"], "range": 3, "type": "S"},
        }
    elif version == 2:
        scales = {
            "QL2": {"items": ["q29", "q30"], "range": 6, "type": "QL"},
            "PF": {"items": ["q1", "q2", "q3", "q4", "q5"], "range": 1, "type": "F"},
            "RF2": {"items": ["q6", "q7"], "range": 3, "type": "F"},
            "EF": {"items": ["q21", "q22", "q23", "q24"], "range": 3, "type": "F"},
            "CF": {"items": ["q20", "q25"], "range": 3, "type": "F"},
            "SF": {"items": ["q26", "q27"], "range": 3, "type": "F"},
            "FA": {"items": ["q10", "q12", "q18"], "range": 3, "type": "S"},
            "NV": {"items": ["q14", "q15"], "range": 3, "type": "S"},
            "PA": {"items": ["q9", "q19"], "range": 3, "type": "S"},
            "DY": {"items": ["q8"], "range": 3, "type": "S"},
            "SL": {"items": ["q11"], "range": 3, "type": "S"},
            "AP": {"items": ["q13"], "range": 3, "type": "S"},
            "CO": {"items": ["q16"], "range": 3, "type": "S"},
            "DI": {"items": ["q17"], "range": 3, "type": "S"},
            "FI": {"items": ["q28"], "range": 3, "type": "S"},
        }
    elif version == 1:
        scales = {
            "QL": {"items": ["q29", "q30"], "range": 6, "type": "QL"},
            "PF": {"items": ["q1", "q2", "q3", "q4", "q5"], "range": 1, "type": "F"},
            "RF": {"items": ["q6", "q7"], "range": 1, "type": "F"},
            "EF": {"items": ["q21", "q22", "q23", "q24"], "range": 3, "type": "F"},
            "CF": {"items": ["q20", "q25"], "range": 3, "type": "F"},
            "SF": {"items": ["q26", "q27"], "range": 3, "type": "F"},
            "FA": {"items": ["q10", "q12", "q18"], "range": 3, "type": "S"},
            "NV": {"items": ["q14", "q15"], "range": 3, "type": "S"},
            "PA": {"items": ["q9", "q19"], "range": 3, "type": "S"},
            "DY": {"items": ["q8"], "range": 3, "type": "S"},
            "SL": {"items": ["q11"], "range": 3, "type": "S"},
            "AP": {"items": ["q13"], "range": 3, "type": "S"},
            "CO": {"items": ["q16"], "range": 3, "type": "S"},
            "DI": {"items": ["q17"], "range": 3, "type": "S"},
            "FI": {"items": ["q28"], "range": 3, "type": "S"},
        }
    elif version == 303:
        scales = {
            "QL": {"items": ["q31", "q33"], "range": 6, "type": "QL"},
            "QL2": {"items": ["q32", "q33"], "range": 6, "type": "QL"},
            "PF": {"items": ["q1", "q2", "q3", "q4", "q5"], "range": 1, "type": "F"},
            "RF": {"items": ["q6", "q7"], "range": 1, "type": "F"},
            "RF2": {"items": ["q26", "q27"], "range": 3, "type": "F"},
            "EF": {"items": ["q21", "q22", "q23", "q24"], "range": 3, "type": "F"},
            "CF": {"items": ["q20", "q25"], "range": 3, "type": "F"},
            "SF": {"items": ["q28", "q29"], "range": 3, "type": "F"},
            "FA": {"items": ["q10", "q12", "q18"], "range": 3, "type": "S"},
            "NV": {"items": ["q14", "q15"], "range": 3, "type": "S"},
            "PA": {"items": ["q9", "q19"], "range": 3, "type": "S"},
            "DY": {"items": ["q8"], "range": 3, "type": "S"},
            "SL": {"items": ["q11"], "range": 3, "type": "S"},
            "AP": {"items": ["q13"], "range": 3, "type": "S"},
            "CO": {"items": ["q16"], "range": 3, "type": "S"},
            "DI": {"items": ["q17"], "range": 3, "type": "S"},
            "FI": {"items": ["q30"], "range": 3, "type": "S"},
        }
    else:
        raise ValueError("Invalid version. Choose 1, 2, 3, or 303")

    return scales