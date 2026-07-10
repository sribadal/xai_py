"""
NKR treatment code lookup tables.

Maps raw NKR codes to human-readable treatment names for systemic
therapy (ATC codes), surgical procedures, tumor destruction methods,
surgical radicality, radiotherapy types, and metastatic therapy.
"""
SYSTEMIC_TREATMENT_DICT = {
    "690420": "Chemotherapy Abroad",
    "699004": "Study Medication (Chemotherapy)",
    "L01": "Systemic Chemotherapy",
    "L01BA04": "Pemetrexed",
    "L01BC02": "5-FU",
    "L01BC05": "Gemcitabine",
    "L01BC06": "Capecitabine",
    "L01BC53": "Tegafur/Gimeracil/Oteracil",
    "L01CD01": "Paclitaxel",
    "L01CD01_A": "Nab-Paclitaxel",
    "L01CD02": "Docetaxel",
    "L01CE02": "Irinotecan",
    "L01XA01": "Cisplatin",
    "L01XA02": "Carboplatin",
    "L01XA03": "Oxaliplatin",
}

CHIR_TYPE_DICT = {
    21: "Whipple",
    22: "PPPD (Pylorus-Preserving Pancreaticoduodenectomy)",
    23: "PRPD (Pylorus-Reconstructing Pancreaticoduodenectomy)",
    24: "NNO (Non-Specified)",
    25: "Distal Pancreatectomy (Corpus/Tail Resection)",
    26: "Total Pancreatectomy",
    27: "Central Pancreatectomy (Head-Tail Sparing)",
    29: "Pancreatic Resection (NNO)",
    98: "Surgery Abroad",
}

DESTR_TYPE_DICT = {
    1: "RFA (Radiofrequency Ablation)",
    3: "IRE (Irreversible Electroporation)",
    8: "Tumor Destruction Abroad",
}

CHIR_RAD_DICT = {
    0: "Microscopic Radical Resection",
    1: "Microscopic Iradical Resection",
    2: "Macroscopic Iradical Resection",
    9: "Unknown Resection Radicality",
}

RT_TYPE_DICT = {
    1: "Chemoradiation",
    3: "SBRT (Stereotactic Body Radiation Therapy)",
    7: "Other Specific RT",
    8: "Radiotherapy Abroad",
    9: "NNO (Non-Specified)",
}

META_THER_DICT = {
    "411C22M": "TACE (Transarterial Chemoembolization) - Liver Metastases",
    "410002": "HIPEC (Hyperthermic Intraperitoneal Chemotherapy)",
    "320C22": "Radiotherapy - Liver Metastases",
    "320C34": "Radiotherapy - Lung Metastases",
    "320C41": "Radiotherapy - Bone Metastases",
    "320C44": "Radiotherapy - Skin Metastases",
    "320C71": "Radiotherapy - Brain Metastases (Nonspecific)",
    "320C71_S": "Stereotactic Radiotherapy - Brain Metastases",
    "320C71_W": "Whole Brain Radiotherapy - Brain Metastases",
    "320C77": "Radiotherapy - Lymph Node Metastases",
    "320000": "Radiotherapy - Metastases (Nonspecific)",
}