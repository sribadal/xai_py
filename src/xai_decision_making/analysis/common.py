"""Shared constants, paths, palettes, and data-loading for the Results pipeline."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "survey_responses.csv"
OUT = SCRIPT_DIR / "output"
OUT.mkdir(exist_ok=True)

LIKERT_MAP = {"Strongly disagree": 1, "Disagree": 2, "Neutral": 3,
              "Agree": 4, "Strongly agree": 5}
LIKERT_LABELS = list(LIKERT_MAP)

Q_SHORT = {"675fc11e": "Understandability & Trust",
           "0992b918": "Critical Assessment",
           "cb295bcb": "Actionability & Clinical Fit"}
Q_ORDER = list(Q_SHORT.values())

MODALITY_ORDER = ["Attribution", "Case-based", "Counterfactual", "Rule-based"]
MODEL_ORDER = ["RSF", "CONDOR"]

DEM_IDS = {"60c1c2ca": "role", "7298568d": "role_specify",
           "286de972": "experience", "a11f23d8": "ai_familiarity",
           "45018e92": "specialty"}

RANK_IDS = {"5fd09cc1": "Overall preference",
            "8ca42642": "Patient communication"}

PAL = {"Attribution": "#1B4332", "Case-based": "#2D6A4F",
       "Counterfactual": "#52B788", "Rule-based": "#95D5B2"}
MODEL_PAL = {"RSF": "#264653", "CONDOR": "#E76F51"}
DIV_COLORS = ["#C1121F", "#E5756D", "#CCCCCC", "#74C69D", "#2D6A4F"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("xai", ["#F8F0E3", "#1B4332"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def load():
    raw = pd.read_csv(DATA_PATH)
    raw = raw.dropna(subset=["session_id"])
    raw = raw[raw["session_id"].astype(str).str.strip() != ""].copy()
    raw["pid"] = raw["session_id"].str[:8]

    demo_rows = []
    for pid, g in raw.groupby("pid"):
        row = {"pid": pid}
        for prefix, col in DEM_IDS.items():
            hit = g[g["question_id"].str.startswith(prefix)]
            if len(hit):
                val = hit.iloc[0]["answer_label"]
                if pd.isna(val) or str(val).strip() == "":
                    val = hit.iloc[0]["open_text"]
                row[col] = str(val).strip() if pd.notna(val) else ""
        demo_rows.append(row)
    demo = pd.DataFrame(demo_rows)
    demo["role_clean"] = demo.apply(
        lambda r: r.get("role_specify") if str(r.get("role_specify", "")) not in ("", "nan")
        else r.get("role", ""), axis=1)

    fix = {"Attribution": "Attribution", "Case-Based": "Case-based",
           "Counterfactual": "Counterfactual", "Rule-Based": "Rule-based"}
    lik = raw[(raw["question_type"] == "LIKERT")
              & raw["model"].notna() & (raw["model"] != "")].copy()
    lik["q_short"] = lik["question_id"].str[:8].map(Q_SHORT)
    lik = lik[lik["q_short"].notna()].copy()
    lik["modality"] = lik["modality"].str.replace("_", "-").str.title().map(fix)
    lik["model"] = lik["model"].str.upper()
    lik["score"] = lik["answer_label"].map(LIKERT_MAP)
    lik = lik[["pid", "modality", "model", "q_short", "score", "answer_label"]]

    ranks = raw[raw["question_type"] == "RANKING"][["pid", "question_id", "ranking_order"]].copy()
    comments = (raw[raw["question_id"].str.startswith("723f6669")]["open_text"]
                .dropna().tolist())

    return SimpleNamespace(likert=lik, demo=demo, ranks=ranks,
                           comments=comments, N=demo["pid"].nunique())