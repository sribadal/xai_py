"""Paths, study constants and the survey loader shared by both analyses."""
import io
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
SHEET_DIR = DATA_DIR / "coding_sheets"
OUTPUT_DIR = PACKAGE_DIR / "output"
QUANTITATIVE_DIR = OUTPUT_DIR / "quantitative"
QUALITATIVE_DIR = OUTPUT_DIR / "qualitative"

# All files matching this pattern are read, in filename order, so the
# follow-up export is picked up by putting it next to the first one.
SURVEY_FILES = "survey_responses*.csv"

# --- Survey items ------------------------------------------------------------

LIKERT_MAP = {"Strongly disagree": 1, "Disagree": 2, "Neutral": 3,
              "Agree": 4, "Strongly agree": 5}
LIKERT_LABELS = list(LIKERT_MAP)

DIMENSIONS = {"675fc11e": "Understandability",
              "0992b918": "Critical Assessment",
              "cb295bcb": "Actionability"}
DIMENSION_ORDER = list(DIMENSIONS.values())

BACKGROUND_ITEMS = {"60c1c2ca": "role", "7298568d": "role_specify",
                    "286de972": "experience", "a11f23d8": "ai_familiarity",
                    "45018e92": "specialty"}
RANKINGS = {"5fd09cc1": "Overall preference",
            "8ca42642": "Patient communication"}
COMMENTS_ITEM = "723f6669"

# The follow-up survey (P10, P11) was a new build in the app, so every item
# got a new id. They are mapped back to the original ids when loading.
FOLLOW_UP_IDS = {
    "13d14f2c": "675fc11e",  # understandability
    "c0184c62": "0992b918",  # critical assessment
    "b2232b6e": "cb295bcb",  # actionability
    "4399b8ec": "60c1c2ca",  # role
    "96b4bb65": "7298568d",  # role, specify
    "07ca0fff": "286de972",  # experience
    "9aca0c13": "a11f23d8",  # AI familiarity
    "befcff51": "45018e92",  # specialty
    "ccc437f8": "5fd09cc1",  # overall preference
    "2d4e213e": "8ca42642",  # patient communication
}

MODALITY_ORDER = ["Attribution", "Case-based", "Counterfactual", "Rule-based"]
MODALITY_NAMES = {"ATTRIBUTION": "Attribution", "CASE_BASED": "Case-based",
                  "COUNTERFACTUAL": "Counterfactual", "RULE_BASED": "Rule-based"}
MODEL_ORDER = ["RSF", "CONDOR"]

# --- Participants ------------------------------------------------------------

# Session key (first 8 characters of the session id) to participant number.
# The numbers match the coding sheets.
PARTICIPANTS = {
    "11273fa2": "P1",
    "4bebd520": "P2",
    "5047b1dc": "P3",
    "792708fe": "P4",
    "80a6a7c6": "P5",
    "86f11375": "P6",
    "b55e9e5d": "P7",
    "ee3f8abe": "P8",
    "f16fbc6d": "P9",
    "62e4f35a": "P10",
    "584d91e1": "P11",
}
PARTICIPANT_ORDER = [f"P{i}" for i in range(1, 12)]

# Two incomplete test sessions run by the first author.
TEST_SESSIONS = {"7e6f6c86", "e2a08e2b"}

# P1-P9 saw interface v1; P10 and P11 saw v2 (see CHANGELOG_v1_v2.md at the
# repository root). The panel changes were to attribution and counterfactual.
# Version is kept as a column for the sensitivity checks; two participants
# are not a group.
INTERFACE_VERSION = {p: ("v2" if p in {"P10", "P11"} else "v1")
                     for p in PARTICIPANT_ORDER}
V1_PARTICIPANTS = [p for p, v in INTERFACE_VERSION.items() if v == "v1"]
V2_PARTICIPANTS = [p for p, v in INTERFACE_VERSION.items() if v == "v2"]
CHANGED_MODALITIES = ["Attribution", "Counterfactual"]

# --- Figures -----------------------------------------------------------------

PAL = {"Attribution": "#1B4332", "Case-based": "#2D6A4F",
       "Counterfactual": "#52B788", "Rule-based": "#95D5B2"}
DIV_COLORS = ["#C1121F", "#E5756D", "#CCCCCC", "#74C69D", "#2D6A4F"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


_figure_viewers = []


def show_figures_inline(dpi=110):
    """In a notebook, also display each figure when it is saved."""
    from IPython.display import Image, display

    def view(fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        display(Image(buffer.getvalue()))
    _figure_viewers[:] = [view]


def save_figure(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    for view in _figure_viewers:
        view(fig)
    plt.close(fig)


def write_workbook(path, tables, descriptions):
    """One .xlsx with a contents tab followed by one tab per table.

    tables: name -> DataFrame. descriptions: name -> one line for the contents.
    """
    from openpyxl.styles import Font
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = pd.DataFrame([{"tab": name, "description": descriptions.get(name, ""),
                              "rows": len(frame), "columns": frame.shape[1]}
                             for name, frame in tables.items()])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in {"contents": contents, **tables}.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for column in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column)
                sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 8), 60)


def participant_number(pid):
    return int(str(pid)[1:])


# --- Loading -----------------------------------------------------------------

def _read_survey_files():
    paths = sorted(DATA_DIR.glob(SURVEY_FILES))
    if not paths:
        raise FileNotFoundError(f"No file matching {SURVEY_FILES} in {DATA_DIR}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["export_file"] = path.name
        frames.append(frame)
    columns = [set(f.columns) for f in frames]
    if set.union(*columns) != set.intersection(*columns):
        print(f"WARNING survey files differ in columns: "
              f"{sorted(set.union(*columns) - set.intersection(*columns))}")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    # A later export can repeat earlier sessions; keep the latest copy.
    key = [c for c in ["session_id", "question_id", "modality", "model"]
           if c in raw.columns]
    before = len(raw)
    raw = raw.drop_duplicates(subset=key, keep="last")
    if len(raw) != before:
        print(f"  dropped {before - len(raw)} duplicate rows across exports")
    return raw


def load():
    """Read the survey exports into ratings, rankings and background."""
    raw = _read_survey_files()
    raw = raw.dropna(subset=["session_id"])
    raw = raw[raw["session_id"].astype(str).str.strip() != ""].copy()
    raw["session"] = raw["session_id"].str[:8]
    raw = raw[~raw["session"].isin(TEST_SESSIONS)].copy()

    raw["question_id"] = raw["question_id"].map(
        lambda q: FOLLOW_UP_IDS.get(str(q)[:8], str(q)[:8]) + str(q)[8:])
    raw["pid"] = raw["session"].map(PARTICIPANTS)
    unknown = sorted(raw.loc[raw["pid"].isna(), "session"].unique())
    if unknown:
        raise KeyError(f"Sessions not in PARTICIPANTS: {unknown}")

    background = _background(raw)
    ratings = _ratings(raw)
    ranks = raw[raw["question_type"] == "RANKING"][
        ["pid", "question_id", "ranking_order"]].copy()
    ranks["version"] = ranks["pid"].map(INTERFACE_VERSION)
    comments = (raw[raw["question_id"].str.startswith(COMMENTS_ITEM)]["open_text"]
                .dropna().tolist())

    _check(raw, ratings, background)
    return SimpleNamespace(likert=ratings, demo=background, ranks=ranks,
                           comments=comments, N=background["pid"].nunique())


def _background(raw):
    rows = []
    for pid, answers in raw.groupby("pid"):
        row = {"pid": pid, "session": answers["session"].iloc[0]}
        for prefix, column in BACKGROUND_ITEMS.items():
            hit = answers[answers["question_id"].str.startswith(prefix)]
            if len(hit):
                value = hit.iloc[0]["answer_label"]
                if pd.isna(value) or str(value).strip() == "":
                    value = hit.iloc[0]["open_text"]
                row[column] = str(value).strip() if pd.notna(value) else ""
        rows.append(row)
    demo = pd.DataFrame(rows)
    demo = (demo.assign(_n=demo["pid"].map(participant_number))
                .sort_values("_n").drop(columns="_n").reset_index(drop=True))
    # The free-text role where given, otherwise the option they picked.
    demo["role_clean"] = demo.apply(
        lambda r: r.get("role_specify")
        if str(r.get("role_specify", "")) not in ("", "nan") else r.get("role", ""),
        axis=1)
    demo["version"] = demo["pid"].map(INTERFACE_VERSION)
    return demo


def _ratings(raw):
    lik = raw[(raw["question_type"] == "LIKERT")
              & raw["model"].notna() & (raw["model"] != "")].copy()
    lik["q_short"] = lik["question_id"].str[:8].map(DIMENSIONS)
    lik = lik[lik["q_short"].notna()].copy()
    lik["modality"] = lik["modality"].str.upper().map(MODALITY_NAMES)
    lik["model"] = lik["model"].str.upper()
    lik["score"] = lik["answer_label"].map(LIKERT_MAP)
    lik["version"] = lik["pid"].map(INTERFACE_VERSION)
    lik["modality_changed_in_v2"] = lik["modality"].isin(CHANGED_MODALITIES)
    return lik[["pid", "modality", "model", "q_short", "score", "answer_label",
                "version", "modality_changed_in_v2"]]


def _check(raw, ratings, background):
    """Print warnings for anything that would silently shrink the sample."""
    no_ratings = sorted(set(background["pid"]) - set(ratings["pid"]),
                        key=participant_number)
    if no_ratings:
        print(f"WARNING {no_ratings} have no usable ratings; check their question ids.")
    unmapped = raw[(raw["question_type"] == "LIKERT")
                   & ~raw["question_id"].str[:8].isin(DIMENSIONS)
                   & ~raw["question_id"].str[:8].isin(BACKGROUND_ITEMS)]
    if len(unmapped):
        print(f"WARNING {len(unmapped)} rating rows with unknown question ids "
              f"{sorted(unmapped['question_id'].str[:8].unique())} were dropped.")
    for column in ["role", "experience", "ai_familiarity"]:
        if column not in background.columns:
            print(f"WARNING no {column} recorded for anyone.")
            continue
        blank = background.loc[background[column].isna() | (background[column] == ""), "pid"]
        if len(blank):
            print(f"WARNING no {column} for {sorted(blank, key=participant_number)}.")
    counts = ratings.groupby("pid").size()
    short = counts[counts != 24]
    if len(short):
        print("WARNING incomplete ratings (24 expected): "
              + ", ".join(f"{p}={n}" for p, n in short.items()))
    per_file = raw.groupby("export_file")["session_id"].nunique()
    print("  sessions per export: "
          + ", ".join(f"{f}={n}" for f, n in per_file.items()))


# --- Subsets and rankings ----------------------------------------------------

def subset(data, keep=None, drop=None):
    """The same data restricted to some participants."""
    def pick(frame):
        mask = pd.Series(True, index=frame.index)
        if keep is not None:
            mask &= frame["pid"].isin(keep)
        if drop is not None:
            mask &= ~frame["pid"].isin(drop)
        return frame[mask].copy()
    likert = pick(data.likert)
    return SimpleNamespace(likert=likert, demo=pick(data.demo), ranks=pick(data.ranks),
                           comments=data.comments, N=likert["pid"].nunique())


def v1_only(data):
    return subset(data, keep=V1_PARTICIPANTS)


def rank_positions(ranks, prefix):
    """One row per participant and type: the position given (1 = top)."""
    names = {m.lower(): m for m in MODALITY_ORDER}
    rows = []
    for _, r in ranks[ranks["question_id"].str.startswith(prefix)].iterrows():
        if pd.isna(r.get("ranking_order")):
            continue
        for position, item in enumerate(r["ranking_order"].split(">"), 1):
            name = names.get(item.strip().lower(), item.strip())
            rows.append({"pid": r["pid"], "modality": name, "position": position})
    return pd.DataFrame(rows, columns=["pid", "modality", "position"])


def participant_means(likert):
    """Each participant's two models averaged, per type and dimension."""
    return likert.groupby(["pid", "modality", "q_short"], as_index=False)["score"].mean()
