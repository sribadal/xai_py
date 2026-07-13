"""Demographics table and bar charts for Study 2 participants."""
import matplotlib.pyplot as plt
from xai_decision_making.analysis.common import load, save, OUT


def build():
    d = load()
    demo = d.demo.reset_index(drop=True)
    demo.insert(0, "participant", [f"P{i+1}" for i in range(len(demo))])
    table = demo[["participant", "role_clean", "experience",
                  "ai_familiarity", "specialty"]]
    table.columns = ["Participant", "Role", "Experience", "AI familiarity", "Specialty"]
    table.to_csv(OUT / "table_participants.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes,
        ["experience", "ai_familiarity", "role"],
        ["Years of experience", "AI / prediction tool familiarity", "Professional role"],
    ):
        counts = demo[col].value_counts()
        ax.barh(counts.index, counts.values, color="#2D6A4F", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(counts.values):
            ax.text(v + 0.05, i, str(v), va="center", fontweight="bold")
    fig.suptitle(f"Participant background (N={d.N})", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig_demographics.png")
    return table


if __name__ == "__main__":
    build()