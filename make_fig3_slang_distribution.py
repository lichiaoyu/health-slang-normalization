"""
Figure 3（slang term distribution，依 expression type 上色），用去重後
修正的資料重新產出。

用法（在你本機、health_slang_dataset 資料夾底下跑）：
    pip install matplotlib --break-system-packages
    python make_fig3_slang_distribution.py
"""

import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "data_raw/youtube_annotation_pool_cleaned.csv"  # 換成你實際的 547 筆檔案路徑

PAPER = "#F6F3EC"
INK = "#262421"
INK_SOFT = "#5B564C"
LINE = "#DDD6C7"

EXPR_COLORS = {
    "internet_slang": "#3F6B5E",
    "informal_symptom_language": "#D9A441",
    "ambiguous": "#A6473B",
}


def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"找不到 {INPUT_PATH}，請確認路徑指向你去重後的 547 筆 annotation pool。")

    df = pd.read_csv(INPUT_PATH)
    counts = df.groupby(["primary_slang", "expression_type"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=True)

    total_n = len(df)
    labels = counts["primary_slang"].tolist()
    values = counts["n"].tolist()
    colors = [EXPR_COLORS.get(e, "#999") for e in counts["expression_type"]]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.65, zorder=3)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va="center", ha="left", fontsize=10, color=INK, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.set_xlabel("Number of annotated rows", fontsize=12, color=INK)
    fig.suptitle(f"Slang term distribution (n={total_n}), colored by expression type",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.spines["left"].set_color(LINE)
    ax.tick_params(axis="both", colors=INK_SOFT)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in EXPR_COLORS.values()]
    ax.legend(legend_handles, EXPR_COLORS.keys(), loc="lower right", fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/fig3_slang_distribution.png", dpi=300, facecolor=PAPER)
    plt.close()

    print(f"總筆數: {total_n}")
    print("依 expression_type 分布:")
    print(df["expression_type"].value_counts())
    print("\n已存檔: figures/fig3_slang_distribution.png")


if __name__ == "__main__":
    main()
