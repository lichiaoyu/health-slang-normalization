"""
Figure 1 (domain distribution)，用去重後修正的 n=547 數字，
提供 bar chart 跟 pie chart 兩個版本。

用法：
    python make_fig1_domain_distribution.py
"""

import matplotlib.pyplot as plt

PAPER = "#F6F3EC"
INK = "#262421"
INK_SOFT = "#5B564C"
MOSS = "#3F6B5E"
LINE = "#DDD6C7"

DOMAIN_PALETTE = [
    "#3F6B5E", "#5C8A7A", "#D9A441", "#E6BE6E",
    "#A6473B", "#C17268", "#7A6A4F", "#9C8F6E",
]

DOMAIN_DATA = [
    ("diet_weight_loss", 97),
    ("body_image_transformation", 96),
    ("supplements", 90),
    ("fitness_gym", 70),
    ("medication_side_effects", 59),
    ("sleep_fatigue", 52),
    ("mental_cognitive", 43),
    ("gut_health", 40),
]

TOTAL_N = sum(n for _, n in DOMAIN_DATA)
assert TOTAL_N == 547


def clean_label(name):
    return name.replace("_", " ")


def make_bar_chart():
    labels = [clean_label(d) for d, _ in DOMAIN_DATA]
    values = [n for _, n in DOMAIN_DATA]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=MOSS, height=0.62, zorder=3)

    for i, v in enumerate(values):
        ax.text(v + 1.5, i, str(v), va="center", ha="left", fontsize=12, color=INK, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Number of annotated rows", fontsize=12, color=INK)
    fig.suptitle(f"Domain distribution in the balanced dataset (n={TOTAL_N})",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.spines["left"].set_color(LINE)
    ax.tick_params(axis="both", colors=INK_SOFT)
    ax.set_xlim(0, max(values) * 1.18)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("figures/fig1_domain_distribution_bar.png", dpi=300, facecolor=PAPER)
    plt.close()
    print("已存檔: figures/fig1_domain_distribution_bar.png")


def make_pie_chart():
    labels = [clean_label(d) for d, _ in DOMAIN_DATA]
    values = [n for _, n in DOMAIN_DATA]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    wedges, texts, autotexts = ax.pie(
        values, colors=DOMAIN_PALETTE, autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.78, startangle=90,
        wedgeprops={"edgecolor": PAPER, "linewidth": 2},
        textprops={"fontsize": 10, "color": PAPER, "fontweight": "bold"},
    )
    ax.legend(wedges, [f"{l} ({v})" for l, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10.5, frameon=False, labelcolor=INK)
    ax.set_title(f"Domain distribution in the balanced dataset (n={TOTAL_N}), alternate view",
                 fontsize=14, color=INK, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig("figures/fig1_domain_distribution_pie.png", dpi=300, facecolor=PAPER, bbox_inches="tight")
    plt.close()
    print("已存檔: figures/fig1_domain_distribution_pie.png")


if __name__ == "__main__":
    make_bar_chart()
    make_pie_chart()
