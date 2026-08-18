"""
Figure 6 (pipeline diagram)，兩軌設計，每個階段標註對應的 research question。

用法：
    python make_fig6_pipeline_detailed.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PAPER = "#F6F3EC"
INK = "#262421"
INK_SOFT = "#5B564C"
MOSS = "#3F6B5E"
MUSTARD = "#D9A441"
CLAY = "#A6473B"

GROUP_COLOR = {"collection": MOSS, "slang_track": MUSTARD, "corpus_track": CLAY}
FIG_W, FIG_H = 14.5, 6.6


def draw_box(ax, x, y, w, h, title, subtitle, rq, color, title_fs=11, sub_fs=8.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=1.4, edgecolor=color, facecolor="#FCFAF5", zorder=3, clip_on=False)
    ax.add_patch(box)
    top_bar = FancyBboxPatch((x, y + h - 0.09), w, 0.09, boxstyle="round,pad=0,rounding_size=0.05",
                              linewidth=0, facecolor=color, zorder=4, clip_on=False)
    ax.add_patch(top_bar)
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=INK, zorder=5, clip_on=False)
    ax.text(x + w / 2, y + h * 0.40, subtitle, ha="center", va="center", fontsize=sub_fs,
            color=INK_SOFT, zorder=5, clip_on=False, linespacing=1.4)
    if rq:
        ax.text(x + w / 2, y + 0.15, rq, ha="center", va="center", fontsize=9, color=color,
                fontweight="bold", style="italic", zorder=5, clip_on=False)


def draw_arrow(ax, start, end, color=INK_SOFT):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                  linewidth=1.3, color=color, zorder=2, clip_on=False))


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    fig.suptitle("Dataset construction pipeline, split by downstream research question",
                 fontsize=14.5, color=INK, fontweight="bold", x=0.02, ha="left", y=0.985)

    box_h = 1.4
    box_w_shared = 2.15
    y_top = 4.75

    draw_box(ax, 0.4, y_top, box_w_shared, box_h, "Collection",
             "Track A/B/C query design,\navoids self-fulfilling bias", "→ RQ2", GROUP_COLOR["collection"])
    draw_box(ax, 3.0, y_top, box_w_shared, box_h, "Domain\nResolution",
             "Query-intent ground truth\n> keyword fallback", "→ RQ2", GROUP_COLOR["collection"])
    draw_arrow(ax, (0.4 + box_w_shared, y_top + box_h / 2), (3.0, y_top + box_h / 2))

    fork_x = 3.0 + box_w_shared + 0.35
    fork_y = y_top + box_h / 2
    draw_arrow(ax, (3.0 + box_w_shared, fork_y), (fork_x, fork_y))

    y_corpus = y_top + 0.05
    draw_arrow(ax, (fork_x, fork_y), (fork_x, y_corpus + box_h / 2))
    corpus_x = fork_x + 0.05
    draw_arrow(ax, (fork_x, y_corpus + box_h / 2), (corpus_x, y_corpus + box_h / 2))
    corpus_w = 5.6
    draw_box(ax, corpus_x, y_corpus, corpus_w, box_h, "Corpus-Wide Analyses",
             "Engagement · Temporal · Demographics ·\nAudience Overlap · Co-occurrence\n(all videos/comments, no slang filter needed)",
             "→ RQ3, RQ4, RQ5", GROUP_COLOR["corpus_track"], sub_fs=8.2)

    y_slang = 2.1
    draw_arrow(ax, (fork_x, fork_y), (fork_x, y_slang + box_h / 2))
    slang_start_x = fork_x + 0.05
    draw_arrow(ax, (fork_x, y_slang + box_h / 2), (slang_start_x, y_slang + box_h / 2))

    slang_boxes = [
        ("Filtering", "Slang lexicon +\nhealth keyword\nfilter"),
        ("Cleaning", "Remove literal\nfalse positives\n(e.g., \"dying\")"),
        ("Balancing", "Per-term cap +\nfloor-then-\ncompetition"),
        ("Annotation\nPool", "n=547, expression-\ntype taxonomy"),
        ("Normalization\nPrototype", "LLM literal/\nnonliteral +\nparaphrase"),
    ]
    n_boxes = len(slang_boxes)
    gap = 0.22
    box_w_slang = (corpus_w - (n_boxes - 1) * gap) / n_boxes
    x_cursor = slang_start_x
    for i, (title, subtitle) in enumerate(slang_boxes):
        rq = "→ RQ1" if i >= 2 else None
        draw_box(ax, x_cursor, y_slang, box_w_slang, box_h, title, subtitle, rq,
                 GROUP_COLOR["slang_track"], title_fs=10, sub_fs=7.8)
        if i < n_boxes - 1:
            draw_arrow(ax, (x_cursor + box_w_slang, y_slang + box_h / 2),
                       (x_cursor + box_w_slang + gap, y_slang + box_h / 2))
        x_cursor += box_w_slang + gap

    legend_items = [
        (GROUP_COLOR["collection"], "Collection & domain resolution — feeds RQ2 (bias correction)"),
        (GROUP_COLOR["slang_track"], "Slang annotation track — feeds RQ1 (taxonomy & ambiguity)"),
        (GROUP_COLOR["corpus_track"], "Corpus-wide analysis track — feeds RQ3, RQ4, RQ5"),
    ]
    legend_y_start = 0.85
    for i, (color, label) in enumerate(legend_items):
        ly = legend_y_start - i * 0.32
        ax.add_patch(mpatches.Rectangle((0.4, ly), 0.28, 0.22, facecolor=color, edgecolor="none", clip_on=False))
        ax.text(0.78, ly + 0.11, label, fontsize=10, color=INK, va="center", ha="left", clip_on=False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig("figures/fig6_pipeline_detailed.png", dpi=300, facecolor=PAPER)
    plt.close()
    print("已存檔: figures/fig6_pipeline_detailed.png")


if __name__ == "__main__":
    main()
