"""
Figure 5（declared geographic location: creators vs. commenters），重新產出。

沿用 enrich_geo.py 產生的資料，直接輸出圖表。

用法（在你本機、health_slang_dataset 資料夾底下跑，需要先跑過
enrich_geo.py 產生 channel_geo_lookup.jsonl 跟 video_channel_lookup.jsonl）：
    pip install matplotlib --break-system-packages
    python make_fig5_demographics.py
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAPER = "#F6F3EC"
INK = "#262421"
INK_SOFT = "#5B564C"
MOSS = "#3F6B5E"
MUSTARD = "#D9A441"


def main():
    geo_path = "data_raw/channel_geo_lookup.jsonl"
    video_channel_path = "data_raw/video_channel_lookup.jsonl"
    for p in [geo_path, video_channel_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"找不到 {p}，請先跑過 enrich_geo.py")

    country_by_channel = {}
    with open(geo_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            country_by_channel[item["channel_id"]] = item.get("country") or "UNKNOWN"

    creator_channel_ids = set()
    with open(video_channel_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = item.get("channel_id")
            if cid:
                creator_channel_ids.add(cid)

    commenter_channel_ids = set()
    for p in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("author_channel_id")
                if cid:
                    commenter_channel_ids.add(cid)

    creator_countries = Counter(country_by_channel.get(cid, "UNKNOWN") for cid in creator_channel_ids)
    commenter_countries = Counter(country_by_channel.get(cid, "UNKNOWN") for cid in commenter_channel_ids)

    top_countries = [c for c, _ in creator_countries.most_common(5) if c != "UNKNOWN"]

    def pct_series(counter, total, countries):
        vals = [counter.get(c, 0) / total * 100 for c in countries]
        vals.append(counter.get("UNKNOWN", 0) / total * 100)
        return vals

    categories = top_countries + ["Unknown"]
    creator_pcts = pct_series(creator_countries, len(creator_channel_ids), top_countries)
    commenter_pcts = pct_series(commenter_countries, len(commenter_channel_ids), top_countries)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, creator_pcts, width=w, color=MOSS, label=f"Video creators (n={len(creator_channel_ids)})", zorder=3)
    ax.bar(x + w / 2, commenter_pcts, width=w, color=MUSTARD, label=f"Commenters (n={len(commenter_channel_ids)})", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color=INK)
    ax.set_ylabel("% of channels", fontsize=12, color=INK)
    fig.suptitle("Declared geographic location: creators vs. commenters",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_SOFT)
    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figures/fig5_geo_creator_vs_commenter.png", dpi=300, facecolor=PAPER)
    plt.close()

    known_creator = 1 - creator_countries.get("UNKNOWN", 0) / len(creator_channel_ids)
    known_commenter = 1 - commenter_countries.get("UNKNOWN", 0) / len(commenter_channel_ids)
    print(f"創作者填寫率: {known_creator*100:.1f}% (n={len(creator_channel_ids)})")
    print(f"留言者填寫率: {known_commenter*100:.1f}% (n={len(commenter_channel_ids)})")
    print("\n已存檔: figures/fig5_geo_creator_vs_commenter.png")


if __name__ == "__main__":
    main()
