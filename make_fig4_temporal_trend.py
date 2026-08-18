"""
Figure 4（health-related slang rate over time），重新產出。

沿用 analyze_temporal_trend.py 的邏輯，但額外做跨 batch 去重
（同一個上游問題：留言可能同時被 batch_01 跟 batch_02 收集到），
並直接輸出圖表。

用法（在你本機、health_slang_dataset 資料夾底下跑）：
    pip install matplotlib --break-system-packages
    python make_fig4_temporal_trend.py
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PAPER = "#F6F3EC"
INK = "#262421"
INK_SOFT = "#5B564C"
MOSS = "#2E4F45"
MUSTARD = "#D9A441"
CLAY = "#A6473B"
SHADE = "#E5DFD0"


def make_merge_key(df: pd.DataFrame) -> pd.Series:
    comment_id = df["comment_id"].astype(str) if "comment_id" in df.columns else pd.Series([""] * len(df))
    video_id = df["video_id"].astype(str) if "video_id" in df.columns else pd.Series([""] * len(df))
    is_comment_row = (
        comment_id.notna() & (comment_id.str.strip() != "") & (comment_id.str.lower() != "nan")
    )
    key = np.where(is_comment_row, "c_" + comment_id, "v_" + video_id)
    return pd.Series(key, index=df.index)


def load_all_raw_comments_by_year():
    year_counter = Counter()
    seen_ids = set()
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("comment_id")
                if cid and cid in seen_ids:
                    continue
                if cid:
                    seen_ids.add(cid)
                pub = item.get("published_at")
                if pub:
                    year = pd.to_datetime(pub, errors="coerce", utc=True).year
                    if pd.notna(year):
                        year_counter[int(year)] += 1
    return year_counter


def main():
    cleaned_path = "data_raw/youtube_candidate_slang_cleaned.csv"
    if not Path(cleaned_path).exists():
        raise FileNotFoundError(f"找不到 {cleaned_path}，請先跑過 clean_candidates.py")

    cleaned = pd.read_csv(cleaned_path)
    n_before = len(cleaned)
    cleaned["_merge_key"] = make_merge_key(cleaned)
    cleaned = cleaned.drop_duplicates(subset="_merge_key", keep="first")
    print(f"去重：{n_before} -> {len(cleaned)}")

    cleaned["published_at_parsed"] = pd.to_datetime(cleaned["published_at"], errors="coerce", utc=True)
    cleaned["year"] = cleaned["published_at_parsed"].dt.year

    total_by_year = load_all_raw_comments_by_year()
    cleaned_by_year = cleaned["year"].value_counts().sort_index()

    rate_rows = []
    for year, total in sorted(total_by_year.items()):
        n_slang = int(cleaned_by_year.get(year, 0))
        rate = n_slang / total * 100 if total > 0 else 0
        rate_rows.append({"year": year, "total_comments": total, "health_slang_comments": n_slang, "health_slang_rate_pct": rate})

    rate_df = pd.DataFrame(rate_rows).sort_values("year")
    print(rate_df.to_string(index=False))
    rate_df.to_csv("data_raw/temporal_trend_health_slang_rate.csv", index=False)

    years = rate_df["year"].tolist()
    rates = rate_df["health_slang_rate_pct"].tolist()
    sparse_cutoff = 2021
    # 只在樣本量可靠的年份範圍（2021 年以後）裡找峰值，避免像 2017 年
    # 這種「57 則留言裡剛好 1 則命中」造成的小樣本假象被誤標成全局峰值
    # ——這正是圖上網底區塊本來就在警告的問題，峰值標註邏輯不該忽略它。
    reliable_mask = [y >= sparse_cutoff for y in years]
    reliable_indices = [i for i, ok in enumerate(reliable_mask) if ok]
    peak_idx = max(reliable_indices, key=lambda i: rates[i])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    ax.axvspan(min(years) - 0.5, sparse_cutoff - 0.5, color=SHADE, zorder=1)
    ax.text(min(years) + 0.1, max(rates) * 0.92, "sparse early samples\n(n<800/yr before 2021)",
            fontsize=10, color=INK_SOFT)

    ax.plot(years, rates, color=MOSS, linewidth=2, zorder=3)
    ax.scatter(years, rates, color=MUSTARD, edgecolor=MOSS, s=60, zorder=4)

    ax.annotate(f"peak: {years[peak_idx]}", xy=(years[peak_idx], rates[peak_idx]),
                xytext=(years[peak_idx] + 0.6, rates[peak_idx] + max(rates) * 0.2),
                fontsize=11, color=CLAY, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=CLAY, lw=1))

    ax.set_xlabel("Year", fontsize=12, color=INK)
    ax.set_ylabel("Health-slang rate (%)", fontsize=12, color=INK)
    fig.suptitle("Health-related slang rate over time (corrected, deduplicated)",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_SOFT)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figures/fig4_temporal_trend.png", dpi=300, facecolor=PAPER)
    plt.close()
    print("\n已存檔: figures/fig4_temporal_trend.png")


if __name__ == "__main__":
    main()