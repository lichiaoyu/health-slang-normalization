"""
影片長度分析：長度分桶 x 觀看數、長度分桶 x health slang 出現率。

需要先跑過：
    1. enrich_video_duration.py（產生 video_duration_lookup.jsonl）
    2. enrich_video_stats.py（產生 video_stats_lookup.jsonl，engagement 數據）

呼應你列的兩個觀察點：
    - Shorts（1 分鐘以內）佔比 + 平均觀看數
    - 長影片的 health slang 出現率

「slang rate」這裡的定義是：該長度分桶裡，所有留言中有命中 slang 詞的
比例（分子用 youtube_candidate_slang_cleaned.csv 已經去重、清理過的
候選池，分母用該分桶影片底下的全部留言數），跟 analyze_temporal_trend.py
算「health slang rate」的邏輯是同一套定義，方便互相對照。

用法：
    python analyze_video_length.py
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

DURATION_PATH = "data_raw/video_duration_lookup.jsonl"
STATS_PATH = "data_raw/video_stats_lookup.jsonl"
CLEANED_CANDIDATES_PATH = "data_raw/youtube_candidate_slang_cleaned.csv"


def make_merge_key_comments(df: pd.DataFrame) -> pd.Series:
    comment_id = df["comment_id"].astype(str) if "comment_id" in df.columns else pd.Series([""] * len(df))
    return "c_" + comment_id


def load_all_comments_per_video():
    """統計每支影片底下總共有幾則不重複留言，當作 slang rate 的分母。"""
    seen_ids = set()
    counts = {}
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("comment_id")
                vid = item.get("video_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    counts[vid] = counts.get(vid, 0) + 1
    return counts


def main():
    for path in [DURATION_PATH, STATS_PATH, CLEANED_CANDIDATES_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到 {path}。請先跑過 enrich_video_duration.py、"
                f"enrich_video_stats.py，並確認 clean_candidates.py 已執行過。"
            )

    duration_df = pd.read_json(DURATION_PATH, lines=True).drop_duplicates(subset="video_id")
    stats_df = pd.read_json(STATS_PATH, lines=True).drop_duplicates(subset="video_id")

    merged = duration_df.merge(stats_df, on="video_id", how="inner")
    print(f"成功合併長度 + engagement 數據的影片數: {len(merged)}")

    # ---------- 1. 長度分桶 x 觀看數 ----------
    bucket_order = ["short_under_1min", "short_1_to_3min", "medium_3_to_10min", "long_over_10min", "unknown"]
    merged["duration_bucket"] = pd.Categorical(merged["duration_bucket"], categories=bucket_order, ordered=True)

    engagement_summary = merged.groupby("duration_bucket", observed=True).agg(
        n_videos=("video_id", "count"),
        pct_of_total=("video_id", lambda s: len(s) / len(merged) * 100),
        mean_views=("view_count", "mean"),
        median_views=("view_count", "median"),
    ).round(1)

    print(f"\n{'=' * 70}")
    print("長度分桶 x 觀看數")
    print(f"{'=' * 70}")
    print(engagement_summary)

    # ---------- 2. 長度分桶 x health slang 出現率 ----------
    comments_per_video = load_all_comments_per_video()

    candidates = pd.read_csv(CLEANED_CANDIDATES_PATH)
    candidates["_merge_key"] = make_merge_key_comments(candidates)
    candidates = candidates.drop_duplicates(subset="_merge_key", keep="first")
    slang_comments_per_video = candidates[candidates["source_file"].str.contains("comments", na=False)] \
        .groupby("video_id").size().to_dict()

    rate_rows = []
    for vid, bucket in zip(merged["video_id"], merged["duration_bucket"]):
        total_comments = comments_per_video.get(vid, 0)
        slang_comments = slang_comments_per_video.get(vid, 0)
        rate_rows.append({
            "video_id": vid,
            "duration_bucket": bucket,
            "total_comments": total_comments,
            "slang_comments": slang_comments,
        })

    rate_df = pd.DataFrame(rate_rows)
    rate_summary = rate_df.groupby("duration_bucket", observed=True).agg(
        total_comments=("total_comments", "sum"),
        slang_comments=("slang_comments", "sum"),
    )
    rate_summary["slang_rate_pct"] = (rate_summary["slang_comments"] / rate_summary["total_comments"] * 100).round(2)

    print(f"\n{'=' * 70}")
    print("長度分桶 x health slang 出現率")
    print(f"{'=' * 70}")
    print(rate_summary)

    output = engagement_summary.join(rate_summary[["slang_rate_pct"]])
    output.to_csv("data_raw/video_length_analysis.csv")

    print(f"\n已存檔: data_raw/video_length_analysis.csv")
    print("\n⚠️ 方法學註記：'long_over_10min' 這個分桶的影片很可能不是真正的")
    print("YouTube Shorts（你的 query 沒有強制篩選 Shorts-only），如果這個分桶")
    print("的影片數很少，slang rate 的數字會很不穩定，寫進論文時記得附上")
    print("這個分桶的 n_videos，不要只報比例。")


if __name__ == "__main__":
    main()