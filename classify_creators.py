"""
創作者層級分類（正面散播者 / 負面散播者 / 中立教育者）。

呼應你朋友文件裡的 Creator Analysis：把情感分析結果依 channel_id
聚合，看哪些創作者的內容底下留言普遍偏正面/負面/中性。

需要先跑過：
    1. sentiment_analysis_v2.py（產生留言層級的情感標籤）
    2. enrich_geo.py（產生 video_id -> channel_id 對照表）

用法：
    python classify_creators.py
"""

from pathlib import Path

import pandas as pd

SENTIMENT_PATH = "data_raw/sentiment_analysis_vader.csv"
VIDEO_CHANNEL_LOOKUP_PATH = "data_raw/video_channel_lookup.jsonl"
OUTPUT_PATH = "data_raw/creator_classification.csv"

# 呼應你朋友文件「至少要有 3 支影片才納入可靠分類」的門檻
MIN_VIDEOS_FOR_CLASSIFICATION = 3

# 分類閾值：正面比例減負面比例，超過這個差距才算「有傾向」的創作者，
# 否則歸類為中立教育者。這個閾值目前是我們自己設的合理起點，
# 建議之後可以畫分布圖看看門檻設在哪裡比較符合資料的自然斷點。
LEAN_THRESHOLD = 0.15


def classify(row):
    diff = row["pct_positive"] - row["pct_negative"]
    if diff >= LEAN_THRESHOLD:
        return "positive_spreader"
    elif diff <= -LEAN_THRESHOLD:
        return "negative_spreader"
    return "neutral_educator"


def main():
    for path in [SENTIMENT_PATH, VIDEO_CHANNEL_LOOKUP_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到 {path}。請先跑過 sentiment_analysis_v2.py 和 enrich_geo.py。"
            )

    sentiment = pd.read_csv(SENTIMENT_PATH)
    channel_lookup = pd.read_json(VIDEO_CHANNEL_LOOKUP_PATH, lines=True)
    channel_lookup = channel_lookup.drop_duplicates(subset="video_id")

    # sentiment（來自 cleaned csv）從收集階段就帶有自己的 channel_title 欄位，
    # 跟 channel_lookup 的 channel_title 撞名，合併後 pandas 會自動變成
    # channel_title_x / channel_title_y。這裡先改名避免衝突，統一用
    # channel_lookup 這邊查到的版本（來自 videos().list，較新鮮）。
    channel_lookup = channel_lookup.rename(columns={"channel_title": "creator_channel_title"})

    merged = sentiment.merge(channel_lookup, on="video_id", how="left")
    merged = merged[merged["channel_id"].notna()]

    print(f"情感分析資料 {len(sentiment)} 筆，成功對到 channel_id 的有 {len(merged)} 筆 "
          f"({len(merged)/len(sentiment)*100:.1f}%)")

    grouped = merged.groupby("channel_id").agg(
        channel_title=("creator_channel_title", "first"),
        n_videos=("video_id", "nunique"),
        n_comments=("comment_id", "count") if "comment_id" in merged.columns else ("video_id", "count"),
        pct_positive=("sentiment_label", lambda s: (s == "positive").mean()),
        pct_negative=("sentiment_label", lambda s: (s == "negative").mean()),
        pct_neutral=("sentiment_label", lambda s: (s == "neutral").mean()),
    ).reset_index()

    reliable = grouped[grouped["n_videos"] >= MIN_VIDEOS_FOR_CLASSIFICATION].copy()
    print(f"\n共 {len(grouped)} 個不重複頻道，其中 {len(reliable)} 個頻道有 "
          f">= {MIN_VIDEOS_FOR_CLASSIFICATION} 支影片、可納入可靠分類")

    reliable["classification"] = reliable.apply(classify, axis=1)
    reliable.to_csv(OUTPUT_PATH, index=False)

    print("\n創作者分類分布:")
    dist = reliable["classification"].value_counts()
    print(dist)
    print("\n各分類佔比 (%):")
    print((dist / len(reliable) * 100).round(1))

    print(f"\n已存檔: {OUTPUT_PATH}")
    print(f"\n⚠️ 方法學註記：閾值 {LEAN_THRESHOLD} 是目前自訂的起點，不是統計上推導出來的切點，")
    print("建議之後畫出 pct_positive - pct_negative 的分布圖，確認斷點位置是否合理，")
    print("這點你朋友的文件裡也沒有交代閾值怎麼選的，你這裡註明反而比他更嚴謹。")


if __name__ == "__main__":
    main()