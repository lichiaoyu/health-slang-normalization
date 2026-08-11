"""
Hashtag 分析。完全離線，不呼叫任何 API。

從現有的 title + description 裡用 regex 抓 hashtag，統計整體分布、
跟 collection_domain 交叉、並標記哪些 hashtag 本身就是 slang 詞
（例如 #glowup、#girldinner），呼應你朋友文件裡的 hashtag 分析。

用法：
    python extract_hashtags.py
"""

import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd

HASHTAG_PATTERN = re.compile(r"#(\w+)")

# 從 clean_candidates.py 的 SLANG_TERMS 轉成「無空白」版本，
# 方便跟 hashtag（hashtag 本身不能有空白）比對。
SLANG_TERMS_NO_SPACE = {
    "dead", "dying", "slay", "bussin", "hitsdifferent", "lowkey", "highkey",
    "fr", "frfr", "nocap", "lockedin", "delulu", "girldinner", "snatched",
    "glowup", "brainfog", "fightingformylife", "knockedmeout", "hadmedying",
    "hadmeinshambles", "sentme", "tookmeout", "messedmeup", "wreckedme",
}


def load_unique_videos():
    seen = set()
    videos = []
    for path in sorted(Path("data_raw").glob("*_youtube_videos.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                vid = item.get("video_id")
                if vid and vid not in seen:
                    seen.add(vid)
                    videos.append(item)
    return videos


def main():
    videos = load_unique_videos()
    print(f"不重複影片數: {len(videos)}")

    records = []
    videos_with_hashtag = set()

    for v in videos:
        text = " ".join([str(v.get("title", "")), str(v.get("description", ""))])
        tags = [t.lower() for t in HASHTAG_PATTERN.findall(text)]
        if tags:
            videos_with_hashtag.add(v.get("video_id"))
        for t in tags:
            records.append({
                "video_id": v.get("video_id"),
                "collection_domain": v.get("collection_domain", ""),
                "hashtag": t,
            })

    df = pd.DataFrame(records)
    df["is_slang_related"] = df["hashtag"].isin(SLANG_TERMS_NO_SPACE)
    df.to_csv("data_raw/hashtag_mentions.csv", index=False)

    coverage = len(videos_with_hashtag) / len(videos) * 100 if videos else 0
    print(f"有 hashtag 的影片數: {len(videos_with_hashtag)} ({coverage:.1f}%)")
    print(f"總 hashtag 提及次數: {len(df)}")
    print(f"不重複 hashtag 數: {df['hashtag'].nunique()}")

    print("\nTop 20 hashtag:")
    print(df["hashtag"].value_counts().head(20))

    slang_related = df["is_slang_related"].sum()
    slang_pct = slang_related / len(df) * 100 if len(df) else 0
    print(f"\n跟 slang 詞庫相關的 hashtag 提及次數: {slang_related} ({slang_pct:.1f}%)")

    if "collection_domain" in df.columns and df["collection_domain"].astype(bool).any():
        print("\n各 domain 的 hashtag 提及量（僅 batch_02 有 collection_domain 標籤，"
              "batch_01 這欄會是空的，這是已知限制）:")
        print(df[df["collection_domain"] != ""].groupby("collection_domain")["hashtag"].count())

    print(f"\n已存檔: data_raw/hashtag_mentions.csv")


if __name__ == "__main__":
    main()
