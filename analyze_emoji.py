"""
Emoji 分析。完全離線，不呼叫任何 API。

從現有留言文字裡偵測 emoji，統計整體使用率、最常見的 emoji，
呼應你朋友文件裡「31.6% 留言含 emoji」那類分析。

用法：
    python analyze_emoji.py
"""

import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd

# 涵蓋常見 emoji 的 Unicode 區段（表情符號、符號與圖形、旗幟等）。
# 不是每一個 emoji 變體都涵蓋得到，但足以抓到絕大多數常見用法
# （骷髏頭、哭臉、愛心、火焰等）。
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)


def load_unique_comments():
    seen = set()
    comments = []
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("comment_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    comments.append(item)
    return comments


def main():
    comments = load_unique_comments()
    print(f"不重複留言數: {len(comments)}")

    records = []
    emoji_counter = Counter()
    n_with_emoji = 0

    for c in comments:
        text = str(c.get("text", "") or "")
        found = EMOJI_PATTERN.findall(text)
        # findall 會把連續 emoji 抓成一整串（例如 "😂😂😂"），
        # 這裡再拆成單一字元方便統計個別 emoji 出現次數。
        single_emojis = []
        for chunk in found:
            single_emojis.extend(list(chunk))

        has_emoji = len(single_emojis) > 0
        if has_emoji:
            n_with_emoji += 1
            emoji_counter.update(single_emojis)

        records.append({
            "comment_id": c.get("comment_id"),
            "collection_domain": c.get("collection_domain", ""),
            "has_emoji": has_emoji,
            "emoji_count": len(single_emojis),
        })

    df = pd.DataFrame(records)
    df.to_csv("data_raw/comment_emoji_flags.csv", index=False)

    pct_with_emoji = n_with_emoji / len(comments) * 100 if comments else 0
    print(f"\n含 emoji 的留言數: {n_with_emoji} ({pct_with_emoji:.1f}%)")

    print("\nTop 20 emoji:")
    for emoji_char, n in emoji_counter.most_common(20):
        print(f"  {emoji_char}  {n}")

    if "collection_domain" in df.columns and df["collection_domain"].astype(bool).any():
        print("\n各 domain 的 emoji 使用率（僅 batch_02 有 collection_domain 標籤，"
              "batch_01 這欄會是空的，這是已知限制）:")
        sub = df[df["collection_domain"] != ""]
        print(sub.groupby("collection_domain")["has_emoji"].mean() * 100)

    print(f"\n已存檔: data_raw/comment_emoji_flags.csv")


if __name__ == "__main__":
    main()
