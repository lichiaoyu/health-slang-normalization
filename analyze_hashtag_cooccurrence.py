"""
Hashtag 共現分析 + 跨 domain 分布分析。

跟 extract_hashtags.py 不一樣的地方：extract_hashtags.py 是把每個 hashtag
攤平成一列（方便算單一 hashtag 的頻率），這裡改成保留「同一支影片裡完整
的 hashtag 集合」，才能算兩兩共現，以及一個 hashtag 到底橫跨了幾個不同的
health domain（用來驗證「#shorts 和 #genz 幾乎每個類別都會出現」這類觀察
是不是在你的資料裡也成立）。

用法：
    python analyze_hashtag_cooccurrence.py
"""

import html
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

HASHTAG_PATTERN = re.compile(r"#(\w+)")

# 跟 engagement_by_domain.py / analyze_commenter_overlap.py 用同一份對照表。
V1_QUERY_TO_DOMAIN = {
    '"ozempic side effects" #shorts': "medication_side_effects",
    '"glp1 journey" #shorts': "medication_side_effects",
    '"mounjaro experience" #shorts': "medication_side_effects",
    '"semaglutide nausea" #shorts': "medication_side_effects",
    '"weight loss medication" #shorts': "medication_side_effects",
    '"medication side effects" #shorts': "medication_side_effects",
    '"supplement review" #shorts': "supplements",
    '"protein powder review" #shorts': "supplements",
    '"creatine transformation" #shorts': "supplements",
    '"pre workout review" #shorts': "supplements",
    '"pre workout side effects" #shorts': "supplements",
    '"fat burner review" #shorts': "supplements",
    '"greens powder review" #shorts': "supplements",
    '"ashwagandha review" #shorts': "supplements",
    '"magnesium sleep" #shorts': "supplements",
    '"melatonin review" #shorts': "supplements",
    '"calorie deficit meals" #shorts': "diet_weight_loss",
    '"what I eat in a day weight loss" #shorts': "diet_weight_loss",
    '"weight loss journey" #shorts': "diet_weight_loss",
    '"intermittent fasting results" #shorts': "diet_weight_loss",
    '"keto diet results" #shorts': "diet_weight_loss",
    '"high protein meals" #shorts': "diet_weight_loss",
    '"girl dinner" #shorts': "diet_weight_loss",
    '"meal prep weight loss" #shorts': "diet_weight_loss",
    '"gymtok" #shorts': "fitness_gym",
    '"gym transformation" #shorts': "fitness_gym",
    '"body recomp" #shorts': "fitness_gym",
    '"bulking diet" #shorts': "fitness_gym",
    '"cutting diet" #shorts': "fitness_gym",
    '"leg day" #shorts': "fitness_gym",
    '"gym motivation" #shorts': "fitness_gym",
    '"fitness transformation" #shorts': "fitness_gym",
    '"gut health tips" #shorts': "gut_health",
    '"bloating remedies" #shorts': "gut_health",
    '"probiotics review" #shorts': "gut_health",
    '"IBS symptoms" #shorts': "gut_health",
    '"constipation relief" #shorts': "gut_health",
    '"digestion problems" #shorts': "gut_health",
    '"detox drink" #shorts': "gut_health",
    '"sleep tips" #shorts': "sleep_fatigue",
    '"insomnia tips" #shorts': "sleep_fatigue",
    '"melatonin gummies" #shorts': "sleep_fatigue",
    '"sleep supplement" #shorts': "sleep_fatigue",
    '"chronic fatigue" #shorts': "sleep_fatigue",
    '"energy crash" #shorts': "sleep_fatigue",
    '"brain fog" #shorts': "mental_cognitive",
    '"anxiety symptoms" #shorts': "mental_cognitive",
    '"burnout symptoms" #shorts': "mental_cognitive",
    '"cortisol levels" #shorts': "mental_cognitive",
    '"hormone imbalance" #shorts': "mental_cognitive",
    '"PCOS symptoms" #shorts': "mental_cognitive",
    '"weight loss glow up" #shorts': "body_image_transformation",
    '"body transformation" #shorts': "body_image_transformation",
    '"before and after weight loss" #shorts': "body_image_transformation",
    '"snatched waist" #shorts': "body_image_transformation",
    '"summer body" #shorts': "body_image_transformation",
}


def resolve_domain(video):
    cd = video.get("collection_domain")
    if cd and str(cd).strip():
        return cd
    query = video.get("query")
    if query in V1_QUERY_TO_DOMAIN:
        return V1_QUERY_TO_DOMAIN[query]
    return None


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

    video_hashtags = []
    for v in videos:
        # 先解碼 HTML entity（例如標題裡未解碼的 &#39; 撇號），否則
        # regex 會把 "&#39;" 裡的 "#39" 誤判成一個真的 hashtag，
        # 這就是「39」這個假 hashtag 橫跨全部 domain 的真正原因，
        # 不是真的有人用了 #39 這個標籤。
        raw_text = " ".join([str(v.get("title", "")), str(v.get("description", ""))])
        text = html.unescape(raw_text)
        tags = sorted(set(t.lower() for t in HASHTAG_PATTERN.findall(text)))
        domain = resolve_domain(v)
        video_hashtags.append({"video_id": v.get("video_id"), "domain": domain, "hashtags": tags})

    multi_tag_videos = [v for v in video_hashtags if len(v["hashtags"]) >= 2]
    print(f"帶有 2 個以上 hashtag 的影片數: {len(multi_tag_videos)}")

    # ---------- 1. 共現 ----------
    pair_counter = Counter()
    for v in multi_tag_videos:
        for pair in combinations(v["hashtags"], 2):
            pair_counter[pair] += 1

    pair_rows = [{"hashtag_a": a, "hashtag_b": b, "co_occurrence_count": n} for (a, b), n in pair_counter.items()]
    pair_df = pd.DataFrame(pair_rows).sort_values("co_occurrence_count", ascending=False)
    pair_df.to_csv("data_raw/hashtag_cooccurrence.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Top 20 最常共現的 hashtag 組合")
    print(f"{'=' * 60}")
    print(pair_df.head(20).to_string(index=False))

    # ---------- 2. 跨 domain 分布 ----------
    hashtag_domains = {}
    for v in video_hashtags:
        if v["domain"] is None:
            continue
        for tag in v["hashtags"]:
            hashtag_domains.setdefault(tag, set()).add(v["domain"])

    n_total_domains = len(set(v["domain"] for v in video_hashtags if v["domain"]))
    spread_rows = [
        {"hashtag": tag, "n_domains": len(domains), "pct_of_domains": round(len(domains) / n_total_domains * 100, 1)}
        for tag, domains in hashtag_domains.items()
    ]
    spread_df = pd.DataFrame(spread_rows).sort_values("n_domains", ascending=False)
    spread_df.to_csv("data_raw/hashtag_domain_spread.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"橫跨最多 domain 的 hashtag（總共 {n_total_domains} 個 domain）")
    print(f"{'=' * 60}")
    print(spread_df.head(15).to_string(index=False))

    print(f"\n已存檔: data_raw/hashtag_cooccurrence.csv, data_raw/hashtag_domain_spread.csv")
    print("\n⚠️ 方法學註記：像 #shorts 這種平台通用標籤橫跨所有 domain 是預期中的",
          "事（它不是內容標籤，是格式標籤），論文裡如果要討論「哪些 hashtag",
          "跨類別」，建議先排除這種平台層級的通用標籤，只看內容相關的",
          "hashtag（例如 #fitness vs #mentalhealth）才有實際分析意義。")


if __name__ == "__main__":
    main()