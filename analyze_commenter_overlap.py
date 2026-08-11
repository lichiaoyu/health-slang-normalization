"""
留言者跨 health domain 的重疊分析（"audience silo" 分析）。

注意：這裡刻意用「全部留言」而不是只用 youtube_candidate_slang_cleaned.csv
（那份只留有命中 slang 詞的留言）。原因是：受眾重不重疊，問的是「同一個人
會不會同時看/留言在不同健康主題的影片底下」，這跟他當初留言裡有沒有講
slang 無關——用全部留言當分母，才是真正在回答「audience 是不是 siloed」
這個問題，而不是「有講 slang 的那群人是不是 siloed」（範圍會窄很多，
也偏離題目本身要問的東西）。

domain 的解析邏輯跟 engagement_by_domain.py 完全一致（collection_domain
優先，V1 query 對照表其次），確保跟你論文其他地方引用的 domain 標籤是
同一套標準，不會兩邊對不上。

用法：
    python analyze_commenter_overlap.py
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

# 跟 engagement_by_domain.py / balance_candidates.py 裡的對照表完全一致。
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

MIN_COMMENTERS_FOR_DOMAIN = 20  # domain 留言者太少的話重疊率會很不穩定，低於這個門檻只列出但特別註記


def load_unique_videos():
    seen = set()
    videos = {}
    for path in sorted(Path("data_raw").glob("*_youtube_videos.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                vid = item.get("video_id")
                if vid and vid not in seen:
                    seen.add(vid)
                    videos[vid] = item
    return videos


def resolve_domain(video):
    cd = video.get("collection_domain")
    if cd and str(cd).strip():
        return cd
    query = video.get("query")
    if query in V1_QUERY_TO_DOMAIN:
        return V1_QUERY_TO_DOMAIN[query]
    return None


def load_comments_dedup():
    """
    跨 batch 去重（跟 dedup_annotation_pool.py / merge_sentiment_with_taxonomy.py
    是同一個上游問題：同一則留言可能同時被兩個 batch 收集到）。
    這裡用 comment_id 本身去重，因為留言本來就有唯一 ID，不需要像
    候選池那樣用 text+video_url 湊 key。
    """
    seen_ids = set()
    comments = []
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("comment_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    comments.append(item)
    return comments


def main():
    videos = load_unique_videos()
    comments = load_comments_dedup()

    print(f"不重複影片數: {len(videos)}")
    print(f"不重複留言數（已跨 batch 去重）: {len(comments)}")

    video_domain = {vid: resolve_domain(v) for vid, v in videos.items()}
    unresolved = sum(1 for d in video_domain.values() if d is None)
    print(f"無法解析 domain 的影片數: {unresolved}（會被排除在這個分析之外）")

    domain_commenters = {}
    excluded_no_author = 0

    for c in comments:
        vid = c.get("video_id")
        author = c.get("author_channel_id")
        domain = video_domain.get(vid)
        if domain is None:
            continue
        if not author:
            excluded_no_author += 1
            continue
        domain_commenters.setdefault(domain, set()).add(author)

    print(f"缺少 author_channel_id 而被排除的留言數: {excluded_no_author}")
    print()
    print("各 domain 的不重複留言者數:")
    for domain, authors in sorted(domain_commenters.items(), key=lambda x: -len(x[1])):
        flag = "" if len(authors) >= MIN_COMMENTERS_FOR_DOMAIN else "  ⚠️ 樣本數過小，重疊率不穩定"
        print(f"  {domain}: {len(authors)}{flag}")

    rows = []
    domains = sorted(domain_commenters.keys())
    for d1, d2 in combinations(domains, 2):
        set1, set2 = domain_commenters[d1], domain_commenters[d2]
        overlap = set1 & set2
        overlap_coef = len(overlap) / min(len(set1), len(set2)) * 100 if min(len(set1), len(set2)) > 0 else 0
        jaccard = len(overlap) / len(set1 | set2) * 100 if len(set1 | set2) > 0 else 0
        rows.append({
            "domain_a": d1,
            "domain_b": d2,
            "n_a": len(set1),
            "n_b": len(set2),
            "n_overlap": len(overlap),
            "overlap_coefficient_pct": round(overlap_coef, 2),
            "jaccard_pct": round(jaccard, 2),
        })

    overlap_df = pd.DataFrame(rows).sort_values("overlap_coefficient_pct", ascending=False)
    overlap_df.to_csv("data_raw/commenter_domain_overlap.csv", index=False)

    print(f"\n{'=' * 70}")
    print("Domain 兩兩之間的留言者重疊（overlap coefficient = 交集 / 較小那組的大小）")
    print(f"{'=' * 70}")
    print(overlap_df.to_string(index=False))

    max_row = overlap_df.iloc[0]
    print(f"\n最高重疊: {max_row['domain_a']} x {max_row['domain_b']} "
          f"= {max_row['overlap_coefficient_pct']:.2f}% (overlap coefficient)")

    # 額外指標：每個留言者總共出現在幾個 domain 裡，用來回答
    # 「大部分人是不是只活動在一個健康主題底下」這個更直觀的問題。
    author_domain_count = {}
    for domain, authors in domain_commenters.items():
        for a in authors:
            author_domain_count[a] = author_domain_count.get(a, set())
            author_domain_count[a].add(domain)

    n_domains_per_author = pd.Series([len(v) for v in author_domain_count.values()])
    single_domain_pct = (n_domains_per_author == 1).mean() * 100

    print(f"\n只出現在單一 domain 底下的留言者比例: {single_domain_pct:.1f}%")
    print("留言者橫跨的 domain 數分布:")
    print(n_domains_per_author.value_counts().sort_index())

    print(f"\n已存檔: data_raw/commenter_domain_overlap.csv")
    print("\n⚠️ 方法學註記：這裡假設同一個 author_channel_id 代表同一個真實使用者，")
    print("這在 YouTube 上大致成立，但無法排除少數人使用多個帳號留言的情況，")
    print("這跟你在 gender inference 那節遇到的「顯示名稱不等於真實身份」是同一類限制。")


if __name__ == "__main__":
    main()