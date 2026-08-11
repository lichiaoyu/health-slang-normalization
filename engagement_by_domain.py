"""
Engagement × Domain 交叉分析。完全離線（讀取 enrich_video_stats.py 跟
enrich_geo.py / 收集階段已經產生的資料，不需要額外呼叫 API）。

呼應你朋友文件裡「Sports_fitness 平均觀看數 15.8M，最高」那類分析，
但這裡额外加上中位數——觀看數這種指標通常極度右偏（少數爆紅影片
把平均值拉得很高），只看平均值容易誤導，你朋友的圖表沒有呈現這點，
這是你可以做得更嚴謹的地方。

用法：
    python engagement_by_domain.py
"""

import json
from pathlib import Path

import pandas as pd

VIDEO_STATS_PATH = "data_raw/video_stats_lookup.jsonl"

# 跟 balance_candidates.py 裡的 V1_QUERY_TO_DOMAIN 完全一致，
# 用來把 batch_01（沒有 collection_domain 欄位）的影片也解析出 domain。
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

    '"side effects" "fighting for my life" #shorts': "medication_side_effects",
    '"side effects" "messed me up" #shorts': "medication_side_effects",
    '"side effects" "wrecked me" #shorts': "medication_side_effects",
    '"side effects" "took me out" #shorts': "medication_side_effects",
    '"supplement" "had me dying" #shorts': "supplements",
    '"pre workout" "fighting for my life" #shorts': "supplements",
    '"pre workout" "messed me up" #shorts': "supplements",

    '"melatonin" "knocked me out" #shorts': "sleep_fatigue",
    '"sleep supplement" "knocked me out" #shorts': "sleep_fatigue",
    '"magnesium" "knocked me out" #shorts': "sleep_fatigue",
    '"ashwagandha" "messed me up" #shorts': "sleep_fatigue",

    '"girl dinner" "calorie deficit" #shorts': "diet_weight_loss",
    '"weight loss" "glow up" #shorts': "body_image_transformation",
    '"fitness" "glow up" #shorts': "body_image_transformation",
    '"gymtok" "locked in" #shorts': "fitness_gym",
    '"gymtok" "cooked" #shorts': "fitness_gym",
    '"weight loss" "snatched" #shorts': "body_image_transformation",

    '"protein powder" "bussin" #shorts': "supplements",
    '"gut health" "hits different" #shorts': "gut_health",
    '"healthy recipe" "bussin" #shorts': "diet_weight_loss",
    '"meal prep" "hits different" #shorts': "diet_weight_loss",

    # Track C 稀有詞補強 query
    '"creatine" "fighting for my life" #shorts': "supplements",
    '"pre workout" "had me in shambles" #shorts': "supplements",
    '"protein powder" "no cap" #shorts': "supplements",
    '"keto" "fighting for my life" #shorts': "diet_weight_loss",
    '"diet" "had me in shambles" #shorts': "diet_weight_loss",
    '"weight loss" "no cap" #shorts': "diet_weight_loss",
    '"weight loss" "slay" #shorts': "diet_weight_loss",
    '"leg day" "fighting for my life" #shorts': "fitness_gym",
    '"gym" "had me in shambles" #shorts': "fitness_gym",
    '"gym routine" "no cap" #shorts': "fitness_gym",
    '"gym motivation" "slay" #shorts': "fitness_gym",
    '"detox" "fighting for my life" #shorts': "gut_health",
    '"side effects" "had me in shambles" #shorts': "medication_side_effects",
    '"glow up" "slay" #shorts': "body_image_transformation",
    '"summer body" "slay" #shorts': "body_image_transformation",
    '"glow up" "no cap" #shorts': "body_image_transformation",
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


def resolve_domain(video):
    cd = video.get("collection_domain")
    if cd and str(cd).strip():
        return cd, "collection_ground_truth"
    query = video.get("query")
    if query in V1_QUERY_TO_DOMAIN:
        return V1_QUERY_TO_DOMAIN[query], "v1_query_inferred"
    return "unclassified", "unclassified"


def main():
    if not Path(VIDEO_STATS_PATH).exists():
        raise FileNotFoundError(f"找不到 {VIDEO_STATS_PATH}，請先跑過 enrich_video_stats.py")

    videos = load_unique_videos()
    print(f"不重複影片數: {len(videos)}")

    rows = []
    for v in videos:
        domain, source = resolve_domain(v)
        rows.append({
            "video_id": v.get("video_id"),
            "health_domain": domain,
            "domain_source": source,
        })
    video_domain_df = pd.DataFrame(rows)

    unclassified_n = (video_domain_df["health_domain"] == "unclassified").sum()
    if unclassified_n:
        print(f"⚠️ 有 {unclassified_n} 支影片無法解析 domain（query 不在對照表裡），"
              f"會被排除在交叉分析之外。")

    stats_df = pd.read_json(VIDEO_STATS_PATH, lines=True)
    stats_df = stats_df.drop_duplicates(subset="video_id")

    merged = video_domain_df.merge(stats_df, on="video_id", how="inner")
    merged = merged[merged["health_domain"] != "unclassified"]

    print(f"成功合併 engagement 數據的影片數: {len(merged)}")

    summary = merged.groupby("health_domain").agg(
        n_videos=("video_id", "count"),
        mean_views=("view_count", "mean"),
        median_views=("view_count", "median"),
        mean_likes=("like_count", "mean"),
        median_likes=("like_count", "median"),
        mean_comments=("comment_count", "mean"),
    ).sort_values("mean_views", ascending=False)

    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print("\n各 domain 的 engagement 統計（依平均觀看數排序）:")
    print(summary)

    summary.to_csv("data_raw/engagement_by_domain.csv")
    print(f"\n已存檔: data_raw/engagement_by_domain.csv")

    print("\n⚠️ 方法學提醒：平均值（mean）容易被少數爆紅影片拉高，")
    print("中位數（median）更能反映「一般」影片的實際表現，")
    print("你朋友的圖表只呈現平均值，建議你的報告兩者都放，")
    print("如果兩者差距很大，那本身就是一個值得討論的分布特性。")


if __name__ == "__main__":
    main()