"""
Figure 2（engagement mean vs. median by domain），重新產出。

沿用 engagement_by_domain.py 的邏輯（domain 解析、去重），但這裡直接
輸出圖表而不是只印文字報告。

用法（在你本機、health_slang_dataset 資料夾底下跑，需要先跑過
enrich_video_stats.py 產生 video_stats_lookup.jsonl）：
    pip install matplotlib --break-system-packages
    python make_fig2_engagement.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

VIDEO_STATS_PATH = "data_raw/video_stats_lookup.jsonl"

PAPER = "#F6F3EC"
INK = "#262421"
INK_SOFT = "#5B564C"
MOSS = "#3F6B5E"
MUSTARD = "#D9A441"
LINE = "#DDD6C7"

# 跟 balance_candidates.py / engagement_by_domain.py 用同一份對照表
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
    # Track C 稀有詞補強 query（之前這份對照表漏抄了這一段，
    # 導致這些 query 收集到的影片解析不出 domain，被排除在合併之外，
    # 這就是 make_fig2_engagement.py 算出 n=5,014 而不是完整的
    # n=5,635 的真正原因，跟 video_stats_lookup.jsonl 有沒有收集完整
    # 無關）。
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
        return cd
    query = video.get("query")
    if query in V1_QUERY_TO_DOMAIN:
        return V1_QUERY_TO_DOMAIN[query]
    return None


def main():
    if not Path(VIDEO_STATS_PATH).exists():
        raise FileNotFoundError(f"找不到 {VIDEO_STATS_PATH}，請先跑過 enrich_video_stats.py")

    videos = load_unique_videos()
    rows = []
    for v in videos:
        domain = resolve_domain(v)
        if domain:
            rows.append({"video_id": v.get("video_id"), "health_domain": domain})
    domain_df = pd.DataFrame(rows)

    stats_df = pd.read_json(VIDEO_STATS_PATH, lines=True).drop_duplicates(subset="video_id")
    merged = domain_df.merge(stats_df, on="video_id", how="inner")
    merged = merged[merged["view_count"].notna()]

    summary = merged.groupby("health_domain")["view_count"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    print(f"合併後影片數: {len(merged)}")
    print(summary)

    domains = summary.index.tolist()
    means = summary["mean"].tolist()
    medians = summary["median"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    y_pos = np.arange(len(domains))
    bar_h = 0.35
    ax.barh(y_pos + bar_h / 2, means, height=bar_h, color=MOSS, label="Mean views", zorder=3)
    ax.barh(y_pos - bar_h / 2, medians, height=bar_h, color=MUSTARD, label="Median views", zorder=3)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d.replace("_", " ") for d in domains], fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Views (log scale)", fontsize=12, color=INK)
    fig.suptitle(f"Mean vs. median view count by domain (n={len(merged)} videos)",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_SOFT)
    ax.legend(frameon=False, fontsize=11, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figures/fig2_engagement_mean_median.png", dpi=300, facecolor=PAPER)
    plt.close()
    print("\n已存檔: figures/fig2_engagement_mean_median.png")


if __name__ == "__main__":
    main()