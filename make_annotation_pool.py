import pandas as pd

df = pd.read_csv("data_raw/youtube_candidate_slang_balanced.csv")

# =============================================================================
# EXPRESSION_TYPE_MAP
#
# 把 SLANG_TERMS 清單裡的詞分成三類，區分「潮流性/社交語氣導向的
# internet slang」跟「非正式但症狀導向、跟醫學語彙有重疊的
# informal symptom language」。這個區分很重要，因為像 brain fog 這種詞
# 本質上更接近患者社群長期使用的症狀自陳語，而不是短命的流行語，
# 混在一起會模糊掉「health slang」這個研究構念真正在測量的是什麼。
#
# - internet_slang：潮流性、社交語氣詞，通常不專屬於健康情境
# - informal_symptom_language：非正式但描述具體生理/認知狀態的詞
# - ambiguous：語意會因上下文在兩類之間切換，需要標註者逐筆判斷
#
# 未列在這裡的詞會標成 "unclassified"，代表詞庫清單本身如果之後又
# 擴充了新詞，需要回來更新這份對照表。
# =============================================================================
EXPRESSION_TYPE_MAP = {
    # --- internet_slang ---
    "dead": "internet_slang",
    "dying": "internet_slang",
    "slay": "internet_slang",
    "bussin": "internet_slang",
    "hits different": "internet_slang",
    "lowkey": "internet_slang",
    "highkey": "internet_slang",
    "fr": "internet_slang",
    "fr fr": "internet_slang",
    "no cap": "internet_slang",
    "locked in": "internet_slang",
    "delulu": "internet_slang",
    "girl dinner": "internet_slang",
    "snatched": "internet_slang",
    "glow up": "internet_slang",
    "cooked": "internet_slang",
    "brainrot": "internet_slang",
    "had me dying": "internet_slang",
    "had me in shambles": "internet_slang",
    "sent me": "internet_slang",
    "took me out": "internet_slang",

    # --- informal_symptom_language ---
    "brain fog": "informal_symptom_language",
    "food noise": "informal_symptom_language",
    "crashed": "informal_symptom_language",
    "energy crash": "informal_symptom_language",
    "wired": "informal_symptom_language",
    "jittery": "informal_symptom_language",
    "shaky": "informal_symptom_language",
    "heart racing": "informal_symptom_language",
    "bloated": "informal_symptom_language",
    "bloated af": "informal_symptom_language",
    "couldn't function": "informal_symptom_language",
    "could barely function": "informal_symptom_language",
    "felt awful": "informal_symptom_language",
    "felt terrible": "informal_symptom_language",
    "stomach was wrecked": "informal_symptom_language",
    "wrecked my stomach": "informal_symptom_language",
    "messed up my stomach": "informal_symptom_language",
    "destroyed my stomach": "informal_symptom_language",
    "couldn't eat": "informal_symptom_language",
    "could not eat": "informal_symptom_language",
    "couldn't sleep": "informal_symptom_language",
    "slept all day": "informal_symptom_language",
    "out cold": "informal_symptom_language",
    "zombie": "informal_symptom_language",
    "nuked my appetite": "informal_symptom_language",
    "killed my appetite": "informal_symptom_language",

    # --- ambiguous（需要語境才能判斷，標註時建議特別留意） ---
    "knocked me out": "ambiguous",
    "fighting for my life": "ambiguous",
    "messed me up": "ambiguous",
    "wrecked me": "ambiguous",
    "felt like i was dying": "ambiguous",
    "felt like death": "ambiguous",
}

df["expression_type"] = df["primary_slang"].map(EXPRESSION_TYPE_MAP).fillna("unclassified")

annotation_cols = [
    "annotation_id",
    "platform",
    "source_file",
    "batch_name",
    "query",
    "published_at",
    "video_id",
    "comment_id",
    "video_url",
    "video_title",
    "title",
    "text",
    "primary_slang",
    "clean_slang_hits",
    "expression_type",
    "health_domain",
    "domain_source",
    "collection_domain",
    "collection_track",
    "priority_score",

    "slang_span",
    "normalized_meaning",
    "normalized_sentence",
    "health_category",
    "literal_or_nonliteral",
    "ambiguity_level",
    "requires_visual_context",
    "requires_audio_context",
    "requires_emoji_context",
    "requires_comment_context",
    "keep_final",
    "notes"
]

df["annotation_id"] = [f"yt_{i:04d}" for i in range(len(df))]
df["platform"] = "youtube"

for col in annotation_cols:
    if col not in df.columns:
        df[col] = ""

df[annotation_cols].to_csv("data_raw/youtube_annotation_pool.csv", index=False)
print("Saved annotation pool:", len(df))

# expression_type 分布快速檢查。如果 unclassified 的數量不是 0，
# 代表 primary_slang 裡出現了 EXPRESSION_TYPE_MAP 沒涵蓋到的詞，
# 需要回來補上對照表。
print("\nExpression type 分布:")
print(df["expression_type"].value_counts())

if (df["expression_type"] == "unclassified").any():
    unclassified_terms = df.loc[df["expression_type"] == "unclassified", "primary_slang"].unique()
    print(f"\n⚠️ 有 {len(unclassified_terms)} 個詞沒被 EXPRESSION_TYPE_MAP 涵蓋到：{list(unclassified_terms)}")
    print("請回到 make_annotation_pool.py 補上這幾個詞的分類。")

# 快速檢查：domain 標籤來源分布，讓你在標註前先知道每個 domain 的
# 標籤有多少比例是 ground truth（收集階段記錄），多少是關鍵字猜的。
# 標註時如果對 health_domain 有疑慮，可以優先檢查 keyword_fallback 的列。
if "domain_source" in df.columns and df["domain_source"].notna().any():
    print("\nDomain label 來源分布（整個標註池）:")
    print(df["domain_source"].value_counts())

    print("\n各 domain 中 ground truth 比例:")
    print(
        df.groupby("health_domain")["domain_source"]
        .apply(lambda s: (s == "collection_ground_truth").mean())
        .sort_values(ascending=False)
    )