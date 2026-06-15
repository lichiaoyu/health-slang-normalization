import pandas as pd

df = pd.read_csv("data_raw/youtube_candidate_slang_balanced.csv")

annotation_cols = [
    "annotation_id",
    "platform",
    "source_file",
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
    "health_domain",
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