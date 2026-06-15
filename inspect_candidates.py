import pandas as pd

df = pd.read_csv("data_raw/youtube_candidate_slang.csv")

print("Total candidate rows:", len(df))
print("\nSource distribution:")
print(df["source_file"].value_counts())

print("\nSlang distribution:")
print(df["slang_hits"].value_counts().head(30))

if "text" in df.columns:
    comment_df = df[df["source_file"].str.contains("comments", na=False)]
    print("\nCandidate comments:", len(comment_df))
    if len(comment_df) > 0:
        print(comment_df[["query", "slang_hits", "text", "video_title"]].head(20).to_string())

video_df = df[df["source_file"].str.contains("videos", na=False)]
print("\nCandidate videos:", len(video_df))
if len(video_df) > 0:
    print(video_df[["query", "slang_hits", "title", "description"]].head(20).to_string())