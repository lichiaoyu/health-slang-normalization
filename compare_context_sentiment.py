from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_FILE = Path("data_raw/youtube_candidate_slang_enriched.csv")
OUTPUT_FILE = Path("analysis_output/context_sentiment_comparison.csv")

analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> str:
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


df = pd.read_csv(INPUT_FILE)

comment_text = df.get("text", pd.Series("", index=df.index)).fillna("")
video_title = df.get("video_title", pd.Series("", index=df.index)).fillna("")
title = df.get("title", pd.Series("", index=df.index)).fillna("")
description = df.get("description", pd.Series("", index=df.index)).fillna("")

df["comment_only_sentiment"] = comment_text.apply(classify_sentiment)

df["title_context_sentiment"] = (
    comment_text + " " + video_title + " " + title
).apply(classify_sentiment)

df["full_context_sentiment"] = (
    comment_text + " " + video_title + " " + title + " " + description
).apply(classify_sentiment)

df["title_changed_sentiment"] = (
    df["comment_only_sentiment"] != df["title_context_sentiment"]
)

df["full_context_changed_sentiment"] = (
    df["comment_only_sentiment"] != df["full_context_sentiment"]
)

print("Comment → title-context change rate:")
print(df["title_changed_sentiment"].mean())

print("\nComment → full-context change rate:")
print(df["full_context_changed_sentiment"].mean())

print("\nComment-only vs full-context matrix:")
matrix = pd.crosstab(
    df["comment_only_sentiment"],
    df["full_context_sentiment"],
)
print(matrix)

chi2, p_value, dof, _ = chi2_contingency(matrix)

print(f"\nChi-square: {chi2:.3f}")
print(f"p-value: {p_value:.6g}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")