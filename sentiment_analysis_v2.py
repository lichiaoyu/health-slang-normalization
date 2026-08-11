"""
正式、可重現的情感分析方法。

你之前的 sentiment analysis 文件裡的結果是用某種「自動」方法產生的，
但沒有明確說明用了什麼工具/模型，這在方法學上站不住腳——審稿人會問
「你怎麼定義正負面」。

這裡改用 VADER（Valence Aware Dictionary and sEntiment Reasoner），
一個專門為社群媒體/非正式文字設計的詞典式情感分析工具（Hutto & Gilbert,
2014），對 emoji、誇飾語氣、大寫強調都有一定程度的處理，比一般新聞語料
訓練的情感模型更適合這份留言資料，而且是可重現、有文獻依據的方法，
可以直接寫進論文方法學章節。

用法：
    pip install vaderSentiment --break-system-packages
    python sentiment_analysis_v2.py
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_PATH = "data_raw/youtube_candidate_slang_cleaned.csv"
OUTPUT_PATH = "data_raw/sentiment_analysis_vader.csv"

analyzer = SentimentIntensityAnalyzer()


def safe_text_for_sentiment(row):
    """
    留言優先用留言本文；影片列（沒有留言文字）則用標題+描述代替。
    """
    text = row.get("text")
    if pd.notna(text) and str(text).strip():
        return str(text)
    parts = [str(row.get("title", "")), str(row.get("description", ""))]
    return " ".join(p for p in parts if p and p != "nan")


def classify_sentiment(compound_score):
    """
    VADER 官方建議的閾值（compound score 介於 -1 到 1 之間）：
    >= 0.05 正面，<= -0.05 負面，中間為中性。這是 VADER 文件裡
    公開建議的切點，不是我們自己隨意設定的，方法學上站得住腳。
    """
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(INPUT_PATH)
    df["text_for_sentiment"] = df.apply(safe_text_for_sentiment, axis=1)

    scores = df["text_for_sentiment"].apply(analyzer.polarity_scores)
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_pos"] = scores.apply(lambda s: s["pos"])
    df["vader_neu"] = scores.apply(lambda s: s["neu"])
    df["vader_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_label"] = df["vader_compound"].apply(classify_sentiment)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"共分析 {len(df)} 筆")
    print("\n整體情感分布:")
    print(df["sentiment_label"].value_counts())
    print(f"\n已存檔: {OUTPUT_PATH}")

    if "health_domain" in df.columns:
        print("\n各 domain 的情感分布 (%):")
        domain_sentiment = pd.crosstab(df["health_domain"], df["sentiment_label"], normalize="index") * 100
        print(domain_sentiment.round(1))

    if "clean_slang_hits" in df.columns:
        # primary_slang 可能還沒算過（balance_candidates.py 才會算），
        # 這裡簡單取 clean_slang_hits 的第一個詞當代表
        import ast

        def first_hit(x):
            try:
                hits = ast.literal_eval(str(x))
                return hits[0] if hits else "unknown"
            except Exception:
                return "unknown"

        df["_primary_slang_tmp"] = df["clean_slang_hits"].apply(first_hit)
        print("\n各 slang 詞的情感分布 (%，只列前 10 個最常見的詞):")
        top_terms = df["_primary_slang_tmp"].value_counts().head(10).index
        slang_sentiment = pd.crosstab(
            df[df["_primary_slang_tmp"].isin(top_terms)]["_primary_slang_tmp"],
            df[df["_primary_slang_tmp"].isin(top_terms)]["sentiment_label"],
            normalize="index",
        ) * 100
        print(slang_sentiment.round(1))

    print("\n⚠️ 方法學註記：VADER 是詞典式方法，對反諷、誇飾 slang（例如")
    print("「這個護膚品害我死掉了」實際是稱讚）的判斷不一定準確，")
    print("這點在你自己 sentiment analysis 文件的 Limitation 段落裡也提過，")
    print("VADER 並沒有解決這個根本限制，只是提供一個可重現、有文獻依據的基準方法。")


if __name__ == "__main__":
    main()
