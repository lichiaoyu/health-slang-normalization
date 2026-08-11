"""
在已經跑好的雙模型情感比對結果上，先去除跨 batch 重複留言，
再重新計算「整體」的一致率、Cohen's Kappa、完全相反極性的筆數。

背景：sentiment_analysis_roberta.py 是直接對 clean_candidates.py 產生的
2,009 筆「清理後候選池」跑的，那個階段還沒做跨 batch 去重（跟 annotation
pool 遇到的是同一個上游問題：同一則留言可能同時被 batch_health_broad_01
跟 batch_health_broad_02 收集到）。merge_sentiment_with_taxonomy.py 執行
時發現整份 2,009 筆裡有 145 筆是這種重複列，這代表先前算出來的整體數字
（n=2009、58.7%、Kappa=0.318）也需要在去重後的版本上重新算一次，才能跟
annotation pool 的 547 筆口徑一致。

這裡不需要重新呼叫 RoBERTa 模型（那一步比較慢），因為
sentiment_analysis_dual_model.csv 已經存了每一筆的 vader_label 跟
roberta_label，只要去重、重新統計即可。

用法：
    python recompute_overall_sentiment_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

INPUT_PATH = "data_raw/sentiment_analysis_dual_model.csv"
OUTPUT_PATH = "data_raw/sentiment_analysis_dual_model_deduped.csv"


def make_merge_key(df: pd.DataFrame) -> pd.Series:
    """跟 merge_sentiment_with_taxonomy.py 用的是同一套邏輯，確保兩支
    腳本判斷「哪些列是重複」的標準一致。"""
    comment_id = df["comment_id"].astype(str)
    video_id = df["video_id"].astype(str)
    is_comment_row = (
        df["comment_id"].notna()
        & (comment_id.str.strip() != "")
        & (comment_id.str.lower() != "nan")
    )
    key = np.where(is_comment_row, "c_" + comment_id, "v_" + video_id)
    return pd.Series(key, index=df.index)


def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"找不到 {INPUT_PATH}，請先跑過 sentiment_analysis_roberta.py")

    df = pd.read_csv(INPUT_PATH)
    n_before = len(df)

    df["_merge_key"] = make_merge_key(df)
    deduped = df.drop_duplicates(subset="_merge_key", keep="first").copy()
    n_after = len(deduped)

    print(f"去重前筆數: {n_before}")
    print(f"去重後筆數: {n_after}（移除 {n_before - n_after} 筆跨 batch 重複留言）")

    deduped.to_csv(OUTPUT_PATH, index=False)

    n_agree = deduped["models_agree"].sum()
    agree_pct = n_agree / n_after * 100
    kappa = cohen_kappa_score(deduped["sentiment_label"], deduped["roberta_label"])

    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(deduped["sentiment_label"], deduped["roberta_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"vader_{l}" for l in labels], columns=[f"roberta_{l}" for l in labels])

    # 完全相反極性：VADER 負面但 RoBERTa 正面，或反過來
    opposite = (
        ((deduped["sentiment_label"] == "negative") & (deduped["roberta_label"] == "positive"))
        | ((deduped["sentiment_label"] == "positive") & (deduped["roberta_label"] == "negative"))
    )
    n_opposite = opposite.sum()
    opposite_pct = n_opposite / n_after * 100

    vader_dist = deduped["sentiment_label"].value_counts(normalize=True).mul(100).round(1)
    roberta_dist = deduped["roberta_label"].value_counts(normalize=True).mul(100).round(1)

    print(f"\n{'=' * 60}")
    print("去重後的整體一致性（取代先前用未去重資料算出的數字）")
    print(f"{'=' * 60}")
    print(f"總筆數: {n_after}")
    print(f"一致筆數: {n_agree} ({agree_pct:.1f}%)")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"\n完全相反極性筆數: {n_opposite} ({opposite_pct:.1f}%)")
    print(f"\nVADER 分布 (%):\n{vader_dist}")
    print(f"\nRoBERTa 分布 (%):\n{roberta_dist}")
    print(f"\n混淆矩陣:\n{cm_df}")

    print(f"\n已存檔去重後的完整結果: {OUTPUT_PATH}")
    print("\n把上面「去重後」這幾個數字拿去跟 Claude 說，")
    print("用來取代論文 Sentiment Analysis 小節裡原本用未去重資料算出的版本。")


if __name__ == "__main__":
    main()