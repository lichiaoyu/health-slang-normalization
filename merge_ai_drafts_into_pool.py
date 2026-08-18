"""
把 normalize_slang.py 的預測結果，接進 annotation pool CSV，變成
annotator 工具裡看得到的「AI 草稿建議」欄位。

重要設計決定：這裡新增的是 ai_draft_literal_or_nonliteral /
ai_draft_normalized_meaning 兩個獨立欄位，**不會**直接寫進
literal_or_nonliteral / normalized_meaning 這兩個真正的標註欄位。

原因：如果直接把 AI 預測填進真正的標註欄位，annotator 工具會把這些
列誤判成「已經標註過」（因為 isAnnotated() 檢查的就是
literal_or_nonliteral 有沒有值），你會在完全沒有人看過的情況下，
讓進度條顯示成已完成。真正的標註欄位只有在你在工具裡按下
「Accept AI suggestion」或自己輸入判斷之後才會被填上，這樣才符合
Wang et al. (2024, CHI) 的人機協作標註設計：LLM 產生草稿，人類主動
驗證或覆蓋，而不是被動接受。

用法：
    python merge_ai_drafts_into_pool.py
"""

import json
from pathlib import Path

import pandas as pd

POOL_PATH = "data_raw/youtube_annotation_pool_cleaned.csv"
PREDICTIONS_PATH = "data_raw/normalizer_predictions.jsonl"
OUTPUT_PATH = "data_raw/youtube_annotation_pool_with_ai_drafts.csv"


def main():
    for path in [POOL_PATH, PREDICTIONS_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到 {path}。請先跑過 normalize_slang.py 產生預測結果。"
            )

    pool = pd.read_csv(POOL_PATH)
    preds = pd.read_json(PREDICTIONS_PATH, lines=True)

    preds = preds.rename(columns={
        "model_literal_or_nonliteral": "ai_draft_literal_or_nonliteral",
        "model_normalized_meaning": "ai_draft_normalized_meaning",
    })

    merged = pool.merge(
        preds[["annotation_id", "ai_draft_literal_or_nonliteral", "ai_draft_normalized_meaning"]],
        on="annotation_id",
        how="left",
    )

    n_with_draft = merged["ai_draft_literal_or_nonliteral"].notna().sum()
    print(f"Annotation pool 筆數: {len(merged)}")
    print(f"有 AI 草稿建議的筆數: {n_with_draft}")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n已存檔: {OUTPUT_PATH}")
    print("把這份檔案上傳到 annotator 工具，草稿建議會顯示在每個判斷按鈕上方，")
    print("按 Accept 才會真正填進標註欄位並存檔。")


if __name__ == "__main__":
    main()