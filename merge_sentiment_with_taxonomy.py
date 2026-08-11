"""
把雙模型情感比對結果（sentiment_analysis_roberta.py 的輸出）跟
annotation pool 的 expression_type / health_domain / primary_slang
標籤接起來，這樣才能算出「依 expression_type 拆解的一致率」。

背景：sentiment_analysis_v2.py（以及接在它後面跑的
sentiment_analysis_roberta.py）是對 clean_candidates.py 產生的
2,009 筆「清理後候選資料」跑的，那個階段還在 balancing/annotation
pool 產生之前，所以檔案裡沒有 expression_type 這個欄位。
expression_type 是後來 make_annotation_pool.py 才標上去的，
而且只標在最終平衡後的 547 筆（或你手上還沒 dedupe 的 574 筆）。

這支腳本用 comment_id（留言列）或 video_id（純標題/描述列，
comment_id 是空的）當 join key，把兩份資料接起來。因為 annotation
pool 是從候選池「取樣後平衡」出來的子集，所以合併後的筆數會比
2,009 少很多，只有同時出現在兩份檔案裡的那些列才會有結果
——這是預期中的正常現象，不是錯誤。

用法：
    python merge_sentiment_with_taxonomy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENTIMENT_PATH = "data_raw/sentiment_analysis_dual_model.csv"

# 指向你目前手上「已經 dedupe 過的 547 筆」annotation pool 檔案。
# 如果你還沒把 Claude 給你的 youtube_annotation_pool_cleaned.csv 存回
# data_raw/ 資料夾，先把它存過去，或者把下面這行路徑改成你實際存放的位置。
ANNOTATION_POOL_PATH = "data_raw/youtube_annotation_pool_cleaned.csv"

OUTPUT_PATH = "data_raw/sentiment_by_expression_type.csv"


def make_merge_key(df: pd.DataFrame) -> pd.Series:
    """
    留言列用 comment_id；純標題/描述列（comment_id 是空值）改用
    video_id 當 key，並加前綴避免不同性質的 key 混在一起誤配對。

    用 np.where 一次算出最終字串，避免布林值跟字串混在一起相加
    （上一版在這裡有 bug，把布林遮罩直接跟字串相加導致 TypeError）。
    """
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
    for path in [SENTIMENT_PATH, ANNOTATION_POOL_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到 {path}。請確認 sentiment_analysis_roberta.py 已經跑過，"
                f"且 annotation pool 檔案路徑正確（見腳本開頭的 ANNOTATION_POOL_PATH）。"
            )

    sentiment = pd.read_csv(SENTIMENT_PATH)
    pool = pd.read_csv(ANNOTATION_POOL_PATH)

    sentiment["_merge_key"] = make_merge_key(sentiment)
    pool["_merge_key"] = make_merge_key(pool)

    # 上游的 2,009 筆清理後候選池，跟 annotation pool 有同一個根源的問題：
    # 同一則留言可能同時被 batch_health_broad_01 跟 batch_health_broad_02
    # 收集到，兩邊都沒有先做跨 batch 的去重。這裡先去重，否則一筆
    # annotation pool row 會對到兩筆重複的 sentiment row，讓配對筆數
    # 超過 annotation pool 本身的筆數（等於是同一筆資料被算了兩次）。
    n_before_dedup = len(sentiment)
    sentiment = sentiment.drop_duplicates(subset="_merge_key", keep="first")
    n_dropped = n_before_dedup - len(sentiment)
    if n_dropped > 0:
        print(f"⚠️ 情感分析資料裡發現 {n_dropped} 筆跨 batch 重複留言，已在合併前移除。")
        print("   這跟 annotation pool 遇到的是同一個上游問題（filter_candidates.py")
        print("   沒有跨 batch 去重），建議之後在 clean_candidates.py 的輸出階段")
        print("   就處理掉，而不是每次分析前都要補一次 dedup。")
        print()

    pool_labels = pool[["_merge_key", "expression_type", "health_domain", "primary_slang", "annotation_id"]].drop_duplicates("_merge_key")

    merged = sentiment.merge(pool_labels, on="_merge_key", how="inner")

    print(f"情感比對資料筆數: {len(sentiment)}")
    print(f"Annotation pool 筆數: {len(pool)}")
    print(f"成功配對筆數: {len(merged)} ({len(merged) / len(pool) * 100:.1f}% 的 annotation pool)")

    if len(merged) == 0:
        print("\n⚠️ 沒有任何一筆配對成功。最常見的原因：")
        print("  1. ANNOTATION_POOL_PATH 指到錯的檔案（例如指到還沒跑過 sentiment 的舊版）")
        print("  2. 兩份檔案來自不同批次的收集資料，comment_id/video_id 對不起來")
        return

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print("依 expression_type 拆解的雙模型一致率")
    print(f"{'=' * 60}")
    by_type = merged.groupby("expression_type")["models_agree"].agg(["mean", "count"])
    by_type["mean"] = (by_type["mean"] * 100).round(1)
    by_type.columns = ["agreement_pct", "n"]
    by_type = by_type.sort_values("agreement_pct")
    print(by_type)

    print(f"\n已存檔: {OUTPUT_PATH}")
    print("\n把上面這張表的數字拿去跟 Claude 說，就能把論文裡 Sentiment Analysis")
    print("小節的 [TYPE_AGREE_PCT] / [SLANG_AGREE_PCT] 括號填上真實數字。")


if __name__ == "__main__":
    main()