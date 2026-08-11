"""
第二個獨立的情感分析方法（跟 sentiment_analysis_v2.py 的 VADER 對照）。

背景：VADER 是詞典式方法，對反諷、誇飾語氣的 slang（例如「這個護膚品害我
死掉了」實際是稱讚）的判斷不一定準確，這點你自己在 sentiment_analysis_v2.py
的方法學註記裡已經寫過。單一方法沒辦法回答「這個限制實際上影響了多少筆
資料」，所以這裡加入第二個獨立訓練的模型（cardiffnlp/twitter-roberta-base-
sentiment-latest，專門在推特語料上訓練的 transformer），跟 VADER 做逐筆比對，
用 Cohen's Kappa 衡量兩個方法的一致性。

這不是要決定「誰才是對的」——沒有人工標註的 ground truth，兩個方法哪個更準
無法直接判定。這裡的重點是：兩個方法「不一致」的比例本身就是一個發現，
代表 slang/反諷語氣造成的判斷困難不是 VADER 特有的問題，而是情感分析方法
普遍的限制。不一致的那些筆數，剛好也是最適合挑出來做質性討論的候選案例
（跟你 Qualitative Analysis 那節的精神一致）。

前置需求：
    先跑過 sentiment_analysis_v2.py，產生 data_raw/sentiment_analysis_vader.csv
    （這支腳本會直接讀那份檔案的 text_for_sentiment 欄位，確保兩個模型吃進去
    的是完全相同的文字，比較才公平）。

用法：
    pip install transformers torch scikit-learn pandas --break-system-packages
    python sentiment_analysis_roberta.py

注意：
    CPU 就能跑，不需要 GPU。幾千筆資料在一般筆電上大概幾分鐘內能跑完；
    如果資料量到了萬筆以上，建議調大 BATCH_SIZE 或考慮用 Colab 的 GPU。
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from transformers import pipeline

INPUT_PATH = "data_raw/sentiment_analysis_vader.csv"
OUTPUT_PATH = "data_raw/sentiment_analysis_dual_model.csv"
DISAGREEMENT_PATH = "data_raw/sentiment_disagreement_cases.csv"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16
MAX_LENGTH = 512


def classify_batch(classifier, texts):
    """
    這個模型的原生輸出標籤就是 'negative' / 'neutral' / 'positive'，
    跟 VADER 的三分類命名一致，不需要額外的標籤對照表
    （舊版 twitter-roberta-base-sentiment 用的是 LABEL_0/1/2，
    這裡特別選 -latest 版本就是為了避免這個額外的轉換步驟）。
    """
    results = classifier(texts, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
    return [r["label"] for r in results]


def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(
            f"找不到 {INPUT_PATH}，請先跑過 sentiment_analysis_v2.py"
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"讀取 {len(df)} 筆已有 VADER 結果的資料")

    texts = df["text_for_sentiment"].fillna("").astype(str).tolist()

    print(f"\n載入模型: {MODEL_NAME}（第一次執行會自動下載，約 500MB）")
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,  # CPU；如果有 GPU 且已裝好 CUDA，改成 device=0 會快很多
    )

    print("跑第二個模型（RoBERTa）...")
    roberta_labels = []
    for start in range(0, len(texts), 200):
        chunk = texts[start:start + 200]
        roberta_labels.extend(classify_batch(classifier, chunk))
        print(f"  進度: {min(start + 200, len(texts))}/{len(texts)}")

    df["roberta_label"] = roberta_labels
    df["models_agree"] = df["sentiment_label"] == df["roberta_label"]

    df.to_csv(OUTPUT_PATH, index=False)

    n_total = len(df)
    n_agree = df["models_agree"].sum()
    agree_pct = n_agree / n_total * 100

    kappa = cohen_kappa_score(df["sentiment_label"], df["roberta_label"])

    print(f"\n{'=' * 60}")
    print("兩個模型的整體一致性")
    print(f"{'=' * 60}")
    print(f"總筆數: {n_total}")
    print(f"一致筆數: {n_agree} ({agree_pct:.1f}%)")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print("  （經驗法則：<0=poor, 0.01-0.20=slight, 0.21-0.40=fair,")
    print("   0.41-0.60=moderate, 0.61-0.80=substantial, 0.81-1.00=almost perfect）")

    print(f"\n{'=' * 60}")
    print("混淆矩陣（VADER 列 x RoBERTa 欄）")
    print(f"{'=' * 60}")
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(df["sentiment_label"], df["roberta_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"vader_{l}" for l in labels], columns=[f"roberta_{l}" for l in labels])
    print(cm_df)

    if "expression_type" in df.columns:
        print(f"\n{'=' * 60}")
        print("依 expression_type 拆解一致性（這是最值得寫進論文的部分）")
        print(f"{'=' * 60}")
        by_type = df.groupby("expression_type")["models_agree"].agg(["mean", "count"])
        by_type["mean"] = (by_type["mean"] * 100).round(1)
        by_type.columns = ["agreement_pct", "n"]
        print(by_type)
        print("\n預期：ambiguous / informal_symptom_language 類別的一致性應該")
        print("低於 internet_slang，因為這些類別更常出現字面意思跟情感語氣")
        print("相反的情況（例如 yt_0545 那種「副作用描述包在正面語氣裡」的案例）。")
        print("如果觀察到這個模式，這是支持你 taxonomy 有實際分析價值的額外證據。")

    disagree = df[~df["models_agree"]].copy()
    disagree.to_csv(DISAGREEMENT_PATH, index=False)
    print(f"\n不一致案例數: {len(disagree)}，已存檔: {DISAGREEMENT_PATH}")
    print("建議從這份檔案裡挑幾筆放進 Qualitative Analysis 或 Sentiment Analysis")
    print("的討論段落，具體展示「情感分析對 slang/反諷語氣不可靠」這個限制，")
    print("而不是只用一句話輕描淡寫地提過去。")

    print(f"\n已存檔完整比對結果: {OUTPUT_PATH}")
    print("\n⚠️ 方法學註記：這個比對本身不能告訴我們哪個模型「更準確」，")
    print("因為沒有人工標註的 ground truth 可以對照。這裡的論點是：")
    print("兩個獨立訓練的模型在同一批資料上的不一致率，反映的是 slang/反諷")
    print("語氣本身對情感分析方法造成的普遍困難，不是 VADER 特有的缺陷。")


if __name__ == "__main__":
    main()