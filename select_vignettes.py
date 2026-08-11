"""
挑選質性分析用的 vignette 候選案例。

CHI 論文很吃「具體例子 + 分析」的呈現方式，這支腳本從標註池裡
自動挑出最適合的候選：
    1. 優先挑 expression_type == "ambiguous" 的（knocked me out,
       fighting for my life, messed me up, wrecked me 這類需要語境
       才能判斷的詞），每個詞挑留言最長、priority_score 最高的幾筆
    2. 也從 informal_symptom_language 挑幾筆（brain fog 等），
       跟 ambiguous 對照，展示「明確症狀語」vs「語意浮動」的差異
    3. 從 internet_slang 挑 1-2 筆當作對照組（純粹社交語氣，非模糊）

不會自動幫你寫分析，只是把候選案例整理出來，方便你（或跟我一起）
挑選、閱讀完整脈絡後，寫成質性 vignette。

用法：
    python select_vignettes.py
"""

from pathlib import Path

import pandas as pd

ANNOTATION_POOL_PATH = "data_raw/youtube_annotation_pool.csv"

# 每個 expression_type 各挑幾筆候選
N_PER_AMBIGUOUS_TERM = 3
N_INFORMAL_SYMPTOM = 5
N_INTERNET_SLANG_CONTRAST = 2


def main():
    if not Path(ANNOTATION_POOL_PATH).exists():
        raise FileNotFoundError(f"找不到 {ANNOTATION_POOL_PATH}，請先跑過 make_annotation_pool.py")

    df = pd.read_csv(ANNOTATION_POOL_PATH)
    df = df[df["text"].notna()].copy()
    df["text_len"] = df["text"].astype(str).str.len()

    candidates = []

    # ---------- 1. ambiguous：每個詞各挑幾筆 ----------
    ambiguous_df = df[df["expression_type"] == "ambiguous"]
    print(f"ambiguous 類別總筆數: {len(ambiguous_df)}")
    print(f"ambiguous 詞彙分布:\n{ambiguous_df['primary_slang'].value_counts()}\n")

    for term, group in ambiguous_df.groupby("primary_slang"):
        top = group.sort_values(["priority_score", "text_len"], ascending=False).head(N_PER_AMBIGUOUS_TERM)
        for _, row in top.iterrows():
            candidates.append({**row.to_dict(), "vignette_category": "ambiguous"})

    # ---------- 2. informal_symptom_language：對照組 ----------
    informal_df = df[df["expression_type"] == "informal_symptom_language"]
    top_informal = informal_df.sort_values(["priority_score", "text_len"], ascending=False).head(N_INFORMAL_SYMPTOM)
    for _, row in top_informal.iterrows():
        candidates.append({**row.to_dict(), "vignette_category": "informal_symptom_language"})

    # ---------- 3. internet_slang：對照組 ----------
    internet_df = df[df["expression_type"] == "internet_slang"]
    top_internet = internet_df.sort_values(["priority_score", "text_len"], ascending=False).head(N_INTERNET_SLANG_CONTRAST)
    for _, row in top_internet.iterrows():
        candidates.append({**row.to_dict(), "vignette_category": "internet_slang_contrast"})

    result_df = pd.DataFrame(candidates)
    result_df = result_df.drop_duplicates(subset=["annotation_id"])

    output_cols = [
        "vignette_category", "primary_slang", "health_domain", "expression_type",
        "priority_score", "text_len", "video_title", "text",
    ]
    output_cols = [c for c in output_cols if c in result_df.columns]
    result_df[output_cols].to_csv("data_raw/vignette_candidates.csv", index=False)

    print(f"共挑出 {len(result_df)} 筆候選 vignette，已存檔: data_raw/vignette_candidates.csv")
    print("\n" + "=" * 80)
    print("完整候選案例內容（複製貼給 Claude 一起討論質性分析）：")
    print("=" * 80)

    for category, group in result_df.groupby("vignette_category"):
        print(f"\n\n【{category}】")
        for _, row in group.iterrows():
            print(f"\n--- {row.get('primary_slang', '')} | {row.get('health_domain', '')} "
                  f"| priority_score={row.get('priority_score', '')} ---")
            print(f"影片標題: {row.get('video_title', '')}")
            print(f"留言: {row.get('text', '')}")

    print("\n\n⚠️ 提醒：這些是真實使用者的留言。之後在論文裡引用時，")
    print("建議做輕度改寫/去識別化（不要附上原始帳號或影片連結），")
    print("這是保護留言者隱私的常見學術倫理做法，值得在方法學章節提一句。")


if __name__ == "__main__":
    main()
