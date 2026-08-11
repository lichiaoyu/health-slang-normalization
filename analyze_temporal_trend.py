"""
時間趨勢分析。完全離線，不呼叫任何 API。

呼應你朋友文件裡的 Task 3（slang 隨時間成長 4 倍），但這裡算的是
「健康相關 slang」的比率，跟他的「泛用 slang」比率是不同的分母/分子組合：

    health slang rate（某年）= 該年通過清理階段的健康相關 slang 留言數
                                ÷ 該年全部原始留言數

用法：
    python analyze_temporal_trend.py
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd


def load_all_raw_comments_by_year():
    """讀全部原始留言（不管有沒有命中 slang），依 published_at 算每年總留言數，
    當作時間趨勢分析的分母。"""
    year_counter = Counter()
    for path in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                pub = item.get("published_at")
                if pub:
                    year = pd.to_datetime(pub, errors="coerce", utc=True).year
                    if pd.notna(year):
                        year_counter[int(year)] += 1
    return year_counter


def main():
    cleaned_path = "data_raw/youtube_candidate_slang_cleaned.csv"
    if not Path(cleaned_path).exists():
        raise FileNotFoundError(f"找不到 {cleaned_path}，請先跑過 clean_candidates.py")

    cleaned = pd.read_csv(cleaned_path)
    cleaned["published_at_parsed"] = pd.to_datetime(cleaned["published_at"], errors="coerce", utc=True)
    cleaned["year"] = cleaned["published_at_parsed"].dt.year

    total_by_year = load_all_raw_comments_by_year()
    cleaned_by_year = cleaned["year"].value_counts().sort_index()

    print("原始留言總數（分母），依年份:")
    for year in sorted(total_by_year):
        print(f"  {year}: {total_by_year[year]}")

    print("\n健康相關 slang 留言數（分子），依年份:")
    print(cleaned_by_year)

    rate_rows = []
    for year, total in sorted(total_by_year.items()):
        n_slang = int(cleaned_by_year.get(year, 0))
        rate = n_slang / total * 100 if total > 0 else 0
        rate_rows.append({"year": year, "total_comments": total, "health_slang_comments": n_slang, "health_slang_rate_pct": rate})

    rate_df = pd.DataFrame(rate_rows).sort_values("year")
    rate_df.to_csv("data_raw/temporal_trend_health_slang_rate.csv", index=False)

    print("\n健康相關 slang 使用率（%），依年份:")
    print(rate_df.to_string(index=False))

    # 額外：expression_type 隨時間的變化（如果 make_annotation_pool.py 已經跑過）
    annotation_path = "data_raw/youtube_annotation_pool.csv"
    if Path(annotation_path).exists():
        ann = pd.read_csv(annotation_path)
        if "published_at" in ann.columns and "expression_type" in ann.columns:
            ann["year"] = pd.to_datetime(ann["published_at"], errors="coerce", utc=True).dt.year
            print("\nexpression_type 隨年份分布（僅限最終 574 筆標註池，樣本較小，僅供參考）:")
            print(ann.groupby(["year", "expression_type"]).size().unstack(fill_value=0))

    print(f"\n已存檔: data_raw/temporal_trend_health_slang_rate.csv")
    print("\n⚠️ 提醒：你的 query 設計沒有像你朋友那樣刻意用「每年分開查詢」的方式，")
    print("避免近期資料淹沒早期資料，所以這份時間趨勢的早期年份筆數可能偏少，")
    print("解讀早期年份的比率時要更保守，這點值得在方法學裡註明。")


if __name__ == "__main__":
    main()
