"""
重新跑「Slang 詞 x Domain」卡方獨立性檢定，改成讀去重後的 547 筆
annotation pool（原本 statistical_tests.py 的 test_slang_domain_association()
讀的是還沒去重的 youtube_annotation_pool.csv，574 筆）。

用法：
    python statistical_tests_chi_square_fixed.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# 換成你實際存放去重後檔案的路徑
ANNOTATION_POOL_PATH = "data_raw/youtube_annotation_pool_cleaned.csv"


def cramers_v(chi2, n, r, c):
    return np.sqrt((chi2 / n) / (min(r - 1, c - 1)))


def main():
    if not Path(ANNOTATION_POOL_PATH).exists():
        raise FileNotFoundError(
            f"找不到 {ANNOTATION_POOL_PATH}，請確認去重後的 547 筆檔案路徑正確。"
        )

    df = pd.read_csv(ANNOTATION_POOL_PATH)
    print(f"讀取筆數: {len(df)}（應該是 547，不是 574）")

    contingency = pd.crosstab(df["primary_slang"], df["health_domain"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    v = cramers_v(chi2, n, contingency.shape[0], contingency.shape[1])

    low_expected_pct = (expected < 5).sum() / expected.size * 100

    print(f"\n卡方 = {chi2:.2f}, df = {dof}, p = {p_value:.2e}")
    print(f"Cramér's V = {v:.3f}")
    print(f"expected cell < 5 的比例: {low_expected_pct:.1f}%")


if __name__ == "__main__":
    main()
