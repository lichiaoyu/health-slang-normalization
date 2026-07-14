from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


INPUT_FILE = Path("data_raw/youtube_candidate_slang_enriched.csv")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)


def cramers_v(table: pd.DataFrame) -> float:
    """Effect size for a chi-square association."""
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    rows, cols = table.shape
    denominator = min(rows - 1, cols - 1)

    if n == 0 or denominator <= 0:
        return np.nan

    return np.sqrt((chi2 / n) / denominator)


def run_chi_square(row_col: str, col_col: str, filename: str) -> None:
    table = pd.crosstab(df[row_col], df[col_col])

    # Remove empty rows/columns
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]

    chi2, p_value, dof, expected = chi2_contingency(table)
    effect = cramers_v(table)

    print("\n" + "=" * 80)
    print(f"{row_col} × {col_col}")
    print("=" * 80)
    print(f"Chi-square: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.6g}")
    print(f"Cramer's V: {effect:.3f}")

    low_expected_rate = (expected < 5).mean()
    print(f"Expected cells below 5: {low_expected_rate:.1%}")

    table.to_csv(OUTPUT_DIR / filename)


run_chi_square(
    "primary_slang",
    "sentiment_label",
    "slang_by_sentiment_test.csv",
)

run_chi_square(
    "primary_slang",
    "health_domain_enriched",
    "slang_by_domain_test.csv",
)

run_chi_square(
    "primary_slang",
    "sentiment_target",
    "slang_by_target_test.csv",
)


# Compare sentiment scores across broader health domains
domain_groups = [
    group["sentiment_compound"].dropna().to_numpy()
    for _, group in df.groupby("health_domain_enriched")
    if len(group) >= 5
]

if len(domain_groups) >= 2:
    statistic, p_value = kruskal(*domain_groups)

    print("\n" + "=" * 80)
    print("Sentiment score differences across health domains")
    print("=" * 80)
    print(f"Kruskal-Wallis statistic: {statistic:.3f}")
    print(f"p-value: {p_value:.6g}")