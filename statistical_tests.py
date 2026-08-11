"""
論文用的統計檢定。針對四個主張分別跑對應的檢定，並且每個都同時報告
p 值跟效果量——樣本數大（尤其原始留言等級有 11 萬多筆）時，p 值幾乎
必然顯著，效果量才能真正告訴你「這個差異在實務上有沒有意義」。

用法：
    pip install scipy pandas --break-system-packages   # 通常已經有了
    python statistical_tests.py
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from importlib import util as _importlib_util


def _load_engagement_by_domain_module():
    spec = _importlib_util.spec_from_file_location("engagement_by_domain", "engagement_by_domain.py")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cramers_v(chi2, n, r, c):
    """Cramér's V：卡方檢定的標準效果量，範圍 0-1，不受樣本數大小影響，
    比單看卡方值或 p 值更能反映關聯強度。"""
    return np.sqrt((chi2 / n) / (min(r - 1, c - 1)))


def cohens_h(p1, p2):
    """Cohen's h：兩個比例之間差異的效果量，適合創作者 vs 留言者
    填寫率這種「兩個獨立比例比較」的情境。"""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def epsilon_squared(h_stat, n):
    """Kruskal-Wallis 的效果量（epsilon-squared），可以理解成
    「組別可以解釋多少排名變異」，類似 ANOVA 的 eta-squared 但適用於
    不假設常態分布的資料。"""
    return h_stat / (n - 1)


def dunn_test_holm(groups: dict):
    """
    Dunn's test 事後成對比較，搭配 Holm-Bonferroni 校正。
    手動實作，不依賴 scikit-posthocs（環境裡可能沒裝）。

    groups: {group_name: np.array of values}
    回傳: DataFrame，欄位為 (group_a, group_b, z, p_raw, p_holm)
    """
    all_values = np.concatenate(list(groups.values()))
    ranks = stats.rankdata(all_values)

    idx = 0
    rank_groups = {}
    for name, vals in groups.items():
        rank_groups[name] = ranks[idx: idx + len(vals)]
        idx += len(vals)

    N = len(all_values)
    _, counts = np.unique(all_values, return_counts=True)
    tie_correction = 1 - (np.sum(counts**3 - counts) / (N**3 - N))

    names = list(groups.keys())
    rows = []
    for a, b in combinations(names, 2):
        n_a, n_b = len(groups[a]), len(groups[b])
        mean_rank_a = rank_groups[a].mean()
        mean_rank_b = rank_groups[b].mean()

        se = np.sqrt(
            tie_correction * (N * (N + 1) / 12.0) * (1.0 / n_a + 1.0 / n_b)
        )
        z = (mean_rank_a - mean_rank_b) / se
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({"group_a": a, "group_b": b, "z": z, "p_raw": p_raw})

    df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(df)
    df["p_holm"] = [min(1.0, p * (m - i)) for i, p in enumerate(df["p_raw"])]
    df["p_holm"] = df["p_holm"].cummax()
    return df.sort_values(["group_a", "group_b"]).reset_index(drop=True)


def test_engagement_by_domain():
    print("=" * 70)
    print("檢定一：Domain 之間的觀看數差異（Kruskal-Wallis H 檢定）")
    print("=" * 70)

    mod = _load_engagement_by_domain_module()
    videos = mod.load_unique_videos()
    stats_df = pd.read_json("data_raw/video_stats_lookup.jsonl", lines=True)
    stats_df = stats_df.drop_duplicates(subset="video_id")

    rows = []
    for v in videos:
        domain, _ = mod.resolve_domain(v)
        rows.append({"video_id": v.get("video_id"), "health_domain": domain})
    domain_df = pd.DataFrame(rows)

    merged = domain_df.merge(stats_df, on="video_id", how="inner")
    merged = merged[merged["health_domain"] != "unclassified"]
    merged = merged[merged["view_count"].notna()]

    groups = {
        domain: sub["view_count"].values
        for domain, sub in merged.groupby("health_domain")
    }

    h_stat, p_value = stats.kruskal(*groups.values())
    n_total = sum(len(v) for v in groups.values())
    eps_sq = epsilon_squared(h_stat, n_total)

    print(f"樣本數: {n_total}，組數: {len(groups)}")
    print(f"H 統計量 = {h_stat:.2f}, p = {p_value:.2e}")
    print(f"效果量 (epsilon-squared) = {eps_sq:.4f}")
    print("  （epsilon-squared 大致參考：0.01=小, 0.06=中, 0.14=大，"
          "這是 Cohen 對類似效果量的經驗法則，非嚴格公式）")

    if p_value < 0.05:
        print("\n事後成對比較（Dunn's test + Holm 校正）：")
        posthoc = dunn_test_holm(groups)
        print(posthoc.to_string(index=False))
        print("\n只有 p_holm < 0.05 的配對，才代表校正多重比較後仍然顯著的差異。")

    return {"h_stat": h_stat, "p_value": p_value, "epsilon_squared": eps_sq}


def test_temporal_trend():
    print("\n" + "=" * 70)
    print("檢定二：健康 slang 使用率隨年份的趨勢（線性迴歸 + Mann-Kendall）")
    print("=" * 70)

    path = "data_raw/temporal_trend_health_slang_rate.csv"
    if not Path(path).exists():
        print(f"找不到 {path}，請先跑過 analyze_temporal_trend.py")
        return None

    df = pd.read_csv(path).sort_values("year")

    def run_trend(sub_df, label):
        if len(sub_df) < 3:
            print(f"{label}: 樣本點太少（<3 年），跳過檢定")
            return
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            sub_df["year"], sub_df["health_slang_rate_pct"]
        )
        tau, mk_p = stats.kendalltau(sub_df["year"], sub_df["health_slang_rate_pct"])

        print(f"\n{label}（n={len(sub_df)} 年）:")
        print(f"  線性迴歸: slope = {slope:.4f}/年, R² = {r_value**2:.3f}, p = {p_value:.4f}")
        print(f"  Mann-Kendall: tau = {tau:.3f}, p = {mk_p:.4f}")

    run_trend(df, "完整期間 (2017-2026)")
    run_trend(df[df["year"] >= 2021], "排除早期稀疏樣本後 (2021-2026)")

    print("\n⚠️ 完整期間的檢定會被 2017-2020 極少量樣本干擾（你自己的圖裡"
          "已經標示過這段是探索性的），論文裡建議主要引用 2021-2026 子集的結果，"
          "完整期間結果放在附錄或腳注說明敏感度分析。")


def test_geo_disclosure():
    print("\n" + "=" * 70)
    print("檢定三：創作者 vs 留言者的國家填寫率差異（兩比例卡方檢定）")
    print("=" * 70)

    path = "data_raw/channel_geo_lookup.jsonl"
    video_channel_path = "data_raw/video_channel_lookup.jsonl"
    if not (Path(path).exists() and Path(video_channel_path).exists()):
        print(f"找不到 {path} 或 {video_channel_path}，請先跑過 enrich_geo.py")
        return None

    country_by_channel = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            country_by_channel[item["channel_id"]] = item.get("country")

    creator_channel_ids = set()
    with open(video_channel_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = item.get("channel_id")
            if cid:
                creator_channel_ids.add(cid)

    commenter_channel_ids = set()
    for p in sorted(Path("data_raw").glob("*_youtube_comments.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cid = item.get("author_channel_id")
                if cid:
                    commenter_channel_ids.add(cid)

    creator_known = sum(1 for c in creator_channel_ids if country_by_channel.get(c))
    creator_total = len(creator_channel_ids)
    commenter_known = sum(1 for c in commenter_channel_ids if country_by_channel.get(c))
    commenter_total = len(commenter_channel_ids)

    table = np.array([
        [creator_known, creator_total - creator_known],
        [commenter_known, commenter_total - commenter_known],
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    p1 = creator_known / creator_total
    p2 = commenter_known / commenter_total
    h = cohens_h(p1, p2)

    print(f"創作者填寫率: {p1*100:.1f}% (n={creator_total})")
    print(f"留言者填寫率: {p2*100:.1f}% (n={commenter_total})")
    print(f"卡方 = {chi2:.2f}, df = {dof}, p = {p_value:.2e}")
    print(f"Cohen's h = {h:.3f}")
    print("  （Cohen's h 經驗法則：0.2=小, 0.5=中, 0.8=大）")


def test_slang_domain_association():
    print("\n" + "=" * 70)
    print("檢定四：Slang 詞 × Domain 的關聯性（卡方獨立性檢定 + Cramér's V）")
    print("=" * 70)

    path = "data_raw/youtube_annotation_pool.csv"
    if not Path(path).exists():
        print(f"找不到 {path}，請先跑過 make_annotation_pool.py")
        return None

    df = pd.read_csv(path)
    contingency = pd.crosstab(df["primary_slang"], df["health_domain"])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    v = cramers_v(chi2, n, contingency.shape[0], contingency.shape[1])

    low_expected_pct = (expected < 5).sum() / expected.size * 100

    print(f"卡方 = {chi2:.2f}, df = {dof}, p = {p_value:.2e}")
    print(f"Cramér's V = {v:.3f}")
    print("  （Cramér's V 經驗法則，自由度調整後粗略參考：0.1=小, 0.3=中, 0.5=大）")
    print(f"\n⚠️ expected cell < 5 的比例: {low_expected_pct:.1f}%")
    if low_expected_pct > 20:
        print("這個比例偏高（超過 20%），卡方檢定的可靠度會打折扣，")
        print("論文裡這個結果應該標註為 exploratory，不是 confirmatory，")
        print("這跟你之前在其他分析裡遇到的 sparse cell 問題是同一類限制。")


if __name__ == "__main__":
    results = {}
    results["engagement"] = test_engagement_by_domain()
    test_temporal_trend()
    test_geo_disclosure()
    test_slang_domain_association()

    print("\n" + "=" * 70)
    print("全部檢定跑完。把上面的數字對應貼進論文對應章節：")
    print("  - Kruskal-Wallis 結果 → Results/Engagement 小節")
    print("  - 趨勢檢定結果 → Results/Temporal Trend 小節")
    print("  - 兩比例卡方結果 → Results/Demographics 小節")
    print("  - Slang×Domain 卡方結果 → Results 或 Discussion 的 expression-type 小節")
    print("=" * 70)