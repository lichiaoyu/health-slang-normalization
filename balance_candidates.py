import pandas as pd
import ast
import re

INPUT = "data_raw/youtube_candidate_slang_cleaned.csv"
OUTPUT = "data_raw/youtube_candidate_slang_balanced.csv"

# 每個 slang 最多保留幾筆，避免 glow up 淹掉 dataset
MAX_PER_SLANG = {
    "glow up": 80,
    "brain fog": 80,
    "girl dinner": 80,
    "hits different": 80,
    "knocked me out": 80,
    "fighting for my life": 80,
    "cooked": 80,
    "dying": 80,
    "dead": 40,
    "locked in": 60,
    "bussin": 60,
    "fr": 40,
    "lowkey": 40,
    "no cap": 40,
    "snatched": 40,
    "slay": 40,
    "delulu": 40,
}

# 每個 health domain 最多保留幾筆，避免只剩 weight-loss/glow-up
MAX_PER_DOMAIN = {
    "medication_side_effects": 100,
    "supplements": 100,
    "diet_weight_loss": 100,
    "fitness_gym": 100,
    "gut_health": 100,
    "sleep_fatigue": 100,
    "mental_cognitive": 100,
    "body_image_transformation": 100,
    "other_health": 50,
}

def parse_hits(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []

def safe_text(row):
    parts = []
    for col in ["query", "title", "video_title", "description", "text"]:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]).lower())
    return " ".join(parts)

def classify_domain(text):
    if any(k in text for k in [
        "ozempic", "glp1", "mounjaro", "semaglutide", "wegovy",
        "zepbound", "side effect", "nausea", "medication"
    ]):
        return "medication_side_effects"

    if any(k in text for k in [
        "protein powder", "creatine", "pre workout", "fat burner",
        "greens powder", "ashwagandha", "magnesium", "supplement"
    ]):
        return "supplements"

    if any(k in text for k in [
        "calorie deficit", "what i eat", "weight loss", "keto",
        "fasting", "meal prep", "girl dinner", "diet"
    ]):
        return "diet_weight_loss"

    if any(k in text for k in [
        "gymtok", "gym", "workout", "body recomp", "bulking",
        "cutting", "leg day", "fitness"
    ]):
        return "fitness_gym"

    if any(k in text for k in [
        "gut health", "bloating", "probiotic", "ibs",
        "constipation", "digestion", "detox"
    ]):
        return "gut_health"

    if any(k in text for k in [
        "sleep", "melatonin", "insomnia", "fatigue",
        "energy crash", "knocked me out"
    ]):
        return "sleep_fatigue"

    if any(k in text for k in [
        "brain fog", "anxiety", "burnout", "cortisol",
        "hormone", "pcos", "mood"
    ]):
        return "mental_cognitive"

    if any(k in text for k in [
        "glow up", "snatched", "body transformation",
        "before and after", "summer body"
    ]):
        return "body_image_transformation"

    return "other_health"

def priority_score(row):
    """
    Higher score = more useful for annotation.
    Prioritize:
    - comments over video titles
    - ambiguous health expressions
    - longer context
    """
    text = safe_text(row)
    source = str(row.get("source_file", ""))

    score = 0

    if "comments" in source:
        score += 3

    useful_terms = [
        "brain fog", "fighting for my life", "knocked me out",
        "cooked", "dying", "hits different", "girl dinner",
        "messed me up", "wrecked me", "took me out",
        # 原本這份清單漏了這三個稀有詞，導致它們在 domain 平衡階段
        # 因為分數天生較低，被其他詞系統性擠掉，即使清理階段其實
        # 留有足夠候選也一樣。
        "slay", "no cap", "had me in shambles",
    ]

    for term in useful_terms:
        if term in text:
            score += 2

    # longer comments often give better context
    if len(text) > 150:
        score += 1
    if len(text) > 400:
        score += 1

    # penalize likely noisy transformation spam
    if "glow up" in text and "weight loss" not in text and "fitness" not in text:
        score -= 2

    return score

df = pd.read_csv(INPUT)

hit_col = "clean_slang_hits" if "clean_slang_hits" in df.columns else "slang_hits"
df["parsed_hits"] = df[hit_col].apply(parse_hits)
df["primary_slang"] = df["parsed_hits"].apply(lambda xs: xs[0] if xs else "unknown")
df["combined_text"] = df.apply(safe_text, axis=1)

# =============================================================================
# V1（collect_youtube.py / batch_health_broad_01）的 query → domain 對照表。
#
# V1 收集當時沒有把 domain 存進資料欄位，但 query 本身在設計時就已經
# 按照 collect_youtube.py 裡的區塊註解分好組了（例如
# "# Medication / GLP-1 / side effects"）。這裡把當初的分組意圖
# 重建成對照表，回填給 batch_01 的資料，可信度等同於 batch_02 的
# collection_domain ground truth，且完全不需要重新呼叫 API。
# =============================================================================
V1_QUERY_TO_DOMAIN = {
    # Medication / GLP-1 / side effects
    '"ozempic side effects" #shorts': "medication_side_effects",
    '"glp1 journey" #shorts': "medication_side_effects",
    '"mounjaro experience" #shorts': "medication_side_effects",
    '"semaglutide nausea" #shorts': "medication_side_effects",
    '"weight loss medication" #shorts': "medication_side_effects",
    '"medication side effects" #shorts': "medication_side_effects",

    # Supplements
    '"supplement review" #shorts': "supplements",
    '"protein powder review" #shorts': "supplements",
    '"creatine transformation" #shorts': "supplements",
    '"pre workout review" #shorts': "supplements",
    '"pre workout side effects" #shorts': "supplements",
    '"fat burner review" #shorts': "supplements",
    '"greens powder review" #shorts': "supplements",
    '"ashwagandha review" #shorts': "supplements",
    '"magnesium sleep" #shorts': "supplements",
    '"melatonin review" #shorts': "supplements",

    # Diet / weight loss / food behavior
    '"calorie deficit meals" #shorts': "diet_weight_loss",
    '"what I eat in a day weight loss" #shorts': "diet_weight_loss",
    '"weight loss journey" #shorts': "diet_weight_loss",
    '"intermittent fasting results" #shorts': "diet_weight_loss",
    '"keto diet results" #shorts': "diet_weight_loss",
    '"high protein meals" #shorts': "diet_weight_loss",
    '"girl dinner" #shorts': "diet_weight_loss",
    '"meal prep weight loss" #shorts': "diet_weight_loss",

    # Fitness / GymTok
    '"gymtok" #shorts': "fitness_gym",
    '"gym transformation" #shorts': "fitness_gym",
    '"body recomp" #shorts': "fitness_gym",
    '"bulking diet" #shorts': "fitness_gym",
    '"cutting diet" #shorts': "fitness_gym",
    '"leg day" #shorts': "fitness_gym",
    '"gym motivation" #shorts': "fitness_gym",
    '"fitness transformation" #shorts': "fitness_gym",

    # Gut health / digestion
    '"gut health tips" #shorts': "gut_health",
    '"bloating remedies" #shorts': "gut_health",
    '"probiotics review" #shorts': "gut_health",
    '"IBS symptoms" #shorts': "gut_health",
    '"constipation relief" #shorts': "gut_health",
    '"digestion problems" #shorts': "gut_health",
    '"detox drink" #shorts': "gut_health",

    # Sleep / fatigue
    '"sleep tips" #shorts': "sleep_fatigue",
    '"insomnia tips" #shorts': "sleep_fatigue",
    '"melatonin gummies" #shorts': "sleep_fatigue",
    '"sleep supplement" #shorts': "sleep_fatigue",
    '"chronic fatigue" #shorts': "sleep_fatigue",
    '"energy crash" #shorts': "sleep_fatigue",

    # Mental / cognitive state
    '"brain fog" #shorts': "mental_cognitive",
    '"anxiety symptoms" #shorts': "mental_cognitive",
    '"burnout symptoms" #shorts': "mental_cognitive",
    '"cortisol levels" #shorts': "mental_cognitive",
    '"hormone imbalance" #shorts': "mental_cognitive",
    '"PCOS symptoms" #shorts': "mental_cognitive",

    # Body image / transformation
    '"weight loss glow up" #shorts': "body_image_transformation",
    '"body transformation" #shorts': "body_image_transformation",
    '"before and after weight loss" #shorts': "body_image_transformation",
    '"snatched waist" #shorts': "body_image_transformation",
    '"summer body" #shorts': "body_image_transformation",

    # Slang-targeted queries：依照內容本身的主題判斷 domain
    # （原本的區塊註解是跨 domain 混合的，這裡拆到各自對應的 domain）
    '"side effects" "fighting for my life" #shorts': "medication_side_effects",
    '"side effects" "messed me up" #shorts': "medication_side_effects",
    '"side effects" "wrecked me" #shorts': "medication_side_effects",
    '"side effects" "took me out" #shorts': "medication_side_effects",
    '"supplement" "had me dying" #shorts': "supplements",
    '"pre workout" "fighting for my life" #shorts': "supplements",
    '"pre workout" "messed me up" #shorts': "supplements",

    '"melatonin" "knocked me out" #shorts': "sleep_fatigue",
    '"sleep supplement" "knocked me out" #shorts': "sleep_fatigue",
    '"magnesium" "knocked me out" #shorts': "sleep_fatigue",
    '"ashwagandha" "messed me up" #shorts': "sleep_fatigue",

    '"girl dinner" "calorie deficit" #shorts': "diet_weight_loss",
    '"weight loss" "glow up" #shorts': "body_image_transformation",
    '"fitness" "glow up" #shorts': "body_image_transformation",
    '"gymtok" "locked in" #shorts': "fitness_gym",
    '"gymtok" "cooked" #shorts': "fitness_gym",
    '"weight loss" "snatched" #shorts': "body_image_transformation",

    '"protein powder" "bussin" #shorts': "supplements",
    '"gut health" "hits different" #shorts': "gut_health",
    '"healthy recipe" "bussin" #shorts': "diet_weight_loss",
    '"meal prep" "hits different" #shorts': "diet_weight_loss",
}


def resolve_domain(row):
    """
    三層優先順序，數字越小信心越高：
    1. collection_domain（batch_02 收集當下記錄的意圖，ground truth）
    2. V1_QUERY_TO_DOMAIN（batch_01 的 query 事後對照，等同 ground truth，
       只是回填的時間點不同）
    3. classify_domain()（關鍵字猜測，僅在前兩者都沒有對應資料時才使用，
       例如 query 是空值，或出現不在對照表裡的 query）

    這樣做的原因：classify_domain() 是依序判斷、第一個命中就回傳的
    if/elif 邏輯，而 combined_text 混雜了留言全文，容易被留言裡
    無關的詞（例如 brain fog 的留言底下有人回「應該是飲食問題」）
    誤判成排序較前面的類別（如 diet_weight_loss），
    導致 mental_cognitive / body_image_transformation 這類排序較後面
    的類別被系統性低估。
    """
    if "collection_domain" in row and pd.notna(row["collection_domain"]) and str(row["collection_domain"]).strip():
        return row["collection_domain"]

    query = row.get("query")
    if pd.notna(query) and query in V1_QUERY_TO_DOMAIN:
        return V1_QUERY_TO_DOMAIN[query]

    return classify_domain(row["combined_text"])


def resolve_domain_source(row):
    if "collection_domain" in row and pd.notna(row["collection_domain"]) and str(row["collection_domain"]).strip():
        return "collection_ground_truth"

    query = row.get("query")
    if pd.notna(query) and query in V1_QUERY_TO_DOMAIN:
        return "v1_query_inferred"

    return "keyword_fallback"

df["health_domain"] = df.apply(resolve_domain, axis=1)
df["domain_source"] = df.apply(resolve_domain_source, axis=1)
df["priority_score"] = df.apply(priority_score, axis=1)

print("\nDomain label 來源分布:")
print(df["domain_source"].value_counts())

# 先按 slang balance
slang_balanced_parts = []
for slang, group in df.groupby("primary_slang"):
    max_n = MAX_PER_SLANG.get(slang, 50)
    group = group.sort_values("priority_score", ascending=False)
    slang_balanced_parts.append(group.head(max_n))

slang_balanced = pd.concat(slang_balanced_parts, ignore_index=True)

# 再按 domain balance
#
# 舊邏輯的問題：domain 名額直接用 priority_score 排序取前 N 名，
# 完全不管這些名額被哪些 slang 詞佔走。結果是只要調整某個詞的
# priority_score 加分，就會在 domain 名額有限的前提下，把其他詞
# 排擠出局（例如修好 slay/no cap 後，girl dinner 從 35 筆崩到 5 筆）。
#
# 新邏輯：「先保底、後競爭」
#   1. 每個 domain 裡，對「該 domain 出現過的每一個 slang 詞」，
#      先保留最多 MIN_PER_SLANG_PER_DOMAIN 筆（依 priority_score 排序取最好的幾筆）
#      —— 這一步保證任何詞都不會被完全擠到個位數
#   2. 保底名額用完後，domain 剩餘的名額，才用 priority_score
#      從「保底後剩下的候選」裡搶——這一步維持原本鼓勵高品質資料的精神
#
# 這樣之後不管再怎麼調整某個詞的 priority_score 加分，最多只影響
# 「競爭名額」怎麼分配，不會再讓某個詞被徹底清空。
MIN_PER_SLANG_PER_DOMAIN = 5

domain_balanced_parts = []
for domain, group in slang_balanced.groupby("health_domain"):
    max_n = MAX_PER_DOMAIN.get(domain, 50)

    floor_parts = []
    remainder_parts = []
    for slang, sub in group.groupby("primary_slang"):
        sub_sorted = sub.sort_values("priority_score", ascending=False)
        floor_parts.append(sub_sorted.head(MIN_PER_SLANG_PER_DOMAIN))
        remainder_parts.append(sub_sorted.iloc[MIN_PER_SLANG_PER_DOMAIN:])

    floor_df = pd.concat(floor_parts, ignore_index=True) if floor_parts else group.iloc[0:0]
    remainder_df = pd.concat(remainder_parts, ignore_index=True) if remainder_parts else group.iloc[0:0]

    if len(floor_df) > max_n:
        # 極端狀況：光是保底名額加起來就超過 domain 上限，
        # 此時保底名額本身也要依 priority_score 砍到上限，
        # 但至少每個詞被砍掉的機會是均等的，不會系統性偏袒某一詞。
        floor_df = floor_df.sort_values("priority_score", ascending=False).head(max_n)
        remaining_slots = 0
    else:
        remaining_slots = max_n - len(floor_df)

    if remaining_slots > 0 and len(remainder_df) > 0:
        fill_df = remainder_df.sort_values("priority_score", ascending=False).head(remaining_slots)
        domain_final = pd.concat([floor_df, fill_df], ignore_index=True)
    else:
        domain_final = floor_df

    domain_balanced_parts.append(domain_final)

balanced = pd.concat(domain_balanced_parts, ignore_index=True)
balanced = balanced.sort_values(
    ["health_domain", "primary_slang", "priority_score"],
    ascending=[True, True, False]
)

balanced.to_csv(OUTPUT, index=False)

print("Input cleaned rows:", len(df))
print("Balanced rows:", len(balanced))

print("\nBalanced slang distribution:")
print(balanced["primary_slang"].value_counts())

print("\nBalanced health domain distribution:")
print(balanced["health_domain"].value_counts())

print(f"\nSaved: {OUTPUT}")