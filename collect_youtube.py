import os
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm


load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
BATCH_NAME = "batch_health_broad_02"  # 新版本，不覆蓋舊資料，方便之後 V1 vs V2 比較

if not YOUTUBE_API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY. Please check your .env file.")

VIDEOS_PATH = f"data_raw/{BATCH_NAME}_youtube_videos.jsonl"
COMMENTS_PATH = f"data_raw/{BATCH_NAME}_youtube_comments.jsonl"
PROGRESS_PATH = f"data_raw/{BATCH_NAME}_progress.json"


class QuotaExceededError(Exception):
    """quota 用盡時拋出，讓 main() 可以優雅停下並保留已收集的資料。"""
    pass


# =============================================================================
# TRACK A — 純主題查詢（Organic Discovery）
#
# 設計原則：
#   1. Query 字串裡「絕對不能」出現任何 slang 詞（glow up, girl dinner,
#      locked in, hits different ...）。目的是讓 slang 用法自然出現在
#      留言/標題裡，而不是被我們用 slang 詞搜出來 —— 這是 V1 最大的偏誤來源。
#   2. 每個 health domain 分配「大致相同數量」的 query，避免像 V1 一樣
#      diet/fitness 的 query 數量遠多於 mental_cognitive/sleep，
#      導致收集階段就已經決定了 domain 分布。
# =============================================================================

TRACK_A_QUERIES = {
    "medication_side_effects": [
        '"ozempic experience" #shorts',
        '"mounjaro experience" #shorts',
        '"semaglutide journey" #shorts',
        '"glp1 update" #shorts',
        '"weight loss medication" #shorts',
        '"medication update" #shorts',
        '"zepbound journey" #shorts',
        '"wegovy update" #shorts',
    ],
    "supplements": [
        '"supplement routine" #shorts',
        '"protein powder review" #shorts',
        '"creatine routine" #shorts',
        '"pre workout review" #shorts',
        '"fat burner review" #shorts',
        '"greens powder review" #shorts',
        '"ashwagandha routine" #shorts',
        '"magnesium routine" #shorts',
    ],
    "diet_weight_loss": [
        '"calorie deficit meals" #shorts',
        '"what I eat in a day" #shorts',
        '"weight loss journey" #shorts',
        '"intermittent fasting" #shorts',
        '"keto diet results" #shorts',
        '"high protein meals" #shorts',
        '"meal prep routine" #shorts',
        '"low calorie recipes" #shorts',
    ],
    "fitness_gym": [
        '"gym routine" #shorts',
        '"gym transformation" #shorts',
        '"body recomp" #shorts',
        '"bulking routine" #shorts',
        '"cutting routine" #shorts',
        '"leg day routine" #shorts',
        '"gym motivation" #shorts',
        '"fitness transformation" #shorts',
    ],
    "gut_health": [
        '"gut health tips" #shorts',
        '"bloating remedies" #shorts',
        '"probiotics routine" #shorts',
        '"ibs symptoms" #shorts',
        '"constipation relief" #shorts',
        '"digestion tips" #shorts',
        '"detox drink" #shorts',
        '"gut health routine" #shorts',
    ],
    "sleep_fatigue": [
        '"sleep tips" #shorts',
        '"insomnia tips" #shorts',
        '"melatonin routine" #shorts',
        '"sleep supplement" #shorts',
        '"chronic fatigue" #shorts',
        '"energy crash" #shorts',
        '"sleep routine" #shorts',
        '"night routine sleep" #shorts',
    ],
    "mental_cognitive": [
        '"brain fog symptoms" #shorts',
        '"anxiety symptoms" #shorts',
        '"burnout symptoms" #shorts',
        '"cortisol levels" #shorts',
        '"hormone imbalance" #shorts',
        '"pcos symptoms" #shorts',
        '"mental health update" #shorts',
        '"stress symptoms" #shorts',
    ],
    "body_image_transformation": [
        '"body transformation" #shorts',
        '"before and after weight loss" #shorts',
        '"summer body prep" #shorts',
        '"body composition update" #shorts',
        '"fitness progress" #shorts',
        '"transformation update" #shorts',
        '"weight loss progress" #shorts',
        '"body change routine" #shorts',
    ],
}

# =============================================================================
# TRACK B — 稀有 domain 補強（Targeted Boost）
#
# 只用在 V1 資料明顯稀疏、且對研究有價值的模糊/嚴重類別：
#   medication_side_effects, sleep_fatigue, mental_cognitive, gut_health
#
# 不再對 diet_weight_loss / fitness_gym / body_image_transformation 做
# targeted boost —— V1 顯示這幾個 domain 已經過飽和（78.9%），
# 再撈只會讓不平衡更嚴重。
# =============================================================================

TRACK_B_QUERIES = {
    "medication_side_effects": [
        '"ozempic" "fighting for my life" #shorts',
        '"mounjaro" "messed me up" #shorts',
        '"semaglutide" "wrecked me" #shorts',
        '"glp1" "took me out" #shorts',
    ],
    "sleep_fatigue": [
        '"melatonin" "knocked me out" #shorts',
        '"sleep supplement" "knocked me out" #shorts',
        '"magnesium" "knocked me out" #shorts',
        '"insomnia" "couldn\'t function" #shorts',
    ],
    "mental_cognitive": [
        '"brain fog" "couldn\'t function" #shorts',
        '"burnout" "fighting for my life" #shorts',
        '"anxiety" "wrecked me" #shorts',
        '"cortisol" "messed me up" #shorts',
    ],
    "gut_health": [
        '"gut health" "wrecked my stomach" #shorts',
        '"bloating" "couldn\'t function" #shorts',
        '"probiotics" "messed me up" #shorts',
        '"ibs" "fighting for my life" #shorts',
    ],
}

# =============================================================================
# TRACK C — 稀有 slang 詞補強（Rare-Term Boost）
#
# 上一輪平衡結果顯示這幾個詞樣本數過低，不足以支撐後續統計分析：
#   fighting for my life: 6, slay: 7, had me in shambles: 1, no cap: 未達門檻
#
# 策略：把這幾個詞分別跟「不同 domain 的主題詞」組合，而不是只跟單一
# domain 綁在一起，增加命中率，同時讓這幾個稀有詞能分散到多個 domain，
# 不會讓某個 domain 因此被單一詞主導。
# =============================================================================

TRACK_C_QUERIES = {
    "supplements": [
        '"creatine" "fighting for my life" #shorts',
        '"pre workout" "had me in shambles" #shorts',
        '"protein powder" "no cap" #shorts',
    ],
    "diet_weight_loss": [
        '"keto" "fighting for my life" #shorts',
        '"diet" "had me in shambles" #shorts',
        '"weight loss" "no cap" #shorts',
        '"weight loss" "slay" #shorts',
    ],
    "fitness_gym": [
        '"leg day" "fighting for my life" #shorts',
        '"gym" "had me in shambles" #shorts',
        '"gym routine" "no cap" #shorts',
        '"gym motivation" "slay" #shorts',
    ],
    "gut_health": [
        '"detox" "fighting for my life" #shorts',
    ],
    "medication_side_effects": [
        '"side effects" "had me in shambles" #shorts',
    ],
    "body_image_transformation": [
        '"glow up" "slay" #shorts',
        '"summer body" "slay" #shorts',
        '"glow up" "no cap" #shorts',
    ],
}


def build_query_list():
    """
    回傳 (query, domain, track) 的 tuple 列表，方便後續在 metadata
    裡直接記錄「這筆資料是從哪個 domain / track 收集來的」，
    讓 domain 標籤不再只靠事後關鍵字分類，而有收集階段的依據可以交叉驗證。
    """
    queries = []
    for domain, qs in TRACK_A_QUERIES.items():
        for q in qs:
            queries.append((q, domain, "track_a_organic"))
    for domain, qs in TRACK_B_QUERIES.items():
        for q in qs:
            queries.append((q, domain, "track_b_boost"))
    for domain, qs in TRACK_C_QUERIES.items():
        for q in qs:
            queries.append((q, domain, "track_c_rare_slang_boost"))
    return queries


QUERY_LIST = build_query_list()


def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_videos(youtube, query: str, max_results: int = 50):
    videos = []
    next_page_token = None

    while len(videos) < max_results:
        try:
            response = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=min(50, max_results - len(videos)),
                order="relevance",
                safeSearch="moderate",
                relevanceLanguage="en",
                regionCode="US",
                pageToken=next_page_token
            ).execute()
        except HttpError as e:
            if "quota" in str(e).lower():
                raise QuotaExceededError(str(e))
            print(f"Search failed for query={query}: {e}")
            return videos

        for item in response.get("items", []):
            video_id = item["id"].get("videoId")
            snippet = item.get("snippet", {})

            if not video_id:
                continue

            videos.append({
                "platform": "youtube",
                "query": query,
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)

    return videos


def get_comments(youtube, video_id: str, max_comments: int = 100, relevance_ratio: float = 0.6):
    """
    改善點：V1 只用 order="relevance"，只會抓到熱門留言，
    容易漏掉長尾、比較少人按讚但可能包含稀有 slang 的留言。
    這裡混合抓取：一部分 relevance（熱門）+ 一部分 time（最新），
    增加留言的語言/用詞多樣性。
    """
    comments = []
    n_relevance = int(max_comments * relevance_ratio)
    n_time = max_comments - n_relevance

    for order, n_target in [("relevance", n_relevance), ("time", n_time)]:
        next_page_token = None
        collected = 0

        while collected < n_target:
            try:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, n_target - collected),
                    textFormat="plainText",
                    order=order,
                    pageToken=next_page_token
                ).execute()
            except HttpError as e:
                if "quota" in str(e).lower():
                    raise QuotaExceededError(str(e))
                break

            items = response.get("items", [])
            if not items:
                break

            for item in items:
                comment = item["snippet"]["topLevelComment"]["snippet"]

                comments.append({
                    "video_id": video_id,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "text": comment.get("textDisplay"),
                    "like_count": comment.get("likeCount"),
                    "published_at": comment.get("publishedAt"),
                    "author_channel_id": comment.get("authorChannelId", {}).get("value"),
                    "fetch_order": order,
                })
                collected += 1

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.1)

    # 去重（同一則留言可能同時出現在 relevance 和 time 排序裡）
    seen_ids = set()
    deduped = []
    for c in comments:
        if c["comment_id"] not in seen_ids:
            seen_ids.add(c["comment_id"])
            deduped.append(c)

    return deduped


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """附加寫入，而不是每次都整批覆蓋。這樣就算跑到一半中斷，
    已經寫進檔案的資料不會遺失。"""
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_queries": [], "seen_video_ids": []}


def save_progress(progress: Dict[str, Any]) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    os.makedirs("data_raw", exist_ok=True)

    youtube = get_youtube_client()

    progress = load_progress()
    completed = set(progress["completed_queries"])
    seen_video_ids = set(progress["seen_video_ids"])

    is_fresh_run = len(completed) == 0
    if is_fresh_run:
        # 全新開始，確保輸出檔案是空的（避免跟舊的殘留內容混在一起）
        open(VIDEOS_PATH, "w", encoding="utf-8").close()
        open(COMMENTS_PATH, "w", encoding="utf-8").close()
    else:
        print(f"偵測到先前進度：已完成 {len(completed)} 個 query，將從中斷處繼續。")

    remaining = [(q, d, t) for (q, d, t) in QUERY_LIST if q not in completed]

    print(f"Total queries: {len(QUERY_LIST)}")
    print(f"  已完成: {len(completed)}")
    print(f"  剩餘待跑: {len(remaining)}")
    print(f"  Track A (organic, no-slang): {sum(1 for _, _, t in remaining if t == 'track_a_organic')}")
    print(f"  Track B (targeted boost):    {sum(1 for _, _, t in remaining if t == 'track_b_boost')}")
    print(f"  Track C (rare-term boost):   {sum(1 for _, _, t in remaining if t == 'track_c_rare_slang_boost')}")

    # 粗估 quota：每個 query 的 search.list 約 100 units/次(每50筆一頁)，
    # commentThreads.list 屬於 quota 較低的 list 呼叫。收集前建議先確認
    # 每日 quota 上限（預設 10,000 units）是否足夠跑完整批。

    total_new_videos = 0
    total_new_comments = 0
    stopped_early = False

    for query, domain, track in tqdm(remaining, desc="Searching YouTube"):
        query_videos = []
        query_comments = []

        try:
            videos = search_videos(youtube, query=query, max_results=50)

            for video in videos:
                if video["video_id"] in seen_video_ids:
                    continue

                seen_video_ids.add(video["video_id"])
                video["collection_domain"] = domain
                video["collection_track"] = track
                query_videos.append(video)

                comments = get_comments(youtube, video["video_id"], max_comments=100)

                for comment in comments:
                    comment["query"] = query
                    comment["video_url"] = video["url"]
                    comment["video_title"] = video["title"]
                    comment["collection_domain"] = domain
                    comment["collection_track"] = track

                query_comments.extend(comments)
                time.sleep(0.2)

        except QuotaExceededError as e:
            print(f"\n⚠️ Quota 已用盡，停止收集：{e}")
            print(f"目前這個 query（{query!r}）不會被標記為已完成，下次執行會從這裡重跑。")
            print("已收集的資料都已存檔，不會遺失。YouTube API quota 通常在美國太平洋時間"
                  "午夜重置，可以明天再繼續跑同一支腳本。")
            stopped_early = True
            break

        # 每跑完一個 query 就立刻存檔，不等全部跑完才寫檔
        append_jsonl(VIDEOS_PATH, query_videos)
        append_jsonl(COMMENTS_PATH, query_comments)

        total_new_videos += len(query_videos)
        total_new_comments += len(query_comments)

        completed.add(query)
        progress["completed_queries"] = list(completed)
        progress["seen_video_ids"] = list(seen_video_ids)
        save_progress(progress)

    grand_total_videos = count_lines(VIDEOS_PATH)
    grand_total_comments = count_lines(COMMENTS_PATH)

    print(f"\n本次新增 videos: {total_new_videos}, comments: {total_new_comments}")
    print(f"累積總計 videos: {grand_total_videos}, comments: {grand_total_comments}")
    print(f"Output folder: data_raw/")

    if stopped_early:
        print(f"\n尚未跑完，剩餘 {len(QUERY_LIST) - len(completed)} 個 query。"
              f"直接重新執行這支腳本即可自動從中斷處繼續，不會重複收集。")
    else:
        print("\n✅ 全部 query 已完成。")

    # domain 分布快速檢查（收集階段就先看一次，及早發現偏誤）
    from collections import Counter
    import json as _json

    domain_counter = Counter()
    track_counter = Counter()
    with open(VIDEOS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = _json.loads(line)
            domain_counter[item.get("collection_domain")] += 1
            track_counter[item.get("collection_track")] += 1

    print("\nVideo count by collection_domain (累積至今):")
    print(domain_counter)
    print("\nVideo count by collection_track (累積至今):")
    print(track_counter)


if __name__ == "__main__":
    main()