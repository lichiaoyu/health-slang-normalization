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
BATCH_NAME = "batch_health_broad_01"

if not YOUTUBE_API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY. Please check your .env file.")


BROAD_HEALTH_QUERIES = [
    # Medication / GLP-1 / side effects
    '"ozempic side effects" #shorts',
    '"glp1 journey" #shorts',
    '"mounjaro experience" #shorts',
    '"semaglutide nausea" #shorts',
    '"weight loss medication" #shorts',
    '"medication side effects" #shorts',

    # Supplements
    '"supplement review" #shorts',
    '"protein powder review" #shorts',
    '"creatine transformation" #shorts',
    '"pre workout review" #shorts',
    '"pre workout side effects" #shorts',
    '"fat burner review" #shorts',
    '"greens powder review" #shorts',
    '"ashwagandha review" #shorts',
    '"magnesium sleep" #shorts',
    '"melatonin review" #shorts',

    # Diet / weight loss / food behavior
    '"calorie deficit meals" #shorts',
    '"what I eat in a day weight loss" #shorts',
    '"weight loss journey" #shorts',
    '"intermittent fasting results" #shorts',
    '"keto diet results" #shorts',
    '"high protein meals" #shorts',
    '"girl dinner" #shorts',
    '"meal prep weight loss" #shorts',

    # Fitness / GymTok
    '"gymtok" #shorts',
    '"gym transformation" #shorts',
    '"body recomp" #shorts',
    '"bulking diet" #shorts',
    '"cutting diet" #shorts',
    '"leg day" #shorts',
    '"gym motivation" #shorts',
    '"fitness transformation" #shorts',

    # Gut health / digestion
    '"gut health tips" #shorts',
    '"bloating remedies" #shorts',
    '"probiotics review" #shorts',
    '"IBS symptoms" #shorts',
    '"constipation relief" #shorts',
    '"digestion problems" #shorts',
    '"detox drink" #shorts',

    # Sleep / fatigue
    '"sleep tips" #shorts',
    '"insomnia tips" #shorts',
    '"melatonin gummies" #shorts',
    '"sleep supplement" #shorts',
    '"chronic fatigue" #shorts',
    '"energy crash" #shorts',

    # Mental / cognitive state
    '"brain fog" #shorts',
    '"anxiety symptoms" #shorts',
    '"burnout symptoms" #shorts',
    '"cortisol levels" #shorts',
    '"hormone imbalance" #shorts',
    '"PCOS symptoms" #shorts',

    # Body image / transformation
    '"weight loss glow up" #shorts',
    '"body transformation" #shorts',
    '"before and after weight loss" #shorts',
    '"snatched waist" #shorts',
    '"summer body" #shorts',
]

SLANG_TARGETED_QUERIES = [
    # ambiguous physical discomfort
    '"side effects" "fighting for my life" #shorts',
    '"side effects" "messed me up" #shorts',
    '"side effects" "wrecked me" #shorts',
    '"side effects" "took me out" #shorts',
    '"supplement" "had me dying" #shorts',
    '"pre workout" "fighting for my life" #shorts',
    '"pre workout" "messed me up" #shorts',

    # sleep / fatigue slang
    '"melatonin" "knocked me out" #shorts',
    '"sleep supplement" "knocked me out" #shorts',
    '"magnesium" "knocked me out" #shorts',
    '"ashwagandha" "messed me up" #shorts',

    # diet / body culture slang
    '"girl dinner" "calorie deficit" #shorts',
    '"weight loss" "glow up" #shorts',
    '"fitness" "glow up" #shorts',
    '"gymtok" "locked in" #shorts',
    '"gymtok" "cooked" #shorts',
    '"weight loss" "snatched" #shorts',

    # food/product review slang
    '"protein powder" "bussin" #shorts',
    '"gut health" "hits different" #shorts',
    '"healthy recipe" "bussin" #shorts',
    '"meal prep" "hits different" #shorts',
]

HEALTH_QUERIES = BROAD_HEALTH_QUERIES + SLANG_TARGETED_QUERIES

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

def get_comments(youtube, video_id: str, max_comments: int = 100):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                textFormat="plainText",
                order="relevance",
                pageToken=next_page_token
            ).execute()
        except HttpError:
            return comments

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "video_id": video_id,
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "text": comment.get("textDisplay"),
                "like_count": comment.get("likeCount"),
                "published_at": comment.get("publishedAt"),
                "author_channel_id": comment.get("authorChannelId", {}).get("value"),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)

    return comments

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    os.makedirs("data_raw", exist_ok=True)

    youtube = get_youtube_client()

    all_videos = []
    all_comments = []
    seen_video_ids = set()

    for query in tqdm(HEALTH_QUERIES, desc="Searching YouTube"):
        videos = search_videos(youtube, query=query, max_results=50)

        for video in videos:
            if video["video_id"] in seen_video_ids:
                continue

            seen_video_ids.add(video["video_id"])
            all_videos.append(video)

            comments = get_comments(youtube, video["video_id"], max_comments=100)

            for comment in comments:
                comment["query"] = query
                comment["video_url"] = video["url"]
                comment["video_title"] = video["title"]

            all_comments.extend(comments)
            time.sleep(0.2)

    if len(all_videos) == 0 and len(all_comments) == 0:
        print("No data collected. Skip writing output files to avoid overwriting old data.")
        return

    write_jsonl(f"data_raw/{BATCH_NAME}_youtube_videos.jsonl", all_videos)
    write_jsonl(f"data_raw/{BATCH_NAME}_youtube_comments.jsonl", all_comments)
    print(f"Saved {len(all_videos)} unique videos.")
    print(f"Saved {len(all_comments)} comments.")
    print("Output folder: data_raw/")


if __name__ == "__main__":
    main()